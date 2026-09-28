use arko_index::{backend::*, chunker::*, packing::*, source::*, *};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

/// How the fake backend rejects the chunk responses it is told to reject.
#[derive(Clone, Copy)]
enum Rejection {
    /// A wrong tool call: transient, retried after the usual backoff.
    WrongCall,
    /// A wrong tool call the provider asks to retry in an hour.
    WrongCallRetryLater,
    /// A refusal: permanent.
    Refusal,
}
struct Fake {
    profile: EmbeddingProfile,
    calls: Mutex<Vec<ModelRequest>>,
    embeds: AtomicUsize,
    queries: AtomicUsize,
    active: AtomicUsize,
    max: AtomicUsize,
    finished: AtomicUsize,
    delay: Duration,
    fail_file: Mutex<Option<String>>,
    invalid_embedding: AtomicUsize,
    invalid_repair_replies: AtomicUsize,
    /// When > 1, each chunk call waits (bounded) until this many calls have overlapped,
    /// so concurrency assertions don't depend on how fast the host schedules work.
    overlap: usize,
    /// Chunk calls that return a billed response the backend rejects (see
    /// `rejection`), before normal replies.
    rejected_chunks: AtomicUsize,
    rejection: Rejection,
    /// Embedding requests that return a billed response the backend rejects
    /// (transient), before normal replies.
    rejected_embeds: AtomicUsize,
    /// Report a NaN cost with each chunk reply.
    nan_cost: bool,
    /// What the backend reports about the chunk model.
    chunk_input: ChunkInput,
}
impl Fake {
    fn new() -> Self {
        Self {
            profile: EmbeddingProfile {
                dimensions: 2,
                ..Default::default()
            },
            calls: Mutex::new(vec![]),
            embeds: AtomicUsize::new(0),
            queries: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            max: AtomicUsize::new(0),
            finished: AtomicUsize::new(0),
            delay: Duration::from_millis(10),
            fail_file: Mutex::new(None),
            invalid_embedding: AtomicUsize::new(0),
            invalid_repair_replies: AtomicUsize::new(0),
            overlap: 0,
            rejected_chunks: AtomicUsize::new(0),
            rejection: Rejection::WrongCall,
            rejected_embeds: AtomicUsize::new(0),
            nan_cost: false,
            chunk_input: ChunkInput::default(),
        }
    }
}
#[async_trait::async_trait]
impl Backend for Fake {
    async fn chunk(&self, r: ModelRequest) -> Result<ModelReply> {
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max.fetch_max(active, Ordering::SeqCst);
        self.calls.lock().unwrap().push(r.clone());
        let overlap_deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while self.max.load(Ordering::SeqCst) < self.overlap
            && tokio::time::Instant::now() < overlap_deadline
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        tokio::time::sleep(self.delay).await;
        self.active.fetch_sub(1, Ordering::SeqCst);
        if self
            .rejected_chunks
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
            .is_ok()
        {
            let wrong_call = arko_ai::Error::UnexpectedToolCalls {
                expected: r.tool.clone(),
                stop_reason: arko_ai::StopReason::EndTurn,
                found: vec![],
            };
            let error = match self.rejection {
                Rejection::WrongCall => Error::from(wrong_call),
                Rejection::WrongCallRetryLater => Error::Provider {
                    message: wrong_call.to_string(),
                    transient: true,
                    retry_after: Some(Duration::from_secs(3600)),
                    source: None,
                    usage: None,
                },
                Rejection::Refusal => Error::from(arko_ai::Error::Refusal(Box::new(
                    arko_ai::Refusal::new(arko_ai::RefusalSource::Model),
                ))),
            };
            return Err(error.with_usage(chunk_usage()));
        }
        self.finished.fetch_add(1, Ordering::SeqCst);
        let mut out = serde_json::Map::new();
        for (id, _) in r.schema["properties"].as_object().unwrap() {
            if r.tool == "repair_cuts" {
                let invalid = self
                    .invalid_repair_replies
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                    .is_ok();
                out.insert(
                    id.clone(),
                    if invalid {
                        json!(["INVALID_BOUNDARY"])
                    } else {
                        json!([])
                    },
                );
            } else if self
                .fail_file
                .lock()
                .unwrap()
                .as_ref()
                .is_some_and(|p| r.prompt.contains(p))
            {
                out.insert(
                    id.clone(),
                    json!({"cuts":[{"after_line":99999,"tail":"bad"}]}),
                );
            } else {
                out.insert(id.clone(), json!({"cuts":[]}));
            }
        }
        let mut usage = chunk_usage();
        if self.nan_cost {
            usage.total_cost = f64::NAN;
            usage.billed_cost = Some(f64::NAN);
        }
        Ok(ModelReply {
            value: Value::Object(out),
            usage: json!({}),
            usage_info: Some(usage),
        })
    }
    async fn embed(&self, texts: Vec<String>, query: bool) -> Result<VectorReply> {
        if query {
            self.queries.fetch_add(1, Ordering::SeqCst);
        } else {
            self.embeds.fetch_add(texts.len(), Ordering::SeqCst);
        }
        assert_eq!(
            self.active.load(Ordering::SeqCst),
            0,
            "embedding crossed chunking barrier"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
        if !query
            && self
                .rejected_embeds
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                .is_ok()
        {
            return Err(Error::Provider {
                message: "rejected embedding response".into(),
                transient: true,
                retry_after: None,
                source: None,
                usage: Some(Box::new(arko_ai::UsageInfo::for_embedding(500, 0.06))),
            });
        }
        let invalid = self.invalid_embedding.load(Ordering::SeqCst) > 0;
        Ok(VectorReply {
            vectors: texts
                .into_iter()
                .map(|t| {
                    if invalid {
                        vec![0.0]
                    } else if t.contains("alpha") {
                        vec![1.0, 0.0]
                    } else {
                        vec![0.0, 1.0]
                    }
                })
                .collect(),
            billed_tokens: 0,
            usage_info: Some(arko_ai::UsageInfo::for_embedding(1_000, 0.06)),
        })
    }
    fn tokens(&self, text: &str) -> Result<usize> {
        Ok(text.chars().count().div_ceil(4))
    }
    fn limits(&self) -> Limits {
        Limits {
            item_tokens: 32000,
            batch_tokens: 320000,
            batch_items: 1000,
        }
    }
    fn profile(&self) -> &EmbeddingProfile {
        &self.profile
    }
    fn output_tokenizer(&self) -> String {
        "fake-chars-div-4".into()
    }
    fn chunk_model_info(&self) -> Option<ChunkModelInfo> {
        Some(ChunkModelInfo {
            model: "fake-chunker".into(),
            reasoning: json!("none"),
        })
    }
    async fn chunk_input(&self) -> ChunkInput {
        self.chunk_input
    }
}
/// What each fake chunk call reports: a subscription call with a list-price estimate.
fn chunk_usage() -> arko_ai::UsageInfo {
    arko_ai::UsageInfo {
        total_cost: 0.25,
        ..arko_ai::UsageInfo::new(100, 10)
    }
    .with_billing(arko_ai::BillingKind::Subscription)
}
fn config(fake: &Fake) -> Config {
    Config {
        embedding: fake.profile.clone(),
        ..Default::default()
    }
}
async fn settle(index: &ArkoIndex, job: &str) -> JobStatus {
    tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            let s = index.status(job).await.unwrap();
            if matches!(
                s.phase.as_str(),
                "ready" | "blocked" | "paused" | "failed" | "cancelled"
            ) {
                return s;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("test job hung")
}
/// Every eligible result in `mode`, with no size filter.
fn options(mode: SearchMode) -> SearchOptions {
    SearchOptions {
        mode,
        top_k: None,
        min_chunk_bytes: 0,
        rebuild_file_limit: None,
        ..Default::default()
    }
}

struct TestJudge<F> {
    calls: AtomicUsize,
    select: F,
}
impl<F> TestJudge<F> {
    fn new(select: F) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            select,
        }
    }
}
#[async_trait::async_trait]
impl<F> SearchJudge for TestJudge<F>
where
    F: Fn(usize, &[SearchHit]) -> Result<JudgeSelection> + Send + Sync,
{
    async fn select(&self, query: &str, hits: &[SearchHit]) -> Result<JudgeSelection> {
        assert_eq!(query, "alpha");
        (self.select)(self.calls.fetch_add(1, Ordering::SeqCst), hits)
    }
}
async fn judge_fixture() -> (
    tempfile::TempDir,
    tempfile::TempDir,
    ArkoIndex,
    JobHandle,
    Arc<Fake>,
) {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    for i in 0..5 {
        std::fs::write(
            root.path().join(format!("{i}.mdx")),
            format!("alpha example {i} with complete instructions and source code"),
        )
        .unwrap();
    }
    std::fs::write(root.path().join("tiny.mdx"), "alpha").unwrap();
    let fake = Arc::new(Fake::new());
    let index = ArkoIndex::open(db.path().join("i.db"), fake.clone(), config(&fake))
        .await
        .unwrap();
    let job = index
        .start_build(BuildSpec::new("judge-test", root.path()))
        .await
        .unwrap();
    assert_eq!(settle(&index, &job.job).await.phase, "ready");
    (root, db, index, job, fake)
}

#[tokio::test]
async fn judged_search_filters_candidates_then_caps_and_preserves_sources() {
    let (_root, _db, index, job, fake) = judge_fixture().await;
    for mode in [
        SearchMode::Lexical,
        SearchMode::Semantic,
        SearchMode::Hybrid,
    ] {
        let opts = SearchOptions {
            mode,
            top_k: Some(1),
            min_chunk_bytes: 40,
            ..Default::default()
        };
        let raw = index
            .search(
                &job.index,
                "alpha",
                SearchOptions {
                    top_k: Some(3),
                    ..opts.clone()
                },
            )
            .await
            .unwrap();
        assert!(raw.judge.is_none());
        assert!(serde_json::to_value(&raw).unwrap().get("judge").is_none());
        let judge = TestJudge::new(|_, hits: &[SearchHit]| {
            assert_eq!(hits.len(), 3);
            assert!(hits.iter().all(|h| h.text.len() >= 40));
            Ok(JudgeSelection {
                ordered_ids: vec![hits[2].id.clone(), hits[0].id.clone()],
                excluded: vec![JudgeExclusion {
                    id: hits[1].id.clone(),
                    reason: "unrelated".into(),
                }],
            })
        });
        let page = index
            .search_with_judge(
                &job.index,
                "alpha",
                opts,
                &judge,
                JudgeOptions {
                    candidate_limit: 3,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(page.hits.len(), 1);
        assert_eq!(
            serde_json::to_value(&page.hits[0]).unwrap(),
            serde_json::to_value(&raw.hits[2]).unwrap()
        );
        assert_eq!(page.generation, raw.generation);
        assert_eq!(page.mode, raw.mode);
        assert_eq!(page.search_tokens, fake.tokens(&page.hits[0].text).unwrap());
        assert_eq!(page.candidate_count, 5);
        assert_eq!(page.omitted_hits, 4);
        assert!(page.next.is_none());
        let report = page.judge.unwrap();
        assert_eq!(
            (report.examined_candidates, report.unexamined_candidates),
            (3, 2)
        );
        assert_eq!(
            (
                report.selected_count,
                report.selected_beyond_limit,
                report.excluded.len(),
                report.attempts
            ),
            (2, 1, 1, 1)
        );
        // Judging must not mutate the database or raw ranking/paging.
        let again = index
            .search(
                &job.index,
                "alpha",
                SearchOptions {
                    mode: raw.mode,
                    top_k: Some(3),
                    min_chunk_bytes: 40,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(
            serde_json::to_value(again.hits).unwrap(),
            serde_json::to_value(raw.hits).unwrap()
        );
        assert!(again.next.is_some());
    }
}

#[tokio::test]
async fn judged_search_rejects_invalid_partitions_and_never_hides_provider_errors() {
    let (_root, _db, index, job, _) = judge_fixture().await;
    let opts = options(SearchMode::Lexical);
    let judging = JudgeOptions {
        candidate_limit: 3,
        max_attempts: 2,
    };
    for failure in [
        "unknown",
        "duplicate",
        "missing",
        "empty-reason",
        "invalid-json",
    ] {
        let judge = TestJudge::new(move |_, hits: &[SearchHit]| {
            let mut selection = JudgeSelection {
                ordered_ids: hits.iter().map(|h| h.id.clone()).collect(),
                excluded: vec![],
            };
            match failure {
                "unknown" => selection.ordered_ids[0] = "invented-id".into(),
                "duplicate" => selection.ordered_ids[1] = selection.ordered_ids[0].clone(),
                "missing" => {
                    selection.ordered_ids.pop();
                }
                "empty-reason" => selection.excluded.push(JudgeExclusion {
                    id: selection.ordered_ids.pop().unwrap(),
                    reason: " ".into(),
                }),
                "invalid-json" => return Ok(serde_json::from_str("{broken")?),
                _ => unreachable!(),
            }
            Ok(selection)
        });
        assert!(
            index
                .search_with_judge(&job.index, "alpha", opts.clone(), &judge, judging.clone())
                .await
                .is_err(),
            "{failure}"
        );
        assert_eq!(judge.calls.load(Ordering::SeqCst), 2);
    }
    let recovering = TestJudge::new(|attempt, hits: &[SearchHit]| {
        if attempt == 0 {
            return Err(Error::Invalid("invalid first response".into()));
        }
        Ok(JudgeSelection {
            ordered_ids: hits.iter().map(|h| h.id.clone()).collect(),
            excluded: vec![],
        })
    });
    let page = index
        .search_with_judge(
            &job.index,
            "alpha",
            opts.clone(),
            &recovering,
            judging.clone(),
        )
        .await
        .unwrap();
    assert_eq!(page.judge.unwrap().attempts, 2);
    assert_eq!(page.hits.len(), 3);
    let provider_failure = TestJudge::new(|_, _: &[SearchHit]| {
        Err(Error::Provider {
            message: "rate limit".into(),
            transient: true,
            retry_after: Some(Duration::from_secs(2)),
            source: None,
            usage: None,
        })
    });
    assert!(matches!(
        index
            .search_with_judge(&job.index, "alpha", opts, &provider_failure, judging)
            .await,
        Err(Error::Provider {
            transient: true,
            ..
        })
    ));
    assert_eq!(provider_failure.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn judged_search_handles_empty_results_filters_and_invalid_options() {
    let (_root, _db, index, job, _) = judge_fixture().await;
    let reject = TestJudge::new(|_, hits: &[SearchHit]| {
        Ok(JudgeSelection {
            ordered_ids: vec![],
            excluded: hits
                .iter()
                .map(|h| JudgeExclusion {
                    id: h.id.clone(),
                    reason: "no useful answer".into(),
                })
                .collect(),
        })
    });
    let opts = SearchOptions {
        mode: SearchMode::Lexical,
        path_prefix: Some("0.mdx".into()),
        literal: Some("complete instructions".into()),
        ..Default::default()
    };
    let page = index
        .search_with_judge(
            &job.index,
            "alpha",
            opts.clone(),
            &reject,
            JudgeOptions::default(),
        )
        .await
        .unwrap();
    assert!(page.hits.is_empty());
    assert!(page.next.is_none());
    assert_eq!(page.search_tokens, 0);
    assert_eq!(page.judge.unwrap().excluded.len(), 1);
    let empty = index
        .search_with_judge(
            &job.index,
            "alpha",
            SearchOptions {
                path_prefix: Some("absent/".into()),
                ..opts.clone()
            },
            &reject,
            JudgeOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(empty.judge.unwrap().attempts, 0);
    assert_eq!(reject.calls.load(Ordering::SeqCst), 1);
    for (top_k, candidate_limit, max_attempts) in
        [(Some(0), 2, 1), (Some(3), 2, 1), (None, 0, 1), (None, 2, 0)]
    {
        assert!(matches!(
            index
                .search_with_judge(
                    &job.index,
                    "alpha",
                    SearchOptions {
                        top_k,
                        ..opts.clone()
                    },
                    &reject,
                    JudgeOptions {
                        candidate_limit,
                        max_attempts
                    }
                )
                .await,
            Err(Error::Invalid(_))
        ));
    }
    let raw = index
        .search(
            &job.index,
            "alpha",
            SearchOptions {
                top_k: Some(1),
                ..options(SearchMode::Lexical)
            },
        )
        .await
        .unwrap();
    assert!(matches!(
        index
            .search_with_judge(
                &job.index,
                "alpha",
                SearchOptions {
                    cursor: raw.next,
                    ..opts
                },
                &reject,
                JudgeOptions::default()
            )
            .await,
        Err(Error::Invalid(_))
    ));
    assert_eq!(reject.calls.load(Ordering::SeqCst), 1);
}
#[test]
fn normalized_rows_and_exact_unbounded_tails() {
    let s = Snapshot::new("a.rs", "α\r\nβ\rγ\n\nend".as_bytes()).unwrap();
    assert_eq!(s.text, "α\nβ\nγ\n\nend");
    assert_eq!(s.rows(), 5);
    assert_eq!(s.line_starts, vec![0, 3, 6, 9, 10]);
    assert!(s.tail_matches(2, "α\nβ"));
    assert!(!s.tail_matches(2, "β\n"));
    let hundred = Snapshot::new("a.txt", ("line\n".repeat(100)).as_bytes()).unwrap();
    assert!(hundred.numbered(1, 2).starts_with("001 | line\n002 |"));
    assert_eq!(hundred.rows(), 100);
    let long = "x".repeat(5000);
    let s = Snapshot::new("a.txt", format!("{long}\nnext").as_bytes()).unwrap();
    assert!(s.tail_matches(1, &long));
    assert_eq!(Snapshot::new("empty", b"").unwrap().rows(), 0);
}
#[test]
fn guard_and_targeted_repair_preserve_accepted_cuts() {
    let s = Snapshot::new(
        "x.rs",
        b"fn one() {\n    println!(\"x\");\n}\n\nfn two() {}\n\nfn three() {}\n",
    )
    .unwrap();
    let guard = structure(&s);
    let w = Window::new(0, s.text.len());
    let cuts = vec![
        Cut {
            after_line: 2,
            tail: ";".into(),
        },
        Cut {
            after_line: 5,
            tail: "}".into(),
        },
    ];
    assert_eq!(
        failures(&s, &guard, &w, &cuts)
            .into_iter()
            .collect::<Vec<_>>(),
        vec![0]
    );
    let repair = repair(&s, &guard, &w, &cuts).unwrap();
    assert!(!repair.regions["r0"].candidates.contains_key("L2"));
    let fixed = apply_repair(&s, &guard, &w, &cuts, &repair, &json!({"r0":["L3"]})).unwrap();
    assert_eq!(fixed[1], cuts[1]);
    assert!(failures(&s, &guard, &w, &fixed).is_empty());
    let reconstructed: String = spans(&s, &guard, &w, &fixed)
        .unwrap()
        .into_iter()
        .map(|r| s.text[r].to_owned())
        .collect();
    assert_eq!(reconstructed, s.text);
    assert!(apply_repair(&s, &guard, &w, &fixed, &repair, &json!({"r0":["L3"]})).is_err());
    assert!(apply_repair(&s, &guard, &w, &cuts, &repair, &json!({"r0":["L2"]})).is_err());
}
#[test]
fn impossible_cut_removal_requires_the_entire_interval_and_preserves_source() {
    let s = Snapshot::new(
        "x.rs",
        b"fn first() {}\n\nfn last() {\n    println!(\"last\");\n}\n",
    )
    .unwrap();
    let guard = structure(&s);
    let w = Window::new(0, s.text.len());
    let cuts = vec![
        Cut {
            after_line: 1,
            tail: "}".into(),
        },
        Cut {
            after_line: 4,
            tail: ";".into(),
        },
    ];
    let fixed = remove_impossible_cuts(&s, &guard, &w, &cuts).unwrap();
    assert_eq!(fixed, cuts[..1]);
    let text: String = spans(&s, &guard, &w, &fixed)
        .unwrap()
        .into_iter()
        .map(|r| &s.text[r])
        .collect();
    assert_eq!(text, s.text);
    // EOF is implicit even if the anchor itself matches it exactly.
    let eof = vec![
        cuts[0].clone(),
        Cut {
            after_line: 5,
            tail: "}".into(),
        },
    ];
    assert_eq!(remove_impossible_cuts(&s, &guard, &w, &eof).unwrap(), fixed);
    // No local candidates is insufficient: a legal cut exists remotely.
    let text = format!("legal\n{}", "body\n".repeat(200));
    let s = Snapshot::new("x.txt", text.as_bytes()).unwrap();
    let guard = Structure {
        status: "test".into(),
        protected: std::iter::once(s.line_end(1).unwrap()..s.text.len()).collect(),
    };
    let w = Window::new(0, s.text.len());
    let cuts = vec![Cut {
        after_line: 100,
        tail: "body".into(),
    }];
    assert_eq!(
        repair(&s, &guard, &w, &cuts).unwrap().regions["r0"]
            .candidates
            .len(),
        1
    );
    assert_eq!(remove_impossible_cuts(&s, &guard, &w, &cuts).unwrap(), cuts);
}
#[test]
fn grouped_repairs_have_complete_context_and_preserve_multiple_boundaries() {
    let text: String = (1..=1000).map(|i| format!("source-{i}\n")).collect();
    let s = Snapshot::new("x.txt", text.as_bytes()).unwrap();
    let guard = structure(&s);
    let w = Window::new(0, s.text.len());
    let cuts = vec![
        Cut {
            after_line: 100,
            tail: "wrong".into(),
        },
        Cut {
            after_line: 400,
            tail: "wrong".into(),
        },
        Cut {
            after_line: 900,
            tail: "source-900".into(),
        },
    ];
    let r = repair(&s, &guard, &w, &cuts).unwrap();
    for n in [1, 100, 400, 650, 800, 1000] {
        assert!(r.request.prompt.contains(&format!("{n:04} | source-{n}")));
    }
    assert!(r.regions["r0"].candidates.contains_key("L800"));
    assert!(!r.regions["r0"].candidates.contains_key("L900"));
    let expected_ids: Vec<_> = (1..900).map(|row| format!("L{row}")).collect();
    let menu = r
        .request
        .prompt
        .lines()
        .find_map(|line| line.strip_prefix("Legal boundary IDs="))
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Vec<String>>(menu).unwrap(),
        expected_ids
    );
    assert_eq!(
        r.request.schema["properties"]["r0"]["items"]["enum"],
        json!(expected_ids)
    );
    let fixed = apply_repair(&s, &guard, &w, &cuts, &r, &json!({"r0":["L100","L400"]})).unwrap();
    assert_eq!(
        fixed.iter().map(|c| c.after_line).collect::<Vec<_>>(),
        vec![100, 400, 900]
    );
    assert!(failures(&s, &guard, &w, &fixed).is_empty());
    assert_eq!(fixed[2], cuts[2]);
    assert!(apply_repair(&s, &guard, &w, &cuts, &r, &json!({"r0":["L400","L100"]})).is_err());
    assert!(apply_repair(&s, &guard, &w, &cuts, &r, &json!({"r0":["L100","L100"]})).is_err());
    let removed = apply_repair(&s, &guard, &w, &cuts, &r, &json!({"r0":[]})).unwrap();
    assert_eq!(removed, vec![cuts[2].clone()]);
}
#[test]
fn complete_import_statements_are_protected_and_can_be_grouped() {
    let text = "/// Platform imports.\n#[cfg(any(unix, windows))]\nuse crate::{\n    first::Thing,\n    second::{One, Two as Alias},\n};\n\npub use other::{\n    A,\n    B,\n};\n\nfn unrelated() {}\n";
    let s = Snapshot::new("imports.rs", text.as_bytes()).unwrap();
    let guard = structure(&s);
    assert_eq!(guard.status, "rust-guarded");
    for row in [1, 2, 3, 4, 5, 8, 9, 10] {
        assert!(
            !guard.legal(s.line_end(row).unwrap()),
            "import split after {row}"
        );
    }
    assert!(guard.legal(s.line_end(6).unwrap()));
    assert!(guard.legal(s.line_end(11).unwrap()));
    let w = Window::new(0, s.text.len());
    let cuts = vec![Cut {
        after_line: 3,
        tail: "{".into(),
    }];
    let r = repair(&s, &guard, &w, &cuts).unwrap();
    assert!(!r.regions["r0"].candidates.contains_key("L3"));
    let fixed = apply_repair(&s, &guard, &w, &cuts, &r, &json!({"r0":["L11"]})).unwrap();
    let ranges = spans(&s, &guard, &w, &fixed).unwrap();
    assert_eq!(
        &s.text[ranges[0].clone()],
        &s.text[..s.line_end(11).unwrap()]
    );
    assert_eq!(
        ranges
            .iter()
            .map(|r| &s.text[r.clone()])
            .collect::<String>(),
        s.text
    );
    assert!(spans(&s, &guard, &w, &[]).is_ok());
}
#[test]
fn old_saved_configuration_does_not_restore_removed_limits() {
    let config: Config = serde_json::from_value(
        json!({"small_file_chars":2400,"max_repair_rounds":2,"request_byte_budget":700000}),
    )
    .unwrap();
    assert_eq!(config, Config::default());
    let stored = serde_json::to_value(config).unwrap();
    for key in [
        "small_file_chars",
        "max_repair_rounds",
        "request_byte_budget",
    ] {
        assert!(stored.get(key).is_none());
    }
}
#[test]
fn markdown_fences_headings_and_unsupported_languages() {
    let s = Snapshot::new(
        "x.md",
        b"# Title\n\nParagraph.\n\n```rust\nlet x = 1;\n```\n\nTail\n",
    )
    .unwrap();
    let guard = structure(&s);
    assert!(!guard.legal(s.line_end(1).unwrap()));
    assert!(!guard.legal(s.line_end(6).unwrap()));
    assert!(guard.legal(s.line_end(7).unwrap()));
    assert_eq!(
        structure(&Snapshot::new("x.py", b"def x():\n    pass\n").unwrap()).status,
        "unchecked"
    );
}
#[test]
fn packing_exact_caps_and_vector_validation() {
    let item = |id: &str, tokens| Payload {
        id: id.into(),
        tokens,
        text: id.into(),
    };
    let limits = Limits {
        item_tokens: 10,
        batch_tokens: 10,
        batch_items: 2,
    };
    let bins = pack(
        vec![item("a", 6), item("b", 6), item("c", 4), item("d", 4)],
        limits,
    )
    .unwrap();
    assert_eq!(
        bins.iter()
            .map(|b| b.iter().map(|p| p.id.as_str()).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        vec![vec!["a", "c"], vec!["b", "d"]]
    );
    assert!(pack(vec![item("x", 11)], limits).is_err());
    assert!(normalize(vec![0.0, 0.0], 2).is_err());
    assert!(normalize(vec![f32::NAN, 1.0], 2).is_err());
    assert!(normalize(vec![1.0], 2).is_err());
    assert_eq!(normalize(vec![3.0, 4.0], 2).unwrap(), vec![0.6, 0.8]);
}
#[test]
fn event_coalescing_uses_final_identity() {
    let old = BTreeMap::from([("a".into(), "same".into()), ("b".into(), "old".into())]);
    let new = BTreeMap::from([("c".into(), "same".into()), ("b".into(), "new".into())]);
    assert_eq!(
        coalesce(&old, &new),
        vec![
            FileChange::Modified { path: "b".into() },
            FileChange::Moved {
                from: "a".into(),
                to: "c".into()
            }
        ]
    );
    assert!(coalesce(&old, &old).is_empty());
    let duplicates = BTreeMap::from([("a".into(), "same".into()), ("d".into(), "same".into())]);
    let added = BTreeMap::from([("x".into(), "same".into())]);
    assert!(
        coalesce(&duplicates, &added)
            .iter()
            .all(|c| !matches!(c, FileChange::Moved { .. }))
    );
}
#[tokio::test]
async fn six_file_batches_barrier_and_hybrid_search() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    for n in 0..13 {
        std::fs::write(
            root.path().join(format!("file{n:02}.txt")),
            format!("alpha marker{n}\n{}", "context line\n".repeat(250)),
        )
        .unwrap();
    }
    std::fs::write(root.path().join("small.md"), "# Small\n\nalpha tips").unwrap();
    std::fs::write(root.path().join("empty.txt"), " \n").unwrap();
    let mut f = Fake::new();
    f.overlap = 2;
    let fake = Arc::new(f);
    let index = ArkoIndex::open(db.path().join("index.db"), fake.clone(), config(&fake))
        .await
        .unwrap();
    let handle = index
        .start_build(BuildSpec::new("test", root.path()))
        .await
        .unwrap();
    let status = settle(&index, &handle.job).await;
    assert_eq!(status.phase, "ready", "{:?}", status.problem);
    assert_eq!(status.validated, 15);
    {
        // The three batches are prepared concurrently, so they can reach the
        // backend in any order.
        let calls = fake.calls.lock().unwrap();
        let mut batches: Vec<_> = calls
            .iter()
            .map(|r| r.schema["properties"].as_object().unwrap().len())
            .collect();
        batches.sort_unstable();
        assert_eq!(batches, [1, 6, 6]);
    }
    assert_eq!(fake.max.load(Ordering::SeqCst), 2);
    assert_eq!(fake.embeds.load(Ordering::SeqCst), 14);
    let page = index
        .search(&handle.index, "alpha", options(SearchMode::Hybrid))
        .await
        .unwrap();
    assert!(!page.hits.is_empty());
    assert_eq!(page.hits.len(), 14);
    let ids: Vec<_> = page.hits.iter().map(|h| h.id.clone()).collect();
    let chunks = index
        .read_chunks(&handle.index, &page.generation, &ids, 0)
        .await
        .unwrap();
    assert!(chunks.iter().all(|c| c.text.contains("alpha")));
    let a = index.search(&handle.index, "alpha", options(SearchMode::Semantic));
    let b = index.search(&handle.index, "alpha", options(SearchMode::Semantic));
    let (a, b) = tokio::join!(a, b);
    a.unwrap();
    b.unwrap();
    assert_eq!(fake.queries.load(Ordering::SeqCst), 1);
}
#[tokio::test]
async fn rename_delete_atomic_generations_and_rebuild_warning() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("a.txt"), "alpha original").unwrap();
    std::fs::write(root.path().join("delete.txt"), "obsolete").unwrap();
    let mut f = Fake::new();
    f.delay = Duration::from_millis(150);
    let fake = Arc::new(f);
    let index = ArkoIndex::open(db.path().join("i.db"), fake.clone(), config(&fake))
        .await
        .unwrap();
    let first = index
        .start_build(BuildSpec::new("test", root.path()))
        .await
        .unwrap();
    assert_eq!(settle(&index, &first.job).await.phase, "ready");
    let before = fake.embeds.load(Ordering::SeqCst);
    let old = index
        .search(&first.index, "alpha", options(SearchMode::Lexical))
        .await
        .unwrap();
    std::fs::rename(root.path().join("a.txt"), root.path().join("renamed.txt")).unwrap();
    std::fs::remove_file(root.path().join("delete.txt")).unwrap();
    std::fs::write(root.path().join("new.txt"), "alpha new\n".repeat(400)).unwrap();
    let update = index.start_update(&first.index).await.unwrap();
    while index.status(&update.job).await.unwrap().active_chunking == 0 {
        tokio::task::yield_now().await;
    }
    let during = index
        .search(&first.index, "alpha", options(SearchMode::Lexical))
        .await
        .unwrap();
    assert_eq!(during.generation, first.job);
    assert_eq!(during.rebuild.as_ref().unwrap().affected_files.len(), 4);
    let limited = index
        .search(
            &first.index,
            "alpha",
            SearchOptions {
                rebuild_file_limit: Some(1),
                ..options(SearchMode::Lexical)
            },
        )
        .await
        .unwrap();
    assert_eq!(limited.rebuild.unwrap().omitted_files, 3);
    assert_eq!(settle(&index, &update.job).await.phase, "ready");
    assert_eq!(fake.embeds.load(Ordering::SeqCst), before + 1);
    let after = index
        .search(&first.index, "alpha", options(SearchMode::Lexical))
        .await
        .unwrap();
    assert_eq!(after.generation, update.job);
    assert!(
        after
            .hits
            .iter()
            .all(|h| h.path != "a.txt" && h.path != "delete.txt")
    );
    let old_text = index
        .read_chunks(&first.index, &first.job, &[old.hits[0].id.clone()], 0)
        .await
        .unwrap();
    assert_eq!(old_text[0].text, "alpha original");
    let no_change = index.start_update(&first.index).await.unwrap();
    assert_eq!(settle(&index, &no_change.job).await.phase, "ready");
    assert_eq!(fake.embeds.load(Ordering::SeqCst), before + 1);
}
#[tokio::test]
async fn repair_is_local_and_accepted_work_survives_pause_and_restart() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    let path = db.path().join("i.db");
    for n in 0..7 {
        std::fs::write(
            root.path().join(format!("{n}.txt")),
            format!("alpha {n}\n{}", "line\n".repeat(600)),
        )
        .unwrap();
    }
    let mut f = Fake::new();
    f.delay = Duration::from_millis(100);
    *f.fail_file.lock().unwrap() = Some("0.txt".into());
    let fake = Arc::new(f);
    let index = ArkoIndex::open(&path, fake.clone(), config(&fake))
        .await
        .unwrap();
    let h = index
        .start_build(BuildSpec::new("test", root.path()))
        .await
        .unwrap();
    while fake.active.load(Ordering::SeqCst) == 0 {
        tokio::task::yield_now().await;
    }
    index.pause(&h.job).await.unwrap();
    assert_eq!(settle(&index, &h.job).await.phase, "paused");
    assert_eq!(fake.embeds.load(Ordering::SeqCst), 0);
    drop(index);
    let index = ArkoIndex::open(&path, fake.clone(), config(&fake))
        .await
        .unwrap();
    index.resume(&h.job).await.unwrap();
    let done = settle(&index, &h.job).await;
    assert_eq!(done.phase, "ready", "{:?}", done.problem);
    let calls = fake.calls.lock().unwrap().clone();
    let repairs: Vec<_> = calls.iter().filter(|r| r.tool == "repair_cuts").collect();
    assert!(!repairs.is_empty());
    assert!(repairs.iter().all(|r| !r.prompt.contains("FILE f")));
    // Usage adds up every completed call, before the pause and after the
    // restart, and events carry each call's share.
    let usage = &done.usage;
    assert_eq!(usage.chunk_calls, calls.len());
    let chunking = usage.chunking.as_ref().unwrap();
    assert_eq!(chunking.input_tokens, 100 * calls.len());
    assert_eq!(chunking.billing, arko_ai::BillingKind::Subscription);
    assert_eq!(chunking.total_cost, 0.25 * calls.len() as f64);
    let events = index.events(&h.job, 0, 10_000).await.unwrap();
    // Requests the pause stopped before they reached the backend have none.
    let finished: Vec<_> = events
        .iter()
        .filter(|e| e.kind == "request_finished" && !e.detail["usage"].is_null())
        .collect();
    assert_eq!(finished.len(), calls.len());
    assert!(
        finished
            .iter()
            .all(|e| e.detail["usage"]["input_tokens"] == 100)
    );
    let embedded = events.iter().filter(|e| e.kind == "embedded").count();
    assert!(embedded > 0);
    assert_eq!(usage.embedding_calls, embedded);
    assert_eq!(
        usage.embedding.as_ref().unwrap().input_tokens,
        1_000 * embedded
    );
    assert_eq!(usage.unreported_calls, 0);
    assert!((usage.billed() - 0.00006 * embedded as f64).abs() < 1e-12);
}
/// Input tokens and number of the usages a job's events carry: accepted
/// chunk replies (`request_finished`), rejected chunk responses (failed
/// `request_attempt` events), and embedding responses, accepted or rejected.
fn event_usage(events: &[Event]) -> (usize, usize) {
    let mut usages: Vec<&Value> = vec![];
    for event in events {
        let detail = &event.detail;
        let rejected = detail["rejected_usage"].as_array().into_iter().flatten();
        match event.kind.as_str() {
            "request_finished" => usages.push(&detail["usage"]),
            "request_attempt" if detail["event"]["phase"] == "failed" => {
                usages.push(&detail["event"]["detail"]["usage"])
            }
            "embedded" | "embedding_rejected" => {
                usages.push(&detail["usage"]);
                usages.extend(rejected);
            }
            "embedding_failed" => usages.extend(rejected),
            _ => {}
        }
    }
    usages.retain(|usage| !usage.is_null());
    let tokens = usages.iter().map(|u| u["input_tokens"].as_u64().unwrap());
    (tokens.sum::<u64>() as usize, usages.len())
}
/// Summed over the events, usage matches the job's totals.
fn assert_events_match_totals(status: &JobStatus, events: &[Event]) {
    let usage = &status.usage;
    let responses =
        usage.chunk_calls + usage.embedding_calls + usage.rejected_calls - usage.unreported_calls;
    let tokens = usage.total().map_or(0, |total| total.input_tokens);
    assert_eq!(event_usage(events), (tokens, responses));
}
/// Waits until the job's chunk request has been rejected and is waiting to
/// retry.
async fn wait_for_retry(index: &ArkoIndex, job: &str) {
    tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            let events = index.events(job, 0, 1000).await.unwrap();
            if events
                .iter()
                .any(|e| e.kind == "request_attempt" && e.detail["event"]["phase"] == "retry_wait")
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("no retry wait")
}
/// Builds one small file with `fake` and returns the settled status and the
/// chunk request's artifact.
async fn build_one_file(fake: Arc<Fake>, config: Config) -> (JobStatus, Vec<Event>, Value) {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("a.txt"), "alpha\nbeta\n").unwrap();
    let index = ArkoIndex::open(db.path().join("i.db"), fake, config)
        .await
        .unwrap();
    let job = index
        .start_build(BuildSpec::new("test", root.path()))
        .await
        .unwrap();
    let status = settle(&index, &job.job).await;
    let events = index.events(&job.job, 0, 1000).await.unwrap();
    let finished: Vec<_> = events
        .iter()
        .filter(|e| e.kind == "request_finished")
        .collect();
    assert_eq!(finished.len(), 1);
    let artifact = index
        .request_artifact(&job.job, finished[0].detail["artifact"].as_i64().unwrap())
        .await
        .unwrap();
    (status, events, artifact)
}
#[tokio::test]
async fn rejected_chunk_responses_count_when_a_retry_succeeds() {
    let fake = Arc::new(Fake::new());
    fake.rejected_chunks.store(1, Ordering::SeqCst);
    let (status, events, artifact) = build_one_file(fake.clone(), config(&fake)).await;
    assert_eq!(status.phase, "ready", "{:?}", status.problem);
    assert_eq!(fake.calls.lock().unwrap().len(), 2);
    // Both billed responses count; only the second was accepted.
    let usage = &status.usage;
    assert_eq!((usage.chunk_calls, usage.rejected_calls), (1, 1));
    let chunking = usage.chunking.as_ref().unwrap();
    assert_eq!(chunking.input_tokens, 200);
    assert_eq!(chunking.total_cost, 0.5);
    let finished = events
        .iter()
        .find(|e| e.kind == "request_finished")
        .unwrap();
    assert_eq!(finished.detail["usage"]["input_tokens"], 100);
    assert_events_match_totals(&status, &events);
    let failures = artifact["response"]["usage"]["retry_failures"]
        .as_array()
        .unwrap();
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0]["usage"]["input_tokens"], 100);
}
#[tokio::test]
async fn rejected_chunk_responses_count_when_retries_run_out() {
    let fake = Arc::new(Fake::new());
    fake.rejected_chunks.store(usize::MAX, Ordering::SeqCst);
    let config = Config {
        max_transient_attempts: 2,
        ..config(&fake)
    };
    let (status, events, artifact) = build_one_file(fake.clone(), config).await;
    assert_eq!(status.phase, "blocked");
    assert_eq!(fake.calls.lock().unwrap().len(), 2);
    let usage = &status.usage;
    assert_eq!((usage.chunk_calls, usage.rejected_calls), (0, 2));
    assert_eq!(usage.chunking.as_ref().unwrap().input_tokens, 200);
    let finished = events
        .iter()
        .find(|e| e.kind == "request_finished")
        .unwrap();
    assert!(finished.detail["usage"].is_null());
    assert_events_match_totals(&status, &events);
    // The failed request's artifact keeps every attempt with its usage.
    assert!(artifact["response"].is_null());
    let failed: Vec<_> = artifact["attempts"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|a| a["phase"] == "failed")
        .collect();
    assert_eq!(failed.len(), 2);
    assert!(
        failed
            .iter()
            .all(|a| a["detail"]["usage"]["input_tokens"] == 100)
    );
}
#[tokio::test]
async fn a_permanent_rejection_counts_once_and_is_not_retried() {
    let mut fake = Fake::new();
    fake.rejection = Rejection::Refusal;
    let fake = Arc::new(fake);
    fake.rejected_chunks.store(1, Ordering::SeqCst);
    let (status, events, artifact) = build_one_file(fake.clone(), config(&fake)).await;
    assert_eq!(status.phase, "blocked");
    assert_eq!(fake.calls.lock().unwrap().len(), 1);
    let usage = &status.usage;
    assert_eq!((usage.chunk_calls, usage.rejected_calls), (0, 1));
    assert_eq!(usage.chunking.as_ref().unwrap().input_tokens, 100);
    assert_events_match_totals(&status, &events);
    let attempts = artifact["attempts"].as_array().unwrap();
    assert!(attempts.iter().all(|a| a["phase"] != "retry_wait"));
    let failed = attempts.iter().find(|a| a["phase"] == "failed").unwrap();
    assert_eq!(failed["detail"]["usage"]["input_tokens"], 100);
}
async fn start_one_file(
    fake: &Arc<Fake>,
    path: &std::path::Path,
    root: &std::path::Path,
) -> (ArkoIndex, JobHandle) {
    std::fs::write(root.join("a.txt"), "alpha\nbeta\n").unwrap();
    let index = ArkoIndex::open(path, fake.clone(), config(fake))
        .await
        .unwrap();
    let job = index
        .start_build(BuildSpec::new("test", root))
        .await
        .unwrap();
    (index, job)
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_job_seen_paused_can_be_resumed_at_once() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    // Every chunk request waits an hour to retry, so the job only stops when paused.
    let mut fake = Fake::new();
    fake.rejection = Rejection::WrongCallRetryLater;
    fake.delay = Duration::ZERO;
    let fake = Arc::new(fake);
    fake.rejected_chunks.store(usize::MAX, Ordering::SeqCst);
    let (index, job) = start_one_file(&fake, &db.path().join("i.db"), root.path()).await;
    for iteration in 0..500 {
        index.pause(&job.job).await.unwrap();
        // Resume the moment the pause shows, with no settling delay.
        let phase = loop {
            let phase = index.status(&job.job).await.unwrap().phase;
            if matches!(
                phase.as_str(),
                "paused" | "blocked" | "failed" | "cancelled"
            ) {
                break phase;
            }
            tokio::task::yield_now().await;
        };
        assert_eq!(phase, "paused", "iteration {iteration}");
        if let Err(error) = index.resume(&job.job).await {
            panic!("iteration {iteration}: resume after the pause showed: {error}");
        }
    }
    index.cancel(&job.job).await.unwrap();
    assert_eq!(settle(&index, &job.job).await.phase, "cancelled");
}
#[tokio::test]
async fn a_pause_after_a_cancel_does_not_undo_it() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    // The chunk request waits an hour to retry, so the job runs until stopped.
    let mut fake = Fake::new();
    fake.rejection = Rejection::WrongCallRetryLater;
    let fake = Arc::new(fake);
    fake.rejected_chunks.store(usize::MAX, Ordering::SeqCst);
    let (index, job) = start_one_file(&fake, &db.path().join("i.db"), root.path()).await;
    wait_for_retry(&index, &job.job).await;
    index.cancel(&job.job).await.unwrap();
    // Refused while the job drains, or because it has already stopped.
    assert!(index.pause(&job.job).await.is_err());
    assert_eq!(settle(&index, &job.job).await.phase, "cancelled");
    let events = index.events(&job.job, 0, 1000).await.unwrap();
    assert!(events.iter().all(|e| e.kind != "pause_requested"));
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_job_seen_ready_is_no_longer_running() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    let mut fake = Fake::new();
    fake.delay = Duration::ZERO;
    let fake = Arc::new(fake);
    let (index, first) = start_one_file(&fake, &db.path().join("i.db"), root.path()).await;
    assert_eq!(settle(&index, &first.job).await.phase, "ready");
    for iteration in 0..50 {
        let update = index.start_update(&first.index).await.unwrap();
        while index.status(&update.job).await.unwrap().phase != "ready" {
            tokio::task::yield_now().await;
        }
        // Neither can reach the finished job's control any more.
        assert!(
            index.pause(&update.job).await.is_err(),
            "iteration {iteration}"
        );
        assert!(
            matches!(index.cancel(&update.job).await, Err(Error::Invalid(ref m)) if m.contains("completed")),
            "iteration {iteration}"
        );
    }
}
#[tokio::test]
async fn a_pause_during_a_retry_wait_counts_the_rejected_attempt_once() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    let mut fake = Fake::new();
    fake.rejection = Rejection::WrongCallRetryLater;
    let fake = Arc::new(fake);
    fake.rejected_chunks.store(1, Ordering::SeqCst);
    let (index, job) = start_one_file(&fake, &db.path().join("i.db"), root.path()).await;
    wait_for_retry(&index, &job.job).await;
    index.pause(&job.job).await.unwrap();
    let paused = settle(&index, &job.job).await;
    assert_eq!(paused.phase, "paused");
    assert_eq!(
        (paused.usage.chunk_calls, paused.usage.rejected_calls),
        (0, 1)
    );
    index.resume(&job.job).await.unwrap();
    let done = settle(&index, &job.job).await;
    assert_eq!(done.phase, "ready", "{:?}", done.problem);
    assert_eq!((done.usage.chunk_calls, done.usage.rejected_calls), (1, 1));
    assert_eq!(done.usage.chunking.as_ref().unwrap().input_tokens, 200);
    let events = index.events(&job.job, 0, 1000).await.unwrap();
    assert_events_match_totals(&done, &events);
}
#[test]
fn a_crash_during_a_retry_wait_keeps_the_rejected_attempts_usage() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    let path = db.path().join("i.db");
    let mut fake = Fake::new();
    fake.rejection = Rejection::WrongCallRetryLater;
    let fake = Arc::new(fake);
    fake.rejected_chunks.store(1, Ordering::SeqCst);
    let runtime = || {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
    };
    let first = runtime();
    let job = first.block_on(async {
        let (index, job) = start_one_file(&fake, &path, root.path()).await;
        wait_for_retry(&index, &job.job).await;
        job
    });
    // Like a killed process: every task stops where it is, mid-wait.
    drop(first);
    runtime().block_on(async {
        let index = ArkoIndex::open(&path, fake.clone(), config(&fake))
            .await
            .unwrap();
        let interrupted = index.status(&job.job).await.unwrap();
        assert_eq!(interrupted.phase, "paused");
        let usage = &interrupted.usage;
        assert_eq!((usage.chunk_calls, usage.rejected_calls), (0, 1));
        assert_eq!(usage.chunking.as_ref().unwrap().input_tokens, 100);
        // Resuming sends a new request and doesn't count the old one again.
        index.resume(&job.job).await.unwrap();
        let done = settle(&index, &job.job).await;
        assert_eq!(done.phase, "ready", "{:?}", done.problem);
        assert_eq!((done.usage.chunk_calls, done.usage.rejected_calls), (1, 1));
        assert_eq!(done.usage.chunking.as_ref().unwrap().input_tokens, 200);
        let events = index.events(&job.job, 0, 1000).await.unwrap();
        assert_events_match_totals(&done, &events);
    });
}
#[tokio::test]
async fn rejected_embedding_responses_count_when_a_retry_succeeds() {
    let fake = Arc::new(Fake::new());
    fake.rejected_embeds.store(1, Ordering::SeqCst);
    let (status, events, _) = build_one_file(fake.clone(), config(&fake)).await;
    assert_eq!(status.phase, "ready", "{:?}", status.problem);
    let usage = &status.usage;
    assert_eq!((usage.embedding_calls, usage.rejected_calls), (1, 1));
    assert_eq!(usage.embedding.as_ref().unwrap().input_tokens, 1_500);
    let embedded = events.iter().find(|e| e.kind == "embedded").unwrap();
    assert_eq!(embedded.detail["rejected_usage"][0]["input_tokens"], 500);
    assert_events_match_totals(&status, &events);
}
#[tokio::test]
async fn rejected_embedding_responses_count_when_retries_run_out() {
    let fake = Arc::new(Fake::new());
    fake.rejected_embeds.store(usize::MAX, Ordering::SeqCst);
    let config = Config {
        max_transient_attempts: 2,
        ..config(&fake)
    };
    let (status, events, _) = build_one_file(fake.clone(), config).await;
    assert_eq!(status.phase, "blocked");
    let usage = &status.usage;
    assert_eq!((usage.embedding_calls, usage.rejected_calls), (0, 2));
    assert_eq!(usage.embedding.as_ref().unwrap().input_tokens, 1_000);
    let failed = events
        .iter()
        .find(|e| e.kind == "embedding_failed")
        .unwrap();
    assert_eq!(failed.detail["rejected_usage"].as_array().unwrap().len(), 2);
    assert!(
        failed.detail["error"]
            .as_str()
            .unwrap()
            .contains("rejected")
    );
    assert_events_match_totals(&status, &events);
}
#[tokio::test]
async fn non_finite_costs_from_a_backend_do_not_break_status() {
    let mut fake = Fake::new();
    fake.nan_cost = true;
    let fake = Arc::new(fake);
    let (status, _, artifact) = build_one_file(fake.clone(), config(&fake)).await;
    assert_eq!(status.phase, "ready", "{:?}", status.problem);
    let chunking = status.usage.chunking.as_ref().unwrap();
    assert_eq!(chunking.input_tokens, 100);
    assert_eq!(chunking.total_cost, 0.0);
    assert_eq!(chunking.billed(), 0.0);
    // The saved reply loads as a ModelReply again.
    let reply: ModelReply = serde_json::from_value(artifact["response"].clone()).unwrap();
    assert_eq!(reply.usage_info.unwrap().total_cost, 0.0);
}
#[tokio::test]
async fn usage_totals_load_from_databases_saved_before_usage_tracking() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    let path = db.path().join("i.db");
    std::fs::write(root.path().join("a.txt"), "alpha\nbeta\n").unwrap();
    let fake = Arc::new(Fake::new());
    let index = ArkoIndex::open(&path, fake.clone(), config(&fake))
        .await
        .unwrap();
    let job = index
        .start_build(BuildSpec::new("test", root.path()))
        .await
        .unwrap();
    assert_eq!(settle(&index, &job.job).await.phase, "ready");
    drop(index);
    // What an older version left: no usage table, replies without usage.
    let conn = rusqlite::Connection::open(&path).unwrap();
    conn.execute_batch(
        "DROP TABLE job_usage; UPDATE requests SET response=json_remove(response,'$.usage_info');",
    )
    .unwrap();
    let saved: String = conn
        .query_row("SELECT response FROM requests LIMIT 1", [], |r| r.get(0))
        .unwrap();
    let reply: ModelReply = serde_json::from_str(&saved).unwrap();
    assert!(reply.usage_info.is_none());
    drop(conn);
    let index = ArkoIndex::open(&path, fake.clone(), config(&fake))
        .await
        .unwrap();
    let old = index.status(&job.job).await.unwrap();
    assert_eq!(old.phase, "ready");
    assert_eq!(old.usage, JobUsage::default());
    std::fs::write(root.path().join("a.txt"), "alpha\ngamma\n").unwrap();
    let update = index.start_update(&job.index).await.unwrap();
    let new = settle(&index, &update.job).await;
    assert_eq!(new.phase, "ready", "{:?}", new.problem);
    assert_eq!(new.usage.chunk_calls, 1);
    assert_eq!(new.usage.embedding_calls, 1);
    assert_eq!(
        index.status(&job.job).await.unwrap().usage,
        JobUsage::default()
    );
}
#[tokio::test]
async fn malformed_embeddings_do_not_replace_previous_generation() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("a.txt"), "alpha original").unwrap();
    let fake = Arc::new(Fake::new());
    let index = ArkoIndex::open(db.path().join("i.db"), fake.clone(), config(&fake))
        .await
        .unwrap();
    let first = index
        .start_build(BuildSpec::new("test", root.path()))
        .await
        .unwrap();
    assert_eq!(settle(&index, &first.job).await.phase, "ready");
    std::fs::write(root.path().join("a.txt"), "alpha changed").unwrap();
    fake.invalid_embedding.store(1, Ordering::SeqCst);
    let update = index.start_update(&first.index).await.unwrap();
    let blocked = settle(&index, &update.job).await;
    assert_eq!(blocked.phase, "blocked");
    // The rejected request still counts toward the job's usage.
    assert_eq!(blocked.usage.embedding_calls, 1);
    assert_eq!(blocked.usage.embedding.unwrap().input_tokens, 1_000);
    let events = index.events(&update.job, 0, 1000).await.unwrap();
    let rejected: Vec<_> = events
        .iter()
        .filter(|e| e.kind == "embedding_rejected")
        .collect();
    assert_eq!(rejected.len(), 1);
    assert_eq!(rejected[0].detail["usage"]["input_tokens"], 1_000);
    assert!(events.iter().all(|e| e.kind != "embedded"));
    let page = index
        .search(&first.index, "alpha", options(SearchMode::Lexical))
        .await
        .unwrap();
    assert_eq!(page.generation, first.job);
    assert_eq!(page.hits[0].text, "alpha original");
}

#[tokio::test]
async fn final_partial_batches_and_unchanged_line_endings() {
    for count in [0usize, 1, 5, 6, 7, 12, 13] {
        let root = tempfile::tempdir().unwrap();
        let db = tempfile::tempdir().unwrap();
        for n in 0..count {
            std::fs::write(
                root.path().join(format!("{n:02}.txt")),
                format!("file{n}\r\n{}", "row\r\n".repeat(700)),
            )
            .unwrap();
        }
        let fake = Arc::new(Fake::new());
        let index = ArkoIndex::open(db.path().join("i.db"), fake.clone(), config(&fake))
            .await
            .unwrap();
        let h = index
            .start_build(BuildSpec::new("test", root.path()))
            .await
            .unwrap();
        let done = settle(&index, &h.job).await;
        assert_eq!(done.phase, "ready", "{:?}", done.problem);
        assert_eq!(fake.calls.lock().unwrap().len(), count.div_ceil(6));
        let embeddings = fake.embeds.load(Ordering::SeqCst);
        if count > 0 {
            let path = root.path().join("00.txt");
            let content = std::fs::read_to_string(&path)
                .unwrap()
                .replace("\r\n", "\n");
            std::fs::write(path, content).unwrap();
        }
        let update = index.start_update(&h.index).await.unwrap();
        assert_eq!(settle(&index, &update.job).await.phase, "ready");
        assert_eq!(fake.calls.lock().unwrap().len(), count.div_ceil(6));
        assert_eq!(fake.embeds.load(Ordering::SeqCst), embeddings);
    }
}
#[tokio::test]
async fn giant_unicode_line_reconstructs_without_model_rewrites() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    let text = "α😀".repeat(80000);
    std::fs::write(root.path().join("giant.txt"), &text).unwrap();
    let fake = Arc::new(Fake::new());
    let cfg = config(&fake);
    let index = ArkoIndex::open(db.path().join("i.db"), fake.clone(), cfg)
        .await
        .unwrap();
    let h = index
        .start_build(BuildSpec::new("test", root.path()))
        .await
        .unwrap();
    let done = settle(&index, &h.job).await;
    assert_eq!(done.phase, "ready", "{:?}", done.problem);
    let c = rusqlite::Connection::open(db.path().join("i.db")).unwrap();
    let ids: Vec<String> = c
        .prepare("SELECT id FROM chunks WHERE job=?1 ORDER BY start")
        .unwrap()
        .query_map([&h.job], |r| r.get(0))
        .unwrap()
        .collect::<std::result::Result<_, _>>()
        .unwrap();
    let chunks = index.read_chunks(&h.index, &h.job, &ids, 0).await.unwrap();
    assert_eq!(chunks.into_iter().map(|c| c.text).collect::<String>(), text);
}
#[tokio::test]
async fn watch_debounces_atomic_save_bursts() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    let path = root.path().join("a.txt");
    std::fs::write(&path, "alpha original").unwrap();
    let fake = Arc::new(Fake::new());
    let index = ArkoIndex::open(db.path().join("i.db"), fake.clone(), config(&fake))
        .await
        .unwrap();
    let first = index
        .start_build(BuildSpec::new("test", root.path()))
        .await
        .unwrap();
    assert_eq!(settle(&index, &first.job).await.phase, "ready");
    let watch = index
        .watch(
            &first.index,
            WatchOptions::new(Duration::from_millis(150))
                .with_reconcile(Some(Duration::from_secs(3))),
        )
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;
    for n in 0..5 {
        let temporary = root.path().join("save.tmp");
        std::fs::write(
            &temporary,
            format!("alpha final{n}\n{}", "text\n".repeat(600)),
        )
        .unwrap();
        // Windows does not replace an existing destination with std::fs::rename.
        std::fs::remove_file(&path).unwrap();
        std::fs::rename(temporary, &path).unwrap();
        tokio::time::sleep(Duration::from_millis(15)).await;
    }
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let active = index.list().await.unwrap()[0].generation.clone().unwrap();
            if active != first.job {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    watch.stop().await.unwrap();
    assert_eq!(fake.calls.lock().unwrap().len(), 1);
    let page = index
        .search(&first.index, "alpha", options(SearchMode::Lexical))
        .await
        .unwrap();
    assert!(page.hits[0].text.starts_with("alpha final4"));
}

#[tokio::test]
async fn repairs_continue_beyond_former_attempt_limit() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("bad.txt"), "alpha\n".repeat(600)).unwrap();
    let fake = Arc::new(Fake::new());
    *fake.fail_file.lock().unwrap() = Some("bad.txt".into());
    fake.invalid_repair_replies.store(3, Ordering::SeqCst);
    let index = ArkoIndex::open(db.path().join("i.db"), fake.clone(), config(&fake))
        .await
        .unwrap();
    let job = index
        .start_build(BuildSpec::new("test", root.path()))
        .await
        .unwrap();
    assert_eq!(settle(&index, &job.job).await.phase, "ready");
    assert_eq!(
        fake.calls
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.tool == "repair_cuts")
            .count(),
        4
    );
}

#[test]
fn large_code_units_and_markdown_examples_remain_whole() {
    let body = "    // explanation of this operation\n".repeat(200);
    let code = format!(
        "/// Type documentation\nstruct Thing;\n\n/// Related implementation\nimpl Thing {{\n    fn first(&self) {{\n{body}    }}\n    fn second(&self) {{}}\n}}\n"
    );
    let source = Snapshot::new("large.rs", code.as_bytes()).unwrap();
    let guard = structure(&source);
    assert_eq!(guard.status, "rust-guarded");
    assert!(
        !guard.legal(source.line_end(1).unwrap()),
        "docs must stay with their declaration"
    );
    for row in 4..source.rows() {
        assert!(
            !guard.legal(source.line_end(row).unwrap()),
            "impl split at {row}"
        );
    }
    let windows = windows(&source);
    assert_eq!(windows.len(), 1);
    assert_eq!(windows[0].end, code.len());
    assert!(!windows[0].forced);
    let md = format!(
        "# Complete example\n\n```rust\n{}\n```\n\nAnother topic.\n",
        "println!(\"example\");\n".repeat(400)
    );
    let source = Snapshot::new("large.md", md.as_bytes()).unwrap();
    let guard = structure(&source);
    assert!(!guard.legal(source.line_end(200).unwrap()));
}

#[tokio::test]
async fn full_search_and_read_are_not_truncated_by_former_caps() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    for n in 0..121 {
        std::fs::write(
            root.path().join(format!("{n:03}.txt")),
            format!("alpha {n} {} END-{n}", "context ".repeat(1000)),
        )
        .unwrap();
    }
    let fake = Arc::new(Fake::new());
    let path = db.path().join("i.db");
    let index = ArkoIndex::open(&path, fake.clone(), config(&fake))
        .await
        .unwrap();
    let job = index
        .start_build(BuildSpec::new("test", root.path()))
        .await
        .unwrap();
    assert_eq!(settle(&index, &job.job).await.phase, "ready");
    let page = index
        .search(&job.index, "alpha", options(SearchMode::Hybrid))
        .await
        .unwrap();
    assert_eq!(page.hits.len(), 121);
    assert_eq!(page.candidate_count, 121);
    assert_eq!(page.omitted_hits, 0);
    assert!(page.next.is_none());
    assert!(page.search_tokens > 2000);
    for hit in &page.hits {
        assert_eq!(
            hit.text,
            std::fs::read_to_string(root.path().join(&hit.path)).unwrap()
        );
        assert_eq!(hit.text.len(), hit.end_byte - hit.start_byte);
    }
    let ids = page.hits.iter().map(|h| h.id.clone()).collect::<Vec<_>>();
    assert_eq!(
        index
            .read_chunks(&job.index, &job.job, &ids, 20)
            .await
            .unwrap()
            .len(),
        121
    );
    let limited = index
        .search(
            &job.index,
            "alpha",
            SearchOptions {
                top_k: Some(105),
                ..options(SearchMode::Hybrid)
            },
        )
        .await
        .unwrap();
    assert_eq!(limited.hits.len(), 105);
    assert_eq!(limited.candidate_count, 121);
    let remaining = index
        .search(
            &job.index,
            "alpha",
            SearchOptions {
                top_k: Some(105),
                cursor: limited.next,
                ..options(SearchMode::Hybrid)
            },
        )
        .await
        .unwrap();
    assert_eq!(remaining.hits.len(), 16);
    assert_eq!(
        limited
            .hits
            .iter()
            .chain(&remaining.hits)
            .map(|h| h.id.clone())
            .collect::<Vec<_>>(),
        ids
    );
    let query = (0..80)
        .map(|n| format!("missing{n}"))
        .chain(std::iter::once("zfinalmarker".into()))
        .collect::<Vec<_>>()
        .join(" ");
    std::fs::write(root.path().join("query.txt"), "zfinalmarker").unwrap();
    let update = index.start_update(&job.index).await.unwrap();
    assert_eq!(settle(&index, &update.job).await.phase, "ready");
    assert!(
        index
            .search(&job.index, &query, options(SearchMode::Lexical))
            .await
            .unwrap()
            .hits
            .iter()
            .any(|h| h.path == "query.txt")
    );
}

#[tokio::test]
async fn minimum_chunk_size_filters_all_modes_and_preserves_reads() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    let texts = [
        ("fragment.mdx", "<Tabs>\nalpha\n</Tabs>\n".to_owned()),
        ("below.txt", format!("alpha {}", "a".repeat(33))),
        ("exact.txt", format!("alpha {}", "é".repeat(17))),
        ("large.txt", format!("alpha {}", "context ".repeat(20))),
    ];
    assert_eq!(texts[1].1.len(), 39);
    assert_eq!(texts[2].1.len(), 40);
    for (path, text) in &texts {
        std::fs::write(root.path().join(path), text).unwrap();
    }
    let fake = Arc::new(Fake::new());
    let index = ArkoIndex::open(db.path().join("i.db"), fake.clone(), config(&fake))
        .await
        .unwrap();
    let job = index
        .start_build(BuildSpec::new("test", root.path()))
        .await
        .unwrap();
    assert_eq!(settle(&index, &job.job).await.phase, "ready");
    for mode in [
        SearchMode::Lexical,
        SearchMode::Semantic,
        SearchMode::Hybrid,
    ] {
        // Warm the unfiltered vector cache before changing eligibility.
        let all = index
            .search(&job.index, "alpha", options(mode.clone()))
            .await
            .unwrap();
        assert_eq!(all.hits.len(), 4);
        let filtered = SearchOptions {
            min_chunk_bytes: 40,
            top_k: Some(1),
            ..options(mode.clone())
        };
        let first = index
            .search(&job.index, "alpha", filtered.clone())
            .await
            .unwrap();
        assert_eq!(first.candidate_count, 2);
        assert_eq!(first.hits.len(), 1);
        assert_eq!(first.omitted_hits, 1);
        let next = first.next.clone().unwrap();
        let second = index
            .search(
                &job.index,
                "alpha",
                SearchOptions {
                    cursor: Some(next.clone()),
                    ..filtered.clone()
                },
            )
            .await
            .unwrap();
        assert_eq!(second.candidate_count, 2);
        assert_eq!(second.hits.len(), 1);
        assert_eq!(second.omitted_hits, 0);
        assert!(second.next.is_none());
        assert_ne!(first.hits[0].id, second.hits[0].id);
        assert!(
            first
                .hits
                .iter()
                .chain(&second.hits)
                .all(|h| h.text.len() >= 40)
        );
        assert!(matches!(
            index
                .search(
                    &job.index,
                    "alpha",
                    SearchOptions {
                        min_chunk_bytes: 0,
                        cursor: Some(next),
                        ..filtered
                    }
                )
                .await,
            Err(Error::Stale(_))
        ));
        // Small fragments stay available for exact source reads.
        let short_ids: Vec<_> = all
            .hits
            .iter()
            .filter(|h| h.text.len() < 40)
            .map(|h| h.id.clone())
            .collect();
        let reads = index
            .read_chunks(&job.index, &job.job, &short_ids, 0)
            .await
            .unwrap();
        assert_eq!(reads.len(), 2);
        for read in reads {
            assert_eq!(
                read.text,
                std::fs::read_to_string(root.path().join(read.path)).unwrap()
            );
        }
        // Returning to the unfiltered cache key must restore every candidate.
        assert_eq!(
            index
                .search(&job.index, "alpha", options(mode))
                .await
                .unwrap()
                .hits
                .len(),
            4
        );
    }
}

#[tokio::test]
async fn changed_chunk_policy_rebuilds_unchanged_source() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    let path = db.path().join("i.db");
    std::fs::write(root.path().join("a.txt"), "alpha\nbeta\n").unwrap();
    let fake = Arc::new(Fake::new());
    let index = ArkoIndex::open(&path, fake.clone(), config(&fake))
        .await
        .unwrap();
    let job = index
        .start_build(BuildSpec::new("test", root.path()))
        .await
        .unwrap();
    assert_eq!(settle(&index, &job.job).await.phase, "ready");
    let initial = fake.calls.lock().unwrap().len();
    let embedded = fake.embeds.load(Ordering::SeqCst);
    let conn = rusqlite::Connection::open(&path).unwrap();
    conn.execute("UPDATE events SET detail=json_set(detail,'$.chunking_policy','old-policy') WHERE job=?1 AND kind='snapshot'",[&job.job]).unwrap();
    let update = index.start_update(&job.index).await.unwrap();
    assert_eq!(settle(&index, &update.job).await.phase, "ready");
    assert_eq!(fake.calls.lock().unwrap().len(), initial + 1);
    assert_eq!(fake.embeds.load(Ordering::SeqCst), embedded);
    let affected: String = conn
        .query_row(
            "SELECT affected FROM jobs WHERE id=?1",
            [&update.job],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&affected).unwrap(),
        json!(["a.txt"])
    );
    let unchanged = index.start_update(&job.index).await.unwrap();
    assert_eq!(settle(&index, &unchanged.job).await.phase, "ready");
    assert_eq!(fake.calls.lock().unwrap().len(), initial + 1);
}

#[tokio::test]
async fn only_actual_embedding_limits_split_large_impls_and_keep_fitting_methods_intact() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    let method = |name: &str| {
        format!(
            "    fn {name}(&self) {{\n        let _text = \"{}\";\n    }}\n",
            "a".repeat(70000)
        )
    };
    let code = format!(
        "struct Thing;\nimpl Thing {{\n{}{} }}\n",
        method("first"),
        method("second")
    );
    std::fs::write(root.path().join("large.rs"), &code).unwrap();
    let fake = Arc::new(Fake::new());
    let index = ArkoIndex::open(db.path().join("i.db"), fake.clone(), config(&fake))
        .await
        .unwrap();
    // The whole file fits one chunk request.
    let spec = BuildSpec::new("test", root.path()).with_chunking(ChunkingOptions {
        max_request_bytes: Some(1_000_000),
        ..Default::default()
    });
    let job = index.start_build(spec).await.unwrap();
    assert_eq!(settle(&index, &job.job).await.phase, "ready");
    let page = index
        .search(&job.index, "Thing", options(SearchMode::Lexical))
        .await
        .unwrap();
    // Semantic retrieval includes both provider-limited pieces, regardless of lexical mentions.
    let all = index
        .search(&job.index, "Thing", options(SearchMode::Semantic))
        .await
        .unwrap();
    assert_eq!(all.hits.len(), 2);
    assert!(all.hits.iter().any(|h| h.text.contains(&method("first"))));
    assert!(all.hits.iter().any(|h| h.text.contains(&method("second"))));
    let mut ordered = all.hits.clone();
    ordered.sort_by_key(|h| h.start_byte);
    assert_eq!(
        ordered.iter().map(|h| h.text.as_str()).collect::<String>(),
        code
    );
    assert!(!page.hits.is_empty());
}
#[tokio::test]
async fn a_second_open_of_a_live_database_is_rejected() {
    let db = tempfile::tempdir().unwrap();
    let path = db.path().join("index.db");
    let fake = Arc::new(Fake::new());
    let first = ArkoIndex::open(&path, fake.clone(), config(&fake))
        .await
        .unwrap();
    let second = ArkoIndex::open(&path, fake.clone(), config(&fake)).await;
    assert!(matches!(second, Err(Error::Invalid(ref message)) if message.contains("already open")));
    drop(first);
    ArkoIndex::open(&path, fake.clone(), config(&fake))
        .await
        .unwrap();
}
#[tokio::test]
async fn snapshot_records_the_backends_chunk_model() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("a.txt"), "alpha\nbeta\n").unwrap();
    let fake = Arc::new(Fake::new());
    let index = ArkoIndex::open(db.path().join("i.db"), fake.clone(), config(&fake))
        .await
        .unwrap();
    let job = index
        .start_build(BuildSpec::new("test", root.path()))
        .await
        .unwrap();
    assert_eq!(settle(&index, &job.job).await.phase, "ready");
    let events = index.events(&job.job, 0, 1000).await.unwrap();
    let snapshot = events.iter().find(|e| e.kind == "snapshot").unwrap();
    assert_eq!(snapshot.detail["model"], "fake-chunker");
    assert_eq!(snapshot.detail["reasoning"], "none");
}
#[test]
fn job_status_saved_with_the_former_field_name_still_loads() {
    let status: JobStatus = serde_json::from_value(json!({
        "job": "j", "index": "i", "phase": "ready", "problem": null, "files": 1,
        "validated": 1, "bypassed": 0, "unchecked": 0, "active_luna": 2,
        "active_embeddings": 0
    }))
    .unwrap();
    assert_eq!(status.active_chunking, 2);
    assert_eq!(status.usage, JobUsage::default());
}

/// Builds `spec` with `fake` in a new database under `db`.
async fn build_spec(
    fake: &Arc<Fake>,
    db: &std::path::Path,
    spec: BuildSpec,
) -> (ArkoIndex, JobHandle, JobStatus) {
    let index = ArkoIndex::open(db.join("i.db"), fake.clone(), config(fake))
        .await
        .unwrap();
    let job = index.start_build(spec).await.unwrap();
    let status = settle(&index, &job.job).await;
    (index, job, status)
}
fn o200k(text: &str) -> usize {
    tiktoken_rs::o200k_base_singleton().count_ordinary(text)
}
#[tokio::test]
async fn oversize_files_are_split_into_windows_within_the_token_budget() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    let large: String = (0..40)
        .map(|n| {
            format!(
                "/// Function {n}.\nfn f{n}() {{\n{}}}\n\n",
                "    let value = compute(1, 2, 3);\n".repeat(10)
            )
        })
        .collect();
    std::fs::write(root.path().join("large.rs"), &large).unwrap();
    std::fs::write(root.path().join("small.txt"), "alpha\nbeta\n").unwrap();
    // A GPT-family chunk model with a 20,000-token input limit: requests
    // may hold 2,000 o200k tokens.
    let mut fake = Fake::new();
    fake.chunk_input = ChunkInput {
        tokenizer: Some(ChunkTokenizer::O200kBase),
        max_input_tokens: Some(20_000),
    };
    let fake = Arc::new(fake);
    let (index, job, status) =
        build_spec(&fake, db.path(), BuildSpec::new("test", root.path())).await;
    assert_eq!(status.phase, "ready", "{:?}", status.problem);
    let calls = fake.calls.lock().unwrap().clone();
    let windows: Vec<_> = calls
        .iter()
        .filter(|r| r.prompt.contains("path=large.rs"))
        .collect();
    assert!(windows.len() >= 4, "{} requests", windows.len());
    for request in &calls {
        assert!(
            o200k(&request.prompt) <= 2_000,
            "{}",
            o200k(&request.prompt)
        );
        // Windows begin at a function, never inside one.
        for block in request.prompt.split("FILE ").skip(1) {
            if block.contains("path=large.rs") {
                let first_row = block.lines().nth(1).unwrap();
                assert!(
                    first_row.ends_with("| /// Function") || first_row.contains("| /// Function "),
                    "{first_row}"
                );
            }
        }
    }
    let events = index.events(&job.job, 0, 10_000).await.unwrap();
    let snapshot = events.iter().find(|e| e.kind == "snapshot").unwrap();
    assert_eq!(snapshot.detail["chunk_budget"]["tokenizer"], "o200k_base");
    assert_eq!(snapshot.detail["chunk_budget"]["max_tokens"], 2_000);
    assert_eq!(snapshot.detail["oversize_files"], 1);
    let oversize = events.iter().find(|e| e.kind == "oversize").unwrap();
    assert_eq!(oversize.detail["path"], "large.rs");
    assert_eq!(oversize.detail["action"], "split");
    assert_eq!(
        oversize.detail["windows"].as_u64().unwrap() as usize,
        windows.len()
    );
    // The chunks reproduce the file.
    let c = rusqlite::Connection::open(db.path().join("i.db")).unwrap();
    let ids: Vec<String> = c
        .prepare("SELECT id FROM chunks WHERE job=?1 AND path='large.rs' ORDER BY start")
        .unwrap()
        .query_map([&job.job], |r| r.get(0))
        .unwrap()
        .collect::<std::result::Result<_, _>>()
        .unwrap();
    let chunks = index
        .read_chunks(&job.index, &job.job, &ids, 0)
        .await
        .unwrap();
    assert_eq!(
        chunks.into_iter().map(|c| c.text).collect::<String>(),
        large
    );
}
#[tokio::test]
async fn initial_requests_hold_files_within_the_byte_budget() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    for n in 0..6 {
        std::fs::write(
            root.path().join(format!("{n}.txt")),
            format!("alpha {n}\n{}", "some context\n".repeat(40)),
        )
        .unwrap();
    }
    let fake = Arc::new(Fake::new());
    let spec = BuildSpec::new("test", root.path()).with_chunking(ChunkingOptions {
        max_request_bytes: Some(2_500),
        ..Default::default()
    });
    let (_index, _job, status) = build_spec(&fake, db.path(), spec).await;
    assert_eq!(status.phase, "ready", "{:?}", status.problem);
    let calls = fake.calls.lock().unwrap().clone();
    // Each file renders to about 1,000 bytes: two per request, not six.
    assert_eq!(calls.len(), 3);
    for request in &calls {
        assert_eq!(request.schema["properties"].as_object().unwrap().len(), 2);
        assert!(request.prompt.len() <= 2_500);
    }
}
#[tokio::test]
async fn oversize_files_can_be_skipped_or_block_the_job() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("big.txt"), "zulu line\n".repeat(1_000)).unwrap();
    std::fs::write(root.path().join("small.txt"), "alpha\nbeta\n").unwrap();
    let spec = |oversize| {
        BuildSpec::new("test", root.path()).with_chunking(ChunkingOptions {
            max_request_bytes: Some(2_000),
            oversize,
            ..Default::default()
        })
    };
    // Skip: the file is left out and reported; the rest is indexed.
    let db = tempfile::tempdir().unwrap();
    let fake = Arc::new(Fake::new());
    let (index, job, status) = build_spec(&fake, db.path(), spec(OversizePolicy::Skip)).await;
    assert_eq!(status.phase, "ready", "{:?}", status.problem);
    assert_eq!((status.files, status.skipped), (2, 1));
    assert!(
        fake.calls
            .lock()
            .unwrap()
            .iter()
            .all(|r| !r.prompt.contains("big.txt"))
    );
    let events = index.events(&job.job, 0, 1000).await.unwrap();
    let oversize = events.iter().find(|e| e.kind == "oversize").unwrap();
    assert_eq!(oversize.detail["action"], "skipped");
    let c = rusqlite::Connection::open(db.path().join("i.db")).unwrap();
    let excluded: String = c
        .query_row("SELECT excluded FROM jobs WHERE id=?1", [&job.job], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&excluded).unwrap(),
        json!([{"path": "big.txt", "reason": arko_index::source::TOO_LARGE_FOR_A_REQUEST}])
    );
    let page = index
        .search(&job.index, "zulu", options(SearchMode::Lexical))
        .await
        .unwrap();
    assert!(page.hits.is_empty());
    // An update with no changes finds nothing to do.
    let calls = fake.calls.lock().unwrap().len();
    let update = index.start_update(&job.index).await.unwrap();
    assert_eq!(settle(&index, &update.job).await.phase, "ready");
    assert_eq!(fake.calls.lock().unwrap().len(), calls);
    assert!(index.changes(&update.job).await.unwrap().is_empty());
    // Block: the job stops before any model request, naming the file.
    let db = tempfile::tempdir().unwrap();
    let fake = Arc::new(Fake::new());
    let (_index, _job, status) = build_spec(&fake, db.path(), spec(OversizePolicy::Block)).await;
    assert_eq!(status.phase, "blocked");
    let problem = status.problem.unwrap();
    assert!(
        problem.contains("big.txt") && problem.contains("block"),
        "{problem}"
    );
    assert!(fake.calls.lock().unwrap().is_empty());
}
#[test]
fn repairs_show_only_their_window() {
    let text: String = (1..=1000).map(|i| format!("source-{i}\n")).collect();
    let s = Snapshot::new("x.txt", text.as_bytes()).unwrap();
    let guard = structure(&s);
    let w = Window::new(s.line_starts[99], s.line_end(200).unwrap());
    let cuts = vec![Cut {
        after_line: 150,
        tail: "wrong".into(),
    }];
    let r = repair(&s, &guard, &w, &cuts).unwrap();
    assert!(
        r.request
            .prompt
            .contains("(rows 100-200 of 1000, absolute rows)")
    );
    assert!(r.request.prompt.contains("0100 | source-100\n"));
    assert!(r.request.prompt.contains("0200 | source-200\n"));
    assert!(!r.request.prompt.contains("0099 | "));
    assert!(!r.request.prompt.contains("0201 | "));
    let ids = &r.regions["r0"].candidates;
    assert!(ids.contains_key("L100") && ids.contains_key("L199"));
    assert!(!ids.contains_key("L99") && !ids.contains_key("L200"));
}
#[test]
fn chunking_settings_saved_before_they_existed_load_with_defaults() {
    let spec: BuildSpec = serde_json::from_value(json!({
        "name": "docs", "root": "/src", "include": [], "exclude": []
    }))
    .unwrap();
    assert_eq!(spec.max_file_bytes, DEFAULT_MAX_FILE_BYTES);
    assert_eq!(spec.chunking, ChunkingOptions::default());
    assert_eq!(spec.chunking.oversize, OversizePolicy::Split);
    let window: Window = serde_json::from_value(json!({
        "start": 0, "end": 10, "cuts": null, "repair_rounds": 0, "forced": false, "problem": null
    }))
    .unwrap();
    assert_eq!(window.tokens, None);
    assert!(
        BuildSpec::new("x", "/src")
            .with_chunking(ChunkingOptions {
                max_request_tokens: Some(0),
                ..Default::default()
            })
            .validate()
            .is_err()
    );
}

/// Waits for the first watcher event that `matches`.
async fn next_watch_event(
    events: &mut tokio::sync::broadcast::Receiver<WatchEvent>,
    matches: impl Fn(&WatchEvent) -> bool,
) -> WatchEvent {
    tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            match events.recv().await {
                Ok(event) if matches(&event) => return event,
                Ok(_) | Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                Err(error) => panic!("watcher events ended: {error}"),
            }
        }
    })
    .await
    .expect("no such watcher event")
}
#[test]
fn watch_resumes_an_interrupted_update_but_holds_a_paused_one() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    let path = db.path().join("i.db");
    let mut fake = Fake::new();
    fake.rejection = Rejection::WrongCallRetryLater;
    let fake = Arc::new(fake);
    let runtime = || {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
    };
    let first = runtime();
    let (index_id, interrupted) = first.block_on(async {
        let (index, build) = start_one_file(&fake, &path, root.path()).await;
        assert_eq!(settle(&index, &build.job).await.phase, "ready");
        // An update that waits an hour to retry, then a killed process.
        fake.rejected_chunks.store(usize::MAX, Ordering::SeqCst);
        std::fs::write(root.path().join("a.txt"), "alpha\ngamma\n").unwrap();
        let update = index.start_update(&build.index).await.unwrap();
        wait_for_retry(&index, &update.job).await;
        (build.index, update.job)
    });
    drop(first);
    runtime().block_on(async {
        fake.rejected_chunks.store(0, Ordering::SeqCst);
        let index = ArkoIndex::open(&path, fake.clone(), config(&fake))
            .await
            .unwrap();
        assert_eq!(index.status(&interrupted).await.unwrap().phase, "paused");
        let watch = index
            .watch(&index_id, WatchOptions::new(Duration::from_millis(50)))
            .await
            .unwrap();
        let mut events = watch.subscribe();
        let resumed =
            next_watch_event(&mut events, |e| matches!(e, WatchEvent::Resumed { .. })).await;
        assert_eq!(
            resumed,
            WatchEvent::Resumed {
                job: interrupted.clone(),
                reason: "interrupted".into()
            }
        );
        assert_eq!(settle(&index, &interrupted).await.phase, "ready");
        // A job the host paused holds further updates and is left alone.
        watch.stop().await.unwrap();
        fake.rejected_chunks.store(usize::MAX, Ordering::SeqCst);
        std::fs::write(root.path().join("a.txt"), "alpha\ndelta\n").unwrap();
        let update = index.start_update(&index_id).await.unwrap();
        wait_for_retry(&index, &update.job).await;
        index.pause(&update.job).await.unwrap();
        assert_eq!(settle(&index, &update.job).await.phase, "paused");
        fake.rejected_chunks.store(0, Ordering::SeqCst);
        let watch = index
            .watch(&index_id, WatchOptions::new(Duration::from_millis(50)))
            .await
            .unwrap();
        let mut events = watch.subscribe();
        let held = next_watch_event(&mut events, |e| matches!(e, WatchEvent::Held { .. })).await;
        assert_eq!(
            held,
            WatchEvent::Held {
                job: update.job.clone(),
                phase: "paused".into(),
                problem: Some("paused".into())
            }
        );
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(index.status(&update.job).await.unwrap().phase, "paused");
        watch.stop().await.unwrap();
    });
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn watch_retries_a_blocked_update_only_when_asked() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    let fake = Arc::new(Fake::new());
    std::fs::write(root.path().join("a.txt"), "alpha\nbeta\n").unwrap();
    // One transient failure exhausts a request's attempts.
    let config = Config {
        max_transient_attempts: 1,
        ..config(&fake)
    };
    let index = ArkoIndex::open(db.path().join("i.db"), fake.clone(), config)
        .await
        .unwrap();
    let build = index
        .start_build(BuildSpec::new("test", root.path()))
        .await
        .unwrap();
    assert_eq!(settle(&index, &build.job).await.phase, "ready");
    fake.rejected_chunks.store(1, Ordering::SeqCst);
    std::fs::write(root.path().join("a.txt"), "alpha\ngamma\n").unwrap();
    // By default a blocked update is held.
    let options = WatchOptions::new(Duration::from_millis(50));
    let watch = index.watch(&build.index, options.clone()).await.unwrap();
    let mut events = watch.subscribe();
    let WatchEvent::UpdateStarted { job } = next_watch_event(&mut events, |e| {
        matches!(e, WatchEvent::UpdateStarted { .. })
    })
    .await
    else {
        unreachable!()
    };
    let held = next_watch_event(&mut events, |e| matches!(e, WatchEvent::Held { .. })).await;
    assert!(matches!(held, WatchEvent::Held { ref phase, .. } if phase == "blocked"));
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(index.status(&job).await.unwrap().phase, "blocked");
    watch.stop().await.unwrap();
    let phase_event = index
        .events(&job, 0, 1000)
        .await
        .unwrap()
        .into_iter()
        .rfind(|e| e.kind == "phase")
        .unwrap();
    assert_eq!(phase_event.detail["retryable"], true);
    // With retries on, it is resumed after the delay.
    let watch = index
        .watch(
            &build.index,
            options.with_retry_blocked_after(Some(Duration::from_millis(200))),
        )
        .await
        .unwrap();
    let mut events = watch.subscribe();
    let resumed = next_watch_event(&mut events, |e| matches!(e, WatchEvent::Resumed { .. })).await;
    assert_eq!(
        resumed,
        WatchEvent::Resumed {
            job: job.clone(),
            reason: "blocked".into()
        }
    );
    assert_eq!(settle(&index, &job).await.phase, "ready");
    watch.stop().await.unwrap();
}
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn watch_ignores_events_in_skipped_and_ignored_directories() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    for dir in ["src", "target", "build"] {
        std::fs::create_dir_all(root.path().join(dir)).unwrap();
    }
    std::fs::write(root.path().join(".gitignore"), "build/\n").unwrap();
    std::fs::write(root.path().join("src/a.txt"), "alpha\n").unwrap();
    let fake = Arc::new(Fake::new());
    let index = ArkoIndex::open(db.path().join("i.db"), fake.clone(), config(&fake))
        .await
        .unwrap();
    let build = index
        .start_build(BuildSpec::new("test", root.path()))
        .await
        .unwrap();
    assert_eq!(settle(&index, &build.job).await.phase, "ready");
    let watch = index
        .watch(
            &build.index,
            WatchOptions::new(Duration::from_millis(100)).with_reconcile(None),
        )
        .await
        .unwrap();
    let mut events = watch.subscribe();
    // The scan at start.
    next_watch_event(&mut events, |e| matches!(e, WatchEvent::Scanned { .. })).await;
    for n in 0..5 {
        std::fs::write(root.path().join("target/out.txt"), format!("{n}")).unwrap();
        std::fs::write(root.path().join("build/out.txt"), format!("{n}")).unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    tokio::time::sleep(Duration::from_millis(500)).await;
    loop {
        match events.try_recv() {
            Ok(WatchEvent::Scanned { .. }) => {
                panic!("an event in a skipped directory caused a scan")
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }
    std::fs::write(root.path().join("src/b.txt"), "beta\n").unwrap();
    let scanned = next_watch_event(&mut events, |e| matches!(e, WatchEvent::Scanned { .. })).await;
    assert_eq!(
        scanned,
        WatchEvent::Scanned {
            files: 2,
            changes: 1
        }
    );
    watch.stop().await.unwrap();
}

#[tokio::test]
async fn a_new_update_reuses_the_chunking_of_a_cancelled_build() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    for n in 0..4 {
        std::fs::write(
            root.path().join(format!("{n}.txt")),
            format!("alpha {n}\nbeta\n"),
        )
        .unwrap();
    }
    // One request per file; the first is refused, so the build is blocked
    // with three files chunked.
    let mut fake = Fake::new();
    fake.rejection = Rejection::Refusal;
    let fake = Arc::new(fake);
    fake.rejected_chunks.store(1, Ordering::SeqCst);
    let config = Config {
        batch_count: 1,
        ..config(&fake)
    };
    let index = ArkoIndex::open(db.path().join("i.db"), fake.clone(), config)
        .await
        .unwrap();
    let build = index
        .start_build(BuildSpec::new("test", root.path()))
        .await
        .unwrap();
    let blocked = settle(&index, &build.job).await;
    assert_eq!(blocked.phase, "blocked");
    assert_eq!(blocked.validated, 3);
    assert_eq!(fake.calls.lock().unwrap().len(), 4);
    index.cancel(&build.job).await.unwrap();
    // The update chunks only the file the build could not.
    let update = index.start_update(&build.index).await.unwrap();
    let done = settle(&index, &update.job).await;
    assert_eq!(done.phase, "ready", "{:?}", done.problem);
    assert_eq!(fake.calls.lock().unwrap().len(), 5);
    assert_eq!(done.validated, 4);
}
#[tokio::test]
async fn resume_takes_runtime_settings_from_the_current_configuration() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    let path = db.path().join("i.db");
    // Every chunk request waits an hour to retry, so the job runs until paused.
    let mut fake = Fake::new();
    fake.rejection = Rejection::WrongCallRetryLater;
    let fake = Arc::new(fake);
    fake.rejected_chunks.store(usize::MAX, Ordering::SeqCst);
    let (index, job) = start_one_file(&fake, &path, root.path()).await;
    wait_for_retry(&index, &job.job).await;
    index.pause(&job.job).await.unwrap();
    assert_eq!(settle(&index, &job.job).await.phase, "paused");
    drop(index);
    fake.rejected_chunks.store(0, Ordering::SeqCst);
    let changed = Config {
        batch_count: 3,
        concurrency: 1,
        ..config(&fake)
    };
    let index = ArkoIndex::open(&path, fake.clone(), changed.clone())
        .await
        .unwrap();
    index.resume(&job.job).await.unwrap();
    let done = settle(&index, &job.job).await;
    assert_eq!(done.phase, "ready", "{:?}", done.problem);
    let events = index.events(&job.job, 0, 1000).await.unwrap();
    let settings = events
        .iter()
        .find(|e| e.kind == "settings_changed")
        .unwrap();
    assert_eq!(
        settings.detail["changes"],
        json!([
            {"setting": "batch_count", "from": 6, "to": 3},
            {"setting": "concurrency", "from": 2, "to": 1}
        ])
    );
    drop(index);
    // Another embedding profile would mix vectors, so resume refuses it.
    std::fs::write(root.path().join("a.txt"), "alpha\ngamma\n").unwrap();
    fake.rejected_chunks.store(usize::MAX, Ordering::SeqCst);
    let index = ArkoIndex::open(&path, fake.clone(), config(&fake))
        .await
        .unwrap();
    let update = index.start_update(&job.index).await.unwrap();
    wait_for_retry(&index, &update.job).await;
    index.pause(&update.job).await.unwrap();
    assert_eq!(settle(&index, &update.job).await.phase, "paused");
    drop(index);
    let mut other = Fake::new();
    other.profile.dimensions = 3;
    let other = Arc::new(other);
    let index = ArkoIndex::open(&path, other.clone(), config(&other))
        .await
        .unwrap();
    let refused = index.resume(&update.job).await.unwrap_err();
    assert!(refused.to_string().contains("embeds with"), "{refused}");
}

/// Rows of `table` in the database at `path`.
fn rows(path: &std::path::Path, table: &str) -> usize {
    let c = rusqlite::Connection::open(path).unwrap();
    c.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| {
        r.get::<_, i64>(0)
    })
    .unwrap() as usize
}
#[tokio::test]
async fn unchanged_chunks_share_one_text_across_generations() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    let path = db.path().join("i.db");
    for n in 0..3 {
        std::fs::write(root.path().join(format!("{n}.txt")), format!("alpha {n}\n")).unwrap();
    }
    let fake = Arc::new(Fake::new());
    let index = ArkoIndex::open(&path, fake.clone(), config(&fake))
        .await
        .unwrap();
    let build = index
        .start_build(BuildSpec::new("test", root.path()))
        .await
        .unwrap();
    assert_eq!(settle(&index, &build.job).await.phase, "ready");
    assert_eq!(rows(&path, "chunk_texts"), 3);
    std::fs::write(root.path().join("0.txt"), "alpha changed\n").unwrap();
    let update = index.start_update(&build.index).await.unwrap();
    assert_eq!(settle(&index, &update.job).await.phase, "ready");
    // Two generations of three chunks; two chunks are the same in both.
    assert_eq!(rows(&path, "chunks"), 6);
    assert_eq!(rows(&path, "chunk_texts"), 4);
    assert_eq!(rows(&path, "chunk_text_fts"), 4);
    let page = index
        .search(&build.index, "alpha", options(SearchMode::Lexical))
        .await
        .unwrap();
    assert_eq!(page.generation, update.job);
    let mut texts: Vec<_> = page.hits.iter().map(|h| h.text.as_str()).collect();
    texts.sort_unstable();
    assert_eq!(texts, ["alpha 1\n", "alpha 2\n", "alpha changed\n"]);
    let old = index
        .search(&build.index, "changed", options(SearchMode::Lexical))
        .await
        .unwrap();
    assert_eq!(old.hits.len(), 1);
}
#[tokio::test]
async fn delete_removes_only_data_no_other_index_uses() {
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    let path = db.path().join("i.db");
    for root in [&a, &b] {
        std::fs::write(root.path().join("shared.txt"), "alpha shared\n").unwrap();
    }
    std::fs::write(a.path().join("only_a.txt"), "alpha a\n").unwrap();
    std::fs::write(b.path().join("only_b.txt"), "alpha b\n").unwrap();
    let fake = Arc::new(Fake::new());
    let index = ArkoIndex::open(&path, fake.clone(), config(&fake))
        .await
        .unwrap();
    let mut handles = vec![];
    for (name, root) in [("a", &a), ("b", &b)] {
        let job = index
            .start_build(BuildSpec::new(name, root.path()))
            .await
            .unwrap();
        assert_eq!(settle(&index, &job.job).await.phase, "ready");
        handles.push(job);
    }
    // The shared file's snapshot, vector and text are stored once.
    assert_eq!(
        (
            rows(&path, "snapshots"),
            rows(&path, "vectors"),
            rows(&path, "chunk_texts")
        ),
        (3, 3, 3)
    );
    index.delete(&handles[0].index).await.unwrap();
    assert_eq!(
        (
            rows(&path, "snapshots"),
            rows(&path, "vectors"),
            rows(&path, "chunk_texts")
        ),
        (2, 2, 2)
    );
    assert_eq!(rows(&path, "chunk_text_fts"), 2);
    for mode in [SearchMode::Lexical, SearchMode::Semantic] {
        let page = index
            .search(&handles[1].index, "alpha", options(mode))
            .await
            .unwrap();
        let mut texts: Vec<_> = page.hits.iter().map(|h| h.text.as_str()).collect();
        texts.sort_unstable();
        assert_eq!(texts, ["alpha b\n", "alpha shared\n"]);
    }
}
#[tokio::test]
async fn keep_generations_prunes_older_generations_when_asked() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    let path = db.path().join("i.db");
    std::fs::write(root.path().join("a.txt"), "alpha 0\n").unwrap();
    let fake = Arc::new(Fake::new());
    let config = Config {
        keep_generations: Some(2),
        ..config(&fake)
    };
    let index = ArkoIndex::open(&path, fake.clone(), config).await.unwrap();
    let build = index
        .start_build(BuildSpec::new("test", root.path()))
        .await
        .unwrap();
    assert_eq!(settle(&index, &build.job).await.phase, "ready");
    let first = index
        .search(&build.index, "alpha", options(SearchMode::Lexical))
        .await
        .unwrap();
    let mut jobs = vec![build.job.clone()];
    for n in 1..3 {
        std::fs::write(root.path().join("a.txt"), format!("alpha {n}\n")).unwrap();
        let update = index.start_update(&build.index).await.unwrap();
        assert_eq!(settle(&index, &update.job).await.phase, "ready");
        jobs.push(update.job);
    }
    // The newest two are kept.
    assert_eq!(index.status(&jobs[0]).await.unwrap().phase, "pruned");
    assert_eq!(index.status(&jobs[1]).await.unwrap().phase, "ready");
    assert!(matches!(
        index
            .read_chunks(&build.index, &jobs[0], &[first.hits[0].id.clone()], 0)
            .await,
        Err(Error::Stale(_))
    ));
    assert!(index.cancel(&jobs[0]).await.is_err());
    assert_eq!(rows(&path, "chunks"), 2);
    assert_eq!(
        (rows(&path, "chunk_texts"), rows(&path, "snapshots")),
        (2, 2)
    );
    let events = index.events(&jobs[2], 0, 1000).await.unwrap();
    let pruned = events.iter().find(|e| e.kind == "pruned").unwrap();
    assert_eq!(pruned.detail["generations"], json!([jobs[0]]));
    // Updates continue from the kept generations.
    std::fs::write(root.path().join("a.txt"), "alpha 3\n").unwrap();
    let update = index.start_update(&build.index).await.unwrap();
    assert_eq!(settle(&index, &update.job).await.phase, "ready");
}
/// The schema of the first index databases (`user_version` 1).
const V1_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS indexes(id TEXT PRIMARY KEY,name TEXT NOT NULL,spec TEXT NOT NULL,active TEXT,config TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,index_id TEXT NOT NULL REFERENCES indexes(id) ON DELETE CASCADE,phase TEXT NOT NULL,config TEXT NOT NULL,manifest TEXT,excluded TEXT NOT NULL DEFAULT '[]',affected TEXT NOT NULL DEFAULT '[]',changes TEXT NOT NULL DEFAULT '[]',fresh INTEGER NOT NULL DEFAULT 1,problem TEXT);
CREATE TABLE IF NOT EXISTS snapshots(hash TEXT PRIMARY KEY,content TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files(job TEXT REFERENCES jobs(id) ON DELETE CASCADE,path TEXT,hash TEXT NOT NULL REFERENCES snapshots(hash),original_hash TEXT NOT NULL,state TEXT NOT NULL,PRIMARY KEY(job,path));
CREATE TABLE IF NOT EXISTS vectors(id TEXT PRIMARY KEY,profile TEXT NOT NULL,tokens INTEGER NOT NULL,vector BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY,job TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,path TEXT NOT NULL,hash TEXT NOT NULL REFERENCES snapshots(hash),start INTEGER NOT NULL,end INTEGER NOT NULL,first_line INTEGER NOT NULL,last_line INTEGER NOT NULL,payload TEXT NOT NULL REFERENCES vectors(id),structure TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS chunks_job_payload ON chunks(job,payload);
CREATE INDEX IF NOT EXISTS files_hash ON files(hash);
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(id UNINDEXED,job UNINDEXED,path,terms,body,tokenize='unicode61');
CREATE TABLE IF NOT EXISTS events(sequence INTEGER PRIMARY KEY AUTOINCREMENT,job TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,kind TEXT NOT NULL,detail TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS events_job ON events(job,sequence);
CREATE TABLE IF NOT EXISTS requests(id INTEGER PRIMARY KEY AUTOINCREMENT,job TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,request TEXT NOT NULL,response TEXT,error TEXT);
CREATE TABLE IF NOT EXISTS job_usage(job TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,usage TEXT NOT NULL);
PRAGMA user_version=1;
"#;
#[tokio::test]
async fn databases_of_the_first_schema_migrate_without_losing_data() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    let path = db.path().join("i.db");
    let text = "alpha original\n";
    std::fs::write(root.path().join("a.txt"), text).unwrap();
    let fake = Arc::new(Fake::new());
    let snapshot = Snapshot::new("a.txt", text.as_bytes()).unwrap();
    let payload = payload(&*fake, text.to_owned()).unwrap();
    let root_path = std::fs::canonicalize(root.path()).unwrap();
    // As the first version saved them: two complete generations with the
    // same chunk, each with its own full-text copy.
    let spec = json!({"name": "old", "root": root_path, "include": [], "exclude": []});
    let saved_config = json!({"batch_count": 6, "concurrency": 2, "embedding_concurrency": 2,
        "embedding": fake.profile, "vector_cache_bytes": 67108864, "max_transient_attempts": 10});
    let state = json!({"windows": [{"start": 0, "end": text.len(), "cuts": [], "repair_rounds": 0,
        "forced": false, "problem": null}], "structure": {"status": "unchecked", "protected": []},
        "bypass": false, "complete": true});
    let vector: Vec<u8> = [1.0f32, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    {
        let c = rusqlite::Connection::open(&path).unwrap();
        c.execute_batch(V1_SCHEMA).unwrap();
        c.execute(
            "INSERT INTO indexes VALUES('i','old',?1,'g2',?2)",
            rusqlite::params![spec.to_string(), saved_config.to_string()],
        )
        .unwrap();
        c.execute(
            "INSERT INTO snapshots VALUES(?1,?2)",
            rusqlite::params![snapshot.hash, text],
        )
        .unwrap();
        c.execute(
            "INSERT INTO vectors VALUES(?1,?2,?3,?4)",
            rusqlite::params![
                payload.id,
                serde_json::to_string(&fake.profile).unwrap(),
                payload.tokens,
                vector
            ],
        )
        .unwrap();
        for generation in ["g1", "g2"] {
            c.execute(
                "INSERT INTO jobs(id,index_id,phase,config,manifest) VALUES(?1,'i','ready',?2,'m')",
                rusqlite::params![generation, saved_config.to_string()],
            )
            .unwrap();
            c.execute(
                "INSERT INTO files VALUES(?1,'a.txt',?2,?3,?4)",
                rusqlite::params![
                    generation,
                    snapshot.hash,
                    snapshot.original_hash,
                    state.to_string()
                ],
            )
            .unwrap();
            let chunk = format!("chunk-{generation}");
            c.execute(
                "INSERT INTO chunks VALUES(?1,?2,'a.txt',?3,0,?4,1,1,?5,'unchecked')",
                rusqlite::params![chunk, generation, snapshot.hash, text.len(), payload.id],
            )
            .unwrap();
            c.execute(
                "INSERT INTO chunk_fts(id,job,path,terms,body) VALUES(?1,?2,'a.txt','alpha original',?3)",
                rusqlite::params![chunk, generation, text],
            )
            .unwrap();
            c.execute(
                "INSERT INTO events(job,kind,detail) VALUES(?1,'snapshot',?2)",
                rusqlite::params![
                    generation,
                    json!({"chunking_policy": policy_hash()}).to_string()
                ],
            )
            .unwrap();
        }
    }
    let index = ArkoIndex::open(&path, fake.clone(), config(&fake))
        .await
        .unwrap();
    {
        let c = rusqlite::Connection::open(&path).unwrap();
        let version: i64 = c
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(version, 2);
        let old_table: i64 = c
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE name='chunk_fts'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(old_table, 0);
        let linked: i64 = c
            .query_row(
                "SELECT count(*) FROM chunks WHERE text_id IS NOT NULL",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(linked, 2);
    }
    assert_eq!(rows(&path, "chunk_texts"), 1);
    // Both generations stay searchable and readable.
    for mode in [
        SearchMode::Lexical,
        SearchMode::Semantic,
        SearchMode::Hybrid,
    ] {
        let page = index.search("i", "original", options(mode)).await.unwrap();
        assert_eq!(page.generation, "g2");
        assert_eq!(page.hits.len(), 1);
        assert_eq!(page.hits[0].text, text);
    }
    for generation in ["g1", "g2"] {
        let read = index
            .read_chunks("i", generation, &[format!("chunk-{generation}")], 0)
            .await
            .unwrap();
        assert_eq!(read[0].text, text);
    }
    // An unchanged file keeps its boundaries; a changed one is chunked.
    std::fs::write(root.path().join("b.txt"), "alpha new\nbeta\n").unwrap();
    let update = index.start_update("i").await.unwrap();
    let done = settle(&index, &update.job).await;
    assert_eq!(done.phase, "ready", "{:?}", done.problem);
    assert_eq!(fake.calls.lock().unwrap().len(), 1);
    let page = index
        .search("i", "alpha", options(SearchMode::Lexical))
        .await
        .unwrap();
    assert_eq!(page.hits.len(), 2);
}

#[tokio::test]
async fn duplicate_texts_are_one_result_and_pages_match_the_full_ranking() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("copy1.txt"), "alpha same text here\n").unwrap();
    std::fs::write(root.path().join("copy2.txt"), "alpha same text here\n").unwrap();
    for n in 0..4 {
        std::fs::write(
            root.path().join(format!("{n}.txt")),
            format!("alpha {} {n}\n", "word ".repeat(n + 1)),
        )
        .unwrap();
    }
    let fake = Arc::new(Fake::new());
    let (index, job, status) =
        build_spec(&fake, db.path(), BuildSpec::new("test", root.path())).await;
    assert_eq!(status.phase, "ready", "{:?}", status.problem);
    for mode in [
        SearchMode::Lexical,
        SearchMode::Semantic,
        SearchMode::Hybrid,
    ] {
        let all = SearchOptions {
            mode: mode.clone(),
            top_k: None,
            min_chunk_bytes: 0,
            ..Default::default()
        };
        let full = index
            .search(&job.index, "alpha same", all.clone())
            .await
            .unwrap();
        assert_eq!(full.candidate_count, 5, "{mode:?}");
        let copies: Vec<_> = full
            .hits
            .iter()
            .filter(|h| h.path.starts_with("copy"))
            .collect();
        assert_eq!(copies.len(), 1);
        assert_eq!(copies[0].other_locations, 1);
        let mut paged = vec![];
        let mut cursor = None;
        loop {
            let page = index
                .search(
                    &job.index,
                    "alpha same",
                    SearchOptions {
                        top_k: Some(2),
                        cursor,
                        ..all.clone()
                    },
                )
                .await
                .unwrap();
            assert_eq!(page.candidate_count, 5);
            paged.extend(
                page.hits
                    .iter()
                    .map(|h| (h.id.clone(), h.text.clone(), h.score)),
            );
            cursor = page.next;
            if cursor.is_none() {
                break;
            }
        }
        let whole: Vec<_> = full
            .hits
            .iter()
            .map(|h| (h.id.clone(), h.text.clone(), h.score))
            .collect();
        assert_eq!(paged, whole, "{mode:?}");
    }
}

#[tokio::test]
async fn searches_return_ten_substantial_results_by_default() {
    let defaults = SearchOptions::default();
    assert_eq!(
        (
            defaults.top_k,
            defaults.min_chunk_bytes,
            defaults.rebuild_file_limit
        ),
        (Some(10), 40, Some(100))
    );
    // Saved options without these fields load with the same defaults.
    let saved: SearchOptions = serde_json::from_value(json!({"mode": "lexical"})).unwrap();
    assert_eq!(
        (saved.top_k, saved.min_chunk_bytes, saved.rebuild_file_limit),
        (Some(10), 40, Some(100))
    );
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    for n in 0..15 {
        std::fs::write(
            root.path().join(format!("{n:02}.txt")),
            format!("alpha {n} with enough words to pass the size filter\n"),
        )
        .unwrap();
    }
    std::fs::write(root.path().join("tiny.txt"), "alpha\n").unwrap();
    let fake = Arc::new(Fake::new());
    let (index, job, status) =
        build_spec(&fake, db.path(), BuildSpec::new("test", root.path())).await;
    assert_eq!(status.phase, "ready", "{:?}", status.problem);
    let page = index
        .search(&job.index, "alpha", SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(page.hits.len(), 10);
    assert_eq!(page.candidate_count, 15);
    assert!(page.next.is_some());
    assert!(page.hits.iter().all(|h| h.path != "tiny.txt"));
}
#[tokio::test]
async fn hits_from_files_changed_since_indexing_may_be_stale() {
    let root = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();
    for name in ["kept", "edited", "removed", "pending"] {
        std::fs::write(
            root.path().join(format!("{name}.txt")),
            format!("alpha {name}\nbeta\n"),
        )
        .unwrap();
    }
    let mut fake = Fake::new();
    fake.rejection = Rejection::WrongCallRetryLater;
    let fake = Arc::new(fake);
    let (index, job, status) =
        build_spec(&fake, db.path(), BuildSpec::new("test", root.path())).await;
    assert_eq!(status.phase, "ready", "{:?}", status.problem);
    let stale = |page: &SearchPage| -> Vec<(String, bool)> {
        let mut marks: Vec<_> = page
            .hits
            .iter()
            .map(|h| (h.path.clone(), h.may_be_stale))
            .collect();
        marks.sort();
        marks
    };
    let search = || index.search(&job.index, "alpha", options(SearchMode::Lexical));
    assert!(
        stale(&search().await.unwrap())
            .iter()
            .all(|(_, stale)| !stale)
    );
    // Edited or removed on disk, not yet reindexed.
    std::fs::write(root.path().join("edited.txt"), "alpha edited again\nbeta\n").unwrap();
    std::fs::remove_file(root.path().join("removed.txt")).unwrap();
    assert_eq!(
        stale(&search().await.unwrap()),
        [
            ("edited.txt".to_string(), true),
            ("kept.txt".to_string(), false),
            ("pending.txt".to_string(), false),
            ("removed.txt".to_string(), true),
        ]
    );
    // An unfinished update lists its affected files, even ones restored since.
    std::fs::write(
        root.path().join("pending.txt"),
        "alpha pending changed\nbeta\n",
    )
    .unwrap();
    fake.rejected_chunks.store(usize::MAX, Ordering::SeqCst);
    let update = index.start_update(&job.index).await.unwrap();
    wait_for_retry(&index, &update.job).await;
    std::fs::write(root.path().join("pending.txt"), "alpha pending\nbeta\n").unwrap();
    let marks = stale(&search().await.unwrap());
    assert!(
        marks.contains(&("pending.txt".to_string(), true)),
        "{marks:?}"
    );
    index.cancel(&update.job).await.unwrap();
    settle(&index, &update.job).await;
}
