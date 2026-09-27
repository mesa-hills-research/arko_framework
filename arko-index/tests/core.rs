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
        }
    }
}
#[async_trait::async_trait]
impl Backend for Fake {
    async fn chunk(&self, r: ModelRequest) -> Result<ModelReply> {
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max.fetch_max(active, Ordering::SeqCst);
        self.calls.lock().unwrap().push(r.clone());
        tokio::time::sleep(self.delay).await;
        self.active.fetch_sub(1, Ordering::SeqCst);
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
        Ok(ModelReply {
            value: Value::Object(out),
            usage: json!({}),
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
fn options(mode: SearchMode) -> SearchOptions {
    SearchOptions {
        mode,
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
    let fake = Arc::new(Fake::new());
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
        let calls = fake.calls.lock().unwrap();
        assert_eq!(
            calls
                .iter()
                .map(|r| r.schema["properties"].as_object().unwrap().len())
                .collect::<Vec<_>>(),
            vec![6, 6, 1]
        );
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
    while index.status(&update.job).await.unwrap().active_luna == 0 {
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
    let calls = fake.calls.lock().unwrap();
    let repairs: Vec<_> = calls.iter().filter(|r| r.tool == "repair_cuts").collect();
    assert!(!repairs.is_empty());
    assert!(repairs.iter().all(|r| !r.prompt.contains("FILE f")));
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
    assert_eq!(settle(&index, &update.job).await.phase, "blocked");
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
            Duration::from_millis(150),
            Duration::from_secs(3),
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
    let job = index
        .start_build(BuildSpec::new("test", root.path()))
        .await
        .unwrap();
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
