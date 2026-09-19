//! Exact memory views, with delivery-confirmed baselines scoped to a task.
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone)]
struct Snapshot {
    version: usize,
    content: String,
}

#[derive(Debug)]
struct PendingRead {
    output: String,
    snapshot: Snapshot,
}

#[derive(Debug, Default)]
struct TierReads {
    baseline: Option<Snapshot>,
    pending: Vec<PendingRead>,
}

#[derive(Debug, Default)]
pub(super) struct MemoryReads {
    tasks: Mutex<HashMap<Uuid, HashMap<String, TierReads>>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ReadRecord {
    tier: String,
    version: usize,
    #[serde(flatten)]
    change: Change,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Change {
    Full {
        content: String,
    },
    Append {
        base_version: usize,
        content: String,
    },
    Replace {
        base_version: usize,
        start_line: usize,
        delete_lines: usize,
        content: String,
    },
    Unchanged,
    Partial {
        total_chars: usize,
        content: String,
    },
}

fn changed_text(before: &Snapshot, after: &str) -> Change {
    if before.content == after {
        return Change::Unchanged;
    }
    if let Some(appended) = after.strip_prefix(&before.content) {
        return Change::Append {
            base_version: before.version,
            content: appended.to_string(),
        };
    }
    // One exact replacement hunk. Linear work, no diff dependency or quadratic LCS.
    let old = before.content.split_inclusive('\n').collect::<Vec<_>>();
    let new = after.split_inclusive('\n').collect::<Vec<_>>();
    let prefix = old.iter().zip(&new).take_while(|(a, b)| a == b).count();
    let suffix = old[prefix..]
        .iter()
        .rev()
        .zip(new[prefix..].iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    Change::Replace {
        base_version: before.version,
        start_line: prefix,
        delete_lines: old.len() - prefix - suffix,
        content: new[prefix..new.len() - suffix].concat(),
    }
}

impl MemoryReads {
    pub fn reset(&self, task: Uuid) {
        self.tasks.lock().remove(&task);
    }

    pub fn prepare(
        &self,
        task: Uuid,
        tier: &str,
        version: usize,
        content: String,
        full: bool,
        max_chars: Option<usize>,
    ) -> Result<String, serde_json::Error> {
        if let Some(limit) = max_chars {
            let total = content.chars().count();
            if total > limit {
                return serde_json::to_string_pretty(&ReadRecord {
                    tier: tier.into(),
                    version,
                    change: Change::Partial {
                        total_chars: total,
                        content: content.chars().skip(total - limit).collect(),
                    },
                });
            }
        }
        let mut tasks = self.tasks.lock();
        let reads = tasks
            .entry(task)
            .or_default()
            .entry(tier.into())
            .or_default();
        let snapshot = Snapshot { version, content };
        let encode = |change| {
            serde_json::to_string_pretty(&ReadRecord {
                tier: tier.into(),
                version,
                change,
            })
        };
        let full_output = encode(Change::Full {
            content: snapshot.content.clone(),
        })?;
        let output = match reads.baseline.as_ref().filter(|_| !full) {
            Some(before) => {
                let change = changed_text(before, &snapshot.content);
                let unchanged = matches!(change, Change::Unchanged);
                let delta = encode(change)?;
                if unchanged || delta.len() < full_output.len() {
                    delta
                } else {
                    full_output
                }
            }
            None => full_output,
        };
        if !reads.pending.iter().any(|pending| pending.output == output) {
            reads.pending.push(PendingRead {
                output: output.clone(),
                snapshot,
            });
        }
        Ok(output)
    }

    pub fn observe(&self, task: Uuid, message: &arko_ai::Message) {
        if message.role != arko_ai::Role::User {
            return;
        }
        let mut tasks = self.tasks.lock();
        let Some(tiers) = tasks.get_mut(&task) else {
            return;
        };
        for reads in tiers.values_mut() {
            reads.pending.retain(|pending| {
                let delivered = message.content.iter().any(|part| match part {
                    arko_ai::Content::ToolResult(result) => {
                        !result.is_error
                            && result.name == super::tools::READ_MEMORY_TOOL_NAME
                            && (result.output == pending.output
                                || result.output.starts_with(&format!(
                                    "{}\n\nMessage from user on approval:\n",
                                    pending.output
                                )))
                    }
                    arko_ai::Content::Text(text) => text.contains(&pending.output),
                    _ => false,
                });
                if delivered
                    && reads
                        .baseline
                        .as_ref()
                        .is_none_or(|b| b.version <= pending.snapshot.version)
                {
                    reads.baseline = Some(pending.snapshot.clone());
                }
                !delivered
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arko_ai::{Content, Message, ToolResult};

    fn decode(output: &str) -> ReadRecord {
        serde_json::from_str(output).unwrap()
    }
    fn deliver(reads: &MemoryReads, task: Uuid, output: &str) {
        reads.observe(
            task,
            &Message::new_user_with_contents(vec![Content::ToolResult(ToolResult::new(
                Some("call-1".into()),
                "read_memory",
                output,
            ))]),
        );
    }
    // Independent consumer of the public JSON contract.
    fn apply(before: &str, record: ReadRecord) -> String {
        match record.change {
            Change::Full { content } => content,
            Change::Append { content, .. } => format!("{before}{content}"),
            Change::Replace {
                start_line,
                delete_lines,
                content,
                ..
            } => {
                let mut lines = before
                    .split_inclusive('\n')
                    .map(str::to_owned)
                    .collect::<Vec<_>>();
                lines.splice(start_line..start_line + delete_lines, [content]);
                lines.concat()
            }
            Change::Unchanged => before.to_string(),
            Change::Partial { .. } => panic!("partial view cannot reconstruct a snapshot"),
        }
    }

    #[test]
    fn first_full_then_exact_append_unchanged_and_forced_full() {
        let reads = MemoryReads::default();
        let task = Uuid::new_v4();
        let old = "A long accepted requirement.\n".repeat(40);
        let first = reads
            .prepare(task, "requirements", 1, old.clone(), false, None)
            .unwrap();
        assert!(matches!(decode(&first).change, Change::Full { .. }));
        deliver(&reads, task, &first);
        let new = format!("{old}Retain source references.\n");
        let delta = reads
            .prepare(task, "requirements", 2, new.clone(), false, None)
            .unwrap();
        assert!(matches!(
            decode(&delta).change,
            Change::Append {
                base_version: 1,
                ..
            }
        ));
        assert_eq!(apply(&old, decode(&delta)), new);
        assert!(delta.len() < first.len() / 2);
        deliver(&reads, task, &delta);
        let unchanged = reads
            .prepare(task, "requirements", 3, new.clone(), false, None)
            .unwrap();
        assert!(matches!(decode(&unchanged).change, Change::Unchanged));
        deliver(&reads, task, &unchanged);
        let full = reads
            .prepare(task, "requirements", 3, new.clone(), true, None)
            .unwrap();
        assert_eq!(apply("", decode(&full)), new);
        assert!(matches!(decode(&full).change, Change::Full { .. }));
    }

    #[test]
    fn replacements_reconstruct_unicode_crlf_deletions_and_missing_final_newlines() {
        let cases = [
            "",
            "a",
            "a\n",
            "a\r\n",
            "α\n🦀\n終",
            "first\nsecond\nthird\n",
            "first\nthird\n",
            "new\nfirst\nsecond\nthird\n",
        ];
        for old in cases {
            for new in cases {
                let before = Snapshot {
                    version: 4,
                    content: old.into(),
                };
                let record = ReadRecord {
                    tier: "test".into(),
                    version: 5,
                    change: changed_text(&before, new),
                };
                assert_eq!(apply(old, record), new, "{old:?} -> {new:?}");
            }
        }
        let reads = MemoryReads::default();
        let task = Uuid::new_v4();
        let old = format!(
            "{}remove this\n{}",
            "prefix\n".repeat(100),
            "suffix\n".repeat(100)
        );
        let first = reads
            .prepare(task, "state", 1, old.clone(), false, None)
            .unwrap();
        deliver(&reads, task, &first);
        let new = old.replace("remove this\n", "");
        let delta = reads
            .prepare(task, "state", 2, new.clone(), false, None)
            .unwrap();
        assert!(matches!(
            decode(&delta).change,
            Change::Replace {
                delete_lines: 1,
                ..
            }
        ));
        assert_eq!(apply(&old, decode(&delta)), new);
    }

    #[test]
    fn delivery_is_required_and_baselines_are_per_task_and_tier() {
        let reads = MemoryReads::default();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let text = "known facts\n".repeat(60);
        let first = reads
            .prepare(a, "facts", 1, text.clone(), false, None)
            .unwrap();
        let unread = reads
            .prepare(a, "facts", 1, text.clone(), false, None)
            .unwrap();
        assert!(matches!(decode(&unread).change, Change::Full { .. }));
        reads.observe(a, &Message::new_assistant(&first));
        reads.observe(a, &Message::new_user("A summary of the facts."));
        assert!(matches!(
            decode(
                &reads
                    .prepare(a, "facts", 1, text.clone(), false, None)
                    .unwrap()
            )
            .change,
            Change::Full { .. }
        ));
        deliver(&reads, a, &first);
        for (task, tier) in [(b, "facts"), (a, "other")] {
            assert!(matches!(
                decode(
                    &reads
                        .prepare(task, tier, 1, text.clone(), false, None)
                        .unwrap()
                )
                .change,
                Change::Full { .. }
            ));
        }
        reads.reset(a);
        assert!(matches!(
            decode(&reads.prepare(a, "facts", 1, text, false, None).unwrap()).change,
            Change::Full { .. }
        ));
    }

    #[test]
    fn partial_and_failed_results_do_not_establish_or_advance_a_baseline() {
        let reads = MemoryReads::default();
        let task = Uuid::new_v4();
        let text = "αβγ\n".repeat(100);
        let partial = reads
            .prepare(task, "state", 1, text.clone(), false, Some(4))
            .unwrap();
        assert!(matches!(
            decode(&partial).change,
            Change::Partial {
                total_chars: 400,
                ..
            }
        ));
        deliver(&reads, task, &partial);
        let first = reads
            .prepare(task, "state", 1, text.clone(), false, None)
            .unwrap();
        assert!(matches!(decode(&first).change, Change::Full { .. }));
        let mut error = ToolResult::new(None, "read_memory", &first);
        error.is_error = true;
        reads.observe(
            task,
            &Message::new_user_with_contents(vec![Content::ToolResult(error)]),
        );
        assert!(matches!(
            decode(
                &reads
                    .prepare(task, "state", 1, text.clone(), false, None)
                    .unwrap()
            )
            .change,
            Change::Full { .. }
        ));
        // Text-mode tool results contain the exact record inside their wrapper.
        reads.observe(
            task,
            &Message::new_user(&format!("Tool read_memory returned:\n{first}\n")),
        );
        let changed = format!("{text}new");
        let partial = reads
            .prepare(task, "state", 2, changed.clone(), false, Some(4))
            .unwrap();
        deliver(&reads, task, &partial);
        let delta = reads
            .prepare(task, "state", 2, changed, false, None)
            .unwrap();
        assert!(matches!(
            decode(&delta).change,
            Change::Append {
                base_version: 1,
                ..
            }
        ));
    }

    #[test]
    fn small_replacements_fall_back_to_full_and_out_of_order_delivery_cannot_regress() {
        let reads = MemoryReads::default();
        let task = Uuid::new_v4();
        let first = reads
            .prepare(task, "x", 1, "a".into(), false, None)
            .unwrap();
        deliver(&reads, task, &first);
        let second = reads
            .prepare(task, "x", 2, "b".into(), false, None)
            .unwrap();
        let third = reads
            .prepare(task, "x", 3, "c".into(), false, None)
            .unwrap();
        assert!(matches!(decode(&second).change, Change::Full { .. }));
        deliver(&reads, task, &third);
        deliver(&reads, task, &second);
        assert!(matches!(
            decode(
                &reads
                    .prepare(task, "x", 3, "c".into(), false, None)
                    .unwrap()
            )
            .change,
            Change::Unchanged
        ));
    }
}
