//! Replay-based durable steps for code-defined workflows.

use std::future::Future;
use std::path::PathBuf;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::error::WorkflowError;

const JOURNAL_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum DurableRetryPolicy {
    #[default]
    Never,
    WhenDefinitelyNotCompleted {
        max_attempts: u32,
    },
    RetrySafe {
        max_attempts: u32,
    },
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;

    struct TempRun(PathBuf);

    impl TempRun {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("arko-durable-test-{}", Uuid::new_v4()));
            Self(path)
        }
    }

    impl Drop for TempRun {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[tokio::test]
    async fn completed_step_replays_without_executing() {
        let run = TempRun::new();
        let store = DurableStore::new(Uuid::new_v4(), run.0.clone());
        let calls = AtomicUsize::new(0);
        let spec = DurableStepSpec::new("calculate");

        let first: u32 = store
            .execute(&spec, &"input", |_| async {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(42)
            })
            .await
            .unwrap();
        let second: u32 = store
            .execute(&spec, &"input", |_| async {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(99)
            })
            .await
            .unwrap();

        assert_eq!((first, second), (42, 42));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn uncertain_error_is_not_retried_by_default() {
        let run = TempRun::new();
        let store = DurableStore::new(Uuid::new_v4(), run.0.clone());
        let calls = AtomicUsize::new(0);
        let spec = DurableStepSpec::new("send-message");

        let first: Result<String, WorkflowError> = store
            .execute(&spec, &"hello", |_| async {
                calls.fetch_add(1, Ordering::SeqCst);
                Err(WorkflowError::Persistence("connection lost".to_string()))
            })
            .await;
        assert!(matches!(first, Err(WorkflowError::DurableInDoubt { .. })));

        let second: Result<String, WorkflowError> = store
            .execute(&spec, &"hello", |_| async {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok("sent-again".to_string())
            })
            .await;
        assert!(matches!(second, Err(WorkflowError::DurableInDoubt { .. })));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    struct CompletedResolver;

    #[async_trait]
    impl ResolveInDoubt<String, String> for CompletedResolver {
        async fn resolve(
            &self,
            _context: &InDoubtContext,
            input: &String,
        ) -> Result<InDoubtResolution<String>, WorkflowError> {
            Ok(InDoubtResolution::Completed(format!("receipt:{input}")))
        }
    }

    #[tokio::test]
    async fn resolver_can_recover_completed_output() {
        let run = TempRun::new();
        let store = DurableStore::new(Uuid::new_v4(), run.0.clone());
        let spec = DurableStepSpec::new("pay");
        let input = "bill-1".to_string();

        let _: Result<String, WorkflowError> = store
            .execute(&spec, &input, |_| async {
                Err(WorkflowError::Persistence("timeout".to_string()))
            })
            .await;

        let output = store
            .execute_with_resolver(
                &spec,
                &input,
                |_| async { Ok("must-not-run".to_string()) },
                &CompletedResolver,
            )
            .await
            .unwrap();
        assert_eq!(output, "receipt:bill-1");
    }

    #[tokio::test]
    async fn changed_input_is_rejected_for_same_identity() {
        let run = TempRun::new();
        let store = DurableStore::new(Uuid::new_v4(), run.0.clone());
        let spec = DurableStepSpec::new("lookup");
        let _: String = store
            .execute(&spec, &"a", |_| async { Ok("result".to_string()) })
            .await
            .unwrap();

        let changed: Result<String, WorkflowError> = store
            .execute(&spec, &"b", |_| async { Ok("wrong".to_string()) })
            .await;
        assert!(matches!(
            changed,
            Err(WorkflowError::DurableDefinitionMismatch { .. })
        ));
    }

    #[tokio::test]
    async fn concurrent_same_identity_does_not_execute_twice() {
        let run = TempRun::new();
        let store = Arc::new(DurableStore::new(Uuid::new_v4(), run.0.clone()));
        let spec = DurableStepSpec::new("slow-effect");
        let started = Arc::new(tokio::sync::Notify::new());
        let calls = Arc::new(AtomicUsize::new(0));

        let first_store = store.clone();
        let first_spec = spec.clone();
        let first_started = started.clone();
        let first_calls = calls.clone();
        let first = tokio::spawn(async move {
            first_store
                .execute(&first_spec, &"input", |_| async move {
                    first_calls.fetch_add(1, Ordering::SeqCst);
                    first_started.notify_one();
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    Ok::<_, WorkflowError>("done".to_string())
                })
                .await
        });

        started.notified().await;
        let second_calls = calls.clone();
        let second: Result<String, WorkflowError> = store
            .execute(&spec, &"input", |_| async move {
                second_calls.fetch_add(1, Ordering::SeqCst);
                Ok("duplicate".to_string())
            })
            .await;

        assert!(matches!(second, Err(WorkflowError::DurableInDoubt { .. })));
        assert_eq!(first.await.unwrap().unwrap(), "done");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn replay_of_started_record_is_in_doubt_without_running_effect() {
        let run = TempRun::new();
        let run_id = Uuid::new_v4();
        let store = DurableStore::new(run_id, run.0.clone());
        let spec = DurableStepSpec::new("crashed-before-effect");
        let prepared: Preparation<String> = store.prepare(&spec, &"input", false).await.unwrap();
        assert!(matches!(prepared, Preparation::Execute(_)));

        let replay = DurableStore::new(run_id, run.0.clone());
        let calls = AtomicUsize::new(0);
        let result: Result<String, WorkflowError> = replay
            .execute(&spec, &"input", |_| async {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok("duplicate".to_string())
            })
            .await;
        assert!(matches!(result, Err(WorkflowError::DurableInDoubt { .. })));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    struct DefinitelyNotCompleted;

    #[async_trait]
    impl ResolveInDoubt<String, String> for DefinitelyNotCompleted {
        async fn resolve(
            &self,
            _context: &InDoubtContext,
            _input: &String,
        ) -> Result<InDoubtResolution<String>, WorkflowError> {
            Ok(InDoubtResolution::DefinitelyNotCompleted)
        }
    }

    #[tokio::test]
    async fn definitely_not_completed_retries_only_under_matching_policy() {
        let run = TempRun::new();
        let store = DurableStore::new(Uuid::new_v4(), run.0.clone());
        let spec = DurableStepSpec::new("safe-retry")
            .retry_policy(DurableRetryPolicy::WhenDefinitelyNotCompleted { max_attempts: 2 });
        let input = "payment".to_string();
        let _: Result<String, WorkflowError> = store
            .execute(&spec, &input, |_| async {
                Err(WorkflowError::Persistence("unknown".to_string()))
            })
            .await;
        let calls = AtomicUsize::new(0);
        let recovered = store
            .execute_with_resolver(
                &spec,
                &input,
                |_| async {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok("retried".to_string())
                },
                &DefinitelyNotCompleted,
            )
            .await
            .unwrap();
        assert_eq!(recovered, "retried");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    struct StillInDoubt;

    #[async_trait]
    impl ResolveInDoubt<String, String> for StillInDoubt {
        async fn resolve(
            &self,
            _context: &InDoubtContext,
            _input: &String,
        ) -> Result<InDoubtResolution<String>, WorkflowError> {
            Ok(InDoubtResolution::StillInDoubt {
                reason: "provider has no lookup API".to_string(),
            })
        }
    }

    #[tokio::test]
    async fn still_in_doubt_keeps_step_suspended() {
        let run = TempRun::new();
        let store = DurableStore::new(Uuid::new_v4(), run.0.clone());
        let spec = DurableStepSpec::new("unknown-message");
        let input = "hello".to_string();
        let _: Result<String, WorkflowError> = store
            .execute(&spec, &input, |_| async {
                Err(WorkflowError::Persistence("lost response".to_string()))
            })
            .await;
        let calls = AtomicUsize::new(0);
        let result = store
            .execute_with_resolver(
                &spec,
                &input,
                |_| async {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, WorkflowError>("duplicate".to_string())
                },
                &StillInDoubt,
            )
            .await;
        assert!(matches!(result, Err(WorkflowError::DurableInDoubt { .. })));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(store.list_in_doubt().await.unwrap().len(), 1);
    }

    struct BadOutput;

    impl Serialize for BadOutput {
        fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            Err(serde::ser::Error::custom("cannot serialize receipt"))
        }
    }

    impl<'de> Deserialize<'de> for BadOutput {
        fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            Ok(Self)
        }
    }

    #[tokio::test]
    async fn output_serialization_failure_is_uncertain() {
        let run = TempRun::new();
        let store = DurableStore::new(Uuid::new_v4(), run.0.clone());
        let spec = DurableStepSpec::new("effect-with-bad-output");
        let result: Result<BadOutput, WorkflowError> = store
            .execute(&spec, &"input", |_| async { Ok(BadOutput) })
            .await;
        assert!(matches!(result, Err(WorkflowError::DurableInDoubt { .. })));
        assert_eq!(store.list_in_doubt().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn torn_final_record_preserves_completed_replay() {
        let run = TempRun::new();
        let run_id = Uuid::new_v4();
        let store = DurableStore::new(run_id, run.0.clone());
        let spec = DurableStepSpec::new("torn-tail");
        let _: String = store
            .execute(&spec, &"input", |_| async { Ok("committed".to_string()) })
            .await
            .unwrap();
        use std::io::Write;
        let mut journal = std::fs::OpenOptions::new()
            .append(true)
            .open(store.journal_path())
            .unwrap();
        journal.write_all(b"{\"torn\":").unwrap();
        journal.sync_data().unwrap();

        let replay = DurableStore::new(run_id, run.0.clone());
        let output: String = replay
            .execute(&spec, &"input", |_| async { Ok("wrong".to_string()) })
            .await
            .unwrap();
        assert_eq!(output, "committed");
    }

    #[tokio::test]
    async fn operator_resolution_is_append_only_and_replays() {
        let run = TempRun::new();
        let store = DurableStore::new(Uuid::new_v4(), run.0.clone());
        let spec = DurableStepSpec::new("operator-resolution");
        let input = "bill".to_string();
        let _: Result<String, WorkflowError> = store
            .execute(&spec, &input, |_| async {
                Err(WorkflowError::Persistence("ambiguous".to_string()))
            })
            .await;
        store
            .operator_complete(&spec, &input, &"receipt-7", "alice", "checked provider")
            .await
            .unwrap();
        let output: String = store
            .execute(&spec, &input, |_| async { Ok("duplicate".to_string()) })
            .await
            .unwrap();
        assert_eq!(output, "receipt-7");
        let journal = std::fs::read_to_string(store.journal_path()).unwrap();
        assert!(journal.contains("operator:alice"));
        assert!(journal.lines().count() >= 3);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurableStepSpec {
    pub key: String,
    pub instance: Option<String>,
    pub version: String,
    pub retry_policy: DurableRetryPolicy,
}

impl DurableStepSpec {
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            instance: None,
            version: "v1".to_string(),
            retry_policy: DurableRetryPolicy::Never,
        }
    }

    pub fn instance(mut self, instance: impl Into<String>) -> Self {
        self.instance = Some(instance.into());
        self
    }

    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    pub fn retry_policy(mut self, policy: DurableRetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    pub fn display_key(&self) -> String {
        match &self.instance {
            Some(instance) => format!("{}[{instance}]", self.key),
            None => self.key.clone(),
        }
    }

    fn validate(&self) -> Result<(), WorkflowError> {
        if self.key.trim().is_empty() || self.key.contains(['\n', '\r', '\0']) {
            return Err(WorkflowError::BuilderInvalid(
                "durable step key must be non-empty and contain no control separators".to_string(),
            ));
        }
        if self.version.trim().is_empty() {
            return Err(WorkflowError::BuilderInvalid(
                "durable step version must be non-empty".to_string(),
            ));
        }
        if self
            .instance
            .as_ref()
            .is_some_and(|value| value.contains(['\n', '\r', '\0']))
        {
            return Err(WorkflowError::BuilderInvalid(
                "durable step instance contains an invalid control separator".to_string(),
            ));
        }
        Ok(())
    }

    fn identity(&self) -> String {
        format!("{}\0{}", self.key, self.instance.as_deref().unwrap_or(""))
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DurableStepStatus {
    Started,
    Completed,
    InDoubt,
    PermanentlyFailed,
}

#[derive(Debug, Clone)]
pub struct DurableStepContext {
    pub workflow_run_id: Uuid,
    pub step_id: Uuid,
    pub operation_id: String,
    pub attempt: u32,
}

impl DurableStepContext {
    pub fn operation_id(&self) -> &str {
        &self.operation_id
    }
}

#[derive(Debug, Clone)]
pub struct InDoubtContext {
    pub workflow_run_id: Uuid,
    pub step_id: Uuid,
    pub operation_id: String,
    pub attempt: u32,
    pub reason: Option<String>,
}

#[derive(Debug)]
pub enum InDoubtResolution<O> {
    Completed(O),
    DefinitelyNotCompleted,
    StillInDoubt { reason: String },
    PermanentlyFailed { reason: String },
}

#[async_trait]
pub trait ResolveInDoubt<I, O>: Send + Sync {
    async fn resolve(
        &self,
        context: &InDoubtContext,
        input: &I,
    ) -> Result<InDoubtResolution<O>, WorkflowError>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JournalRecord {
    schema_version: u32,
    sequence: u64,
    workflow_run_id: Uuid,
    step_id: Uuid,
    identity: String,
    key: String,
    instance: Option<String>,
    step_version: String,
    input_hash: String,
    operation_id: String,
    attempt: u32,
    status: DurableStepStatus,
    timestamp: chrono::DateTime<chrono::Utc>,
    output: Option<Value>,
    reason: Option<String>,
    resolution_source: Option<String>,
}

enum Preparation<O> {
    Replay(O),
    Execute(DurableStepContext),
    InDoubt(InDoubtContext),
}

/// Append-only durable journal for one workflow run.
pub struct DurableStore {
    workflow_run_id: Uuid,
    journal_path: PathBuf,
    lock: Mutex<()>,
    active_steps: parking_lot::Mutex<std::collections::HashSet<String>>,
}

struct ExecutionLease<'a> {
    active_steps: &'a parking_lot::Mutex<std::collections::HashSet<String>>,
    identity: String,
}

impl Drop for ExecutionLease<'_> {
    fn drop(&mut self) {
        self.active_steps.lock().remove(&self.identity);
    }
}

impl DurableStore {
    pub fn new(workflow_run_id: Uuid, run_dir: PathBuf) -> Self {
        Self {
            workflow_run_id,
            journal_path: run_dir.join("durable").join("journal.jsonl"),
            lock: Mutex::new(()),
            active_steps: parking_lot::Mutex::new(std::collections::HashSet::new()),
        }
    }

    fn execution_lease(&self, spec: &DurableStepSpec) -> Result<ExecutionLease<'_>, WorkflowError> {
        let identity = spec.identity();
        if !self.active_steps.lock().insert(identity.clone()) {
            return Err(WorkflowError::DurableInDoubt {
                step: spec.display_key(),
                operation_id: "active-in-this-process".to_string(),
            });
        }
        Ok(ExecutionLease {
            active_steps: &self.active_steps,
            identity,
        })
    }

    pub fn journal_path(&self) -> &std::path::Path {
        &self.journal_path
    }

    pub fn workflow_run_id(&self) -> Uuid {
        self.workflow_run_id
    }

    async fn records(&self) -> Result<Vec<JournalRecord>, WorkflowError> {
        let bytes = match tokio::fs::read(&self.journal_path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };

        let text = String::from_utf8(bytes)
            .map_err(|error| WorkflowError::Persistence(error.to_string()))?;
        let nonempty = text
            .lines()
            .enumerate()
            .filter(|(_, line)| !line.trim().is_empty())
            .collect::<Vec<_>>();
        let mut records = Vec::with_capacity(nonempty.len());
        for (position, (line_index, line)) in nonempty.iter().enumerate() {
            match serde_json::from_str::<JournalRecord>(line) {
                Ok(record) => records.push(record),
                Err(_) if position + 1 == nonempty.len() => {
                    log::warn!(
                        "Ignoring torn final durable journal record at line {}",
                        line_index + 1
                    );
                }
                Err(error) => {
                    return Err(WorkflowError::Persistence(format!(
                        "invalid durable journal record at line {}: {error}",
                        line_index + 1
                    )));
                }
            }
        }
        Ok(records)
    }

    async fn append_locked(&self, record: &JournalRecord) -> Result<(), WorkflowError> {
        if let Some(parent) = self.journal_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.journal_path)
            .await?;
        let mut encoded = serde_json::to_vec(record)?;
        encoded.push(b'\n');
        file.write_all(&encoded).await?;
        file.flush().await?;
        file.sync_data().await?;
        Ok(())
    }

    fn input_hash<I: Serialize>(input: &I) -> Result<String, WorkflowError> {
        let value = serde_json::to_value(input)?;
        let bytes = serde_json::to_vec(&value)?;
        let digest = Sha256::digest(bytes);
        Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
    }

    fn latest_for<'a>(records: &'a [JournalRecord], identity: &str) -> Option<&'a JournalRecord> {
        records
            .iter()
            .rev()
            .find(|record| record.identity == identity)
    }

    fn validate_match(
        spec: &DurableStepSpec,
        input_hash: &str,
        record: &JournalRecord,
    ) -> Result<(), WorkflowError> {
        if record.step_version != spec.version || record.input_hash != input_hash {
            return Err(WorkflowError::DurableDefinitionMismatch {
                step: spec.display_key(),
                details: format!(
                    "recorded version/hash {}/{}, requested {}/{}",
                    record.step_version, record.input_hash, spec.version, input_hash
                ),
            });
        }
        Ok(())
    }

    fn retry_limit(policy: &DurableRetryPolicy) -> Option<u32> {
        match policy {
            DurableRetryPolicy::Never => None,
            DurableRetryPolicy::WhenDefinitelyNotCompleted { max_attempts }
            | DurableRetryPolicy::RetrySafe { max_attempts } => Some(*max_attempts),
        }
    }

    async fn prepare<I, O>(
        &self,
        spec: &DurableStepSpec,
        input: &I,
        definitely_not_completed: bool,
    ) -> Result<Preparation<O>, WorkflowError>
    where
        I: Serialize,
        O: DeserializeOwned,
    {
        spec.validate()?;
        let _guard = self.lock.lock().await;
        let records = self.records().await?;
        let identity = spec.identity();
        let input_hash = Self::input_hash(input)?;

        let (step_id, operation_id, next_attempt, should_execute) =
            if let Some(record) = Self::latest_for(&records, &identity) {
                Self::validate_match(spec, &input_hash, record)?;
                match record.status {
                    DurableStepStatus::Completed => {
                        let output = record.output.clone().ok_or_else(|| {
                            WorkflowError::Persistence(format!(
                                "completed durable step '{}' has no output",
                                spec.display_key()
                            ))
                        })?;
                        return Ok(Preparation::Replay(serde_json::from_value(output)?));
                    }
                    DurableStepStatus::PermanentlyFailed => {
                        return Err(WorkflowError::DurablePermanentlyFailed {
                            step: spec.display_key(),
                            reason: record.reason.clone().unwrap_or_default(),
                        });
                    }
                    DurableStepStatus::Started => {
                        let in_doubt = JournalRecord {
                            schema_version: JOURNAL_SCHEMA_VERSION,
                            sequence: records.last().map_or(1, |last| last.sequence + 1),
                            workflow_run_id: self.workflow_run_id,
                            step_id: record.step_id,
                            identity: identity.clone(),
                            key: spec.key.clone(),
                            instance: spec.instance.clone(),
                            step_version: spec.version.clone(),
                            input_hash: input_hash.clone(),
                            operation_id: record.operation_id.clone(),
                            attempt: record.attempt,
                            status: DurableStepStatus::InDoubt,
                            timestamp: chrono::Utc::now(),
                            output: None,
                            reason: Some("process ended without a completion record".to_string()),
                            resolution_source: Some("replay-detection".to_string()),
                        };
                        self.append_locked(&in_doubt).await?;
                        (
                            record.step_id,
                            record.operation_id.clone(),
                            record.attempt + 1,
                            matches!(spec.retry_policy, DurableRetryPolicy::RetrySafe { .. }),
                        )
                    }
                    DurableStepStatus::InDoubt => (
                        record.step_id,
                        record.operation_id.clone(),
                        record.attempt + 1,
                        matches!(spec.retry_policy, DurableRetryPolicy::RetrySafe { .. })
                            || (definitely_not_completed
                                && matches!(
                                    spec.retry_policy,
                                    DurableRetryPolicy::WhenDefinitelyNotCompleted { .. }
                                ))
                            || record
                                .resolution_source
                                .as_deref()
                                .is_some_and(|source| source.starts_with("operator-retry:")),
                    ),
                }
            } else {
                (Uuid::new_v4(), Uuid::new_v4().to_string(), 1, true)
            };

        if !should_execute {
            return Ok(Preparation::InDoubt(InDoubtContext {
                workflow_run_id: self.workflow_run_id,
                step_id,
                operation_id,
                attempt: next_attempt.saturating_sub(1),
                reason: Some("outcome is not known to be safe to retry".to_string()),
            }));
        }

        if Self::retry_limit(&spec.retry_policy).is_some_and(|limit| next_attempt > limit.max(1)) {
            return Err(WorkflowError::DurableAttemptsExhausted {
                step: spec.display_key(),
            });
        }

        let started = JournalRecord {
            schema_version: JOURNAL_SCHEMA_VERSION,
            sequence: records.last().map_or(1, |last| last.sequence + 1),
            workflow_run_id: self.workflow_run_id,
            step_id,
            identity,
            key: spec.key.clone(),
            instance: spec.instance.clone(),
            step_version: spec.version.clone(),
            input_hash,
            operation_id: operation_id.clone(),
            attempt: next_attempt,
            status: DurableStepStatus::Started,
            timestamp: chrono::Utc::now(),
            output: None,
            reason: None,
            resolution_source: None,
        };
        self.append_locked(&started).await?;

        Ok(Preparation::Execute(DurableStepContext {
            workflow_run_id: self.workflow_run_id,
            step_id,
            operation_id,
            attempt: next_attempt,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    async fn transition<O: Serialize>(
        &self,
        spec: &DurableStepSpec,
        input_hash: String,
        context: &DurableStepContext,
        status: DurableStepStatus,
        output: Option<&O>,
        reason: Option<String>,
        source: Option<String>,
    ) -> Result<(), WorkflowError> {
        let _guard = self.lock.lock().await;
        let records = self.records().await?;
        let record = JournalRecord {
            schema_version: JOURNAL_SCHEMA_VERSION,
            sequence: records.last().map_or(1, |last| last.sequence + 1),
            workflow_run_id: self.workflow_run_id,
            step_id: context.step_id,
            identity: spec.identity(),
            key: spec.key.clone(),
            instance: spec.instance.clone(),
            step_version: spec.version.clone(),
            input_hash,
            operation_id: context.operation_id.clone(),
            attempt: context.attempt,
            status,
            timestamp: chrono::Utc::now(),
            output: output.map(serde_json::to_value).transpose()?,
            reason,
            resolution_source: source,
        };
        self.append_locked(&record).await
    }

    pub async fn execute<I, O, F, Fut>(
        &self,
        spec: &DurableStepSpec,
        input: &I,
        execute: F,
    ) -> Result<O, WorkflowError>
    where
        I: Serialize,
        O: Serialize + DeserializeOwned,
        F: FnOnce(DurableStepContext) -> Fut,
        Fut: Future<Output = Result<O, WorkflowError>>,
    {
        let _lease = self.execution_lease(spec)?;
        match self.prepare(spec, input, false).await? {
            Preparation::Replay(output) => Ok(output),
            Preparation::InDoubt(context) => Err(WorkflowError::DurableInDoubt {
                step: spec.display_key(),
                operation_id: context.operation_id,
            }),
            Preparation::Execute(context) => match execute(context.clone()).await {
                Ok(output) => {
                    // Serialize before committing completion so serialization
                    // failure leaves the existing Started record in doubt.
                    if let Err(error) = serde_json::to_value(&output) {
                        self.transition::<O>(
                            spec,
                            Self::input_hash(input)?,
                            &context,
                            DurableStepStatus::InDoubt,
                            None,
                            Some(format!("output serialization failed: {error}")),
                            Some("serialization-error".to_string()),
                        )
                        .await?;
                        return Err(WorkflowError::DurableInDoubt {
                            step: spec.display_key(),
                            operation_id: context.operation_id,
                        });
                    }
                    self.transition(
                        spec,
                        Self::input_hash(input)?,
                        &context,
                        DurableStepStatus::Completed,
                        Some(&output),
                        None,
                        Some("execution".to_string()),
                    )
                    .await?;
                    Ok(output)
                }
                Err(error) => {
                    self.transition::<O>(
                        spec,
                        Self::input_hash(input)?,
                        &context,
                        DurableStepStatus::InDoubt,
                        None,
                        Some(error.to_string()),
                        Some("execution-error".to_string()),
                    )
                    .await?;
                    Err(WorkflowError::DurableInDoubt {
                        step: spec.display_key(),
                        operation_id: context.operation_id,
                    })
                }
            },
        }
    }

    pub async fn execute_with_resolver<I, O, F, Fut, R>(
        &self,
        spec: &DurableStepSpec,
        input: &I,
        execute: F,
        resolver: &R,
    ) -> Result<O, WorkflowError>
    where
        I: Serialize + Sync,
        O: Serialize + DeserializeOwned,
        F: FnOnce(DurableStepContext) -> Fut,
        Fut: Future<Output = Result<O, WorkflowError>>,
        R: ResolveInDoubt<I, O>,
    {
        let _lease = self.execution_lease(spec)?;
        match self.prepare(spec, input, false).await? {
            Preparation::Replay(output) => Ok(output),
            Preparation::Execute(context) => {
                self.execute_prepared(spec, input, context, execute).await
            }
            Preparation::InDoubt(in_doubt) => match resolver.resolve(&in_doubt, input).await? {
                InDoubtResolution::Completed(output) => {
                    let context = DurableStepContext {
                        workflow_run_id: in_doubt.workflow_run_id,
                        step_id: in_doubt.step_id,
                        operation_id: in_doubt.operation_id,
                        attempt: in_doubt.attempt,
                    };
                    self.transition(
                        spec,
                        Self::input_hash(input)?,
                        &context,
                        DurableStepStatus::Completed,
                        Some(&output),
                        None,
                        Some("resolver".to_string()),
                    )
                    .await?;
                    Ok(output)
                }
                InDoubtResolution::DefinitelyNotCompleted => {
                    match self.prepare(spec, input, true).await? {
                        Preparation::Execute(context) => {
                            self.execute_prepared(spec, input, context, execute).await
                        }
                        Preparation::Replay(output) => Ok(output),
                        Preparation::InDoubt(context) => Err(WorkflowError::DurableInDoubt {
                            step: spec.display_key(),
                            operation_id: context.operation_id,
                        }),
                    }
                }
                InDoubtResolution::StillInDoubt { reason } => {
                    let context = DurableStepContext {
                        workflow_run_id: in_doubt.workflow_run_id,
                        step_id: in_doubt.step_id,
                        operation_id: in_doubt.operation_id.clone(),
                        attempt: in_doubt.attempt,
                    };
                    self.transition::<O>(
                        spec,
                        Self::input_hash(input)?,
                        &context,
                        DurableStepStatus::InDoubt,
                        None,
                        Some(reason),
                        Some("resolver".to_string()),
                    )
                    .await?;
                    Err(WorkflowError::DurableInDoubt {
                        step: spec.display_key(),
                        operation_id: in_doubt.operation_id,
                    })
                }
                InDoubtResolution::PermanentlyFailed { reason } => {
                    let context = DurableStepContext {
                        workflow_run_id: in_doubt.workflow_run_id,
                        step_id: in_doubt.step_id,
                        operation_id: in_doubt.operation_id,
                        attempt: in_doubt.attempt,
                    };
                    self.transition::<O>(
                        spec,
                        Self::input_hash(input)?,
                        &context,
                        DurableStepStatus::PermanentlyFailed,
                        None,
                        Some(reason.clone()),
                        Some("resolver".to_string()),
                    )
                    .await?;
                    Err(WorkflowError::DurablePermanentlyFailed {
                        step: spec.display_key(),
                        reason,
                    })
                }
            },
        }
    }

    async fn execute_prepared<I, O, F, Fut>(
        &self,
        spec: &DurableStepSpec,
        input: &I,
        context: DurableStepContext,
        execute: F,
    ) -> Result<O, WorkflowError>
    where
        I: Serialize,
        O: Serialize + DeserializeOwned,
        F: FnOnce(DurableStepContext) -> Fut,
        Fut: Future<Output = Result<O, WorkflowError>>,
    {
        match execute(context.clone()).await {
            Ok(output) => {
                if let Err(error) = serde_json::to_value(&output) {
                    self.transition::<O>(
                        spec,
                        Self::input_hash(input)?,
                        &context,
                        DurableStepStatus::InDoubt,
                        None,
                        Some(format!("output serialization failed: {error}")),
                        Some("serialization-error".to_string()),
                    )
                    .await?;
                    return Err(WorkflowError::DurableInDoubt {
                        step: spec.display_key(),
                        operation_id: context.operation_id,
                    });
                }
                self.transition(
                    spec,
                    Self::input_hash(input)?,
                    &context,
                    DurableStepStatus::Completed,
                    Some(&output),
                    None,
                    Some("execution".to_string()),
                )
                .await?;
                Ok(output)
            }
            Err(error) => {
                self.transition::<O>(
                    spec,
                    Self::input_hash(input)?,
                    &context,
                    DurableStepStatus::InDoubt,
                    None,
                    Some(error.to_string()),
                    Some("execution-error".to_string()),
                )
                .await?;
                Err(WorkflowError::DurableInDoubt {
                    step: spec.display_key(),
                    operation_id: context.operation_id,
                })
            }
        }
    }

    pub async fn list_in_doubt(&self) -> Result<Vec<InDoubtContext>, WorkflowError> {
        let _guard = self.lock.lock().await;
        let records = self.records().await?;
        let mut latest = std::collections::HashMap::<String, &JournalRecord>::new();
        for record in &records {
            latest.insert(record.identity.clone(), record);
        }
        Ok(latest
            .values()
            .filter(|record| {
                matches!(
                    record.status,
                    DurableStepStatus::Started | DurableStepStatus::InDoubt
                )
            })
            .map(|record| InDoubtContext {
                workflow_run_id: record.workflow_run_id,
                step_id: record.step_id,
                operation_id: record.operation_id.clone(),
                attempt: record.attempt,
                reason: record.reason.clone(),
            })
            .collect())
    }

    async fn unresolved_record<I: Serialize>(
        &self,
        spec: &DurableStepSpec,
        input: &I,
    ) -> Result<JournalRecord, WorkflowError> {
        spec.validate()?;
        let records = self.records().await?;
        let record = Self::latest_for(&records, &spec.identity())
            .cloned()
            .ok_or_else(|| WorkflowError::DurableDefinitionMismatch {
                step: spec.display_key(),
                details: "no recorded attempt exists".to_string(),
            })?;
        Self::validate_match(spec, &Self::input_hash(input)?, &record)?;
        if !matches!(
            record.status,
            DurableStepStatus::Started | DurableStepStatus::InDoubt
        ) {
            return Err(WorkflowError::DurableDefinitionMismatch {
                step: spec.display_key(),
                details: format!("step is {:?}, not unresolved", record.status),
            });
        }
        Ok(record)
    }

    pub async fn operator_complete<I, O>(
        &self,
        spec: &DurableStepSpec,
        input: &I,
        output: &O,
        actor: &str,
        reason: &str,
    ) -> Result<(), WorkflowError>
    where
        I: Serialize,
        O: Serialize,
    {
        let _guard = self.lock.lock().await;
        let record = self.unresolved_record(spec, input).await?;
        let records = self.records().await?;
        let resolution = JournalRecord {
            schema_version: JOURNAL_SCHEMA_VERSION,
            sequence: records.last().map_or(1, |last| last.sequence + 1),
            status: DurableStepStatus::Completed,
            timestamp: chrono::Utc::now(),
            output: Some(serde_json::to_value(output)?),
            reason: Some(reason.to_string()),
            resolution_source: Some(format!("operator:{actor}")),
            ..record
        };
        self.append_locked(&resolution).await
    }

    pub async fn operator_permanently_fail<I: Serialize>(
        &self,
        spec: &DurableStepSpec,
        input: &I,
        actor: &str,
        reason: &str,
    ) -> Result<(), WorkflowError> {
        let _guard = self.lock.lock().await;
        let record = self.unresolved_record(spec, input).await?;
        let records = self.records().await?;
        let resolution = JournalRecord {
            schema_version: JOURNAL_SCHEMA_VERSION,
            sequence: records.last().map_or(1, |last| last.sequence + 1),
            status: DurableStepStatus::PermanentlyFailed,
            timestamp: chrono::Utc::now(),
            output: None,
            reason: Some(reason.to_string()),
            resolution_source: Some(format!("operator:{actor}")),
            ..record
        };
        self.append_locked(&resolution).await
    }

    pub async fn operator_authorize_retry<I: Serialize>(
        &self,
        spec: &DurableStepSpec,
        input: &I,
        actor: &str,
        reason: &str,
    ) -> Result<(), WorkflowError> {
        let _guard = self.lock.lock().await;
        let record = self.unresolved_record(spec, input).await?;
        let records = self.records().await?;
        let resolution = JournalRecord {
            schema_version: JOURNAL_SCHEMA_VERSION,
            sequence: records.last().map_or(1, |last| last.sequence + 1),
            status: DurableStepStatus::InDoubt,
            timestamp: chrono::Utc::now(),
            output: None,
            reason: Some(reason.to_string()),
            resolution_source: Some(format!("operator-retry:{actor}")),
            ..record
        };
        self.append_locked(&resolution).await
    }
}
