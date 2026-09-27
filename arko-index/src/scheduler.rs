use crate::{
    Error, Result,
    backend::{Backend, ModelReply, ModelRequest, VectorReply},
    usage::finite,
};
use arko_ai::{UsageInfo, retry::MAX_SERVER_DELAY_HINT};
use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::{
    sync::{mpsc, oneshot, watch},
    task::JoinSet,
    time::Instant,
};

/// Delay between transient provider failures: 1 s doubling to a 60 s cap,
/// jittered, and never shorter than the provider's Retry-After. A request
/// fails with its last error after `max_transient_attempts` transient
/// failures; explicit capacity signals wait without consuming that budget.
const RETRY_BACKOFF: arko_ai::BackoffPolicy =
    arko_ai::BackoffPolicy::new(Duration::from_secs(1), Duration::from_secs(60));

/// Delay before retry `attempt`. A backend's hint is capped like the hints
/// arko-ai parses from responses, so the retry deadline stays representable.
fn retry_delay(attempt: u32, hint: Option<Duration>) -> Duration {
    let hint = hint.map(|hint| hint.min(MAX_SERVER_DELAY_HINT));
    RETRY_BACKOFF.next_delay(attempt, hint)
}
/// `now + delay`, saturating instead of panicking on overflow. An
/// unrepresentable deadline becomes a far one (the longest representable
/// delay found by halving from `MAX_SERVER_DELAY_HINT`), never an immediate
/// retry unless no later instant exists at all.
fn deadline(now: Instant, delay: Duration) -> Instant {
    if let Some(deadline) = now.checked_add(delay) {
        return deadline;
    }
    let mut step = delay.min(MAX_SERVER_DELAY_HINT);
    while !step.is_zero() {
        if let Some(deadline) = now.checked_add(step) {
            return deadline;
        }
        step /= 2;
    }
    now
}

#[derive(Clone)]
pub(crate) enum Operation {
    Chunk(ModelRequest),
    Embed(Vec<String>, bool),
}
pub(crate) enum Reply {
    Chunk(ModelReply),
    Embed(VectorReply),
}
/// A request's result, and the usage of every response the backend rejected
/// on the way (see `Error::Provider::usage`), in attempt order: the retried
/// attempts' and, when the request failed with one, the last attempt's.
pub(crate) struct Outcome {
    pub result: Result<Reply>,
    pub rejected: Vec<UsageInfo>,
}
impl Outcome {
    fn failed(error: Error) -> Self {
        Self {
            result: Err(error),
            rejected: vec![],
        }
    }
}
struct Task {
    operation: Operation,
    initial: bool,
    attempts: usize,
    /// Failures that consumed the transient-attempt budget (not capacity).
    transient_failures: usize,
    failures: Vec<serde_json::Value>,
    /// Usage of the responses the backend rejected so far.
    rejected: Vec<UsageInfo>,
    ready: Instant,
    control: watch::Receiver<u8>,
    reply: oneshot::Sender<Outcome>,
    progress: Option<mpsc::UnboundedSender<serde_json::Value>>,
}
impl Task {
    /// Records a failed attempt, with the usage of a rejected response.
    fn record_failure(&mut self, error: &Error) {
        let mut detail = serde_json::json!({"error":error.to_string()});
        let usage = error.usage().map(finite);
        if let Some(usage) = &usage {
            detail["usage"] = serde_json::json!(usage);
        }
        self.event("failed", detail.clone());
        detail["attempt"] = self.attempts.into();
        self.failures.push(detail);
        self.rejected.extend(usage);
    }
    fn outcome(self, result: Result<Reply>) -> (oneshot::Sender<Outcome>, Outcome) {
        let outcome = Outcome {
            result,
            rejected: self.rejected,
        };
        (self.reply, outcome)
    }
    fn event(&self, phase: &str, detail: serde_json::Value) {
        if let Some(sender) = &self.progress {
            let at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            let _=sender.send(serde_json::json!({"phase":phase,"attempt":self.attempts,"unix_ms":at,"detail":detail}));
        }
    }
}
/// 0 running; 1 pause queued work/drain active; 2 cancel.
#[derive(Clone)]
pub(crate) struct Control(pub watch::Sender<u8>);
impl Control {
    pub fn new() -> Self {
        Self(watch::channel(0).0)
    }
    pub fn check(&self) -> Result<()> {
        match *self.0.borrow() {
            1 => Err(Error::Paused),
            2 => Err(Error::Cancelled),
            _ => Ok(()),
        }
    }
    pub fn set(&self, state: u8) {
        self.0.send_replace(state);
    }
}
#[derive(Clone)]
pub(crate) struct Scheduler {
    tx: mpsc::Sender<Task>,
    pub active: Arc<AtomicUsize>,
}
impl Scheduler {
    pub fn new(backend: Arc<dyn Backend>, limit: usize, max_transient_attempts: usize) -> Self {
        let (tx, mut rx) = mpsc::channel::<Task>(64);
        let active = Arc::new(AtomicUsize::new(0));
        let count = active.clone();
        tokio::spawn(async move {
            let mut queue = VecDeque::<Task>::new();
            let mut running = JoinSet::new();
            let mut delivered = vec![];
            let mut open = true;
            loop {
                // Drain prepared requests before selecting priority and refilling.
                while queue.len() < 128 {
                    match rx.try_recv() {
                        Ok(t) => queue.push_back(t),
                        Err(_) => break,
                    }
                }
                while running.len() < limit {
                    let now = Instant::now();
                    let position = queue
                        .iter()
                        .position(|t| t.initial && (t.ready <= now || *t.control.borrow() != 0))
                        .or_else(|| {
                            queue
                                .iter()
                                .position(|t| t.ready <= now || *t.control.borrow() != 0)
                        });
                    let Some(position) = position else {
                        break;
                    };
                    let task = queue.remove(position).unwrap();
                    if task.reply.is_closed() {
                        continue;
                    }
                    let state = *task.control.borrow();
                    if state != 0 {
                        // Keeps the usage of attempts rejected before the wait.
                        delivered.push(task.outcome(Err(if state == 1 {
                            Error::Paused
                        } else {
                            Error::Cancelled
                        })));
                        continue;
                    }
                    task.event("started", serde_json::json!({}));
                    let backend = backend.clone();
                    count.fetch_add(1, Ordering::SeqCst);
                    running.spawn(async move {
                        let mut control = task.control.clone();
                        let cancelled = async {
                            loop {
                                if *control.borrow_and_update() == 2 { break; }
                                if control.changed().await.is_err() { break; }
                            }
                        };
                        let operation = async { match &task.operation {
                            Operation::Chunk(r)=>backend.chunk(r.clone()).await.map(Reply::Chunk),
                            Operation::Embed(texts, query)=>backend.embed(texts.clone(),*query).await.map(Reply::Embed),
                        } };
                        // Dropping the transport future is explicit cancellation, never a timer.
                        let result = tokio::select! { result = operation => result, _ = cancelled => Err(Error::Cancelled) };
                        (task,result)
                    });
                }
                // Provider slots have been refilled before callers start validation/I/O.
                for (reply, outcome) in delivered.drain(..) {
                    let _ = reply.send(outcome);
                }
                if !open && queue.is_empty() && running.is_empty() {
                    break;
                }
                let due = queue
                    .iter()
                    .map(|t| t.ready)
                    .filter(|&t| t > Instant::now())
                    .min();
                let delay = async {
                    match due {
                        Some(t) => tokio::time::sleep_until(t).await,
                        None => std::future::pending().await,
                    }
                };
                let changes = async {
                    use futures::StreamExt;
                    let mut waits: futures::stream::FuturesUnordered<_> = queue
                        .iter()
                        .map(|t| {
                            let mut control = t.control.clone();
                            async move {
                                let _ = control.changed().await;
                            }
                        })
                        .collect();
                    if waits.is_empty() {
                        std::future::pending::<()>().await;
                    } else {
                        waits.next().await;
                    }
                };
                tokio::select! {
                    _ = changes => {},
                    task = rx.recv(), if open && queue.len() < 128 => match task {Some(t)=>queue.push_back(t),None=>open=false},
                    completed = running.join_next(), if !running.is_empty() => {
                        count.fetch_sub(1, Ordering::SeqCst);
                        // A panic closes that request's oneshot; callers persist its worker error.
                        if let Some(Ok((mut task, mut result))) = completed {
                                if let Err(error)=&result {task.record_failure(error);}
                                if let Err(error @ Error::Provider { transient: true, retry_after, .. }) = &result
                                    && *task.control.borrow() == 0 {
                                        // Capacity signals wait without consuming the attempt budget.
                                        let capacity = error.retry_class() == Some(arko_ai::RetryClass::Capacity);
                                        if !capacity { task.transient_failures = task.transient_failures.saturating_add(1); }
                                        if capacity || task.transient_failures < max_transient_attempts {
                                            let attempt = u32::try_from(task.attempts).unwrap_or(u32::MAX);
                                            let delay = retry_delay(attempt, *retry_after);
                                            task.event("retry_wait",serde_json::json!({"delay_ms":delay.as_millis(),"capacity":capacity}));
                                            task.attempts = task.attempts.saturating_add(1); task.initial = false; task.ready = deadline(Instant::now(), delay);
                                            queue.push_back(task); continue;
                                        }
                                        task.event("retries_exhausted",serde_json::json!({"transient_failures":task.transient_failures}));
                                    }
                                if result.is_ok() {task.event("completed",serde_json::json!({}));}
                                match &mut result {
                                    Ok(Reply::Chunk(reply)) => {
                                        reply.usage_info=reply.usage_info.as_ref().map(finite);
                                        reply.usage=serde_json::json!({"provider":reply.usage,"attempts":task.attempts,"retry_failures":task.failures});
                                    }
                                    Ok(Reply::Embed(reply)) => reply.usage_info=reply.usage_info.as_ref().map(finite),
                                    Err(_) => {}
                                }
                                delivered.push(task.outcome(result));
                        }
                    },
                    _ = delay, if due.is_some() && running.len() < limit => {}
                }
            }
        });
        Self { tx, active }
    }
    pub async fn call(&self, operation: Operation, initial: bool, control: &Control) -> Outcome {
        self.call_tracked(operation, initial, control, None).await
    }
    pub async fn call_tracked(
        &self,
        operation: Operation,
        initial: bool,
        control: &Control,
        progress: Option<mpsc::UnboundedSender<serde_json::Value>>,
    ) -> Outcome {
        if let Err(error) = control.check() {
            return Outcome::failed(error);
        }
        let (reply, outcome) = oneshot::channel();
        let task = Task {
            operation,
            initial,
            attempts: 1,
            transient_failures: 0,
            failures: vec![],
            rejected: vec![],
            ready: Instant::now(),
            control: control.0.subscribe(),
            reply,
            progress,
        };
        if self.tx.send(task).await.is_err() {
            return Outcome::failed(Error::Worker("request queue closed".into()));
        }
        outcome.await.unwrap_or_else(|_| {
            Outcome::failed(Error::Worker("provider worker stopped unexpectedly".into()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EmbeddingProfile,
        backend::{Limits, ModelReply, VectorReply},
    };
    use serde_json::json;
    use std::sync::Mutex;
    struct Gate {
        profile: EmbeddingProfile,
        gates: Vec<tokio::sync::Semaphore>,
        started: mpsc::UnboundedSender<usize>,
        tries: Mutex<Vec<usize>>,
        retry_attempts: usize,
        retry_delay: Duration,
        /// Fail with an explicit capacity signal instead of a plain transient error.
        capacity: bool,
        /// Usage of each failed (rejected) response.
        usage: Option<arko_ai::UsageInfo>,
    }
    #[async_trait::async_trait]
    impl Backend for Gate {
        async fn chunk(&self, r: ModelRequest) -> Result<ModelReply> {
            let id: usize = r.prompt.parse().unwrap();
            self.started.send(id).unwrap();
            let attempt = {
                let mut tries = self.tries.lock().unwrap();
                tries[id] += 1;
                tries[id]
            };
            if id == 0 && attempt <= self.retry_attempts {
                return Err(Error::Provider {
                    message: "temporary capacity".into(),
                    transient: true,
                    retry_after: Some(self.retry_delay),
                    source: self.capacity.then(|| {
                        Box::new(arko_ai::Error::ProviderError {
                            message: "overloaded".into(),
                            status: Some(529),
                            code: Some("overloaded_error".into()),
                            retry_after: None,
                        })
                    }),
                    usage: self.usage.clone().map(Box::new),
                });
            }
            self.gates[id].acquire().await.unwrap().forget();
            Ok(ModelReply {
                value: json!({}),
                usage: json!({}),
                usage_info: None,
            })
        }
        async fn embed(&self, _: Vec<String>, _: bool) -> Result<VectorReply> {
            unreachable!()
        }
        fn tokens(&self, _: &str) -> Result<usize> {
            Ok(1)
        }
        fn limits(&self) -> Limits {
            Limits {
                item_tokens: 1,
                batch_tokens: 1,
                batch_items: 1,
            }
        }
        fn profile(&self) -> &EmbeddingProfile {
            &self.profile
        }
        fn output_tokenizer(&self) -> String {
            "test".into()
        }
    }
    fn setup(
        count: usize,
        concurrency: usize,
        retry: bool,
    ) -> (Arc<Gate>, Scheduler, mpsc::UnboundedReceiver<usize>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let gate = Arc::new(Gate {
            profile: Default::default(),
            gates: (0..count).map(|_| tokio::sync::Semaphore::new(0)).collect(),
            started: tx,
            tries: Mutex::new(vec![0; count]),
            retry_attempts: usize::from(retry),
            retry_delay: Duration::from_secs(3600),
            capacity: false,
            usage: None,
        });
        let scheduler = Scheduler::new(gate.clone(), concurrency, 10);
        (gate, scheduler, rx)
    }
    fn enqueue(
        s: &Scheduler,
        id: usize,
        initial: bool,
        c: &Control,
    ) -> tokio::task::JoinHandle<Result<Reply>> {
        let (s, c) = (s.clone(), c.clone());
        tokio::spawn(async move {
            s.call(
                Operation::Chunk(ModelRequest {
                    system: String::new(),
                    prompt: id.to_string(),
                    tool: "test".into(),
                    schema: json!({}),
                }),
                initial,
                &c,
            )
            .await
            .result
        })
    }
    #[tokio::test]
    async fn eager_refill_across_a_hundred_requests() {
        let (gate, s, mut started) = setup(100, 2, false);
        let c = Control::new();
        let first = enqueue(&s, 0, true, &c);
        assert_eq!(started.recv().await, Some(0));
        let second = enqueue(&s, 1, true, &c);
        assert_eq!(started.recv().await, Some(1));
        let mut rest = vec![];
        for id in 2..100 {
            rest.push(enqueue(&s, id, true, &c));
        }
        gate.gates[1].add_permits(1);
        second.await.unwrap().unwrap();
        let next = started.recv().await.unwrap();
        assert!(next >= 2);
        assert!(!first.is_finished());
        assert_eq!(s.active.load(Ordering::SeqCst), 2);
        for id in 0..100 {
            gate.gates[id].add_permits(1);
        }
        first.await.unwrap().unwrap();
        for task in rest {
            task.await.unwrap().unwrap();
        }
        assert_eq!(s.active.load(Ordering::SeqCst), 0);
        assert!(gate.tries.lock().unwrap().iter().all(|&n| n == 1));
    }
    #[tokio::test]
    async fn queued_initial_has_priority_over_repair() {
        let (gate, s, mut started) = setup(3, 1, false);
        let c = Control::new();
        let a = enqueue(&s, 0, true, &c);
        assert_eq!(started.recv().await, Some(0));
        let repair = enqueue(&s, 1, false, &c);
        let b = enqueue(&s, 2, true, &c);
        for _ in 0..5 {
            tokio::task::yield_now().await;
        }
        gate.gates[0].add_permits(1);
        assert_eq!(started.recv().await, Some(2));
        gate.gates[2].add_permits(1);
        assert_eq!(started.recv().await, Some(1));
        gate.gates[1].add_permits(1);
        a.await.unwrap().unwrap();
        b.await.unwrap().unwrap();
        repair.await.unwrap().unwrap();
    }
    #[tokio::test]
    async fn retry_backoff_holds_no_slot_and_pause_interrupts_the_wait() {
        let (gate, s, mut started) = setup(2, 1, true);
        let c = Control::new();
        let retry = enqueue(&s, 0, true, &c);
        assert_eq!(started.recv().await, Some(0));
        let b = enqueue(&s, 1, true, &c);
        assert_eq!(started.recv().await, Some(1));
        gate.gates[1].add_permits(1);
        b.await.unwrap().unwrap();
        assert_eq!(s.active.load(Ordering::SeqCst), 0);
        c.set(1);
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(1), retry)
                .await
                .unwrap()
                .unwrap(),
            Err(Error::Paused)
        ));
        assert_eq!(gate.tries.lock().unwrap()[0], 1);
    }
    #[tokio::test]
    async fn cancellation_closes_active_request() {
        let (_, s, mut started) = setup(1, 1, false);
        let c = Control::new();
        let a = enqueue(&s, 0, true, &c);
        started.recv().await.unwrap();
        c.set(2);
        assert!(matches!(a.await.unwrap(), Err(Error::Cancelled)));
        assert_eq!(s.active.load(Ordering::SeqCst), 0);
    }
    #[tokio::test(start_paused = true)]
    async fn transient_retries_exceed_three_and_report_progress_before_completion() {
        let (started, _started_rx) = mpsc::unbounded_channel();
        let gate = Arc::new(Gate {
            profile: Default::default(),
            gates: vec![tokio::sync::Semaphore::new(0)],
            started,
            tries: Mutex::new(vec![0]),
            retry_attempts: 4,
            retry_delay: Duration::from_secs(10),
            capacity: false,
            usage: None,
        });
        let scheduler = Scheduler::new(gate.clone(), 1, 10);
        let control = Control::new();
        let (progress, mut events) = mpsc::unbounded_channel();
        let request = Operation::Chunk(ModelRequest {
            system: String::new(),
            prompt: "0".into(),
            tool: "test".into(),
            schema: json!({}),
        });
        let task = tokio::spawn(async move {
            scheduler
                .call_tracked(request, true, &control, Some(progress))
                .await
                .result
        });
        let mut observed = vec![];
        loop {
            let event = events.recv().await.unwrap();
            let started_fifth = event["phase"] == "started" && event["attempt"] == 5;
            observed.push(event);
            if started_fifth {
                break;
            }
        }
        assert!(!task.is_finished());
        gate.gates[0].add_permits(1);
        let Reply::Chunk(reply) = task.await.unwrap().unwrap() else {
            panic!("wrong response")
        };
        assert_eq!(reply.usage["attempts"], 5);
        assert_eq!(reply.usage["retry_failures"].as_array().unwrap().len(), 4);
        while let Some(event) = events.recv().await {
            observed.push(event);
        }
        assert_eq!(
            observed
                .iter()
                .filter(|e| e["phase"] == "retry_wait")
                .count(),
            4
        );
        let waits: Vec<_> = observed
            .iter()
            .filter(|e| e["phase"] == "retry_wait")
            .map(|e| e["detail"]["delay_ms"].as_u64().unwrap())
            .collect();
        // Each wait honors the 10 s Retry-After and adds at most half of the
        // attempt's backoff cap (1 s doubling) as jitter above it.
        for (index, wait) in waits.iter().enumerate() {
            let cap = 1_000_u64 << index;
            assert!(
                *wait >= 10_000 && *wait <= 10_000 + cap / 2,
                "retry {index} waited {wait} ms"
            );
        }
        assert!(observed.iter().all(|e| e["unix_ms"].is_u64()));
        assert_eq!(observed.last().unwrap()["phase"], "completed");
    }
    fn failing_gate(
        failures: usize,
        capacity: bool,
    ) -> (Arc<Gate>, mpsc::UnboundedReceiver<usize>) {
        let (started, started_rx) = mpsc::unbounded_channel();
        let gate = Arc::new(Gate {
            profile: Default::default(),
            gates: vec![tokio::sync::Semaphore::new(0)],
            started,
            tries: Mutex::new(vec![0]),
            retry_attempts: failures,
            retry_delay: Duration::from_secs(10),
            capacity,
            usage: None,
        });
        (gate, started_rx)
    }
    fn chunk_request() -> Operation {
        Operation::Chunk(ModelRequest {
            system: String::new(),
            prompt: "0".into(),
            tool: "test".into(),
            schema: json!({}),
        })
    }
    #[tokio::test(start_paused = true)]
    async fn transient_failures_stop_at_the_attempt_limit_with_the_last_error() {
        let (gate, _started) = failing_gate(usize::MAX, false);
        let scheduler = Scheduler::new(gate.clone(), 1, 3);
        let control = Control::new();
        let (progress, mut events) = mpsc::unbounded_channel();
        let result = scheduler
            .call_tracked(chunk_request(), true, &control, Some(progress))
            .await
            .result;
        assert!(matches!(
            result,
            Err(Error::Provider {
                transient: true,
                ..
            })
        ));
        assert_eq!(gate.tries.lock().unwrap()[0], 3);
        let mut observed = vec![];
        while let Ok(event) = events.try_recv() {
            observed.push(event);
        }
        let phases: Vec<_> = observed.iter().map(|e| e["phase"].clone()).collect();
        assert_eq!(phases.iter().filter(|p| *p == "retry_wait").count(), 2);
        assert_eq!(phases.iter().filter(|p| *p == "failed").count(), 3);
        let exhausted = observed.last().unwrap();
        assert_eq!(exhausted["phase"], "retries_exhausted");
        assert_eq!(exhausted["detail"]["transient_failures"], 3);
        assert_eq!(scheduler.active.load(Ordering::SeqCst), 0);
    }
    #[tokio::test(start_paused = true)]
    async fn capacity_waits_do_not_consume_the_attempt_limit() {
        let (gate, _started) = failing_gate(5, true);
        gate.gates[0].add_permits(1);
        let scheduler = Scheduler::new(gate.clone(), 1, 2);
        let control = Control::new();
        let Reply::Chunk(reply) = scheduler
            .call(chunk_request(), true, &control)
            .await
            .result
            .unwrap()
        else {
            panic!("wrong response")
        };
        assert_eq!(gate.tries.lock().unwrap()[0], 6);
        assert_eq!(reply.usage["attempts"], 6);
    }
    #[tokio::test(start_paused = true)]
    async fn huge_backend_retry_hints_are_capped_instead_of_overflowing() {
        // A custom backend may report any Retry-After; adding it to the current
        // time used to panic the scheduler task.
        let (gate, _started) = failing_gate(1, false);
        let gate = Arc::new(Gate {
            retry_delay: Duration::MAX,
            ..Arc::into_inner(gate).unwrap()
        });
        gate.gates[0].add_permits(1);
        let scheduler = Scheduler::new(gate.clone(), 1, 2);
        let (progress, mut events) = mpsc::unbounded_channel();
        let Reply::Chunk(reply) = scheduler
            .call_tracked(chunk_request(), true, &Control::new(), Some(progress))
            .await
            .result
            .unwrap()
        else {
            panic!("wrong response")
        };
        assert_eq!(reply.usage["attempts"], 2);
        assert_eq!(gate.tries.lock().unwrap()[0], 2);
        let mut waits = vec![];
        while let Ok(event) = events.try_recv() {
            if event["phase"] == "retry_wait" {
                waits.push(event["detail"]["delay_ms"].as_u64().unwrap());
            }
        }
        let cap = arko_ai::retry::MAX_SERVER_DELAY_HINT.as_millis() as u64;
        assert_eq!(waits.len(), 1);
        assert!(waits[0] >= cap && waits[0] <= cap + 1_000, "{waits:?}");
    }
    #[test]
    fn retry_deadlines_saturate() {
        let now = Instant::now();
        assert_eq!(
            deadline(now, Duration::from_secs(5)),
            now + Duration::from_secs(5)
        );
        assert!(deadline(now, Duration::MAX) >= now + arko_ai::retry::MAX_SERVER_DELAY_HINT);
        let delay = retry_delay(3, Some(Duration::MAX));
        assert!(delay <= arko_ai::retry::MAX_SERVER_DELAY_HINT + Duration::from_secs(60));
    }
    #[test]
    fn deadlines_near_the_end_of_time_stay_in_the_future() {
        let hint = MAX_SERVER_DELAY_HINT;
        // The latest instant (to the second) that still has room for a full hint.
        let mut edge = Instant::now();
        for shift in (0..64).rev() {
            if let Some(later) = edge.checked_add(Duration::from_secs(1 << shift))
                && later.checked_add(hint).is_some()
            {
                edge = later;
            }
        }
        let near_end = edge + hint / 2;
        assert!(near_end.checked_add(hint).is_none());
        // Not an immediate retry: the farthest deadline halving finds.
        assert_eq!(deadline(near_end, Duration::MAX), edge + hint);
        assert_eq!(deadline(near_end, hint), edge + hint);
    }
    /// A gate whose first `failures` attempts return a rejected response
    /// reporting `usage`.
    fn rejecting_gate(
        failures: usize,
        usage: arko_ai::UsageInfo,
    ) -> (Arc<Gate>, mpsc::UnboundedReceiver<usize>) {
        let (gate, started) = failing_gate(failures, false);
        let gate = Arc::new(Gate {
            usage: Some(usage),
            ..Arc::into_inner(gate).unwrap()
        });
        (gate, started)
    }
    #[tokio::test(start_paused = true)]
    async fn rejected_responses_report_their_usage_when_a_retry_succeeds() {
        let (gate, _started) = rejecting_gate(2, arko_ai::UsageInfo::new(100, 10));
        gate.gates[0].add_permits(1);
        let scheduler = Scheduler::new(gate.clone(), 1, 10);
        let (progress, mut events) = mpsc::unbounded_channel();
        let outcome = scheduler
            .call_tracked(chunk_request(), true, &Control::new(), Some(progress))
            .await;
        assert_eq!(outcome.rejected.len(), 2);
        assert!(outcome.rejected.iter().all(|u| u.input_tokens == 100));
        let Reply::Chunk(reply) = outcome.result.unwrap() else {
            panic!("wrong response")
        };
        // The saved request artifact keeps each rejected attempt's usage.
        let failures = reply.usage["retry_failures"].as_array().unwrap();
        assert_eq!(failures.len(), 2);
        assert!(failures.iter().all(|f| f["usage"]["input_tokens"] == 100));
        let mut failed = 0;
        while let Ok(event) = events.try_recv() {
            if event["phase"] == "failed" {
                assert_eq!(event["detail"]["usage"]["output_tokens"], 10);
                failed += 1;
            }
        }
        assert_eq!(failed, 2);
    }
    #[tokio::test(start_paused = true)]
    async fn exhausted_retries_return_the_usage_of_every_rejected_attempt() {
        let (gate, _started) = rejecting_gate(usize::MAX, arko_ai::UsageInfo::new(100, 10));
        let scheduler = Scheduler::new(gate.clone(), 1, 3);
        let outcome = scheduler.call(chunk_request(), true, &Control::new()).await;
        assert!(matches!(outcome.result, Err(Error::Provider { .. })));
        assert_eq!(gate.tries.lock().unwrap()[0], 3);
        assert_eq!(outcome.rejected.len(), 3);
    }
    #[tokio::test(start_paused = true)]
    async fn a_pause_during_a_retry_wait_keeps_the_rejected_usage() {
        let (gate, _started) = rejecting_gate(usize::MAX, arko_ai::UsageInfo::new(100, 10));
        let scheduler = Scheduler::new(gate.clone(), 1, 10);
        let control = Control::new();
        let (progress, mut events) = mpsc::unbounded_channel();
        let call = {
            let (scheduler, control) = (scheduler.clone(), control.clone());
            tokio::spawn(async move {
                scheduler
                    .call_tracked(chunk_request(), true, &control, Some(progress))
                    .await
            })
        };
        while events.recv().await.unwrap()["phase"] != "retry_wait" {}
        control.set(1);
        let outcome = call.await.unwrap();
        assert!(matches!(outcome.result, Err(Error::Paused)));
        assert_eq!(gate.tries.lock().unwrap()[0], 1);
        assert_eq!(outcome.rejected.len(), 1);
    }
    #[tokio::test(start_paused = true)]
    async fn non_finite_costs_leave_the_scheduler_as_zero() {
        let broken = arko_ai::UsageInfo {
            total_cost: f64::NAN,
            ..arko_ai::UsageInfo::new(100, 10)
        };
        let (gate, _started) = rejecting_gate(1, broken);
        gate.gates[0].add_permits(1);
        let scheduler = Scheduler::new(gate.clone(), 1, 10);
        let outcome = scheduler.call(chunk_request(), true, &Control::new()).await;
        assert_eq!(outcome.rejected[0].total_cost, 0.0);
        let Reply::Chunk(reply) = outcome.result.unwrap() else {
            panic!("wrong response")
        };
        assert_eq!(reply.usage["retry_failures"][0]["usage"]["total_cost"], 0.0);
    }
}
