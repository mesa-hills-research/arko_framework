use crate::{
    Error, Result,
    backend::{Backend, ModelReply, ModelRequest, VectorReply},
};
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

#[derive(Clone)]
pub(crate) enum Operation {
    Chunk(ModelRequest),
    Embed(Vec<String>, bool),
}
pub(crate) enum Reply {
    Chunk(ModelReply),
    Embed(VectorReply),
}
struct Task {
    operation: Operation,
    initial: bool,
    attempts: usize,
    retry_delay: Duration,
    failures: Vec<serde_json::Value>,
    ready: Instant,
    control: watch::Receiver<u8>,
    reply: oneshot::Sender<Result<Reply>>,
    progress: Option<mpsc::UnboundedSender<serde_json::Value>>,
}
impl Task {
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
    pub fn new(backend: Arc<dyn Backend>, limit: usize) -> Self {
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
                        delivered.push((
                            task.reply,
                            Err(if state == 1 {
                                Error::Paused
                            } else {
                                Error::Cancelled
                            }),
                        ));
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
                for (reply, result) in delivered.drain(..) {
                    let _ = reply.send(result);
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
                                if let Err(error)=&result {task.event("failed",serde_json::json!({"error":error.to_string()}));task.failures.push(serde_json::json!({"attempt":task.attempts,"error":error.to_string()}));}
                                if let Err(Error::Provider { transient: true, retry_after, .. }) = &result
                                    && *task.control.borrow() == 0 {
                                        let jitter = (task.attempts.wrapping_mul(137).wrapping_add(queue.len().wrapping_mul(31))) % 251;
                                        let delay = task.retry_delay.saturating_add(Duration::from_millis(1000 + jitter as u64)).max(retry_after.unwrap_or(Duration::ZERO));
                                        task.retry_delay = delay;
                                        task.event("retry_wait",serde_json::json!({"delay_ms":delay.as_millis()}));
                                        task.attempts = task.attempts.saturating_add(1); task.initial = false; task.ready = Instant::now()+delay;
                                        queue.push_back(task); continue;
                                    }
                                if result.is_ok() {task.event("completed",serde_json::json!({}));}
                                if let Ok(Reply::Chunk(reply))=&mut result {
                                    reply.usage=serde_json::json!({"provider":reply.usage,"attempts":task.attempts,"retry_failures":task.failures});
                                }
                                delivered.push((task.reply,result));
                        }
                    },
                    _ = delay, if due.is_some() && running.len() < limit => {}
                }
            }
        });
        Self { tx, active }
    }
    pub async fn call(
        &self,
        operation: Operation,
        initial: bool,
        control: &Control,
    ) -> Result<Reply> {
        self.call_tracked(operation, initial, control, None).await
    }
    pub async fn call_tracked(
        &self,
        operation: Operation,
        initial: bool,
        control: &Control,
        progress: Option<mpsc::UnboundedSender<serde_json::Value>>,
    ) -> Result<Reply> {
        control.check()?;
        let (reply, result) = oneshot::channel();
        self.tx
            .send(Task {
                operation,
                initial,
                attempts: 1,
                retry_delay: Duration::ZERO,
                failures: vec![],
                ready: Instant::now(),
                control: control.0.subscribe(),
                reply,
                progress,
            })
            .await
            .map_err(|_| Error::Worker("request queue closed".into()))?;
        result
            .await
            .map_err(|_| Error::Worker("provider worker stopped unexpectedly".into()))?
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
                });
            }
            self.gates[id].acquire().await.unwrap().forget();
            Ok(ModelReply {
                value: json!({}),
                usage: json!({}),
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
        });
        let scheduler = Scheduler::new(gate.clone(), concurrency);
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
        });
        let scheduler = Scheduler::new(gate.clone(), 1);
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
        assert!(waits[0] >= 10_000);
        assert!(waits.windows(2).all(|pair| pair[1] > pair[0]));
        assert!(observed.iter().all(|e| e["unix_ms"].is_u64()));
        assert_eq!(observed.last().unwrap()["phase"], "completed");
    }
}
