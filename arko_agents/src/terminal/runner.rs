use super::*;
use detection::Declaration;
use std::collections::HashSet;
use std::io::Write;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::task::JoinSet;
use tokio::time::{Instant, sleep_until};

#[derive(Debug)]
enum ReadEvent {
    Output(Vec<u8>),
    Declaration(Vec<u8>),
    Error(String),
}

fn readers(
    process: &mut io::Process,
    sender: mpsc::Sender<ReadEvent>,
) -> (JoinSet<()>, tokio::task::JoinHandle<()>) {
    let mut readers = JoinSet::new();
    for mut reader in process.output.drain(..) {
        let sender = sender.clone();
        readers.spawn(async move {
            let mut bytes = vec![0; 8192];
            loop {
                match reader.read(&mut bytes).await {
                    Ok(0) => break,
                    Ok(n) => {
                        if sender
                            .send(ReadEvent::Output(bytes[..n].to_vec()))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = sender.send(ReadEvent::Error(error.to_string())).await;
                        break;
                    }
                }
            }
        });
    }
    let mut requests = std::mem::replace(&mut process.requests, Box::pin(tokio::io::empty()));
    let requests_task = tokio::spawn(async move {
        let mut bytes = vec![0; 4096];
        loop {
            match requests.read(&mut bytes).await {
                Ok(0) => break,
                Ok(n) => {
                    if sender
                        .send(ReadEvent::Declaration(bytes[..n].to_vec()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
                Err(error) => {
                    let _ = sender.send(ReadEvent::Error(error.to_string())).await;
                    break;
                }
            }
        }
    });
    (readers, requests_task)
}

pub(super) fn output(
    data: &mut JobData,
    bytes: &[u8],
    log: &mut std::fs::File,
    logged: &mut u64,
    config: &TerminalConfig,
) -> Result<()> {
    let keep = bytes
        .len()
        .min(config.max_log_bytes.saturating_sub(*logged) as usize);
    log.write_all(&bytes[..keep])?;
    *logged += keep as u64;
    if keep < bytes.len() {
        data.record.log_truncated = true;
    }
    data.record.output_end += bytes.len() as u64;
    data.output.extend(bytes);
    let extra = data.output.len().saturating_sub(config.output_buffer_bytes);
    data.output.drain(..extra);
    data.record.revision += 1;
    if matches!(
        data.record.input.as_ref().map(|notice| &notice.source),
        Some(InputSource::PromptRule { .. })
    ) {
        data.record.input = None;
        data.last_event = None;
    }
    Ok(())
}

fn event(
    manager: &Weak<TerminalManager>,
    data: &Arc<Mutex<JobData>>,
    config: &TerminalConfig,
) -> Result<()> {
    let (owner, snapshot) = {
        let mut data = data.lock().unwrap();
        persist(config, &data.record)?;
        let snapshot = data.snapshot(config, None, config.excerpt_bytes)?;
        data.last_event = Some(snapshot.clone());
        (data.record.owner, snapshot)
    };
    if let Some(manager) = manager.upgrade() {
        manager.deliver(owner, snapshot);
    }
    Ok(())
}

fn declaration(bytes: &[u8], data: &mut JobData, seen: &mut HashSet<String>) -> Result<bool> {
    let declaration: Declaration = serde_json::from_slice(bytes)?;
    match declaration {
        Declaration::InputRequest { request_id, prompt } => {
            if request_id.is_empty()
                || request_id.len() > 128
                || prompt.is_empty()
                || prompt.len() > 4096
            {
                return Err(TerminalError::Invalid(
                    "input request requires an ID (1–128 bytes) and prompt (1–4096 bytes)".into(),
                ));
            }
            if seen.contains(&request_id) {
                return Ok(false);
            }
            if seen.len() >= 1024 {
                return Err(TerminalError::Invalid("input request limit reached".into()));
            }
            if matches!(
                data.record.input.as_ref().map(|notice| &notice.source),
                Some(InputSource::Program { .. })
            ) {
                return Err(TerminalError::Invalid(
                    "finish or withdraw the pending input request before declaring another".into(),
                ));
            }
            seen.insert(request_id.clone());
            data.record.input = Some(InputNotice {
                source: InputSource::Program { request_id },
                prompt,
                prompt_truncated: false,
            });
        }
        Declaration::InputCancelled { request_id } => {
            if !matches!(data.record.input.as_ref().map(|notice| &notice.source), Some(InputSource::Program { request_id: pending }) if pending == &request_id)
            {
                return Ok(false);
            }
            data.record.input = None;
        }
    }
    data.record.revision += 1;
    Ok(true)
}

struct ActiveProcess {
    process: io::Process,
    readers: JoinSet<()>,
    requests_task: tokio::task::JoinHandle<()>,
    incoming: mpsc::Receiver<ReadEvent>,
}

impl ActiveProcess {
    fn from_process(mut process: io::Process) -> Self {
        let (sender, incoming) = mpsc::channel(32);
        let (readers, requests_task) = readers(&mut process, sender);
        Self {
            process,
            readers,
            requests_task,
            incoming,
        }
    }
}

/// Parameters to respawn a standing line monitor's command when a worker
/// starts.
pub(super) struct LineSpawner {
    pub(super) config: TerminalConfig,
    pub(super) command: String,
    pub(super) cwd: PathBuf,
    pub(super) terminal: bool,
    pub(super) directory: PathBuf,
}

impl LineSpawner {
    fn spawn(&self) -> std::io::Result<ActiveProcess> {
        let process = io::respawn(
            &self.config,
            &self.command,
            &self.cwd,
            self.terminal,
            &self.directory,
        )?;
        Ok(ActiveProcess::from_process(process))
    }
}

/// Terminates a standing line monitor's child process for a pause, draining
/// any buffered output without ending the monitor or sending an exit alert.
#[allow(clippy::too_many_arguments)]
async fn pause_process(
    mut active: ActiveProcess,
    manager: &Weak<TerminalManager>,
    config: &TerminalConfig,
    data: &Arc<Mutex<JobData>>,
    log: &mut std::fs::File,
    logged: &mut u64,
    monitor: &mut Option<monitor::LineWatch>,
    changed: &watch::Sender<u64>,
) -> Option<String> {
    active.requests_task.abort();
    let _ = active.requests_task.await;
    active.process.group.kill();
    let _ = active.process.child.wait().await;
    let mut failure = None;
    let mut wrote = false;
    let drain = async {
        while let Some(item) = active.incoming.recv().await {
            if let ReadEvent::Output(bytes) = item {
                let (offset, result) = {
                    let mut data = data.lock().unwrap();
                    let offset = data.record.output_end;
                    (offset, output(&mut data, &bytes, log, logged, config))
                };
                wrote = true;
                if let Err(error) = result {
                    failure = Some(error.to_string());
                }
                if let Some(watch) = monitor {
                    watch.output(manager, &bytes, offset);
                }
            }
        }
    };
    let _ = tokio::time::timeout(Duration::from_secs(1), drain).await;
    active.readers.abort_all();
    while active.readers.join_next().await.is_some() {}
    if let Some(watch) = monitor {
        watch.flush(manager);
    }
    if wrote {
        changed.send_replace(data.lock().unwrap().record.revision);
    }
    failure
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn run(
    manager: Weak<TerminalManager>,
    config: TerminalConfig,
    detector: Arc<Detector>,
    data: Arc<Mutex<JobData>>,
    process: Option<io::Process>,
    spawner: Option<LineSpawner>,
    log: &mut std::fs::File,
    options: StartOptions,
    mut controls: mpsc::Receiver<Control>,
    mut cancellation: watch::Receiver<bool>,
    changed: watch::Sender<u64>,
    mut monitor: Option<monitor::LineWatch>,
) {
    let mut active_proc = process.map(ActiveProcess::from_process);
    // A monitor's end is its only time limit.
    let timeout = match monitor {
        Some(_) => options.timeout,
        None => options.timeout.or(config.default_timeout),
    };
    let deadline = timeout.and_then(|value| Instant::now().checked_add(value));
    let mut last_output = Instant::now();
    let mut check_prompt = false;
    let mut logged = 0;
    let mut control_buffer = Vec::new();
    let mut seen_requests = HashSet::new();
    let mut stop = None;
    let mut exit_code = None;
    let mut failure = None;
    // How a monitor ended by itself, other than by its command's exit.
    let mut done: Option<monitor::Done> = None;
    let mut timing = monitor
        .as_ref()
        .and(manager.upgrade())
        .map(|manager| manager.watch_monitor_timing());
    let mut workers = workers_watch(&manager, monitor.as_ref().map(|watch| &watch.running));
    let standing = monitor
        .as_ref()
        .is_some_and(|watch| watch.running.lifetime == monitor::MonitorLifetime::Standing);
    let mut paused = active_proc.is_none();
    while stop.is_none() {
        if *cancellation.borrow() {
            stop = Some(JobState::Cancelled);
            break;
        }
        if standing {
            let active = monitor
                .as_ref()
                .is_some_and(|watch| watch.running.workers_active(&manager));
            if active && active_proc.is_none() {
                match spawner
                    .as_ref()
                    .expect("standing line monitor has a spawner")
                    .spawn()
                {
                    Ok(spawned) => {
                        active_proc = Some(spawned);
                        monitor::set_paused(&data, &mut paused, false);
                    }
                    Err(error) => {
                        failure = Some(format!("could not resume monitor: {error}"));
                        stop = Some(JobState::Failed);
                        break;
                    }
                }
            } else if !active && active_proc.is_some() {
                if let Some(proc) = active_proc.take()
                    && let Some(err) = pause_process(
                        proc,
                        &manager,
                        &config,
                        &data,
                        log,
                        &mut logged,
                        &mut monitor,
                        &changed,
                    )
                    .await
                {
                    failure = Some(err);
                    stop = Some(JobState::Failed);
                    break;
                }
                check_prompt = false;
                if monitor.as_ref().is_some_and(monitor::LineWatch::is_done) {
                    done = Some(monitor::Done::Alerted);
                    stop = Some(JobState::Succeeded);
                    break;
                }
                monitor::set_paused(&data, &mut paused, true);
                continue;
            }
        }
        let wait_deadline = async {
            match deadline {
                Some(deadline) => sleep_until(deadline).await,
                None => std::future::pending().await,
            }
        };
        let monitor_event = until(
            monitor
                .as_ref()
                .and_then(|watch| watch.running.next_event(&manager, &data)),
        );
        let partial_line = until(monitor.as_ref().and_then(|watch| watch.partial_due()));
        let (child, incoming, input, replies) = match active_proc.as_mut() {
            Some(active) => (
                Some(&mut active.process.child),
                Some(&mut active.incoming),
                Some(&mut active.process.input),
                Some(&mut active.process.replies),
            ),
            None => (None, None, None, None),
        };
        let has_incoming = incoming
            .as_ref()
            .is_some_and(|rx| !rx.is_closed() || !rx.is_empty());
        let wait_child = async {
            match child {
                Some(child) => child.wait().await,
                None => std::future::pending().await,
            }
        };
        let recv_incoming = async {
            match incoming {
                Some(rx) => rx.recv().await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            _ = cancellation.changed() => { stop = Some(JobState::Cancelled); }
            _ = wait_deadline => { stop = Some(JobState::TimedOut); }
            result = wait_child => {
                match result {
                    Ok(status) => { exit_code = status.code(); stop = Some(if status.success() { JobState::Succeeded } else { JobState::Failed }); }
                    Err(error) => { failure = Some(error.to_string()); stop = Some(JobState::Failed); }
                }
            }
            item = recv_incoming, if has_incoming => {
                match item {
                    Some(ReadEvent::Output(bytes)) => {
                        let (offset, result) = {
                            let mut data = data.lock().unwrap();
                            let offset = data.record.output_end;
                            (offset, output(&mut data, &bytes, log, &mut logged, &config))
                        };
                        if let Err(error) = result { failure = Some(error.to_string()); stop = Some(JobState::Failed); }
                        if let Some(watch) = &mut monitor { watch.output(&manager, &bytes, offset); }
                        if stop.is_none() && monitor.as_ref().is_some_and(monitor::LineWatch::is_done) {
                            done = Some(monitor::Done::Alerted); stop = Some(JobState::Succeeded);
                        }
                        last_output = Instant::now(); check_prompt = options.detect_prompts;
                    }
                    Some(ReadEvent::Declaration(bytes)) => {
                        control_buffer.extend(bytes);
                        while let Some(end) = control_buffer.iter().position(|byte| *byte == b'\n') {
                            if end > 8192 { failure = Some("oversized input declaration".into()); stop = Some(JobState::Failed); break; }
                            let line = control_buffer.drain(..=end).collect::<Vec<_>>();
                            let result = declaration(&line, &mut data.lock().unwrap(), &mut seen_requests);
                            match result {
                                Ok(true) => if let Err(error) = event(&manager, &data, &config) { failure = Some(error.to_string()); stop = Some(JobState::Failed); },
                                Ok(false) => {},
                                Err(error) => { failure = Some(format!("invalid input declaration: {error}")); stop = Some(JobState::Failed); break; }
                            }
                        }
                        if control_buffer.len() > 8192 { failure = Some("oversized input declaration".into()); stop = Some(JobState::Failed); }
                    }
                    Some(ReadEvent::Error(error)) => { failure = Some(error); stop = Some(JobState::Failed); }
                    None => {},
                }
                changed.send_replace(data.lock().unwrap().record.revision);
            }
            _ = sleep_until(last_output + Duration::from_millis(config.prompt_quiet_ms)), if check_prompt => {
                check_prompt = false;
                let matched = {
                    let mut data = data.lock().unwrap();
                    if data.record.input.is_none() {
                        let bytes = data.output.iter().copied().collect::<Vec<_>>();
                        if let Some((rule, mut prompt)) = detector.detect(&bytes) {
                            let prompt_truncated = prompt.len() > config.excerpt_bytes;
                            if prompt_truncated {
                                let mut start = prompt.len() - config.excerpt_bytes;
                                while !prompt.is_char_boundary(start) { start += 1; }
                                prompt = prompt[start..].to_string();
                            }
                            // Heuristic prompt matches inspect already-received output, so
                            // don't bump revision and invalidate terminal_input calls using
                            // the revision the caller already read.
                            data.record.input = Some(InputNotice { source: InputSource::PromptRule { rule }, prompt, prompt_truncated });
                            true
                        } else { false }
                    } else { false }
                };
                if matched {
                    if let Err(error) = event(&manager, &data, &config) { failure = Some(error.to_string()); stop = Some(JobState::Failed); }
                    changed.send_replace(data.lock().unwrap().record.revision);
                }
            }
            () = monitor_event => {
                if let Some(ended) = monitor.as_ref().and_then(|watch| watch.running.timed_event(&manager, &data, &config)) {
                    done = Some(ended);
                    stop = Some(ended.state());
                }
            }
            () = partial_line => {
                if let Some(watch) = &mut monitor { watch.flush(&manager); }
                if monitor.as_ref().is_some_and(monitor::LineWatch::is_done) {
                    done = Some(monitor::Done::Alerted); stop = Some(JobState::Succeeded);
                }
            }
            // Planned again.
            () = retimed(&mut timing) => {
                workers = workers_watch(&manager, monitor.as_ref().map(|watch| &watch.running));
            }
            // Workers started or ended.
            () = workers_changed(&mut workers) => {}
            control = controls.recv() => {
                match control {
                    Some(Control::Input { revision, text, response }) => {
                        let (Some(input), Some(replies)) = (input, replies) else {
                            let _ = response.send(Err(TerminalError::Invalid(
                                "this monitor is paused while no workers run".into(),
                            )));
                            continue;
                        };
                        let (stale, request_id) = {
                            let mut data = data.lock().unwrap();
                            if revision != data.record.revision { (true, None) } else {
                                let request_id = match data.record.input.take().map(|notice|notice.source) {
                                    Some(InputSource::Program { request_id }) => Some(request_id), _ => None,
                                };
                                data.record.revision += 1; data.last_event = None; (false, request_id)
                            }
                        };
                        if stale { let _ = response.send(Err(TerminalError::StaleView)); continue; }
                        check_prompt = false;
                        // Cooperating programs receive the request ID with the
                        // reply and can reject it even if they changed questions
                        // before the host observed their cancellation notice.
                        let payload = request_id.map(|request_id| {
                            let mut payload = serde_json::to_vec(&serde_json::json!({"request_id":request_id,"text":text})).expect("reply serialization");
                            payload.push(b'\n'); payload
                        });
                        let result = match &payload {
                            Some(bytes) => tokio::time::timeout(Duration::from_secs(2), replies.write_all(bytes)).await,
                            None => tokio::time::timeout(Duration::from_secs(2), input.write_all(text.as_bytes())).await,
                        };
                        match result {
                            Ok(Ok(())) => { let _ = response.send(Ok(())); }
                            error => {
                                let message = format!("input delivery uncertain. Job stopped to prevent duplicate input: {error:?}");
                                failure = Some(message.clone()); stop = Some(JobState::Failed);
                                let _ = response.send(Err(TerminalError::Invalid(message)));
                            }
                        }
                        changed.send_replace(data.lock().unwrap().record.revision);
                    }
                    None => { stop = Some(JobState::Cancelled); }
                }
            }
        }
    }
    // The job is ending from now on, whatever its end turns out to be.
    if let Some(stop) = stop {
        data.lock().unwrap().decide(stop);
    }
    #[cfg(test)]
    let barrier = data.lock().unwrap().end_barrier.clone();
    #[cfg(test)]
    if let Some(barrier) = barrier {
        barrier.hold().await;
    }
    if let Some(mut active) = active_proc.take() {
        active.requests_task.abort();
        let _ = active.requests_task.await;
        active.process.group.kill();
        let _ = active.process.child.wait().await;
        // Drain final bytes after process exit, but descendants cannot hold us open.
        let drain = async {
            while let Some(item) = active.incoming.recv().await {
                if let ReadEvent::Output(bytes) = item {
                    let (offset, result) = {
                        let mut data = data.lock().unwrap();
                        let offset = data.record.output_end;
                        (offset, output(&mut data, &bytes, log, &mut logged, &config))
                    };
                    if let Err(error) = result {
                        failure = Some(error.to_string());
                    }
                    if let Some(watch) = &mut monitor {
                        watch.output(&manager, &bytes, offset);
                    }
                }
            }
        };
        if tokio::time::timeout(Duration::from_secs(1), drain)
            .await
            .is_err()
        {
            failure = Some("output drain did not complete. Output may be incomplete".into());
        }
        active.readers.abort_all();
        while active.readers.join_next().await.is_some() {}
    }
    if let Some(watch) = &mut monitor {
        watch.flush(&manager);
    }
    let final_notice = {
        let mut data = data.lock().unwrap();
        let state = if failure.is_some() && stop == Some(JobState::Succeeded) {
            JobState::Failed
        } else {
            stop.unwrap_or(JobState::Failed)
        };
        let by_user =
            stop == Some(JobState::Cancelled) && data.stopped_by == Some(calls::StopCause::User);
        if by_user && failure.is_none() {
            failure = Some("stopped by the user".into());
        }
        let ending = monitor.as_ref().map(|_| {
            monitor::end_reason(state, done, data.stopped_by, exit_code, failure.as_deref())
        });
        if monitor.is_some() && failure.is_none() {
            failure = match (done, data.stopped_by) {
                (Some(done), _) => done.error(),
                (None, Some(calls::StopCause::OwnerEnded)) => {
                    Some("stopped because its agent's run ended".into())
                }
                _ => None,
            };
        }
        data.record.exit_code = exit_code;
        // Publish terminal state only after the last manifest write. A caller
        // can then forget the job without racing a later write from this actor.
        let snapshot = data.finish(&config, state, failure);
        // A monitor tells its owner how it ended with its news.
        let last_lines = match &ending {
            Some(Some(monitor::EndReason::Exited { .. })) => monitor
                .as_ref()
                .map(|watch| monitor::last_lines(&data.output, watch.running.limits.line_bytes))
                .unwrap_or_default(),
            _ => Vec::new(),
        };
        if ending.is_none() {
            data.last_event = Some(snapshot.clone());
        }
        (
            data.record.owner,
            snapshot,
            calls::wakes_owner(data.record.held, data.stopped_by),
            ending,
            last_lines,
        )
    };
    let (owner, snapshot, wakes, ending, last_lines) = final_notice;
    match (ending, &monitor) {
        (Some(ending), Some(watch)) => {
            // Only a command that exited by itself wakes its owner, as
            // other jobs that finish do.
            if let Some(manager) = manager.upgrade()
                && matches!(ending, Some(monitor::EndReason::Exited { .. }))
            {
                let _ = manager.finished.send(JobFinished {
                    owner,
                    job_id: snapshot.job_id,
                    state: snapshot.state,
                });
            }
            if let Some(reason) = ending {
                watch.running.tell(
                    &manager,
                    outbox::News::Ended {
                        reason,
                        at: std::time::SystemTime::now(),
                        last_lines,
                    },
                );
            }
        }
        _ => {
            if let Some(manager) = manager.upgrade() {
                // Otherwise the job's update goes with the owner's next turn.
                if wakes {
                    let _ = manager.finished.send(JobFinished {
                        owner,
                        job_id: snapshot.job_id,
                        state: snapshot.state,
                    });
                }
                manager.deliver(owner, snapshot);
            }
        }
    }
    changed.send_replace(data.lock().unwrap().record.revision);
}

/// Resolves when `timing` changes, or never without it.
pub(super) async fn retimed(timing: &mut Option<watch::Receiver<u64>>) {
    if let Some(receiver) = timing
        && receiver.changed().await.is_ok()
    {
        return;
    }
    std::future::pending().await
}

/// What tells a monitor that follows its owner's workers that they
/// started or ended, one for each [`WorkerLiveness`](monitor::WorkerLiveness),
/// or none for another monitor.
pub(super) fn workers_watch(
    manager: &Weak<TerminalManager>,
    running: Option<&monitor::Running>,
) -> Vec<watch::Receiver<u64>> {
    running
        .filter(|running| running.lifetime.watches_workers())
        .and(manager.upgrade())
        .map(|manager| manager.worker_changes())
        .unwrap_or_default()
}

/// Resolves when one of `workers` changes, and once more when one closes,
/// which is then left out. Never without any.
pub(super) async fn workers_changed(workers: &mut Vec<watch::Receiver<u64>>) {
    if workers.is_empty() {
        return std::future::pending().await;
    }
    let (closed, index) = {
        let changes = workers
            .iter_mut()
            .map(|receiver| Box::pin(receiver.changed()));
        let (result, index, _) = futures::future::select_all(changes).await;
        (result.is_err(), index)
    };
    if closed {
        workers.swap_remove(index);
    }
}

/// Resolves at `deadline`, or never without one.
pub(super) async fn until(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => sleep_until(deadline).await,
        None => std::future::pending().await,
    }
}
