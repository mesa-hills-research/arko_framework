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

#[allow(clippy::too_many_arguments)]
pub(super) async fn run(
    manager: Weak<TerminalManager>,
    config: TerminalConfig,
    detector: Arc<Detector>,
    data: Arc<Mutex<JobData>>,
    mut process: io::Process,
    log: &mut std::fs::File,
    options: StartOptions,
    mut controls: mpsc::Receiver<Control>,
    mut cancellation: watch::Receiver<bool>,
    changed: watch::Sender<u64>,
    mut monitor: Option<monitor::LineWatch>,
) {
    let (sender, mut incoming) = mpsc::channel(32);
    let (mut readers, requests_task) = readers(&mut process, sender);
    let timeout = options.timeout.or(config.default_timeout);
    let deadline = timeout.and_then(|value| Instant::now().checked_add(value));
    let mut last_output = Instant::now();
    let mut check_prompt = false;
    let mut logged = 0;
    let mut control_buffer = Vec::new();
    let mut seen_requests = HashSet::new();
    let mut stop = None;
    let mut exit_code = None;
    let mut failure = None;
    // A monitor that lasted as long as it was armed for.
    let mut expired = false;
    let mut timing = monitor
        .as_ref()
        .and(manager.upgrade())
        .map(|manager| manager.watch_monitor_timing());
    while stop.is_none() {
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
        tokio::select! {
            _ = cancellation.changed() => { stop = Some(JobState::Cancelled); }
            _ = wait_deadline => { stop = Some(JobState::TimedOut); }
            result = process.child.wait() => {
                match result {
                    Ok(status) => { exit_code = status.code(); stop = Some(if status.success() { JobState::Succeeded } else { JobState::Failed }); }
                    Err(error) => { failure = Some(error.to_string()); stop = Some(JobState::Failed); }
                }
            }
            item = incoming.recv(), if !incoming.is_closed() || !incoming.is_empty() => {
                match item {
                    Some(ReadEvent::Output(bytes)) => {
                        let (offset, result) = {
                            let mut data = data.lock().unwrap();
                            let offset = data.record.output_end;
                            (offset, output(&mut data, &bytes, log, &mut logged, &config))
                        };
                        if let Err(error) = result { failure = Some(error.to_string()); stop = Some(JobState::Failed); }
                        if let Some(watch) = &mut monitor { watch.output(&manager, &bytes, offset); }
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
                            data.record.input = Some(InputNotice { source: InputSource::PromptRule { rule }, prompt, prompt_truncated });
                            data.record.revision += 1; true
                        } else { false }
                    } else { false }
                };
                if matched {
                    if let Err(error) = event(&manager, &data, &config) { failure = Some(error.to_string()); stop = Some(JobState::Failed); }
                    changed.send_replace(data.lock().unwrap().record.revision);
                }
            }
            () = monitor_event => {
                if monitor.as_ref().is_some_and(|watch| watch.running.timed_event(&manager, &data, &config)) {
                    expired = true;
                    stop = Some(JobState::TimedOut);
                }
            }
            () = partial_line => {
                if let Some(watch) = &mut monitor { watch.flush(&manager); }
            }
            // Planned again.
            () = retimed(&mut timing) => {}
            control = controls.recv() => {
                match control {
                    Some(Control::Input { revision, text, response }) => {
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
                            Some(bytes) => tokio::time::timeout(Duration::from_secs(2), process.replies.write_all(bytes)).await,
                            None => tokio::time::timeout(Duration::from_secs(2), process.input.write_all(text.as_bytes())).await,
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
    if let Some(watch) = &monitor {
        watch.running.ending(&data);
    }
    requests_task.abort();
    let _ = requests_task.await;
    process.group.kill();
    let _ = process.child.wait().await;
    // Drain final bytes after process exit, but descendants cannot hold us open.
    let drain = async {
        while let Some(item) = incoming.recv().await {
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
    readers.abort_all();
    while readers.join_next().await.is_some() {}
    if let Some(watch) = &mut monitor {
        watch.flush(&manager);
    }
    let final_notice = {
        let mut data = data.lock().unwrap();
        data.record.state = if failure.is_some() && stop == Some(JobState::Succeeded) {
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
            monitor::end_reason(
                data.record.state,
                expired,
                data.stopped_by,
                exit_code,
                failure.as_deref(),
            )
        });
        if monitor.is_some() && failure.is_none() {
            failure = match data.stopped_by {
                _ if expired => Some("expired".into()),
                Some(calls::StopCause::OwnerEnded) => {
                    Some("stopped because its agent's run ended".into())
                }
                _ => None,
            };
        }
        data.record.exit_code = exit_code;
        data.record.error = failure;
        data.record.input = None;
        data.record.revision += 1;
        if let Err(error) = persist(&config, &data.record) {
            data.record.state = JobState::Failed;
            data.record.error = Some(format!("could not persist terminal result: {error}"));
        }
        // Publish terminal state only after the last manifest write. A caller
        // can then forget the job without racing a later write from this actor.
        let snapshot = data
            .snapshot(&config, None, config.excerpt_bytes)
            .expect("valid output window");
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
            data.record.held,
            by_user,
            ending,
            last_lines,
        )
    };
    let (owner, snapshot, held, by_user, ending, last_lines) = final_notice;
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
                // The user's stop wakes no one. The job's update goes with
                // the owner's next turn.
                if !held && !by_user {
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

/// Resolves at `deadline`, or never without one.
pub(super) async fn until(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => sleep_until(deadline).await,
        None => std::future::pending().await,
    }
}
