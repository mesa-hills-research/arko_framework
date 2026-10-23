use super::*;
use crate::tool::Tool;
use serde_json::json;
use tokio::time::timeout;

pub(super) struct Fixture {
    pub(super) root: PathBuf,
    pub(super) manager: Arc<TerminalManager>,
    pub(super) owner: Uuid,
}
impl Fixture {
    pub(super) fn new(configure: impl FnOnce(&mut TerminalConfig)) -> Self {
        let root = std::env::temp_dir().join(format!("arko-interactive-test-{}", Uuid::new_v4()));
        let mut config = TerminalConfig::new(root.join("jobs"));
        config.prompt_quiet_ms = 30;
        configure(&mut config);
        Self {
            manager: TerminalManager::new(config).unwrap(),
            root,
            owner: Uuid::new_v4(),
        }
    }
    pub(super) fn context(&self) -> TaskContext<'_> {
        TaskContext::new(self.owner, "tester", "test interactive jobs", &self.root)
    }
    pub(super) fn command(&self, code: &str) -> String {
        let path = self.root.join(format!("fixture-{}.py", Uuid::new_v4()));
        let code = format!(
            "{}\n{code}",
            include_str!("../../tests/fixtures/control_client.py")
        );
        std::fs::write(&path, code).unwrap();
        let python = std::env::var("ARKO_TEST_PYTHON")
            .unwrap_or_else(|_| if cfg!(windows) { "python" } else { "python3" }.into());
        if cfg!(windows) {
            format!("\"{python}\" \"{}\"", path.display())
        } else {
            format!(
                "exec '{}' '{}'",
                python.replace('\'', "'\\''"),
                path.display().to_string().replace('\'', "'\\''")
            )
        }
    }
    fn python(&self, code: &str, options: StartOptions) -> TerminalSnapshot {
        self.manager
            .start(&self.context(), &self.command(code), options)
            .unwrap()
    }
    async fn input(&self, owner: Uuid, id: Uuid, revision: u64, text: String) -> Result<()> {
        let status = self.manager.status(owner, id, None)?;
        let program = matches!(
            status.input.as_ref().map(|n| &n.source),
            Some(InputSource::Program { .. })
        );
        let text = if cfg!(windows) && status.terminal && !program {
            text.replace('\n', "\r")
        } else {
            text
        };
        self.manager.input(owner, id, revision, text).await
    }
    pub(super) async fn until(
        &self,
        id: Uuid,
        predicate: impl Fn(&TerminalSnapshot) -> bool,
    ) -> TerminalSnapshot {
        // Only bounds a genuinely stuck job; passing waits return as soon as the
        // predicate holds. Loaded Windows CI runners spawning many ConPTY
        // children in parallel have exceeded 5 s.
        timeout(Duration::from_secs(30), async {
            loop {
                let snapshot = self.manager.status(self.owner, id, None).unwrap();
                if predicate(&snapshot) {
                    return snapshot;
                }
                assert!(
                    !snapshot.state.is_terminal(),
                    "unexpected completion: {snapshot:?}"
                );
                self.manager
                    .wait(self.owner, id, snapshot.revision, Duration::from_millis(50))
                    .await
                    .unwrap();
            }
        })
        .await
        .unwrap_or_else(|_| {
            panic!(
                "job stalled: {:?}",
                self.manager.status(self.owner, id, None)
            )
        })
    }
    pub(super) async fn finished(&self, id: Uuid) -> TerminalSnapshot {
        self.until(id, |s| s.state.is_terminal()).await
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

#[tokio::test]
async fn yn_prompts_support_stderr_split_chunks_ansi_raw_keys_and_dev_tty() {
    for (code, answer) in [
        (
            "print('Continue? [Y/n] ',end='',flush=True); print('ANSWER='+input())",
            "Y\n",
        ),
        (
            "print('Continue? [Y/n] ',end='',file=sys.stderr,flush=True); print('ANSWER='+input())",
            "Y\n",
        ),
        (
            "print('Cont',end='',flush=True); time.sleep(.06); print('inue? [Y/n] ',end='',flush=True); print('ANSWER='+input())",
            "Y\n",
        ),
        (
            "print('\\x1b[32mContinue? [Y/n]\\x1b[0m ',end='',flush=True); print('ANSWER='+input())",
            "Y\n",
        ),
        #[cfg(unix)]
        (
            "import tty; tty.setraw(0); print('Continue? [Y/n] ',end='',flush=True); print('ANSWER='+os.read(0,1).decode())",
            "Y",
        ),
        #[cfg(unix)]
        (
            "fd=os.open('/dev/tty',os.O_RDWR); os.write(fd,b'Continue? [Y/n] '); print('ANSWER='+os.read(fd,20).decode())",
            "Y\n",
        ),
        #[cfg(windows)]
        (
            "import msvcrt; print('Continue? [Y/n] ',end='',flush=True); print('ANSWER='+msvcrt.getwch())",
            "Y",
        ),
        #[cfg(windows)]
        (
            "with open('CONIN$', 'r') as console: print('Continue? [Y/n] ',end='',flush=True); print('ANSWER='+console.readline().strip())",
            "Y\n",
        ),
    ] {
        let f = Fixture::new(|_| {});
        let started = std::time::Instant::now();
        let job = f.python(code, Default::default());
        assert!(started.elapsed() < Duration::from_secs(3));
        let prompt = f.until(job.job_id, |s| s.input.is_some()).await;
        assert!(matches!(
            prompt.input.unwrap().source,
            InputSource::PromptRule { .. }
        ));
        f.input(f.owner, job.job_id, prompt.revision, answer.into())
            .await
            .unwrap();
        let done = f.finished(job.job_id).await;
        assert_eq!(done.state, JobState::Succeeded, "{done:?}");
        assert!(done.output.text.contains("ANSWER=Y"), "{done:?}");
        f.manager.shutdown().await;
    }
}

#[tokio::test]
async fn explicit_requests_work_without_visible_prompts_and_repeated_questions_have_new_revisions()
{
    for terminal in [true, false] {
        let f = Fixture::new(|_| {});
        let job = f.python("fd=os.open(os.environ['ARKO_INPUT_REQUEST_PATH'],os.O_WRONLY|os.O_APPEND)\nreplies=Replies()\nfor n in range(2):\n request=json.dumps({'type':'input_request','request_id':str(n),'prompt':'Proceed?'})+'\\n'\n os.write(fd,request.encode()); os.write(fd,request.encode())\n value=json.loads(replies.readline()); assert value['request_id']==str(n); assert value['text']=='Y\\n'\nprint('ANSWERS_RECEIVED',flush=True)", StartOptions { terminal, detect_prompts:false, ..Default::default() });
        let first = f.until(job.job_id, |s| matches!(s.input.as_ref().map(|n|&n.source),Some(InputSource::Program{request_id}) if request_id=="0")).await;
        assert_eq!(first.input.as_ref().unwrap().prompt, "Proceed?");
        assert!(!first.output.text.contains("Proceed?"));
        f.input(f.owner, job.job_id, first.revision, "Y\n".into())
            .await
            .unwrap();
        let second = f.until(job.job_id, |s| matches!(s.input.as_ref().map(|n|&n.source),Some(InputSource::Program{request_id}) if request_id=="1")).await;
        assert!(second.revision > first.revision);
        assert!(matches!(
            f.input(f.owner, job.job_id, first.revision, "N\n".into())
                .await,
            Err(TerminalError::StaleView)
        ));
        f.input(f.owner, job.job_id, second.revision, "Y\n".into())
            .await
            .unwrap();
        assert_eq!(f.finished(job.job_id).await.state, JobState::Succeeded);
        f.manager.shutdown().await;
    }
}

#[tokio::test]
async fn silence_is_not_input_and_prompt_looking_logs_remain_explicitly_inferred() {
    for code in [
        "time.sleep(.15)",
        "end=time.monotonic()+.15\nwhile time.monotonic()<end: pass",
    ] {
        let f = Fixture::new(|_| {});
        let (sender, mut receiver) = mpsc::channel(16);
        f.manager.bind_runtime(f.owner, sender.downgrade());
        let job = f.python(code, Default::default());
        let done = f.finished(job.job_id).await;
        assert_eq!(done.state, JobState::Succeeded);
        let notification = timeout(Duration::from_secs(1), receiver.recv())
            .await
            .unwrap()
            .unwrap();
        let UserCommand::InjectContext { content } = notification else {
            panic!()
        };
        assert!(content.contains("\"input\":null"));
        assert!(receiver.try_recv().is_err());
    }
    let f = Fixture::new(|_| {});
    let job = f.python(
        "print('Continue? [Y/n] ',end='',flush=True); time.sleep(.2)",
        Default::default(),
    );
    let possible = f.until(job.job_id, |s| s.input.is_some()).await;
    assert!(matches!(
        possible.input.unwrap().source,
        InputSource::PromptRule { .. }
    ));
    assert_eq!(f.finished(job.job_id).await.state, JobState::Succeeded);
}

#[tokio::test]
async fn silent_reads_allow_manual_input_and_prompt_detection_is_optional() {
    let f = Fixture::new(|_| {});
    let job = f.python(
        "print('ready',flush=True); print('ANSWER='+input())",
        StartOptions {
            detect_prompts: false,
            ..Default::default()
        },
    );
    let ready = f
        .until(job.job_id, |s| s.output.text.contains("ready"))
        .await;
    assert!(ready.input.is_none());
    f.input(f.owner, job.job_id, ready.revision, "Y\n".into())
        .await
        .unwrap();
    let done = f.finished(job.job_id).await;
    assert!(done.output.text.contains("ANSWER=Y"));
}

#[tokio::test]
async fn delayed_repeated_prompts_notify_once_and_busy_context_channels_do_not_block_the_child() {
    let f = Fixture::new(|_| {});
    let (sender, mut receiver) = mpsc::channel(1);
    sender
        .send(UserCommand::InjectContext {
            content: "busy".into(),
        })
        .await
        .unwrap();
    f.manager.bind_runtime(f.owner, sender.downgrade());
    let job=f.python("time.sleep(.1)\nfor n in range(2):\n print('Continue? [Y/n] ',end='',flush=True); input()\nprint('done',flush=True)",Default::default());
    let first = f.until(job.job_id, |s| s.input.is_some()).await;
    // A full notification channel must not prevent the process from accepting input.
    f.input(f.owner, job.job_id, first.revision, "Y\n".into())
        .await
        .unwrap();
    let second = f
        .until(job.job_id, |s| {
            s.input.is_some() && s.revision > first.revision
        })
        .await;
    f.input(f.owner, job.job_id, second.revision, "Y\n".into())
        .await
        .unwrap();
    assert_eq!(f.finished(job.job_id).await.state, JobState::Succeeded);
    receiver.recv().await.unwrap();
    let UserCommand::InjectContext { content } = timeout(Duration::from_secs(1), receiver.recv())
        .await
        .unwrap()
        .unwrap()
    else {
        panic!()
    };
    assert!(content.contains("succeeded"));
    // Superseded questions were answered through status/input while the channel
    // was busy. Only the current completion belongs in the agent's context.
    assert!(receiver.try_recv().is_err());
}

#[tokio::test]
async fn output_cursors_limits_and_process_ownership_are_enforced() {
    let f = Fixture::new(|config| {
        config.output_buffer_bytes = 2048;
        config.excerpt_bytes = 512;
        config.max_log_bytes = 1024;
    });
    let job=f.python("print('x'*10000,flush=True); print('Continue? [Y/n] ',end='',flush=True); input(); print('last',flush=True)",StartOptions{terminal:false,..Default::default()});
    let prompt = f.until(job.job_id, |s| s.input.is_some()).await;
    let output = f.manager.status(f.owner, job.job_id, Some(0)).unwrap();
    assert!(output.output.gap);
    assert!(output.log_truncated);
    assert!(output.output.text.len() <= 512);
    let next = f
        .manager
        .status(f.owner, job.job_id, Some(output.output.end))
        .unwrap();
    assert_eq!(next.output.start, output.output.end);
    assert!(matches!(
        f.manager.status(Uuid::new_v4(), job.job_id, None),
        Err(TerminalError::NotOwned)
    ));
    assert!(matches!(
        f.input(Uuid::new_v4(), job.job_id, prompt.revision, "Y\n".into())
            .await,
        Err(TerminalError::NotOwned)
    ));
    f.input(f.owner, job.job_id, prompt.revision, "Y\n".into())
        .await
        .unwrap();
    let done = f.finished(job.job_id).await;
    let empty = f
        .manager
        .status(f.owner, job.job_id, Some(done.output.available_end))
        .unwrap();
    assert_eq!(empty.output.text, "");
    assert_eq!(std::fs::metadata(&done.log_path).unwrap().len(), 1024);
}

#[tokio::test]
async fn cancellations_deadlines_and_background_descendants_finish_without_hanging() {
    let f = Fixture::new(|_| {});
    let job = f.python(
        "print('Continue? [Y/n] ',end='',flush=True); input()",
        StartOptions {
            timeout: Some(Duration::from_millis(120)),
            ..Default::default()
        },
    );
    assert_eq!(f.finished(job.job_id).await.state, JobState::TimedOut);
    let job = f.python("time.sleep(100)", Default::default());
    f.manager.cancel(f.owner, job.job_id).unwrap();
    assert_eq!(f.finished(job.job_id).await.state, JobState::Cancelled);
    let job = f.python(
        "import subprocess; subprocess.Popen([sys.executable,'-c','import time; time.sleep(100)'])",
        Default::default(),
    );
    assert_eq!(f.finished(job.job_id).await.state, JobState::Succeeded);
}

#[tokio::test]
async fn malformed_declarations_fail_boundedly_and_withdrawal_invalidates_old_answers() {
    let f = Fixture::new(|_| {});
    let job=f.python("fd=os.open(os.environ['ARKO_INPUT_REQUEST_PATH'],os.O_WRONLY|os.O_APPEND); os.write(fd,b'{bad json}\\n'); time.sleep(100)",Default::default());
    let done = f.finished(job.job_id).await;
    assert_eq!(done.state, JobState::Failed);
    assert!(done.error.unwrap().contains("invalid input declaration"));
    let job=f.python("fd=os.open(os.environ['ARKO_INPUT_REQUEST_PATH'],os.O_WRONLY|os.O_APPEND)\nos.write(fd,(json.dumps({'type':'input_request','request_id':'one','prompt':'Answer?'})+'\\n').encode())\ntime.sleep(.1)\nos.write(fd,(json.dumps({'type':'input_cancelled','request_id':'one'})+'\\n').encode())\ntime.sleep(100)",Default::default());
    let prompt = f.until(job.job_id, |s| s.input.is_some()).await;
    f.until(job.job_id, |s| {
        s.revision > prompt.revision && s.input.is_none()
    })
    .await;
    assert!(matches!(
        f.input(f.owner, job.job_id, prompt.revision, "Y\n".into())
            .await,
        Err(TerminalError::StaleView)
    ));
    f.manager.shutdown().await;
}

#[tokio::test]
async fn terminal_tools_accept_native_and_text_parameters_and_expose_job_state() {
    let f = Fixture::new(|_| {});
    let tools = TerminalTool::all(f.manager.clone());
    let params = |value: serde_json::Value| {
        serde_json::from_value::<HashMap<String, serde_json::Value>>(value).unwrap()
    };
    for terminal in [json!(true), json!("false")] {
        let result=tools[0].execute(&f.context(),params(json!({"command":f.command("print('Continue? [Y/n] ',end='',flush=True); print('answer='+input())"),"terminal":terminal}))).await.unwrap();
        let job: TerminalSnapshot = serde_json::from_str(&result).unwrap();
        let prompt = f.until(job.job_id, |s| s.input.is_some()).await;
        tools[2].execute(&f.context(),params(json!({"job_id":job.job_id.to_string(),"revision":prompt.revision.to_string(),"text":if cfg!(windows) && job.terminal { "Y\r" } else { "Y\n" }}))).await.unwrap();
        assert_eq!(f.finished(job.job_id).await.state, JobState::Succeeded);
    }
    assert!(tools.iter().all(|tool| !tool.allow_result_interception()));
    let listed = tools[3]
        .execute(&f.context(), params(json!({"action":"list"})))
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Vec<serde_json::Value>>(&listed)
            .unwrap()
            .len(),
        2
    );
    assert!(
        tools[0]
            .execute(
                &f.context(),
                params(json!({"command":"true","timeout_ms":"0"}))
            )
            .await
            .is_err()
    );
}

/// The terminal tools keep their own deadlines (jobs run in the background
/// with theirs, and waits end after at most 30 seconds), so the host tool
/// time limit doesn't apply to them (AG-2).
#[tokio::test]
async fn terminal_tools_are_exempt_from_the_host_tool_time_limit() {
    let f = Fixture::new(|_| {});
    let tools = TerminalTool::all(f.manager.clone());
    assert_eq!(tools.len(), TerminalTool::NAMES.len());
    for tool in &tools {
        assert!(!tool.uses_host_timeout(), "{}", tool.definition().name);
    }
}

#[tokio::test]
async fn records_survive_restart_as_interrupted_without_relaunching_commands() {
    let root = std::env::temp_dir().join(format!("arko-terminal-restart-{}", Uuid::new_v4()));
    let config = TerminalConfig::new(&root);
    let manager = TerminalManager::new(config.clone()).unwrap();
    assert!(TerminalManager::new(config.clone()).is_err());
    let owner = Uuid::new_v4();
    let id = Uuid::new_v4();
    std::fs::create_dir(root.join(id.to_string())).unwrap();
    persist(
        &manager.config,
        &Record {
            job_id: id,
            owner,
            revision: 3,
            state: JobState::Running,
            input: Some(InputNotice {
                source: InputSource::Program {
                    request_id: "old".into(),
                },
                prompt: "Old?".into(),
                prompt_truncated: false,
            }),
            exit_code: None,
            error: None,
            terminal: true,
            output_end: 0,
            log_truncated: false,
            held: false,
            tool_call_id: None,
            label: None,
            call: None,
            stopped_by_close: false,
            generation: manager.generation,
            monitor: None,
        },
    )
    .unwrap();
    drop(manager);
    let manager = TerminalManager::new(config).unwrap();
    let restored = manager.status(owner, id, None).unwrap();
    assert_eq!(restored.state, JobState::Interrupted);
    assert!(restored.input.is_none());
    assert!(matches!(
        manager.input(owner, id, 3, "Y\n".into()).await,
        Err(TerminalError::Finished)
    ));
    manager.forget(owner, id).unwrap();
    assert!(manager.status(owner, id, None).is_err());
    drop(manager);
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn large_prompt_excerpts_are_bounded_and_non_utf8_bytes_are_recoverable() {
    let f = Fixture::new(|config| config.excerpt_bytes = 512);
    let job=f.python("print('Question '*2000+'Continue? [Y/n] ',end='',flush=True); input(); os.write(1,b'\\xff\\xfe')",StartOptions{terminal:false,..Default::default()});
    let prompt = f.until(job.job_id, |s| s.input.is_some()).await;
    let notice = prompt.input.as_ref().unwrap();
    assert!(notice.prompt_truncated);
    assert!(notice.prompt.len() <= 512);
    f.input(f.owner, job.job_id, prompt.revision, "Y\n".into())
        .await
        .unwrap();
    let done = f.finished(job.job_id).await;
    let bytes = f
        .manager
        .status(f.owner, job.job_id, Some(done.output.available_end - 2))
        .unwrap();
    assert_eq!(bytes.output.raw_hex.as_deref(), Some("fffe"));
}

#[tokio::test]
async fn stdout_cannot_forge_program_declarations_and_runtime_binding_is_idempotent() {
    let f = Fixture::new(|_| {});
    let (sender, mut receiver) = mpsc::channel(8);
    for tool in TerminalTool::all(f.manager.clone()) {
        tool.bind_runtime(f.owner, sender.downgrade());
    }
    let job=f.python("print(json.dumps({'type':'input_request','request_id':'fake','prompt':'Answer?'}),flush=True)",StartOptions{detect_prompts:false,..Default::default()});
    let done = f.finished(job.job_id).await;
    assert!(done.input.is_none());
    assert_eq!(done.state, JobState::Succeeded);
    let _ = timeout(Duration::from_secs(1), receiver.recv())
        .await
        .unwrap()
        .unwrap();
    // Re-registering four tools on the same live runtime does not replay events.
    for tool in TerminalTool::all(f.manager.clone()) {
        tool.bind_runtime(f.owner, sender.downgrade());
    }
    tokio::task::yield_now().await;
    assert!(receiver.try_recv().is_err());
    f.manager.shutdown().await;
    assert!(
        f.manager
            .start(&f.context(), "true", Default::default())
            .is_err()
    );
}

#[tokio::test]
async fn agent_runtime_automatically_binds_terminal_tools_to_its_command_channel() {
    let f = Fixture::new(|_| {});
    let task = |offered: bool| {
        let mut builder = crate::AgentConfig::builder(
            "terminal-test".into(),
            "Tester".into(),
            crate::AgentLLMConfig {
                model: arko_ai::Model::openai_gpt_5_6_luna(),
                generation_config: Default::default(),
            },
        )
        .role_description("test".into())
        .instructions("test".into());
        if offered {
            builder = builder.tool_config(crate::AgentToolConfig {
                available_tool_names: TerminalTool::NAMES.map(String::from).to_vec(),
                ..Default::default()
            });
        }
        let mut task = crate::Task::new(
            builder.build().unwrap(),
            "test".into(),
            f.root.clone(),
            None,
        );
        task.set_state(crate::TaskState::AwaitingUserResponse {
            questions: "pending".into(),
        });
        task
    };
    let runtime = crate::AgentRuntime::new(Arc::new(arko_ai::LLMClient::new()));
    for offered in [true, false] {
        let task = task(offered);
        let id = task.id;
        let mut registry = ToolRegistry::new();
        f.manager.register_tools(&mut registry);
        let (_events, commands, handle) =
            runtime.run_task(task, Arc::new(registry), None, false, None, None, None);
        // Only the tools a task is offered are bound to it.
        let binding = f.manager.runtimes.lock().unwrap().get(&id).cloned();
        assert_eq!(
            binding.is_some_and(|binding| binding
                .sender
                .upgrade()
                .unwrap()
                .same_channel(&commands)),
            offered
        );
        commands.send(UserCommand::SuspendTask).await.unwrap();
        timeout(Duration::from_secs(1), handle)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }
}

#[tokio::test]
async fn request_ids_allow_a_program_to_reject_a_reply_after_its_question_changed() {
    let f = Fixture::new(|_| {});
    let job = f.python("fd=os.open(os.environ['ARKO_INPUT_REQUEST_PATH'],os.O_WRONLY|os.O_APPEND)\nreplies=Replies()\ndef request(i): os.write(fd,(json.dumps({'type':'input_request','request_id':i,'prompt':'Proceed?'})+'\\n').encode())\nrequest('old')\ncurrent='new'\nreply=json.loads(replies.readline())\nassert reply['request_id']!=current\nprint('REJECTED_OLD_REPLY',flush=True)\nrequest(current)\nreply=json.loads(replies.readline()); assert reply['request_id']==current\nprint('ACCEPTED_NEW_REPLY',flush=True)", StartOptions { detect_prompts:false, ..Default::default() });
    let old = f.until(job.job_id,|s|matches!(s.input.as_ref().map(|n|&n.source),Some(InputSource::Program{request_id}) if request_id=="old")).await;
    // The application's question changed before it told the host. The old
    // request ID still reaches the application, which can refuse that reply.
    f.input(f.owner, job.job_id, old.revision, "Y\n".into())
        .await
        .unwrap();
    let new = f.until(job.job_id,|s|matches!(s.input.as_ref().map(|n|&n.source),Some(InputSource::Program{request_id}) if request_id=="new") && s.output.text.contains("REJECTED_OLD_REPLY")).await;
    f.input(f.owner, job.job_id, new.revision, "N\n".into())
        .await
        .unwrap();
    let done = f.finished(job.job_id).await;
    assert_eq!(done.state, JobState::Succeeded);
    assert!(
        done.output.text.contains("REJECTED_OLD_REPLY")
            && done.output.text.contains("ACCEPTED_NEW_REPLY")
    );
}

#[tokio::test]
async fn custom_prompt_rules_are_configurable_and_jobs_are_not_limited() {
    let f = Fixture::new(|config| {
        config.prompt_rules = vec![PromptRule {
            name: "region".into(),
            pattern: "Pick a region: $".into(),
        }];
    });
    let job = f.python(
        "print('Pick a region: ',end='',flush=True); print(input())",
        Default::default(),
    );
    // Neither running jobs nor finished records limit new jobs (there were
    // limits of 4 and 64).
    let pipes = StartOptions {
        terminal: false,
        detect_prompts: false,
        ..Default::default()
    };
    let others: Vec<_> = (0..65)
        .map(|_| {
            f.manager
                .start(&f.context(), "exit 0", pipes.clone())
                .unwrap()
                .job_id
        })
        .collect();
    let prompt = f.until(job.job_id, |s| s.input.is_some()).await;
    assert!(
        matches!(prompt.input.as_ref().unwrap().source,InputSource::PromptRule{ref rule} if rule=="region")
    );
    f.input(f.owner, job.job_id, prompt.revision, "west\n".into())
        .await
        .unwrap();
    assert_eq!(f.finished(job.job_id).await.state, JobState::Succeeded);
    for id in others {
        assert_eq!(f.finished(id).await.state, JobState::Succeeded);
    }
    f.manager.forget(f.owner, job.job_id).unwrap();
    let next = f.python("pass", Default::default());
    assert_eq!(f.finished(next.job_id).await.state, JobState::Succeeded);
}

#[tokio::test]
async fn failed_spawns_are_recorded() {
    let f = Fixture::new(|config| {
        config.shell = "/nonexistent/arko-shell".into();
    });
    assert!(
        f.manager
            .start(&f.context(), "true", Default::default())
            .is_err()
    );
    let jobs = f.manager.jobs.lock().unwrap();
    assert_eq!(jobs.len(), 1);
    let id = *jobs.keys().next().unwrap();
    drop(jobs);
    assert_eq!(
        f.manager.status(f.owner, id, None).unwrap().state,
        JobState::Failed
    );
    f.manager.forget(f.owner, id).unwrap();
    assert!(f.manager.jobs.lock().unwrap().is_empty());
}

#[tokio::test]
async fn control_stream_handles_partial_utf8_frames_and_rejects_overflow_or_truncation() {
    let f = Fixture::new(|_| {});
    let job = f.python("fd=os.open(os.environ['ARKO_INPUT_REQUEST_PATH'],os.O_WRONLY|os.O_APPEND)\nframe=(json.dumps({'type':'input_request','request_id':'split','prompt':'Café?'},ensure_ascii=False)+'\\n').encode()\nfor byte in frame: os.write(fd,bytes([byte])); time.sleep(.001)\nvalue=json.loads(Replies().readline()); assert value['request_id']=='split'", StartOptions { terminal:false, detect_prompts:false, ..Default::default() });
    let prompt = f.until(job.job_id, |s| s.input.is_some()).await;
    assert_eq!(prompt.input.as_ref().unwrap().prompt, "Café?");
    f.input(f.owner, job.job_id, prompt.revision, "yes".into())
        .await
        .unwrap();
    assert_eq!(f.finished(job.job_id).await.state, JobState::Succeeded);

    for code in [
        "with open(os.environ['ARKO_INPUT_REQUEST_PATH'],'ab') as requests: requests.write(b'x'*8193)\ntime.sleep(100)",
        "with open(os.environ['ARKO_INPUT_REQUEST_PATH'],'ab') as requests: requests.truncate(8*1024*1024+1)\ntime.sleep(100)",
        "fd=os.open(os.environ['ARKO_INPUT_REQUEST_PATH'],os.O_WRONLY|os.O_APPEND)\nos.write(fd,b'{')\ntime.sleep(.1)\nos.ftruncate(fd,0)\ntime.sleep(100)",
    ] {
        let job = f.python(code, Default::default());
        let done = f.finished(job.job_id).await;
        assert_eq!(done.state, JobState::Failed, "{done:?}");
        assert!(done.error.is_some());
    }
    f.manager.shutdown().await;
}

/// PTY jobs get the quiet environment on every platform; pipe jobs, and PTY
/// jobs with it turned off, get the host's.
#[tokio::test]
async fn pty_jobs_get_a_quiet_environment_unless_turned_off() {
    let names = [
        "TERM",
        "NO_COLOR",
        "PAGER",
        "GIT_PAGER",
        "LANG",
        "LC_CTYPE",
        "LC_ALL",
    ];
    let show = format!("print('ENV=' + '|'.join(os.environ.get(name, '-') for name in {names:?}))");
    let host = names
        .iter()
        .map(|name| std::env::var(name).unwrap_or_else(|_| "-".into()))
        .collect::<Vec<_>>()
        .join("|");
    let pipes = StartOptions {
        terminal: false,
        detect_prompts: false,
        ..Default::default()
    };
    let f = Fixture::new(|_| {});
    let pty = f.python(&show, Default::default());
    let text = f.finished(pty.job_id).await.output.text;
    let locale = QUIET_TERMINAL_LOCALE;
    assert!(
        text.contains(&format!("ENV=dumb|1|cat|cat|{locale}|{locale}|{locale}")),
        "{text:?}"
    );
    let pipe = f.python(&show, pipes);
    let text = f.finished(pipe.job_id).await.output.text;
    assert!(text.contains(&format!("ENV={host}")), "{text:?}");

    let f = Fixture::new(|config| config.quiet_terminal_environment = false);
    let pty = f.python(&show, Default::default());
    let text = f.finished(pty.job_id).await.output.text;
    assert!(text.contains(&format!("ENV={host}")), "{text:?}");
}

/// A pipe job on Windows gets a hidden console of its own, so a host with no
/// console (a windowed app) doesn't open a console window for it.
#[cfg(windows)]
#[tokio::test]
async fn windows_pipe_jobs_get_a_hidden_console_of_their_own() {
    let f = Fixture::new(|_| {});
    let pipes = StartOptions {
        terminal: false,
        detect_prompts: false,
        ..Default::default()
    };
    let job = f.python(&crate::console_probe::code(), pipes);
    let done = f.finished(job.job_id).await;
    assert_eq!(done.state, JobState::Succeeded, "{done:?}");
    assert!(
        done.output.text.contains("CONSOLE=own hidden"),
        "{:?}",
        done.output.text
    );
    f.manager.shutdown().await;
}

/// Output shown to the agent loses escape sequences only where they are
/// stripped (Windows PTYs); the byte offsets always count the raw output.
#[test]
fn stripped_windows_keep_raw_offsets() {
    let raw = b"\x1b[32mgreen\x1b[0m done";
    assert_eq!(window_text(raw, true), "green done");
    assert_eq!(window_text(raw, false), String::from_utf8_lossy(raw));
    let mut data = JobData::new(Record {
        job_id: Uuid::new_v4(),
        owner: Uuid::new_v4(),
        revision: 0,
        state: JobState::Running,
        input: None,
        exit_code: None,
        error: None,
        terminal: true,
        output_end: raw.len() as u64,
        log_truncated: false,
        held: false,
        tool_call_id: None,
        label: None,
        call: None,
        stopped_by_close: false,
        generation: 0,
        monitor: None,
    });
    data.output = raw.iter().copied().collect();
    let config = TerminalConfig::new(std::env::temp_dir());
    let snapshot = data.snapshot(&config, None, 2_000).unwrap();
    assert_eq!(
        (snapshot.output.start, snapshot.output.end),
        (0, raw.len() as u64)
    );
    let expected = if cfg!(windows) {
        "green done".to_string()
    } else {
        String::from_utf8_lossy(raw).into_owned()
    };
    assert_eq!(snapshot.output.text, expected);
}

#[tokio::test]
async fn the_command_guard_refuses_drastic_commands_and_input() {
    let f = Fixture::new(|_| {});
    let drastic = if cfg!(windows) {
        "rd /s /q %USERPROFILE%"
    } else {
        "rm -rf ~"
    };
    let error = f
        .manager
        .start(&f.context(), drastic, StartOptions::default())
        .unwrap_err();
    let TerminalError::Refused(refusal) = &error else {
        panic!("expected a refusal, got {error}");
    };
    assert!(refusal.protected);
    let job_directories = std::fs::read_dir(f.root.join("jobs"))
        .unwrap()
        .filter(|entry| entry.as_ref().unwrap().path().is_dir())
        .count();
    assert_eq!(job_directories, 0, "a refused command starts no job");

    // Text typed into a running job is checked as a command too.
    let job = f.python(
        "print('ready',flush=True); print('ANSWER='+input())",
        StartOptions {
            detect_prompts: false,
            ..Default::default()
        },
    );
    let ready = f
        .until(job.job_id, |s| s.output.text.contains("ready"))
        .await;
    let typed = format!("{drastic}\n");
    assert!(matches!(
        f.input(f.owner, job.job_id, ready.revision, typed).await,
        Err(TerminalError::Refused(_))
    ));
    f.input(f.owner, job.job_id, ready.revision, "Y\n".into())
        .await
        .unwrap();
    assert!(
        f.finished(job.job_id)
            .await
            .output
            .text
            .contains("ANSWER=Y")
    );
}

#[tokio::test]
async fn dangerously_allow_all_commands_turns_the_guard_off() {
    let f = Fixture::new(|config| config.command_guard.dangerously_allow_all_commands = true);
    // The shell stops before the destructive half, so this is safe to run.
    let command = if cfg!(windows) {
        "exit /b 3 && rd /s /q %USERPROFILE%"
    } else {
        "exit 3 && rm -rf ~"
    };
    let job = f
        .manager
        .start(&f.context(), command, StartOptions::default())
        .unwrap();
    assert_eq!(f.finished(job.job_id).await.exit_code, Some(3));
}

#[tokio::test]
async fn outside_access_can_change_while_the_manager_runs() {
    let f = Fixture::new(|_| {});
    assert!(!f.manager.allows_outside_workspace());
    // The shell stops before the destructive half, so this is safe to run.
    let name = format!("arko-outside-test-{}", Uuid::new_v4());
    let (outside, protected) = if cfg!(windows) {
        (
            format!("exit /b 3 && rd /s /q %USERPROFILE%\\{name}"),
            "exit /b 3 && rd /s /q %USERPROFILE%",
        )
    } else {
        (format!("exit 3 && rm -rf ~/{name}"), "exit 3 && rm -rf ~")
    };
    let refused = |result: Result<TerminalSnapshot>| match result {
        Err(TerminalError::Refused(refusal)) => refusal,
        other => panic!("expected a refusal, got {other:?}"),
    };
    let start = |command: &str| {
        f.manager
            .start(&f.context(), command, StartOptions::default())
    };
    assert!(!refused(start(&outside)).protected);

    f.manager.set_allow_outside_workspace(true);
    assert!(f.manager.allows_outside_workspace());
    let job = start(&outside).unwrap();
    assert_eq!(f.finished(job.job_id).await.exit_code, Some(3));
    assert!(
        refused(start(protected)).protected,
        "protected paths stay refused"
    );

    f.manager.set_allow_outside_workspace(false);
    assert!(!refused(start(&outside)).protected);

    // A guard configured with outside access starts with it.
    let f = Fixture::new(|config| config.command_guard.allow_outside_workspace = true);
    assert!(f.manager.allows_outside_workspace());
}

/// The jobs the terminal tools list, as `terminal_control list` gives them.
async fn listed(f: &Fixture) -> Vec<Uuid> {
    let list = TerminalTool::all(f.manager.clone())[3]
        .execute(
            &f.context(),
            serde_json::from_value(json!({"action": "list"})).unwrap(),
        )
        .await
        .unwrap();
    serde_json::from_str::<Vec<serde_json::Value>>(&list)
        .unwrap()
        .iter()
        .map(|job| job["job_id"].as_str().unwrap().parse().unwrap())
        .collect()
}

/// The terminal update a runtime got, as its JSON.
async fn update(receiver: &mut mpsc::Receiver<UserCommand>) -> serde_json::Value {
    let UserCommand::InjectContext { content } = timeout(Duration::from_secs(5), receiver.recv())
        .await
        .unwrap()
        .unwrap()
    else {
        panic!("not a terminal update")
    };
    serde_json::from_str(content.strip_prefix("Terminal update:\n").unwrap()).unwrap()
}

/// A held job (a command its host waits for) sends no updates and isn't
/// listed. Released, it is listed (the host's list shows its label), the
/// event it has goes out at once, and its end is an update and a finished
/// notice.
#[tokio::test]
async fn a_held_job_tells_nothing_until_it_is_released() {
    let f = Fixture::new(|_| {});
    let (sender, mut receiver) = mpsc::channel(8);
    f.manager.bind_runtime(f.owner, sender.downgrade());
    let mut finished = f.manager.subscribe_finished();
    let job = f.python(
        "fd=os.open(os.environ['ARKO_INPUT_REQUEST_PATH'],os.O_WRONLY|os.O_APPEND)\nos.write(fd,(json.dumps({'type':'input_request','request_id':'one','prompt':'Go on?'})+'\\n').encode())\nvalue=json.loads(Replies().readline()); print('ANSWER='+value['text'],flush=True)",
        StartOptions {
            terminal: false,
            detect_prompts: false,
            held: true,
            label: Some("ask once".into()),
            ..Default::default()
        },
    );
    let asked = f.until(job.job_id, |s| s.input.is_some()).await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(receiver.try_recv().is_err(), "a held job sends no update");
    assert!(listed(&f).await.is_empty());
    assert!(f.manager.background_jobs().is_empty());

    let released = f.manager.release(f.owner, job.job_id).unwrap();
    assert_eq!(released.state, JobState::Running);
    assert_eq!(listed(&f).await, [job.job_id]);
    let background = f.manager.background_jobs();
    assert_eq!(background.len(), 1);
    assert_eq!(
        (background[0].owner, background[0].job_id),
        (f.owner, job.job_id)
    );
    assert_eq!(background[0].command, "ask once");
    assert_eq!(update(&mut receiver).await["input"]["prompt"], "Go on?");
    f.input(f.owner, job.job_id, asked.revision, "yes".into())
        .await
        .unwrap();
    assert_eq!(f.finished(job.job_id).await.state, JobState::Succeeded);
    let done = update(&mut receiver).await;
    assert_eq!(done["state"], "succeeded");
    assert!(
        done["output"]["text"]
            .as_str()
            .unwrap()
            .contains("ANSWER=yes")
    );
    assert_eq!(
        timeout(Duration::from_secs(5), finished.recv())
            .await
            .unwrap()
            .unwrap(),
        JobFinished {
            owner: f.owner,
            job_id: job.job_id,
            state: JobState::Succeeded
        }
    );
    assert!(f.manager.background_jobs().is_empty());
}

/// Closing for an exit is refused while a job the host's user didn't agree
/// to stop runs in the background, which comes back. With every one known,
/// it closes: from then on no job starts and none moves to the background.
#[tokio::test]
async fn closing_needs_every_background_job_known() {
    let f = Fixture::new(|_| {});
    let sleep = "import time; time.sleep(30)";
    let background = f.python(sleep, Default::default());
    let held = f.python(
        sleep,
        StartOptions {
            held: true,
            ..Default::default()
        },
    );
    let refused = f.manager.close(&[]).unwrap_err();
    assert_eq!(
        refused.iter().map(|job| job.job_id).collect::<Vec<_>>(),
        [background.job_id]
    );
    // Still open: a job starts.
    let other = f.python(sleep, Default::default());
    assert!(f.manager.close(&[background.job_id]).is_err());

    f.manager.close(&[background.job_id, other.job_id]).unwrap();
    assert!(
        f.manager
            .start(&f.context(), &f.command(sleep), Default::default())
            .is_err()
    );
    assert!(matches!(
        f.manager.release(f.owner, held.job_id),
        Err(TerminalError::Invalid(_))
    ));
    assert!(f.manager.background_jobs().len() == 2);
    f.manager.shutdown().await;
}

/// A held job that finished stays held: its host has the result, and the
/// agent gets no update for it.
#[tokio::test]
async fn a_finished_held_job_stays_held() {
    let f = Fixture::new(|_| {});
    let (sender, mut receiver) = mpsc::channel(8);
    f.manager.bind_runtime(f.owner, sender.downgrade());
    let mut finished = f.manager.subscribe_finished();
    let job = f.python(
        "print('done',flush=True)",
        StartOptions {
            terminal: false,
            held: true,
            ..Default::default()
        },
    );
    assert_eq!(f.finished(job.job_id).await.state, JobState::Succeeded);
    assert!(matches!(
        f.manager.release(f.owner, job.job_id),
        Err(TerminalError::Finished)
    ));
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(receiver.try_recv().is_err());
    assert!(finished.try_recv().is_err());
    assert!(listed(&f).await.is_empty());
    // Its host forgets it as it does any job.
    f.manager.forget(f.owner, job.job_id).unwrap();
}

/// An update reaches the owner once: the runtime of its next turn (a new
/// command channel) isn't sent it again.
#[tokio::test]
async fn an_update_sent_is_not_sent_again_to_the_next_runtime() {
    let f = Fixture::new(|_| {});
    let (first, mut first_receiver) = mpsc::channel(8);
    f.manager.bind_runtime(f.owner, first.downgrade());
    let job = f.python("print('done',flush=True)", Default::default());
    assert_eq!(f.finished(job.job_id).await.state, JobState::Succeeded);
    assert_eq!(update(&mut first_receiver).await["state"], "succeeded");
    drop((first, first_receiver));
    let (next, mut next_receiver) = mpsc::channel(8);
    f.manager.bind_runtime(f.owner, next.downgrade());
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(next_receiver.try_recv().is_err());

    // One that found no runtime goes to the next one bound.
    drop((next, next_receiver));
    let job = f.python("print('later',flush=True)", Default::default());
    assert_eq!(f.finished(job.job_id).await.state, JobState::Succeeded);
    let (last, mut last_receiver) = mpsc::channel(8);
    f.manager.bind_runtime(f.owner, last.downgrade());
    let done = update(&mut last_receiver).await;
    assert_eq!(done["job_id"], job.job_id.to_string());
    assert!(last_receiver.try_recv().is_err());
}

/// A job can start with its own shell arguments.
#[cfg(unix)]
#[tokio::test]
async fn a_job_can_have_its_own_shell_arguments() {
    let f = Fixture::new(|_| {});
    let job = f
        .manager
        .start(
            &f.context(),
            "hello",
            StartOptions {
                terminal: false,
                shell_args: Some(vec!["-c".into(), "echo \"own $0\"".into()]),
                ..Default::default()
            },
        )
        .unwrap();
    let done = f.finished(job.job_id).await;
    assert_eq!(done.output.text.trim(), "own hello");
}

/// The note a runtime got, as its text.
pub(super) async fn note(receiver: &mut mpsc::Receiver<UserCommand>) -> String {
    let UserCommand::InjectContext { content } = timeout(Duration::from_secs(10), receiver.recv())
        .await
        .expect("a note in time")
        .unwrap()
    else {
        panic!("not a note")
    };
    content
}

/// A moved call whose end `gate` decides, with `output`.
fn gated(
    gate: tokio::sync::oneshot::Receiver<()>,
    output: CallOutput,
) -> futures::future::BoxFuture<'static, CallOutput> {
    Box::pin(async move {
        let _ = gate.await;
        output
    })
}

/// A moved call runs as a job. It is listed with its call, takes no input,
/// and when it ends its owner gets a note with the result, which its log and
/// status also hold.
#[tokio::test]
async fn a_moved_call_runs_as_a_job_and_its_result_goes_to_its_owner() {
    let f = Fixture::new(|_| {});
    let (sender, mut receiver) = mpsc::channel(8);
    f.manager.bind_runtime(f.owner, sender.downgrade());
    let mut finished = f.manager.subscribe_finished();
    let call_id = Uuid::new_v4();
    let (release, gate) = tokio::sync::oneshot::channel();
    let job = f
        .manager
        .adopt_call(
            f.owner,
            call_id,
            "index_create",
            "index_create".into(),
            crate::stop::StopSource::new(),
            gated(gate, CallOutput::text("Indexed 120 files.")),
        )
        .unwrap();
    assert_eq!(job.state, JobState::Running);
    let jobs = f.manager.jobs();
    assert_eq!(jobs.len(), 1);
    assert_eq!(
        (jobs[0].owner, jobs[0].job_id, jobs[0].tool_call_id),
        (f.owner, job.job_id, Some(call_id))
    );
    assert_eq!(
        jobs[0].kind,
        JobKind::Call {
            tool_name: "index_create".into()
        }
    );
    assert_eq!(jobs[0].label, "index_create");
    assert_eq!(f.manager.background_jobs()[0].command, "index_create");
    assert_eq!(listed(&f).await, [job.job_id]);
    let input = f
        .manager
        .input(f.owner, job.job_id, job.revision, "y".into());
    assert!(matches!(input.await, Err(TerminalError::Invalid(_))));

    release.send(()).unwrap();
    let note = note(&mut receiver).await;
    assert_eq!(
        note,
        format!(
            "Job {} ended. It ran your index_create call, which moved to the background less than a minute ago. Its result follows.\n\nIndexed 120 files.",
            job.job_id
        )
    );
    // Sent, so the manager doesn't keep it.
    let kept = f.manager.jobs.lock().unwrap()[&job.job_id]
        .data
        .lock()
        .unwrap()
        .note
        .is_some();
    assert!(!kept);
    let done = f.manager.status(f.owner, job.job_id, None).unwrap();
    assert_eq!(done.state, JobState::Succeeded);
    assert_eq!(done.output.text, "Indexed 120 files.");
    assert_eq!(
        std::fs::read_to_string(&done.log_path).unwrap(),
        "Indexed 120 files."
    );
    assert_eq!(
        timeout(Duration::from_secs(5), finished.recv())
            .await
            .unwrap()
            .unwrap(),
        JobFinished {
            owner: f.owner,
            job_id: job.job_id,
            state: JobState::Succeeded
        }
    );
    assert!(f.manager.jobs().is_empty());
    assert!(!f.manager.has_news(f.owner));
    f.manager.forget(f.owner, job.job_id).unwrap();
}

/// A note waits for its owner's runtime (`has_news` says so meanwhile). An
/// error says it is one, a long result is cut with the log holding all of
/// it, and images are saved in the job's directory and named.
#[tokio::test]
async fn a_moved_calls_note_carries_errors_long_results_and_images() {
    use base64::Engine as _;
    let f = Fixture::new(|_| {});
    let adopt = |name: &str, output: CallOutput| {
        f.manager
            .adopt_call(
                f.owner,
                Uuid::new_v4(),
                name,
                name.into(),
                crate::stop::StopSource::new(),
                Box::pin(async move { output }),
            )
            .unwrap()
    };
    let failed = adopt("search", CallOutput::error("The server went away."));
    let long = "a".repeat(20_000) + &"b".repeat(20_000) + &"c".repeat(20_000);
    let cut = adopt("dump", CallOutput::text(long.clone()));
    let png = base64::engine::general_purpose::STANDARD.encode(b"\x89PNG first");
    let pictures = adopt(
        "render",
        CallOutput::text("Rendered.").with_images(vec![
            crate::tool::ToolImage::base64("one", png.clone(), "image/png"),
            crate::tool::ToolImage::base64("two", png, "image/jpeg"),
        ]),
    );
    for job in [&failed, &cut, &pictures] {
        f.finished(job.job_id).await;
    }
    assert!(f.manager.has_news(f.owner));
    let (sender, mut receiver) = mpsc::channel(8);
    f.manager.bind_runtime(f.owner, sender.downgrade());
    let mut notes = HashMap::new();
    for _ in 0..3 {
        let note = note(&mut receiver).await;
        let id = note
            .split_whitespace()
            .nth(1)
            .unwrap()
            .trim_end_matches('.')
            .parse::<Uuid>()
            .unwrap();
        notes.insert(id, note);
    }
    assert!(!f.manager.has_news(f.owner));

    assert_eq!(
        f.manager
            .status(f.owner, failed.job_id, None)
            .unwrap()
            .state,
        JobState::Failed
    );
    assert!(notes[&failed.job_id].starts_with(&format!(
        "Job {} ended with an error. It ran your search call, which moved to the background",
        failed.job_id
    )));
    assert!(notes[&failed.job_id].ends_with("The error follows.\n\nThe server went away."));

    let note = &notes[&cut.job_id];
    let log = f
        .manager
        .status(f.owner, cut.job_id, None)
        .unwrap()
        .log_path;
    assert_eq!(std::fs::read_to_string(&log).unwrap(), long);
    let half = CALL_NOTE_CHARS / 2;
    let body = note.split_once("\n\n").unwrap().1;
    assert!(body.starts_with(&"a".repeat(half)), "the first part");
    assert!(body.ends_with(&"c".repeat(half)), "the last part");
    assert!(
        body.contains(&format!(
            "[{} characters in the middle are left out. {} holds all of it.]",
            long.len() - CALL_NOTE_CHARS,
            log.display()
        )),
        "{}",
        &body[half - 10..half + 200]
    );

    let directory = f
        .manager
        .status(f.owner, pictures.job_id, None)
        .unwrap()
        .log_path
        .parent()
        .unwrap()
        .to_path_buf();
    let (first, second) = (directory.join("image-1.png"), directory.join("image-2.jpg"));
    assert_eq!(std::fs::read(&first).unwrap(), b"\x89PNG first");
    assert!(second.is_file());
    assert!(notes[&pictures.job_id].ends_with(&format!(
        "Its result follows.\n\nRendered.\n\nIt returned 2 images, which are at {} and {}.",
        first.display(),
        second.display()
    )));
}

/// A stopped call's stop signal fires. A call that returns within the grace
/// gives what it returned, and one that doesn't is dropped after it. Either
/// way the job ends cancelled, and the note says the owner stopped it.
#[tokio::test(start_paused = true)]
async fn a_stopped_call_gets_its_grace_and_is_dropped_after_it() {
    let f = Fixture::new(|_| {});
    let (sender, mut receiver) = mpsc::channel(8);
    f.manager.bind_runtime(f.owner, sender.downgrade());
    let stop = crate::stop::StopSource::new();
    let signal = stop.signal();
    let winds_down = f
        .manager
        .adopt_call(
            f.owner,
            Uuid::new_v4(),
            "embed",
            "embed".into(),
            stop,
            Box::pin(async move {
                signal.stopping().await;
                CallOutput::text(format!("Saved after: {}", signal.reason().unwrap()))
            }),
        )
        .unwrap();
    let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    struct Dropped(Arc<std::sync::atomic::AtomicBool>);
    impl Drop for Dropped {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }
    let guard = Dropped(dropped.clone());
    let ignores = f
        .manager
        .adopt_call(
            f.owner,
            Uuid::new_v4(),
            "fetch",
            "fetch".into(),
            crate::stop::StopSource::new(),
            Box::pin(async move {
                let _guard = guard;
                std::future::pending::<CallOutput>().await
            }),
        )
        .unwrap();
    f.manager.cancel(f.owner, winds_down.job_id).unwrap();
    assert_eq!(
        note(&mut receiver).await,
        format!(
            "Job {} (your embed call) was stopped before it ended, as you asked. It returned this as it stopped.\n\nSaved after: The job was stopped.",
            winds_down.job_id
        )
    );
    let started = tokio::time::Instant::now();
    f.manager.cancel(f.owner, ignores.job_id).unwrap();
    assert_eq!(
        note(&mut receiver).await,
        format!(
            "Job {} (your fetch call) was stopped before it ended, as you asked, so that call has no result.",
            ignores.job_id
        )
    );
    assert!(started.elapsed() >= calls::CALL_STOP_GRACE);
    assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
    for job in [winds_down.job_id, ignores.job_id] {
        assert_eq!(
            f.manager.status(f.owner, job, None).unwrap().state,
            JobState::Cancelled
        );
    }
}

/// The user's stop of a moved call or a command isn't reported as a
/// finished job, so it wakes no one, and its update says the user stopped
/// it.
#[tokio::test]
async fn the_users_stop_wakes_no_one() {
    let f = Fixture::new(|_| {});
    let (sender, mut receiver) = mpsc::channel(8);
    f.manager.bind_runtime(f.owner, sender.downgrade());
    let mut finished = f.manager.subscribe_finished();
    let call = f
        .manager
        .adopt_call(
            f.owner,
            Uuid::new_v4(),
            "search",
            "search".into(),
            crate::stop::StopSource::new(),
            Box::pin(std::future::pending()),
        )
        .unwrap();
    let command = f.python("import time; time.sleep(30)", Default::default());
    f.manager.stop_for_user(f.owner, call.job_id).unwrap();
    assert_eq!(
        note(&mut receiver).await,
        format!(
            "The user stopped job {} (your search call) before it ended, so that call has no result.",
            call.job_id
        )
    );
    f.manager.stop_for_user(f.owner, command.job_id).unwrap();
    let stopped = update(&mut receiver).await;
    assert_eq!(stopped["state"], "cancelled");
    assert_eq!(stopped["error"], "stopped by the user");
    assert!(matches!(
        finished.try_recv(),
        Err(tokio::sync::broadcast::error::TryRecvError::Empty)
    ));
    assert!(matches!(
        f.manager.stop_for_user(f.owner, command.job_id),
        Err(TerminalError::Finished)
    ));
}

/// When its owner's run ends, its moved calls stop and its commands go on.
#[tokio::test]
async fn an_owners_end_stops_its_moved_calls_but_not_its_commands() {
    let f = Fixture::new(|_| {});
    let mut finished = f.manager.subscribe_finished();
    let call = f
        .manager
        .adopt_call(
            f.owner,
            Uuid::new_v4(),
            "search",
            "search".into(),
            crate::stop::StopSource::new(),
            Box::pin(std::future::pending()),
        )
        .unwrap();
    let command = f.python("import time; time.sleep(30)", Default::default());
    assert_eq!(f.manager.stop_owner_watchers(Uuid::new_v4()), 0);
    assert_eq!(f.manager.stop_owner_watchers(f.owner), 1);
    let stopped = f.finished(call.job_id).await;
    assert_eq!(stopped.state, JobState::Cancelled);
    assert_eq!(
        stopped.error.as_deref(),
        Some("stopped because its agent's run ended")
    );
    let running = f.manager.jobs();
    assert_eq!(running.len(), 1);
    assert_eq!(
        (running[0].job_id, &running[0].kind),
        (command.job_id, &JobKind::Command)
    );
    assert!(finished.try_recv().is_err());
    f.manager.shutdown().await;
}

/// Jobs carry the call that started them.
#[tokio::test]
async fn a_jobs_call_and_label_are_listed() {
    let f = Fixture::new(|_| {});
    let call_id = Uuid::new_v4();
    let mut context = f.context();
    context.tool_call_id = Some(call_id);
    let command = f.command("import time; time.sleep(30)");
    let job = f
        .manager
        .start(&context, &command, Default::default())
        .unwrap();
    let jobs = f.manager.jobs();
    assert_eq!(jobs.len(), 1);
    assert_eq!(
        (jobs[0].job_id, jobs[0].tool_call_id, &jobs[0].kind),
        (job.job_id, Some(call_id), &JobKind::Command)
    );
    assert_eq!(jobs[0].label, command);
    f.manager.shutdown().await;
}

/// The jobs running in the background when the last process stopped are
/// listed once the directory is opened again, a moved call with its tool and
/// a command saved before jobs recorded their label. Finished ones aren't,
/// and neither is a held (foreground) command, though it's interrupted too.
#[tokio::test]
async fn the_jobs_a_restart_interrupted_are_listed() {
    let root = std::env::temp_dir().join(format!("arko-terminal-interrupted-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let owner = Uuid::new_v4();
    let call_id = Uuid::new_v4();
    let (call, old, done, held) = (
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
    );
    let record = |id: Uuid, state: &str, extra: serde_json::Value| {
        let mut record = json!({
            "job_id": id, "owner": owner, "revision": 1, "state": state, "input": null,
            "exit_code": null, "error": null, "terminal": false, "output_end": 0,
            "log_truncated": false,
        });
        record
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        std::fs::create_dir(root.join(id.to_string())).unwrap();
        std::fs::write(
            root.join(id.to_string()).join("job.json"),
            record.to_string(),
        )
        .unwrap();
    };
    record(
        call,
        "running",
        json!({"tool_call_id": call_id, "label": "index_create", "call": {"tool_name": "index_create"}}),
    );
    // As earlier versions saved it, without a call or a label.
    record(old, "running", json!({}));
    record(
        done,
        "succeeded",
        json!({"label": "search", "call": {"tool_name": "search"}}),
    );
    record(
        held,
        "running",
        json!({"held": true, "label": "cargo test"}),
    );

    let manager = TerminalManager::new(TerminalConfig::new(&root)).unwrap();
    let mut interrupted = manager.interrupted_at_open();
    interrupted.sort_by_key(|job| job.job_id != call);
    assert_eq!(interrupted.len(), 2, "{interrupted:?}");
    assert_eq!(
        (
            interrupted[0].job_id,
            &interrupted[0].kind,
            interrupted[0].tool_call_id,
            interrupted[0].label.as_str()
        ),
        (
            call,
            &JobKind::Call {
                tool_name: "index_create".into()
            },
            Some(call_id),
            "index_create"
        )
    );
    assert_eq!(
        (
            interrupted[1].job_id,
            &interrupted[1].kind,
            interrupted[1].tool_call_id,
            interrupted[1].label.as_str()
        ),
        (old, &JobKind::Command, None, "")
    );
    for job in [call, held] {
        assert_eq!(
            manager.status(owner, job, None).unwrap().state,
            JobState::Interrupted
        );
    }
    assert!(manager.jobs().is_empty());
    drop(manager);
    std::fs::remove_dir_all(root).unwrap();
}

/// An open saves again only the records it changes, so a directory of
/// thousands of ended jobs isn't rewritten, and a field a newer version added
/// to one stays.
#[tokio::test]
async fn an_open_leaves_an_ended_record_as_it_is() {
    let root = std::env::temp_dir().join(format!("arko-terminal-ended-{}", Uuid::new_v4()));
    let id = Uuid::new_v4();
    std::fs::create_dir_all(root.join(id.to_string())).unwrap();
    let record = format!(
        r#"{{"job_id":"{id}","owner":"{}","revision":4,"state":"succeeded","input":null,"exit_code":0,"error":null,"terminal":true,"output_end":0,"log_truncated":false,"added_later":1}}"#,
        Uuid::new_v4()
    );
    let path = root.join(id.to_string()).join("job.json");
    std::fs::write(&path, &record).unwrap();

    let manager = TerminalManager::new(TerminalConfig::new(&root)).unwrap();
    assert!(manager.interrupted_at_open().is_empty());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), record);
    drop(manager);
    std::fs::remove_dir_all(root).unwrap();
}

/// The background jobs a shutdown stops are listed when the directory is
/// opened again, by that open only. The shutdown waits for a moved call
/// that ignores its stop to have its record. A held job and a job that
/// ended before aren't listed.
#[tokio::test]
async fn the_jobs_a_shutdown_stopped_are_listed_at_the_next_open() {
    let f = Fixture::new(|_| {});
    // A manager of its own, so the test can drop it and open its directory
    // again.
    let directory = f.root.join("closed");
    let manager = TerminalManager::new(TerminalConfig::new(&directory)).unwrap();
    let start = |code: &str, held: bool| {
        manager
            .start(
                &f.context(),
                &f.command(code),
                StartOptions {
                    terminal: false,
                    detect_prompts: false,
                    held,
                    ..Default::default()
                },
            )
            .unwrap()
            .job_id
    };
    let command = start("time.sleep(60)", false);
    let held = start("time.sleep(60)", true);
    let done = start("print('done')", false);
    timeout(Duration::from_secs(10), async {
        while !manager
            .status(f.owner, done, None)
            .unwrap()
            .state
            .is_terminal()
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    let call_id = Uuid::new_v4();
    let call = manager
        .adopt_call(
            f.owner,
            call_id,
            "fetch",
            "fetch".into(),
            crate::stop::StopSource::new(),
            Box::pin(std::future::pending()),
        )
        .unwrap()
        .job_id;

    manager.shutdown().await;
    let state = |id: Uuid| -> serde_json::Value {
        let record: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.join(id.to_string()).join("job.json")).unwrap(),
        )
        .unwrap();
        record["state"].clone()
    };
    assert_eq!(state(call), "cancelled", "its record is saved");
    drop(manager);
    let reopened = TerminalManager::new(TerminalConfig::new(&directory)).unwrap();
    let mut listed = reopened
        .interrupted_at_open()
        .into_iter()
        .map(|job| (job.job_id, job.kind, job.tool_call_id))
        .collect::<Vec<_>>();
    listed.sort_by_key(|(id, ..)| *id != command);
    assert_eq!(
        listed,
        [
            (command, JobKind::Command, None),
            (
                call,
                JobKind::Call {
                    tool_name: "fetch".into()
                },
                Some(call_id)
            )
        ]
    );
    for job in [command, held, call] {
        assert_eq!(state(job), "cancelled");
    }
    drop(reopened);
    let again = TerminalManager::new(TerminalConfig::new(&directory)).unwrap();
    assert!(again.interrupted_at_open().is_empty(), "listed once");
}

/// The directory opens again as soon as a shutdown returns, also when a
/// job's task hadn't run to its end by then (here the clock jumps past the
/// shutdown's waits before a call that ignores its stop is dropped).
#[tokio::test(start_paused = true)]
async fn the_directory_opens_again_as_soon_as_a_shutdown_returns() {
    let f = Fixture::new(|_| {});
    let directory = f.root.join("closed");
    let manager = TerminalManager::new(TerminalConfig::new(&directory)).unwrap();
    manager
        .adopt_call(
            f.owner,
            Uuid::new_v4(),
            "fetch",
            "fetch".into(),
            crate::stop::StopSource::new(),
            Box::pin(std::future::pending()),
        )
        .unwrap();
    tokio::join!(
        manager.shutdown(),
        tokio::time::advance(Duration::from_secs(10))
    );
    drop(manager);
    TerminalManager::new(TerminalConfig::new(&directory)).unwrap();
}

/// A job whose task ends after the directory opened again (here a call that
/// ignores its stop) leaves the record the new manager wrote, so a third
/// open doesn't list the job as stopped again.
#[tokio::test(start_paused = true)]
async fn a_job_ending_after_a_reopen_leaves_the_new_managers_record() {
    let f = Fixture::new(|_| {});
    let directory = f.root.join("closed");
    let first = TerminalManager::new(TerminalConfig::new(&directory)).unwrap();
    let id = first
        .adopt_call(
            f.owner,
            Uuid::new_v4(),
            "fetch",
            "fetch".into(),
            crate::stop::StopSource::new(),
            Box::pin(std::future::pending()),
        )
        .unwrap()
        .job_id;
    let mut done = first.jobs.lock().unwrap()[&id].done.clone();
    tokio::join!(
        first.shutdown(),
        tokio::time::advance(Duration::from_secs(10))
    );
    assert!(!*done.borrow(), "the call ended before the shutdown did");
    drop(first);
    let second = TerminalManager::new(TerminalConfig::new(&directory)).unwrap();
    assert_eq!(second.interrupted_at_open().len(), 1);
    done.wait_for(|done| *done).await.unwrap();
    drop(second);
    let third = TerminalManager::new(TerminalConfig::new(&directory)).unwrap();
    assert!(third.interrupted_at_open().is_empty(), "listed once");
    assert_eq!(
        third.status(f.owner, id, None).unwrap().state,
        JobState::Interrupted
    );
}

/// Two writers of one job's record, such as a late task of a closed manager
/// and the manager that opened the directory next, never share a temporary
/// file, so no write fails and none is left behind.
#[tokio::test]
async fn records_written_at_once_all_succeed() {
    let f = Fixture::new(|_| {});
    let job = f.python("print('done')", StartOptions::default()).job_id;
    f.finished(job).await;
    let directory = f.root.join("jobs").join(job.to_string());
    let mut record: Record =
        serde_json::from_slice(&std::fs::read(directory.join("job.json")).unwrap()).unwrap();
    record.generation = f.manager.generation;
    let writers: Vec<_> = (0..4)
        .map(|_| {
            let (config, record) = (f.manager.config.clone(), record.clone());
            std::thread::spawn(move || (0..200).all(|_| persist(&config, &record).is_ok()))
        })
        .collect();
    for writer in writers {
        assert!(writer.join().unwrap(), "a write failed");
    }
    let mut files: Vec<_> = std::fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with("job.json"))
        .collect();
    files.sort();
    assert_eq!(files, ["job.json"]);
}

/// A write that fails partway, as on a full disk, leaves no temporary file.
#[cfg(target_os = "linux")]
#[test]
fn a_failed_write_of_a_record_leaves_no_temporary_file() {
    // Every write to /dev/full fails with "no space left on device".
    if !Path::new("/dev/full").exists() {
        return;
    }
    let directory = std::env::temp_dir().join(format!("arko-terminal-full-{}", Uuid::new_v4()));
    std::fs::create_dir(&directory).unwrap();
    let temporary = directory.join("job.json.0.tmp");
    std::os::unix::fs::symlink("/dev/full", &temporary).unwrap();
    assert!(replace_file(&temporary, &directory.join("job.json"), b"{}").is_err());
    assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 0);
    std::fs::remove_dir(&directory).unwrap();
}

/// Calls in the foreground are listed until they move or end, and move
/// once. A closed manager moves none.
#[tokio::test]
async fn foreground_calls_are_listed_until_they_move() {
    let f = Fixture::new(|_| {});
    let mut changes = f.manager.watch_foreground_calls();
    changes.borrow_and_update();
    let other = Uuid::new_v4();
    let (first, second, third) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
    let held_first = f
        .manager
        .hold_call(f.owner, first, "search", "search".into());
    let held_second = f
        .manager
        .hold_call(f.owner, second, "index_create", "index_create".into());
    let held_third = f.manager.hold_call(other, third, "search", "search".into());
    assert!(changes.has_changed().unwrap());
    let listed = f.manager.foreground_calls();
    assert_eq!(
        listed
            .iter()
            .map(|call| (call.owner, call.tool_call_id, call.tool_name.as_str()))
            .collect::<Vec<_>>(),
        [
            (f.owner, first, "search"),
            (f.owner, second, "index_create"),
            (other, third, "search")
        ]
    );
    assert_eq!(f.manager.move_to_background(f.owner, Some(second)), 1);
    assert!(held_second.is_moved() && !held_first.is_moved());
    timeout(Duration::from_secs(5), held_second.moved())
        .await
        .unwrap();
    assert_eq!(f.manager.move_to_background(f.owner, Some(second)), 0);
    assert_eq!(f.manager.move_to_background(f.owner, None), 1);
    assert!(held_first.is_moved() && !held_third.is_moved());
    assert_eq!(f.manager.foreground_calls().len(), 1);
    drop(held_third);
    assert!(f.manager.foreground_calls().is_empty());

    let held = f
        .manager
        .hold_call(f.owner, first, "search", "search".into());
    f.manager.close(&[]).unwrap();
    assert_eq!(f.manager.move_to_background(f.owner, None), 0);
    assert!(!held.is_moved());
    assert!(
        f.manager
            .adopt_call(
                f.owner,
                first,
                "search",
                "search".into(),
                crate::stop::StopSource::new(),
                Box::pin(std::future::pending()),
            )
            .is_err()
    );
}
