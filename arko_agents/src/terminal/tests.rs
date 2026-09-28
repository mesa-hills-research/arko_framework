use super::*;
use crate::tool::Tool;
use serde_json::json;
use tokio::time::timeout;

struct Fixture {
    root: PathBuf,
    manager: Arc<TerminalManager>,
    owner: Uuid,
}
impl Fixture {
    fn new(configure: impl FnOnce(&mut TerminalConfig)) -> Self {
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
    fn context(&self) -> TaskContext<'_> {
        TaskContext {
            task_id: self.owner,
            agent_name: "tester",
            initial_task_prompt: "test interactive jobs",
            current_working_directory: &self.root,
        }
    }
    fn command(&self, code: &str) -> String {
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
    async fn until(
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
    async fn finished(&self, id: Uuid) -> TerminalSnapshot {
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
        let job = f.python("fd=os.open(os.environ['ARKO_INPUT_REQUEST_PATH'],os.O_WRONLY|os.O_APPEND)\nreplies=Replies()\nfor n in range(2):\n request=json.dumps({'type':'input_request','request_id':str(n),'prompt':'Proceed?'})+'\\n'\n os.write(fd,request.encode()); os.write(fd,request.encode())\n value=json.loads(replies.readline()); assert value['request_id']==str(n); assert value['text']=='Y\\n'\nprint('ANSWERS_RECEIVED',flush=True)", StartOptions { terminal, detect_prompts:false, timeout:None });
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
        timeout: None,
    };
    let others: Vec<_> = (0..65)
        .map(|_| {
            f.manager
                .start(&f.context(), "exit 0", pipes)
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
    let job = f.python("fd=os.open(os.environ['ARKO_INPUT_REQUEST_PATH'],os.O_WRONLY|os.O_APPEND)\nframe=(json.dumps({'type':'input_request','request_id':'split','prompt':'Café?'},ensure_ascii=False)+'\\n').encode()\nfor byte in frame: os.write(fd,bytes([byte])); time.sleep(.001)\nvalue=json.loads(Replies().readline()); assert value['request_id']=='split'", StartOptions { terminal:false, detect_prompts:false, timeout:None });
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
