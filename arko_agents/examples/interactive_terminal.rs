//! An offline example in which a test program asks Y/N. The host receives a
//! notice and submits a predetermined answer, without a model or credentials.
#[cfg(any(unix, windows))]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use arko_agents::terminal::{StartOptions, TerminalConfig, TerminalManager};
    use arko_agents::{TaskContext, UserCommand};
    use std::time::Duration;
    use tokio::sync::mpsc;
    use uuid::Uuid;

    let directory = std::env::temp_dir().join(format!("arko-terminal-demo-{}", Uuid::new_v4()));
    let manager = TerminalManager::new(TerminalConfig::new(&directory))?;
    let owner = Uuid::new_v4();
    let cwd = std::env::current_dir()?;
    let context = TaskContext::new(owner, "demo", "Exercise interactive input", &cwd);
    let (sender, mut notices) = mpsc::channel(8);
    manager.bind_runtime(owner, sender.downgrade());
    let fixture = directory.join("question.py");
    std::fs::write(
        &fixture,
        "print('Continue? [Y/n] ', end='', flush=True); print('Received: ' + input())",
    )?;
    let python = std::env::var("ARKO_TEST_PYTHON")
        .unwrap_or_else(|_| if cfg!(windows) { "python" } else { "python3" }.into());
    let command = if cfg!(windows) {
        format!("\"{python}\" \"{}\"", fixture.display())
    } else {
        format!(
            "exec '{}' '{}'",
            python.replace('\'', "'\\''"),
            fixture.display().to_string().replace('\'', "'\\''")
        )
    };
    let job = manager.start(
        &context,
        &command,
        StartOptions {
            // Generous: this runs as a CI smoke test, and loaded Windows runners
            // can take several seconds just to start Python under ConPTY.
            timeout: Some(Duration::from_secs(60)),
            ..Default::default()
        },
    )?;
    println!("Started {}", job.job_id);
    loop {
        let notice = tokio::time::timeout(Duration::from_secs(65), notices.recv())
            .await?
            .ok_or("context channel closed")?;
        if let UserCommand::InjectContext { content } = notice {
            println!("{content}");
        }
        let status = manager.status(owner, job.job_id, None)?;
        if status.state.is_terminal() {
            if status.state != arko_agents::terminal::JobState::Succeeded {
                return Err(format!("fixture failed: {status:?}").into());
            }
            break;
        }
        if status.input.is_some() {
            manager
                .input(
                    owner,
                    job.job_id,
                    status.revision,
                    if cfg!(windows) { "Y\r" } else { "Y\n" }.into(),
                )
                .await?;
        }
    }
    manager.shutdown().await;
    println!("Logs: {}", directory.display());
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn main() {
    eprintln!("The interactive terminal module requires Windows or Unix.");
}
