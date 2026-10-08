//! runctl: post a note to a workflow run from another process, stop one
//! agent's runs while the rest of the workflow runs on, or show the run's
//! status. Works with any arko workflow: RUN_DIR is its persistence
//! directory (`PersistenceConfig::new(RUN_DIR)`).
//!
//! Usage:
//!   cargo run -p arko_workflows --example runctl -- note RUN_DIR "Prefer the orbit method."
//!   cargo run -p arko_workflows --example runctl -- note RUN_DIR --file note.md --to searcher,strategist --once
//!   cargo run -p arko_workflows --example runctl -- stop RUN_DIR --agent worker-12 --reason "moving it to Opus 5.5"
//!   cargo run -p arko_workflows --example runctl -- stop RUN_DIR --run 5f0c…-… --wrap-up 120 --relaunch
//!   cargo run -p arko_workflows --example runctl -- status RUN_DIR
//!
//! `note` needs a workflow built with `WorkflowBuilder::control_folder()`: it
//! appends the note to `RUN_DIR/control/notes.jsonl` under the folder's own
//! lock, never the run's, so it works while the run is live and while it is
//! stopped. The workflow posts the note at its agents' next turn boundaries,
//! or when it resumes, as `Workflow::post_note` does; a note naming an
//! unknown agent is skipped there with a warning in its log. `stop` appends
//! a request to `RUN_DIR/control/stops.jsonl` the same way; the running
//! workflow reads it within a second and stops the runs it names as
//! `Workflow::stop_run` does: each is told to save its work and finish
//! within the wrap-up (600 seconds unless `--wrap-up` or `--now`), then
//! cancelled. The runs are retired unless `--relaunch`. A request written
//! while the workflow isn't running is never applied. `status` reads the
//! run's latest checkpoint; it needs no control folder.
//!
//! Makes no model requests and needs no credentials.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use arko_workflows::RunStopIntent;
use arko_workflows::notes::{Note, NoteAudience, NoteKind, NoteSource, control};
use arko_workflows::persistence::PersistenceManager;
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(about = "Post a note to an arko workflow run, stop agent runs, or show its status")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Post a note to the run's agents (the workflow must read a control folder).
    Note {
        /// The run directory: the workflow's persistence directory.
        run_dir: PathBuf,
        /// The note's text.
        #[arg(required_unless_present = "file", conflicts_with = "file")]
        text: Option<String>,
        /// Read the note's text from this file.
        #[arg(long)]
        file: Option<PathBuf>,
        /// Only these agents (registration names, comma-separated).
        #[arg(long, value_delimiter = ',', conflicts_with = "except")]
        to: Vec<String>,
        /// Every agent except these.
        #[arg(long, value_delimiter = ',')]
        except: Vec<String>,
        /// A one-time note: delivered once to the runs that have started,
        /// not shown to later runs.
        #[arg(long)]
        once: bool,
        /// Label the note as this agent's (an observer, say) instead of the
        /// operator's.
        #[arg(long, conflicts_with = "from_host")]
        from_agent: Option<String>,
        /// Label the note as the workflow's own.
        #[arg(long)]
        from_host: bool,
    },
    /// Stop an agent's running runs (or one run) while the rest of the
    /// workflow runs on (the workflow must read a control folder).
    Stop {
        /// The run directory: the workflow's persistence directory.
        run_dir: PathBuf,
        /// Every running run of this agent (its registration name).
        #[arg(long, required_unless_present = "run", conflicts_with = "run")]
        agent: Option<String>,
        /// One agent run, by its ID.
        #[arg(long)]
        run: Option<uuid::Uuid>,
        /// Seconds the runs get to save their work and finish before they
        /// are cancelled.
        #[arg(long, default_value_t = 600, conflicts_with = "now")]
        wrap_up: u64,
        /// Cancel the runs at once, without a wrap-up.
        #[arg(long)]
        now: bool,
        /// Why, as the runs are told.
        #[arg(long, default_value = "")]
        reason: String,
        /// Keep the runs to restart later (`Workflow::stopped_runs`)
        /// instead of retiring them.
        #[arg(long)]
        relaunch: bool,
    },
    /// Show the run's status from its latest checkpoint.
    Status {
        /// The run directory: the workflow's persistence directory.
        run_dir: PathBuf,
    },
}

#[tokio::main]
async fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Note {
            run_dir,
            text,
            file,
            to,
            except,
            once,
            from_agent,
            from_host,
        } => {
            let text = match (text, file) {
                (Some(text), _) => Ok(text),
                (None, Some(file)) => std::fs::read_to_string(&file)
                    .map_err(|error| format!("{}: {error}", file.display())),
                (None, None) => Err("give the note's text, or --file PATH".to_string()),
            };
            text.and_then(|text| {
                let mut note = Note::new(text);
                if !to.is_empty() {
                    note = note.to(to);
                } else if !except.is_empty() {
                    note = note.except(except);
                }
                if once {
                    note = note.once();
                }
                if let Some(agent) = from_agent {
                    note = note.source(NoteSource::Agent(agent));
                } else if from_host {
                    note = note.source(NoteSource::Host);
                }
                post(&run_dir, &note)
            })
        }
        Command::Stop {
            run_dir,
            agent,
            run,
            wrap_up,
            now,
            reason,
            relaunch,
        } => {
            let mut stop = match (run, agent) {
                (Some(run), _) => control::ControlStop::run(run, reason),
                (None, Some(agent)) => control::ControlStop::agent(agent, reason),
                (None, None) => unreachable!("clap requires --agent or --run"),
            };
            if !now {
                stop = stop.wrap_up(std::time::Duration::from_secs(wrap_up));
            }
            if relaunch {
                stop = stop.intent(RunStopIntent::Relaunch);
            }
            request_stop(&run_dir, &stop)
        }
        Command::Status { run_dir } => status(&run_dir).await,
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("runctl: {error}");
            ExitCode::FAILURE
        }
    }
}

fn post(run_dir: &Path, note: &Note) -> Result<(), String> {
    let written = control::post_note(run_dir, note).map_err(|error| error.to_string())?;
    println!(
        "Note {} written to {}. The workflow posts it at its agents' next turn boundaries, or when it resumes.",
        written.id.map(|id| id.to_string()).unwrap_or_default(),
        control::folder(run_dir).join(control::NOTES_FILE).display()
    );
    Ok(())
}

fn request_stop(run_dir: &Path, stop: &control::ControlStop) -> Result<(), String> {
    let written = control::request_stop(run_dir, stop).map_err(|error| error.to_string())?;
    println!(
        "Stop request {} written to {}. The running workflow reads it within a second; a request written while it isn't running is never applied.",
        written.id.map(|id| id.to_string()).unwrap_or_default(),
        control::folder(run_dir).join(control::STOPS_FILE).display()
    );
    Ok(())
}

async fn status(run_dir: &Path) -> Result<(), String> {
    let pointer = std::fs::read_to_string(run_dir.join("latest-checkpoint")).map_err(|error| {
        format!(
            "{}: no checkpoint to read ({error})",
            run_dir.join("latest-checkpoint").display()
        )
    })?;
    let path = run_dir.join(pointer.trim());
    let state = PersistenceManager::load_checkpoint(&path)
        .await
        .map_err(|error| format!("{}: {error}", path.display()))?;

    println!(
        "Workflow '{}', run {}",
        state.workflow_name, state.workflow_run_id
    );
    println!(
        "Checkpoint {} ({})",
        path.display(),
        state.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
    );
    if let Some(cancellation) = &state.cancellation {
        println!(
            "Stopped at {}: {}",
            cancellation.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
            cancellation.cause.agent_reason()
        );
    }
    let limit = state
        .budget_limits
        .as_ref()
        .and_then(|limits| limits.limit)
        .map_or_else(String::new, |limit| format!(" of ${limit:.2}"));
    println!("Spent ${:.4}{limit}", state.budget_spent);

    let mut runs = state
        .agent_results
        .iter()
        .map(|result| {
            (
                result.agent_name.clone(),
                result.agent_run_id,
                result.status.clone(),
                result.error.clone(),
                result.stopped_by.as_ref().and_then(|stop| stop.intent),
            )
        })
        .chain(state.agent_runs.iter().map(|record| {
            (
                record.agent_name.clone(),
                record.agent_run_id,
                record.status.clone(),
                record.error.clone(),
                record.stopped_by.as_ref().and_then(|stop| stop.intent),
            )
        }))
        .collect::<Vec<_>>();
    runs.sort_by(|a, b| a.0.cmp(&b.0));
    println!("Agent runs: {}", runs.len());
    for (agent, id, status, error, intent) in &runs {
        let intent = match intent {
            Some(RunStopIntent::Retire) => " (stopped alone: retired)",
            Some(RunStopIntent::Relaunch) => " (stopped alone: to relaunch)",
            _ => "",
        };
        let error = error
            .as_deref()
            .and_then(|error| error.lines().next())
            .map_or_else(String::new, |line| format!(": {}", cut(line, 120)));
        println!("  {agent} ({id}): {status:?}{intent}{error}");
    }

    let posted = state
        .notes
        .as_ref()
        .map(|notes| notes.notes.clone())
        .unwrap_or_default();
    println!("Notes posted: {}", posted.len());
    for posted in posted.iter().rev().take(5) {
        let audience = match &posted.note.audience {
            NoteAudience::All => "every agent".to_string(),
            NoteAudience::Only(agents) => agents.iter().cloned().collect::<Vec<_>>().join(", "),
            NoteAudience::Except(agents) => format!(
                "all but {}",
                agents.iter().cloned().collect::<Vec<_>>().join(", ")
            ),
        };
        let source = match &posted.note.source {
            NoteSource::Operator => "operator".to_string(),
            NoteSource::Host => "host".to_string(),
            NoteSource::Agent(name) => format!("agent {name}"),
        };
        let kind = match posted.note.kind {
            NoteKind::Standing => "standing",
            NoteKind::Once => "once",
        };
        println!(
            "  #{} {} from {source} to {audience} ({kind}): {}",
            posted.seq,
            posted.posted_at.format("%Y-%m-%d %H:%M UTC"),
            cut(posted.note.text.lines().next().unwrap_or_default(), 100)
        );
    }
    if control::folder(run_dir).is_dir() {
        let written = control::read_notes(run_dir).map_err(|error| error.to_string())?;
        let waiting = written
            .iter()
            .filter(|entry| {
                entry
                    .id
                    .is_some_and(|id| !posted.iter().any(|posted| posted.id == id))
            })
            .count();
        println!(
            "Control folder: {} notes written, {waiting} not posted as of this checkpoint",
            written.len()
        );
    } else {
        println!("Control folder: none (the workflow doesn't read one)");
    }
    Ok(())
}

/// `text` cut to at most `max` bytes, at a character boundary.
fn cut(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &text[..end])
}
