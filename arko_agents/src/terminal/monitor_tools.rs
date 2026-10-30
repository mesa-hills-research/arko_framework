//! `monitor_start` and `monitor_control`, the tools an agent watches
//! commands with.

use super::monitor::{CheckChange, MonitorLifetime, clock, minutes_left};
use super::tools::{number, parameter, string, tool_error};
use super::*;
use crate::tool::{Tool, ToolDefinition, ToolError, ToolParameterType, ToolTypeDetails};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::time::SystemTime;

#[derive(Debug, Clone, Copy)]
enum Operation {
    Start,
    Control,
}

/// The tools `monitor_start` and `monitor_control`, which start an agent's
/// monitors and list, rearm and stop them. See
/// [`TerminalManager::start_monitor`].
/// [`TerminalManager::register_tools`] registers them with the terminal
/// tools. Add their names to an agent's allowlist to offer them.
#[derive(Debug)]
pub struct MonitorTool {
    manager: Arc<TerminalManager>,
    operation: Operation,
}

impl MonitorTool {
    pub const NAMES: [&'static str; 2] = ["monitor_start", "monitor_control"];

    /// Both tools, on `manager`.
    pub fn all(manager: Arc<TerminalManager>) -> Vec<Self> {
        [Operation::Start, Operation::Control]
            .into_iter()
            .map(|operation| Self {
                manager: manager.clone(),
                operation,
            })
            .collect()
    }
}

/// A whole number of minutes, from 1 to `most`.
fn minutes(
    params: &HashMap<String, Value>,
    name: &str,
    most: u64,
) -> std::result::Result<Option<Duration>, ToolError> {
    match number(params, name)? {
        Some(count) if !(1..=most).contains(&count) => Err(ToolError::ParameterValidation {
            param_name: name.into(),
            message: format!("{name} is from 1 to {most}"),
        }),
        count => Ok(count.map(|count| Duration::from_secs(60 * count))),
    }
}

fn optional_string<'a>(
    params: &'a HashMap<String, Value>,
    name: &str,
) -> std::result::Result<Option<&'a str>, ToolError> {
    match params.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(_) => string(params, name).map(Some),
    }
}

#[async_trait]
impl Tool for MonitorTool {
    fn definition(&self) -> ToolDefinition {
        let limits = self.manager.monitor_limits();
        let longest = limits.longest.as_secs() / 60;
        let interval = match limits.alert_interval.as_secs() {
            60 => "once a minute".to_string(),
            _ => format!("once each {}", calls::describe(limits.alert_interval)),
        };
        // Monitors can last while workers run only where the host says
        // whether they do.
        let workers = self.manager.has_worker_liveness();
        let (name, description, mut parameters) = match self.operation {
            Operation::Start => (
                Self::NAMES[0],
                format!(
                    "Watches a command for you and alerts you while the monitor lasts, so you don't have to poll. Without every_seconds, the command runs until it exits or the monitor ends (a dev server, a build, `tail -F` on a log), and each output line that matches pattern is an alert. With every_seconds, the command is a check that runs again and again (`curl -sf localhost:3000/health`), and you're alerted when its result changes, not while it stays the same. Alerts reach you as notes while you work, and start a turn when you're idle. To save turns they come at most {interval}, with the lines since the last one. A monitor lasts expires_minutes ({longest} at most). Before it ends you're asked whether to keep it, and monitor_control rearm keeps it longer.{} terminal_status reads its whole output.",
                    if workers {
                        " For an event that happens once, such as a CI run finishing, a file appearing or a server starting to answer, set once, and the monitor ends after its first alert. For a watch you need while your workers run, such as disk space or memory during their builds or a dev server's errors, set lasts to while_workers_run. It then needs no rearm, and ends after the last worker finishes."
                    } else {
                        " For an event that happens once, such as a CI run finishing, a file appearing or a server starting to answer, set once, and the monitor ends after its first alert."
                    }
                ),
                vec![
                    parameter(
                        "command",
                        "The command, for the configured shell (by default `/bin/sh -c` on Unix and `cmd.exe /D /S /C` on Windows). A check should finish within a few seconds.",
                        true,
                    ),
                    parameter(
                        "pattern",
                        "A regular expression. An output line that matches it is an alert. Leave it out to be alerted of every line. Write `(?i)` at its start to ignore case. Not for checks.",
                        false,
                    ),
                    parameter(
                        "every_seconds",
                        &format!(
                            "Makes the command a check that runs this often, from {} to {} seconds.",
                            limits.shortest_check.as_secs(),
                            limits.longest_check.as_secs()
                        ),
                        false,
                    ),
                    parameter(
                        "alert_when",
                        "For a check, which change alerts you. `success` when it starts succeeding, `failure` when it starts failing, `change` for either (the default), or `output` when its output changes.",
                        false,
                    ),
                    parameter(
                        "expires_minutes",
                        &if workers {
                            format!(
                                "How long the monitor lasts, from 1 to {longest} minutes ({longest} by default). With lasts, at least this long."
                            )
                        } else {
                            format!(
                                "How long the monitor lasts, from 1 to {longest} minutes ({longest} by default)."
                            )
                        },
                        false,
                    ),
                    parameter(
                        "once",
                        "true ends the monitor and its command after its first alert. For a check that alerts on success or failure, a first result that is already the one it waits for ends it at once.",
                        false,
                    ),
                ]
                .into_iter()
                .chain(workers.then(|| {
                    parameter(
                        "lasts",
                        "while_workers_run keeps the monitor going past expires_minutes while any of your workers run. It ends after the last one finishes, without a warning. Leave it out for a monitor that lasts expires_minutes.",
                        false,
                    )
                }))
                .collect(),
            ),
            Operation::Control => (
                Self::NAMES[1],
                format!(
                    "Lists your monitors (action list), keeps one going for longer (rearm), or stops one and its command (stop). rearm sets its end to minutes from now.{}",
                    if workers {
                        " A monitor that lasts while your workers run goes on past that while they do."
                    } else {
                        ""
                    }
                ),
                vec![
                    parameter("action", "list, rearm or stop.", true),
                    parameter("job_id", "The monitor's job ID (not for list).", false),
                    parameter(
                        "minutes",
                        &format!(
                            "For rearm. How long it lasts from now, from 1 to {longest} minutes ({longest} by default)."
                        ),
                        false,
                    ),
                ],
            ),
        };
        for param in &mut parameters {
            match param.name.as_str() {
                "every_seconds" | "expires_minutes" | "minutes" => {
                    param.param_type = ToolParameterType::Integer
                }
                "once" => param.param_type = ToolParameterType::Bool,
                _ => {}
            }
        }
        ToolDefinition {
            name: name.into(),
            description,
            parameters,
            output_based_example: None,
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            read_only: false,
            raw_json_schema_override: None,
        }
    }

    fn allow_result_interception(&self) -> bool {
        false
    }

    /// A start waits for a check's first run, at most
    /// [`first_check_wait`](MonitorLimits::first_check_wait).
    fn uses_host_timeout(&self) -> bool {
        false
    }

    fn bind_runtime(&self, task_id: Uuid, sender: mpsc::WeakSender<UserCommand>) {
        self.manager.bind_runtime(task_id, sender);
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> std::result::Result<String, ToolError> {
        let limits = self.manager.monitor_limits();
        let longest = limits.longest.as_secs() / 60;
        match self.operation {
            Operation::Start => {
                let command = string(&params, "command")?;
                let pattern = optional_string(&params, "pattern")?;
                let every = number(&params, "every_seconds")?;
                let when = optional_string(&params, "alert_when")?;
                let mut options = match (every, pattern) {
                    (Some(_), Some(_)) => {
                        return Err(ToolError::ParameterValidation {
                            param_name: "pattern".into(),
                            message: "pattern is for watching lines, and every_seconds makes the command a check. Give one or the other".into(),
                        });
                    }
                    (Some(every), None) => {
                        let when = match when {
                            None => CheckChange::Change,
                            Some(name) => CheckChange::from_name(name).ok_or_else(|| {
                                ToolError::ParameterValidation {
                                    param_name: "alert_when".into(),
                                    message: "alert_when is success, failure, change or output"
                                        .into(),
                                }
                            })?,
                        };
                        MonitorOptions::check(every, when)
                    }
                    (None, pattern) => {
                        if when.is_some() {
                            return Err(ToolError::ParameterValidation {
                                param_name: "alert_when".into(),
                                message: "alert_when is for a check, so give every_seconds too"
                                    .into(),
                            });
                        }
                        MonitorOptions::lines(pattern.map(str::to_string))
                    }
                };
                options = options.with_expiry(
                    minutes(&params, "expires_minutes", longest)?.unwrap_or(limits.longest),
                );
                if super::tools::boolean(&params, "once", false)? {
                    options = options.once();
                }
                if let Some(lasts) = optional_string(&params, "lasts")? {
                    options = options.with_lifetime(self.lifetime(lasts)?);
                }
                Ok(self
                    .manager
                    .start_monitor(context, command, options)
                    .await
                    .map_err(tool_error)?
                    .summary)
            }
            Operation::Control => {
                let action = string(&params, "action")?;
                if action == "list" {
                    return serde_json::to_string(&self.list(context.task_id)).map_err(tool_error);
                }
                let id = Uuid::parse_str(string(&params, "job_id")?).map_err(tool_error)?;
                match action {
                    "rearm" => {
                        let lasts = minutes(&params, "minutes", longest)?.unwrap_or(limits.longest);
                        let ends =
                            self.manager
                                .rearm(context.task_id, id, lasts)
                                .map_err(|error| match error {
                                    TerminalError::Finished => tool_error(format!(
                                        "monitor {id} has ended. monitor_start starts a new one"
                                    )),
                                    error => tool_error(error),
                                })?;
                        let (once, lifetime) = self
                            .manager
                            .owned(context.task_id, id)
                            .ok()
                            .and_then(|job| {
                                let data = job.data.lock().unwrap();
                                let monitor = data.record.monitor.as_ref()?;
                                Some((monitor.once, monitor.lifetime))
                            })
                            .unwrap_or_default();
                        Ok(format!(
                            "Monitor {id} now ends at {}, in {}{}.{}",
                            clock(ends),
                            minutes_left(
                                ends.duration_since(SystemTime::now()).unwrap_or_default()
                            ),
                            match lifetime {
                                MonitorLifetime::WhileWorkersRun => {
                                    ", or later while your workers run"
                                }
                                _ => "",
                            },
                            if once {
                                " Its first alert ends it sooner."
                            } else {
                                ""
                            }
                        ))
                    }
                    "stop" => {
                        let (monitor, ended) = {
                            let job = self
                                .manager
                                .owned(context.task_id, id)
                                .map_err(tool_error)?;
                            let data = job.data.lock().unwrap();
                            (
                                data.record.monitor.is_some(),
                                data.record.state.is_terminal(),
                            )
                        };
                        if !monitor {
                            return Err(tool_error(format!(
                                "job {id} isn't a monitor. terminal_control cancel stops other jobs"
                            )));
                        }
                        if ended {
                            return Err(tool_error(format!("monitor {id} has ended already")));
                        }
                        self.manager
                            .cancel(context.task_id, id)
                            .map_err(tool_error)?;
                        Ok(format!("Stopped monitor {id}."))
                    }
                    _ => Err(tool_error("action is list, rearm or stop")),
                }
            }
        }
    }
}

impl MonitorTool {
    /// The lifetime `lasts` names. `while_workers_run` (with spaces or
    /// hyphens too) is the only one, and only where the host says whether
    /// workers run.
    fn lifetime(&self, lasts: &str) -> std::result::Result<MonitorLifetime, ToolError> {
        let name = lasts.trim().to_ascii_lowercase().replace([' ', '-'], "_");
        if name != "while_workers_run" {
            return Err(ToolError::ParameterValidation {
                param_name: "lasts".into(),
                message: "lasts takes while_workers_run, or leave it out".into(),
            });
        }
        if !self.manager.has_worker_liveness() {
            return Err(ToolError::ParameterValidation {
                param_name: "lasts".into(),
                message: "you have no workers here, so leave out lasts".into(),
            });
        }
        Ok(MonitorLifetime::WhileWorkersRun)
    }

    /// `owner`'s running monitors, as `monitor_control list` gives them. A
    /// monitor that alerts once says so, and one that lasts while workers
    /// run says that too, and after its time, while they do, when it ends.
    fn list(&self, owner: Uuid) -> Vec<Value> {
        let now = SystemTime::now();
        let workers = self.manager.workers_running(owner);
        self.manager
            .jobs()
            .into_iter()
            .filter(|job| job.owner == owner)
            .filter_map(|job| match job.kind {
                JobKind::Monitor(info) => {
                    let mut entry = json!({
                        "job_id": job.job_id,
                        "command": job.label,
                        "watching": info.trigger.watching(),
                        "alerts": info.alerts,
                    });
                    let left = info.expires_at.duration_since(now).unwrap_or_default();
                    let while_workers = info.lifetime == MonitorLifetime::WhileWorkersRun;
                    // Past its time, by the clock its timing goes by.
                    let past = || {
                        self.manager.owned(owner, job.job_id).is_ok_and(|job| {
                            job.data
                                .lock()
                                .unwrap()
                                .monitor
                                .as_ref()
                                .is_some_and(|arming| arming.expires <= tokio::time::Instant::now())
                        })
                    };
                    if while_workers && workers && past() {
                        entry["ends"] = json!("after your last worker finishes");
                    } else {
                        entry["ends_in_minutes"] = json!((left.as_secs() + 30) / 60);
                    }
                    if while_workers {
                        entry["lasts"] = json!("while_workers_run");
                    }
                    if info.once {
                        entry["once"] = json!(true);
                    }
                    Some(entry)
                }
                _ => None,
            })
            .collect()
    }
}
