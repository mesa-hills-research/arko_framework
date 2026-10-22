//! The workflow's record of its background subagents, their mailboxes, and
//! the drivers that run them.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use parking_lot::Mutex;
use tokio::sync::{broadcast, watch};
use uuid::Uuid;

use arko_agents::{AgentConfig, Task};

use super::{
    MailItem, MailKind, MailSender, PendingLaunch, StartRequest, SubagentEvent, SubagentEventKind,
    SubagentRecord, SubagentState, SubagentsConfig, SubagentsSnapshot, UPDATE_MAX_MAIL_BYTES,
    WorkspaceRequest,
};
use crate::context::SharedContext;
use crate::error::WorkflowError;
use crate::persistence::PersistenceManager;
use crate::policy::{AgentPause, AgentResume};
use crate::workflow::engine::{
    Recovery, StartOptions, drive_to_result, is_superseded, recorded_result, reopened_message,
};
use crate::workflow::result::{AgentResult, AgentRunStatus};
use crate::workflow::stop::{RunStopIntent, StopKind, StopRecord, StopRequest, WrapUp};

/// How long a subagent's question waits for its lead's answer.
const QUESTION_TIMEOUT: Duration = Duration::from_secs(60 * 60);

/// How long a stopped subagent gets to wrap up before it is cancelled.
const STOP_WRAP_UP: Duration = Duration::from_secs(120);

/// Why a subagent is stopped when its lead's conversation is cut back to
/// before the message that started it.
const CUT_BACK_REASON: &str = "its lead's conversation was cut back to before it started";

/// What a subagent's run is told when it goes on because mail reached it
/// after it reported.
const MAIL_AFTER_REPORT: &str = "You have new mail since your report: it follows. Act on it, then call `complete_task` again with your updated report.";

/// Who a run may message, for `message_agent`.
enum Recipient {
    /// One of its own subagents.
    Subagent(SubagentRecord),
    /// Its lead (it is a subagent).
    Lead(Uuid),
    /// Another subagent of its lead, linked to it.
    Peer(SubagentRecord),
}

/// Who asked for the stop of a subagent, which decides what its lead is told.
#[derive(Clone)]
enum StopOrigin {
    /// The workflow's host.
    Host,
    /// The lead's `stop_agent`, with the reason it gave (maybe none).
    Caller(String),
    /// The subagent's own lead was stopped or ended, in these words (`its
    /// lead r1 was stopped`). A stop reaches the subagents of a run that
    /// stops, and the subagents of those.
    Lead(String),
}

pub(crate) struct Board {
    config: Option<SubagentsConfig>,
    state: Mutex<SubagentsSnapshot>,
    /// Bumped at every change, for the waits.
    version: watch::Sender<u64>,
    events: broadcast::Sender<SubagentEvent>,
    /// The tasks driving the subagents running now.
    drivers: Mutex<HashMap<Uuid, tokio::task::AbortHandle>>,
    /// Where events are logged and checkpoints saved.
    persistence: Option<PersistenceManager>,
    workflow_run_id: Uuid,
    /// Where long mailbox items are saved whole.
    mail_dir: PathBuf,
    /// The configurations the host's hook prepared for runs that start:
    /// the hook runs once per start.
    prepared: Mutex<HashMap<Uuid, AgentConfig>>,
    /// Stops received after launch selection and before run registration.
    starting: Mutex<HashMap<Uuid, Option<(StopRequest, std::time::Instant)>>>,
    /// Why the runs stopped because their lead stopped or ended were
    /// stopped, in words, until they end.
    lead_stops: Mutex<HashMap<Uuid, String>>,
}

impl Board {
    /// The board of a workflow run. `restored` from its checkpoint, with
    /// the results it recorded (`results`): records that were running are
    /// settled by the outcome their runs recorded.
    pub(crate) fn new(
        config: Option<SubagentsConfig>,
        restored: Option<SubagentsSnapshot>,
        results: &[AgentResult],
        workflow_run_id: Uuid,
        persistence: Option<PersistenceManager>,
    ) -> Self {
        let mail_dir = persistence.as_ref().map_or_else(
            || {
                std::env::temp_dir()
                    .join("arko-subagents")
                    .join(workflow_run_id.to_string())
            },
            |persistence| persistence.config().checkpoint_dir.join("subagents"),
        );
        let mut state = restored.unwrap_or_default();
        settle_restored(&mut state, results);
        Self {
            config,
            state: Mutex::new(state),
            version: watch::Sender::new(0),
            events: broadcast::channel(1024).0,
            drivers: Mutex::new(HashMap::new()),
            persistence,
            workflow_run_id,
            mail_dir,
            prepared: Mutex::new(HashMap::new()),
            starting: Mutex::new(HashMap::new()),
            lead_stops: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn config(&self) -> Option<&SubagentsConfig> {
        self.config.as_ref()
    }

    /// The state a checkpoint saves. `None` while there is nothing.
    pub(crate) fn snapshot(&self) -> Option<SubagentsSnapshot> {
        let state = self.state.lock();
        (*state != SubagentsSnapshot::default()).then(|| state.clone())
    }

    pub(crate) fn is_subagent(&self, run_id: Uuid) -> bool {
        self.state
            .lock()
            .runs
            .iter()
            .any(|run| run.run_id == run_id)
    }

    pub(crate) fn record(&self, run_id: Uuid) -> Option<SubagentRecord> {
        self.state
            .lock()
            .runs
            .iter()
            .find(|run| run.run_id == run_id)
            .cloned()
    }

    pub(crate) fn records(&self) -> Vec<SubagentRecord> {
        self.state.lock().runs.clone()
    }

    pub(crate) fn subscribe(&self) -> broadcast::Receiver<SubagentEvent> {
        self.events.subscribe()
    }

    /// The mailbox items for run `to` it hasn't received.
    pub(crate) fn pending(&self, to: Uuid) -> Vec<MailItem> {
        self.state
            .lock()
            .mail
            .iter()
            .filter(|item| item.to == to && item.delivered_at.is_none())
            .cloned()
            .collect()
    }

    /// Takes the mailbox items for run `to` it hasn't received out of the
    /// mailbox: the host delivers them its own way.
    pub(crate) fn take(&self, ctx: &Arc<SharedContext>, to: Uuid) -> Vec<MailItem> {
        let taken = {
            let mut state = self.state.lock();
            let now = Utc::now();
            let mut taken = Vec::new();
            state.mail.retain(|item| {
                if item.to == to && item.delivered_at.is_none() {
                    let mut item = item.clone();
                    item.delivered_at = Some(now);
                    taken.push(item);
                    false
                } else {
                    true
                }
            });
            taken
        };
        if !taken.is_empty() {
            self.delivered(to, &taken);
            self.save_soon(ctx);
        }
        taken
    }

    /// The subagents that run, or wait to run.
    pub(crate) fn active(&self) -> Vec<SubagentRecord> {
        self.state
            .lock()
            .runs
            .iter()
            .filter(|run| run.state.is_active())
            .cloned()
            .collect()
    }

    /// Run `from`'s subagents become run `to`'s, with the mail for `from`
    /// it hasn't received.
    pub(crate) fn transfer(&self, from: Uuid, to: Uuid) -> usize {
        let moved = {
            let mut state = self.state.lock();
            let mut moved = 0;
            for run in &mut state.runs {
                if run.caller_run_id == from {
                    run.caller_run_id = to;
                    moved += 1;
                }
            }
            for item in &mut state.mail {
                if item.to == from && item.delivered_at.is_none() {
                    item.to = to;
                }
            }
            moved
        };
        self.bump();
        moved
    }

    /// Aborts the drivers of the subagents running (the workflow is
    /// dropped).
    pub(crate) fn abort_all(&self) {
        for (_, driver) in self.drivers.lock().drain() {
            driver.abort();
        }
    }

    fn bump(&self) {
        self.version
            .send_modify(|version| *version = version.wrapping_add(1));
    }

    fn emit(&self, record: &SubagentRecord, kind: SubagentEventKind) {
        let event = SubagentEvent {
            timestamp: Utc::now(),
            workflow_run_id: self.workflow_run_id,
            run_id: record.run_id,
            handle: record.handle.clone(),
            agent_name: record.agent.clone(),
            caller_run_id: record.caller_run_id,
            caller_agent: record.caller_agent.clone(),
            kind,
        };
        if let Some(persistence) = &self.persistence {
            persistence.append_subagent_event(&event);
        }
        let _ = self.events.send(event);
    }

    /// Reports that `items` reached run `to`.
    fn delivered(&self, to: Uuid, items: &[MailItem]) {
        if items.is_empty() {
            return;
        }
        let about = {
            let state = self.state.lock();
            about_record(&state, items[0].from, to)
        };
        if let Some(record) = about {
            self.emit(
                &record,
                SubagentEventKind::Delivered {
                    to,
                    mail_ids: items.iter().map(|item| item.id).collect(),
                },
            );
        }
        self.bump();
    }

    /// Saves the change soon: with the live checkpoint while runs are live,
    /// else with a checkpoint.
    fn save_soon(&self, ctx: &Arc<SharedContext>) {
        let Some(persistence) = ctx.persistence.clone() else {
            return;
        };
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let ctx = ctx.clone();
        runtime.spawn(async move {
            let saved = if persistence.has_live_runs() {
                persistence.write_live_checkpoint(&ctx, false).await
            } else {
                persistence.auto_checkpoint(&ctx).await
            };
            if let Err(error) = saved {
                log::warn!(
                    "Workflow '{}': could not save the subagents' state: {error}",
                    ctx.workflow_name
                );
            }
        });
    }

    // ----- the mailbox -----

    /// Adds an item to run `to`'s mailbox (a long body is saved whole in a
    /// file too).
    fn add_mail(
        &self,
        state: &mut SubagentsSnapshot,
        to: Uuid,
        from: MailSender,
        kind: MailKind,
        heading: String,
        body: String,
    ) -> MailItem {
        state.last_mail += 1;
        let id = state.last_mail;
        let file = (body.trim().len() > super::MAX_MAIL_BYTES)
            .then(|| self.save_mail(id, &heading, &body))
            .flatten();
        // Mail from a subagent its lead's conversation no longer started
        // is set aside, not given.
        let withheld = matches!(from, MailSender::Run(sender) if state
            .runs
            .iter()
            .any(|record| record.run_id == sender && record.detached && record.caller_run_id == to));
        let now = Utc::now();
        let item = MailItem {
            id,
            to,
            from,
            kind,
            heading,
            body,
            at: now,
            delivered_at: withheld.then_some(now),
            file,
            withheld,
        };
        state.mail.push(item.clone());
        item
    }

    fn save_mail(&self, id: u64, heading: &str, body: &str) -> Option<PathBuf> {
        let file = self.mail_dir.join(format!("mail-{id}.md"));
        let saved = std::fs::create_dir_all(&self.mail_dir)
            .and_then(|()| std::fs::write(&file, format!("{heading}\n\n{}\n", body.trim())));
        match saved {
            Ok(()) => Some(file),
            Err(error) => {
                log::warn!(
                    "Could not save subagent mail {id} in {}: {error}",
                    file.display()
                );
                None
            }
        }
    }

    /// After `item` was added: the waits are woken, and a recipient that
    /// isn't running gets it as its kind says (see [`Self::wake`]).
    fn posted(&self, ctx: &Arc<SharedContext>, item: &MailItem) {
        self.bump();
        if item.kind.wakes() {
            self.wake(ctx, item.to);
        }
        self.save_soon(ctx);
    }

    /// Run `to` has mail that wakes it: unless it runs (it gets the mail at
    /// its next turn boundary, or its wait returns it), a subagent that
    /// finished, or that its lead's message restarts, goes on, and a run
    /// the host drives gets `NewsWaiting`.
    fn wake(&self, ctx: &Arc<SharedContext>, to: Uuid) {
        if ctx.registry.is_running(to) {
            return;
        }
        let (record, pending) = {
            let state = self.state.lock();
            let pending = state
                .mail
                .iter()
                .filter(|item| item.to == to && item.delivered_at.is_none() && item.kind.wakes())
                .cloned()
                .collect::<Vec<_>>();
            (
                state.runs.iter().find(|run| run.run_id == to).cloned(),
                pending,
            )
        };
        if pending.is_empty() {
            return;
        }
        match record {
            Some(record) => {
                let from_lead = pending
                    .iter()
                    .any(|item| item.from == MailSender::Run(record.caller_run_id));
                let goes_on = match record.state {
                    SubagentState::Finished => true,
                    SubagentState::Stopped | SubagentState::Interrupted | SubagentState::Failed => {
                        from_lead
                    }
                    _ => false,
                };
                if goes_on {
                    self.launch(
                        ctx,
                        to,
                        PendingLaunch::Resume {
                            message: String::new(),
                        },
                    );
                }
            }
            None => {
                let about = {
                    let state = self.state.lock();
                    pending
                        .last()
                        .and_then(|item| about_record(&state, item.from, to))
                };
                if let Some(record) = about {
                    self.emit(
                        &record,
                        SubagentEventKind::NewsWaiting {
                            mail_ids: pending.iter().map(|item| item.id).collect(),
                        },
                    );
                }
            }
        }
    }

    /// The mail for run `task_id`'s next turn boundary, oldest first, as
    /// much as fits ([`UPDATE_MAX_MAIL_BYTES`]), with the items' ids
    /// (`mail:<id>`): it counts as delivered.
    pub(crate) fn update(&self, task_id: Uuid) -> Option<(String, Vec<String>)> {
        let (text, taken) = {
            let mut state = self.state.lock();
            let indices = state
                .mail
                .iter()
                .enumerate()
                .filter(|(_, item)| item.to == task_id && item.delivered_at.is_none())
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            if indices.is_empty() {
                return None;
            }
            let now = Utc::now();
            let mut bytes = 0;
            let mut parts = Vec::new();
            let mut taken = Vec::new();
            for &index in &indices {
                let excerpt = state.mail[index].excerpt();
                if !parts.is_empty() && bytes + excerpt.len() > UPDATE_MAX_MAIL_BYTES {
                    break;
                }
                bytes += excerpt.len();
                parts.push(format!("- {excerpt}"));
                state.mail[index].delivered_at = Some(now);
                taken.push(state.mail[index].clone());
            }
            let more = indices.len() - taken.len();
            let intro = if taken.len() == 1 {
                "New mail from the agent runs you work with."
            } else {
                "New mail from the agent runs you work with, oldest first."
            };
            let mut text = format!(
                "{intro} Reply with message_agent; wait_for_agents waits for more.\n{}",
                parts.join("\n")
            );
            match more {
                0 => {}
                1 => text.push_str("\n- (1 more item follows at your next turn.)"),
                more => text.push_str(&format!(
                    "\n- ({more} more items follow at your next turn.)"
                )),
            }
            (text, taken)
        };
        let ids = taken
            .iter()
            .map(|item| format!("mail:{}", item.id))
            .collect();
        self.delivered(task_id, &taken);
        Some((text, ids))
    }

    /// Run `task.id` ended as `task`: the mail it was given that its task
    /// holds leaves the mailbox, and the mail its task doesn't hold (given
    /// as it ended, or after its last save) is given again. A run the host
    /// drives with mail waiting gets `NewsWaiting`.
    pub(crate) fn settle_run(&self, ctx: &Arc<SharedContext>, task: &Task) {
        let delivered = {
            let state = self.state.lock();
            state
                .mail
                .iter()
                .filter(|item| item.to == task.id && item.delivered_at.is_some())
                .cloned()
                .collect::<Vec<_>>()
        };
        if delivered.is_empty() && self.pending(task.id).is_empty() {
            return;
        }
        // Searched outside the lock: a conversation can be long.
        let held = delivered
            .iter()
            .filter(|item| task_holds(task, item))
            .map(|item| item.id)
            .collect::<Vec<_>>();
        {
            let mut state = self.state.lock();
            state.mail.retain_mut(|item| {
                if item.to != task.id || item.delivered_at.is_none() {
                    return true;
                }
                if held.contains(&item.id) {
                    return false;
                }
                item.delivered_at = None;
                true
            });
        }
        self.bump();
        // A subagent's driver continues it.
        if !self.is_subagent(task.id) {
            self.wake(ctx, task.id);
        }
    }

    /// Run `task.id` was cut back (`ContinueOptions::cut_back_to`). The
    /// subagents it started in the `removed` messages belong to a timeline
    /// the conversation abandoned: they are detached, and their mail (given
    /// or not, and any still to come) is withheld, kept in the records for
    /// the host to show. Those it started in the messages it keeps, or in a
    /// conversation a handover replaced, stay. The mail of the subagents it
    /// still started that the removed messages held is given again. A
    /// detached subagent that still runs or waits for a place is stopped,
    /// with its own subagents, since no one reads its report.
    pub(crate) fn cut_back(
        &self,
        ctx: &Arc<SharedContext>,
        task: &Task,
        removed: &[arko_ai::Message],
    ) {
        let mut state = self.state.lock();
        let mut detached = Vec::new();
        for record in &mut state.runs {
            if record.caller_run_id == task.id
                && !record.detached
                && holds_start(removed, &record.handle)
            {
                record.detached = true;
                detached.push((record.run_id, record.state.is_active()));
            }
        }
        let detached_ids = detached.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        let now = Utc::now();
        for item in &mut state.mail {
            if item.to != task.id || item.withheld {
                continue;
            }
            if matches!(item.from, MailSender::Run(sender) if detached_ids.contains(&sender)) {
                item.withheld = true;
                item.delivered_at.get_or_insert(now);
            } else if item.delivered_at.is_some() && !task_holds(task, item) {
                item.delivered_at = None;
            }
        }
        drop(state);
        self.bump();
        let request = StopRequest::operator(CUT_BACK_REASON).wrap_up(STOP_WRAP_UP);
        for (run_id, active) in detached {
            if active {
                let origin = StopOrigin::Lead(CUT_BACK_REASON.to_string());
                self.request_stop(ctx, run_id, request.clone(), origin);
            }
        }
    }

    /// Run `task.id` is recovered from its task as last saved, which may
    /// lack mail it was given after that save: that mail is given again.
    pub(crate) fn recovered(&self, task: &Task) {
        let delivered = {
            let state = self.state.lock();
            state
                .mail
                .iter()
                .filter(|item| item.to == task.id && item.delivered_at.is_some())
                .cloned()
                .collect::<Vec<_>>()
        };
        let missing = delivered
            .iter()
            .filter(|item| !task_holds(task, item))
            .map(|item| item.id)
            .collect::<Vec<_>>();
        if missing.is_empty() {
            return;
        }
        let mut state = self.state.lock();
        for item in &mut state.mail {
            if missing.contains(&item.id) {
                item.delivered_at = None;
            }
        }
    }

    // ----- starting and running -----

    /// Starts a subagent for caller run `caller` (of `caller_agent`, at
    /// `caller_depth`, working in `caller_dir`): the tool's result, or why
    /// it can't.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn start(
        &self,
        ctx: &Arc<SharedContext>,
        caller: Uuid,
        caller_agent: &str,
        caller_depth: u32,
        caller_dir: &Path,
        agent: &str,
        task: &str,
        workspace: Option<&str>,
        isolated: bool,
        peers: Option<&str>,
        parameters: BTreeMap<String, serde_json::Value>,
    ) -> Result<String, String> {
        let config = self.config.as_ref().ok_or("subagents are off")?;
        let task = task.trim();
        if task.is_empty() {
            return Err("give the agent a task".to_string());
        }
        ctx.run_control
            .check_admission(false)
            .map_err(|_| "the workflow is stopping, so no agent run starts".to_string())?;
        if ctx.run_control.is_run_stopping(caller) {
            return Err(
                "your run is being stopped, so it can't start agent runs; save your progress and call complete_task"
                    .to_string(),
            );
        }
        let (peer_ids, all_peers) = match peers.map(str::trim).filter(|text| !text.is_empty()) {
            None => (Vec::new(), false),
            Some(_) if !config.peer_messages => {
                return Err("messages between agent runs are off in this workflow".to_string());
            }
            Some(text) if text.eq_ignore_ascii_case("all") => (Vec::new(), true),
            Some(text) => {
                let state = self.state.lock();
                let mut ids = Vec::new();
                for id in split_ids(text) {
                    let run = find(&state, &id)
                        .filter(|run| run.caller_run_id == caller)
                        .ok_or_else(|| format!("{id} is not one of your agent runs"))?;
                    ids.push(run.run_id);
                }
                (ids, false)
            }
        };
        let run_id = Uuid::new_v4();
        let handle = {
            let mut state = self.state.lock();
            state.last_handle += 1;
            format!("r{}", state.last_handle)
        };
        let now = Utc::now();
        let mut record = SubagentRecord {
            run_id,
            handle: handle.clone(),
            agent: agent.to_string(),
            caller_run_id: caller,
            caller_agent: caller_agent.to_string(),
            depth: caller_depth + 1,
            task: task.to_string(),
            workspace: caller_dir.to_path_buf(),
            isolated: None,
            parameters,
            peers: peer_ids,
            all_peers,
            state: SubagentState::Queued,
            created: now,
            since: now,
            report: None,
            error: None,
            pending: None,
            stopped_by_caller: None,
            stop_requested: None,
            asking: false,
            detached: false,
        };
        // The host's changes are checked before a workspace is made for the
        // run, or the run is recorded.
        let prepared = self.prepared_config(ctx, &record)?;
        let (workspace, isolated) = match (
            workspace.map(str::trim).filter(|w| !w.is_empty()),
            isolated,
        ) {
            (Some(_), true) => {
                return Err(
                    "give either a workspace or isolated, not both: an isolated run gets a workspace of its own"
                        .to_string(),
                );
            }
            (None, true) => {
                let provider = config
                    .workspace_provider
                    .as_ref()
                    .ok_or("isolated workspaces are not available in this workflow")?;
                let request = WorkspaceRequest {
                    workflow_run_id: ctx.workflow_run_id,
                    run_id,
                    handle: handle.clone(),
                    agent: agent.to_string(),
                    caller_run_id: caller,
                    base: caller_dir.to_path_buf(),
                };
                let created = provider.create(&request).await?;
                if !created.path.is_dir() {
                    let _ = provider.remove(&created).await;
                    return Err(format!(
                        "the isolated workspace {} doesn't exist",
                        created.path.display()
                    ));
                }
                (created.path.clone(), Some(created))
            }
            (Some(path), false) => {
                let resolved = crate::workflow::options::run_workspace(caller_dir, Path::new(path))
                    .map_err(|error| error.to_string())?;
                if !config.workspaces_anywhere && !resolved.starts_with(caller_dir) {
                    return Err(format!(
                        "the workspace must be inside your working directory {}",
                        caller_dir.display()
                    ));
                }
                (resolved, None)
            }
            (None, false) => (caller_dir.to_path_buf(), None),
        };
        record.workspace = workspace.clone();
        record.isolated = isolated.clone();
        let prompt = subagent_prompt(config, self.workflow_run_id, &record);
        self.prepared.lock().insert(run_id, prepared);
        self.state.lock().runs.push(record);
        self.launch(ctx, run_id, PendingLaunch::Start { prompt });
        let queued = self
            .record(run_id)
            .is_some_and(|record| record.state == SubagentState::Queued);
        let place = match &isolated {
            Some(isolated) => match &isolated.branch {
                Some(branch) => format!(
                    " in an isolated workspace, {} (branch `{branch}`)",
                    isolated.path.display()
                ),
                None => format!(" in an isolated workspace, {}", isolated.path.display()),
            },
            None if workspace != caller_dir => format!(" in {}", workspace.display()),
            None => String::new(),
        };
        let start = if queued {
            format!(
                "Queued {handle} (agent `{agent}`){place}: {} agent runs are running, the most this workflow allows, so it starts when one ends.",
                config.max_running.map_or(0, |max| max.get())
            )
        } else {
            format!("Started {handle} (agent `{agent}`){place}.")
        };
        Ok(format!(
            "{start} It works while you go on, and its report reaches you when it finishes (wait_for_agents waits for it). message_agent with to \"{handle}\" sends it a message."
        ))
    }

    /// The configuration `record`'s run starts with: its agent's, as the
    /// host's hook changes it.
    pub(crate) fn prepared_config(
        &self,
        ctx: &SharedContext,
        record: &SubagentRecord,
    ) -> Result<AgentConfig, String> {
        let mut config = ctx
            .agent_configs
            .get(&record.agent)
            .cloned()
            .ok_or_else(|| format!("there is no agent `{}`", record.agent))?;
        if let Some(hook) = self.config.as_ref().and_then(|config| config.hook.as_ref()) {
            hook.prepare(&start_request(self.workflow_run_id, record), &mut config)?;
        }
        Ok(config)
    }

    /// Launches run `run_id` (`launch` says how), or queues it while
    /// [`SubagentsConfig::max_running`] runs are running.
    fn launch(&self, ctx: &Arc<SharedContext>, run_id: Uuid, launch: PendingLaunch) {
        self.launch_if(ctx, run_id, launch, false);
    }

    fn launch_if(
        &self,
        ctx: &Arc<SharedContext>,
        run_id: Uuid,
        launch: PendingLaunch,
        queued_only: bool,
    ) {
        let max = self.config.as_ref().and_then(|config| config.max_running);
        let resumed = matches!(launch, PendingLaunch::Resume { .. });
        let (record, started) = {
            let mut state = self.state.lock();
            let running = state
                .runs
                .iter()
                .filter(|run| run.state == SubagentState::Running)
                .count();
            let Some(record) = state.runs.iter_mut().find(|run| run.run_id == run_id) else {
                return;
            };
            if record.state == SubagentState::Running
                || (queued_only && record.state != SubagentState::Queued)
            {
                return;
            }
            let has_place = max.is_none_or(|max| running < max.get());
            let runtime = tokio::runtime::Handle::try_current().ok();
            record.since = Utc::now();
            record.stopped_by_caller = None;
            record.stop_requested = None;
            record.error = None;
            if has_place && runtime.is_some() {
                record.state = SubagentState::Running;
                record.pending = None;
                self.starting.lock().insert(run_id, None);
                (record.clone(), runtime)
            } else {
                record.state = SubagentState::Queued;
                record.pending = Some(launch.clone());
                (record.clone(), None)
            }
        };
        self.bump();
        let Some(runtime) = started else {
            self.emit(&record, SubagentEventKind::Queued);
            self.save_soon(ctx);
            return;
        };
        self.emit(
            &record,
            if resumed {
                SubagentEventKind::Resumed
            } else {
                SubagentEventKind::Started
            },
        );
        let driver = runtime.spawn(drive(ctx.clone(), run_id, launch));
        self.drivers.lock().insert(run_id, driver.abort_handle());
    }

    /// Starts the queued runs that have a place now, oldest first.
    fn start_queued(&self, ctx: &Arc<SharedContext>) {
        loop {
            let next = {
                let state = self.state.lock();
                let running = state
                    .runs
                    .iter()
                    .filter(|run| run.state == SubagentState::Running)
                    .count();
                let max = self.config.as_ref().and_then(|config| config.max_running);
                if max.is_some_and(|max| running >= max.get()) {
                    return;
                }
                state
                    .runs
                    .iter()
                    .filter(|run| run.state == SubagentState::Queued && run.pending.is_some())
                    .min_by_key(|run| run.since)
                    .map(|run| (run.run_id, run.pending.clone().unwrap()))
            };
            let Some((run_id, launch)) = next else {
                return;
            };
            if ctx.run_control.check_admission(false).is_err() {
                return;
            }
            self.launch_if(ctx, run_id, launch, true);
            if self
                .record(run_id)
                .is_some_and(|record| record.state == SubagentState::Queued)
            {
                return;
            }
        }
    }

    /// Run `run_id` ended with `outcome`: its record says how, its lead gets
    /// the news, and a queued run takes its place.
    async fn ended(
        &self,
        ctx: &Arc<SharedContext>,
        run_id: Uuid,
        outcome: Result<AgentResult, WorkflowError>,
    ) {
        self.starting.lock().remove(&run_id);
        self.drivers.lock().remove(&run_id);
        let lead_stop = self.lead_stops.lock().remove(&run_id);
        let Some(record) = self.record(run_id) else {
            return;
        };
        let result = match &outcome {
            Ok(result) => Some(result.clone()),
            Err(_) => recorded_result(ctx, run_id),
        };
        let workflow_stopped = ctx.run_control.error().is_some();
        let handle = &record.handle;
        let agent = &record.agent;
        let (state, kind, heading, body, event) = match &outcome {
            Ok(result)
                if result.status == AgentRunStatus::Completed
                    && result.stopped_by.is_none()
                    && record.stop_requested.is_none() =>
            {
                // Mail that came as it finished: it goes on, and reports again.
                if !workflow_stopped && self.pending(run_id).iter().any(|item| item.kind.wakes()) {
                    let launch = PendingLaunch::Resume {
                        message: String::new(),
                    };
                    {
                        let mut state = self.state.lock();
                        if let Some(record) = state.runs.iter_mut().find(|run| run.run_id == run_id)
                        {
                            settle_record(
                                record,
                                SubagentState::Queued,
                                Some(result.summary.clone()),
                                None,
                            );
                            record.pending = Some(launch.clone());
                        }
                    }
                    self.bump();
                    self.launch_if(ctx, run_id, launch, true);
                    self.start_queued(ctx);
                    return;
                }
                (
                    SubagentState::Finished,
                    MailKind::Report,
                    format!("{handle} (agent `{agent}`) finished. Its report:"),
                    result.summary.clone(),
                    SubagentEventKind::Finished {
                        summary: result.summary.clone(),
                    },
                )
            }
            _ => {
                let stop = result
                    .as_ref()
                    .and_then(|result| result.stopped_by.clone())
                    .or_else(|| record.stop_requested.clone());
                let stopped = stop.is_some()
                    || matches!(&outcome, Err(WorkflowError::AgentStopped { .. }))
                    || workflow_stopped;
                if stopped {
                    let reason = stop
                        .as_ref()
                        .map(|stop| stop.reason.clone())
                        .or_else(|| match &outcome {
                            Err(WorkflowError::AgentStopped { reason, .. }) => Some(reason.clone()),
                            Err(error) => Some(error.to_string()),
                            Ok(_) => None,
                        })
                        .unwrap_or_default();
                    let reason = reason.trim().trim_end_matches('.').to_string();
                    let by_caller = record.stopped_by_caller.clone();
                    let who = match (&by_caller, workflow_stopped, &lead_stop) {
                        (Some(why), ..) if why.is_empty() => "you stopped it".to_string(),
                        (Some(why), ..) => format!("you stopped it: {why}"),
                        (None, true, _) => format!("it was stopped with the workflow ({reason})"),
                        (None, false, Some(words)) => words.clone(),
                        (None, false, None) => {
                            format!("the workflow's host stopped it ({reason})")
                        }
                    };
                    let retired = stop.as_ref().is_some_and(StopRecord::is_retired);
                    let progress = result
                        .as_ref()
                        .filter(|result| !result.summary.trim().is_empty())
                        .map(|result| {
                            format!("Its progress report:\n\n{}\n\n", result.summary.trim())
                        })
                        .unwrap_or_default();
                    let next = if retired {
                        "The host retired it, so it can't be restarted."
                    } else {
                        "message_agent restarts it where it stopped."
                    };
                    (
                        SubagentState::Stopped,
                        MailKind::Stopped,
                        format!("{handle} (agent `{agent}`) stopped before it finished: {who}."),
                        format!("{progress}{next}"),
                        SubagentEventKind::Stopped {
                            reason: by_caller.unwrap_or(reason),
                            by_caller: record.stopped_by_caller.is_some(),
                        },
                    )
                } else {
                    let error = match &outcome {
                        Ok(result) => result
                            .error
                            .clone()
                            .unwrap_or_else(|| format!("its run ended {:?}", result.status)),
                        Err(error) => error.to_string(),
                    };
                    let resumable = result.as_ref().is_some_and(resumable);
                    let next = if resumable {
                        "\n\nmessage_agent continues it from where it was."
                    } else {
                        "\n\nIts run can't go on: start a new one if the work is still needed."
                    };
                    (
                        SubagentState::Failed,
                        MailKind::Failed,
                        format!("{handle} (agent `{agent}`) stopped with an error:"),
                        format!("{error}{next}"),
                        SubagentEventKind::Failed { error },
                    )
                }
            }
        };
        let report = result
            .as_ref()
            .map(|result| result.summary.clone())
            .filter(|summary| !summary.trim().is_empty());
        let error = match &event {
            SubagentEventKind::Failed { error } => Some(error.clone()),
            SubagentEventKind::Stopped { reason, .. } => Some(reason.clone()),
            _ => None,
        };
        // Publish the terminal state and its news together, so a host
        // checking for idle workers cannot miss the final report.
        let (record, item) = {
            let mut board = self.state.lock();
            let Some(record) = board.runs.iter_mut().find(|run| run.run_id == run_id) else {
                return;
            };
            settle_record(record, state, report, error);
            let record = record.clone();
            let item = self.add_mail(
                &mut board,
                record.caller_run_id,
                MailSender::Run(run_id),
                kind,
                heading,
                body,
            );
            (record, item)
        };
        self.bump();
        self.emit(&record, event);
        if lead_stop.is_some() {
            // Its lead stopped or ended too, and isn't told to start a turn.
            self.posted_quietly(ctx);
        } else {
            self.posted(ctx, &item);
        }
        // Its moved calls' and monitors' notes would reach no one. Its
        // commands go on (a server it started for its lead, say).
        #[cfg(any(unix, windows))]
        if let Some(jobs) = &ctx.background_jobs {
            jobs.stop_owner_watchers(run_id);
        }
        // Its subagents would report to a run that is gone, and spend on
        // work no one reads. When the workflow stops, all runs stop anyway.
        if !workflow_stopped && !ctx.run_control.is_wrapping_up() {
            let kind = record
                .stop_requested
                .as_ref()
                .map_or(StopKind::Operator, |stop| stop.kind);
            self.stop_workers_of(
                ctx,
                &record,
                &StopRequest::new(kind, "").wrap_up(STOP_WRAP_UP),
                &format!("its lead {} ended", record.handle),
            );
        }
        if !workflow_stopped {
            self.start_queued(ctx);
        }
    }

    /// Relaunches the subagents a stop of the workflow ended, or that were
    /// running when it ended: their handles. The news that one stopped,
    /// which its lead hasn't received, is withdrawn: it goes on.
    pub(crate) fn resume_all(&self, ctx: &Arc<SharedContext>) -> Vec<String> {
        let runs = self
            .records()
            .into_iter()
            .filter(|run| {
                matches!(
                    run.state,
                    SubagentState::Interrupted | SubagentState::Queued
                ) || (run.state == SubagentState::Stopped && stopped_with_workflow(ctx, run.run_id))
            })
            .collect::<Vec<_>>();
        let withdrawn = {
            let mut state = self.state.lock();
            let before = state.mail.len();
            state.mail.retain(|item| {
                item.kind != MailKind::Stopped
                    || item.delivered_at.is_some()
                    || !runs
                        .iter()
                        .any(|run| item.from == MailSender::Run(run.run_id))
            });
            state.mail.len() != before
        };
        if withdrawn {
            self.bump();
            self.save_soon(ctx);
        }
        let mut resumed = Vec::new();
        for run in runs {
            let launch = run.pending.clone().unwrap_or(PendingLaunch::Resume {
                message: String::new(),
            });
            if run.state == SubagentState::Queued {
                // Its place comes in turn.
                let mut state = self.state.lock();
                if let Some(record) = state
                    .runs
                    .iter_mut()
                    .find(|record| record.run_id == run.run_id)
                {
                    record.pending = Some(launch);
                }
                resumed.push(run.handle);
                continue;
            }
            self.launch(ctx, run.run_id, launch);
            resumed.push(run.handle);
        }
        self.start_queued(ctx);
        resumed
    }

    // ----- messages, waits and stops -----

    /// Sends `text` from run `from` to `to` (a run id, or `lead`): the
    /// tool's result.
    pub(crate) fn message(
        &self,
        ctx: &Arc<SharedContext>,
        from: Uuid,
        to: &str,
        text: &str,
    ) -> Result<String, String> {
        self.send(ctx, from, to, text, None)
    }

    /// Sends the host's `text` (its user's, say) to subagent `to` as a
    /// message from the subagent's lead goes, under `heading`, with a copy
    /// for the lead under `copy_heading` (a copy doesn't start a turn of
    /// it).
    pub(crate) fn host_message(
        &self,
        ctx: &Arc<SharedContext>,
        to: &str,
        text: &str,
        heading: &str,
        copy_heading: &str,
    ) -> Result<String, String> {
        let lead = {
            let state = self.state.lock();
            find(&state, to).map(|run| run.caller_run_id)
        };
        let lead = lead.ok_or_else(|| format!("there is no agent run {}", to.trim()))?;
        self.send(ctx, lead, to, text, Some((heading, copy_heading)))
    }

    /// [`message`](Self::message), or with `host`'s headings
    /// [`host_message`](Self::host_message).
    fn send(
        &self,
        ctx: &Arc<SharedContext>,
        from: Uuid,
        to: &str,
        text: &str,
        host: Option<(&str, &str)>,
    ) -> Result<String, String> {
        let config = self.config.as_ref().ok_or("subagents are off")?;
        let text = text.trim();
        if text.is_empty() {
            return Err("the message is empty".to_string());
        }
        let recipient = self.recipient(from, to)?;
        let me = self.record(from);
        let my_name = me
            .as_ref()
            .map(|me| format!("{} (agent `{}`)", me.handle, me.agent));
        let (to_id, heading, about) = match &recipient {
            Recipient::Subagent(run) => {
                match run.state {
                    SubagentState::Closed => {
                        return Err(format!(
                            "{}'s workspace was removed, so it can't continue; start a new agent run",
                            run.handle
                        ));
                    }
                    SubagentState::Failed
                        if !recorded_result(ctx, run.run_id)
                            .as_ref()
                            .is_some_and(resumable) =>
                    {
                        return Err(format!(
                            "{} can't continue ({}); start a new agent run",
                            run.handle,
                            run.error.as_deref().unwrap_or("its run failed")
                        ));
                    }
                    SubagentState::Stopped
                        if recorded_result(ctx, run.run_id)
                            .and_then(|result| result.stopped_by)
                            .is_some_and(|stop| stop.is_retired()) =>
                    {
                        return Err(format!(
                            "the workflow's host retired {}, so it can't continue",
                            run.handle
                        ));
                    }
                    _ => {}
                }
                let heading = match host {
                    Some((heading, _)) => heading.to_string(),
                    None if run.asking => "Your lead answers your question:".to_string(),
                    None => "Your lead sent you a message:".to_string(),
                };
                (run.run_id, heading, run.clone())
            }
            Recipient::Lead(lead) => {
                let me = me.clone().ok_or("you have no lead")?;
                (
                    *lead,
                    format!(
                        "{} sent you a message (it goes on working):",
                        my_name.clone().unwrap_or_default()
                    ),
                    me,
                )
            }
            Recipient::Peer(peer) => {
                let me = me.clone().ok_or("you have no lead")?;
                if peer.state == SubagentState::Closed {
                    return Err(format!(
                        "{}'s workspace was removed: it can't run again",
                        peer.handle
                    ));
                }
                (
                    peer.run_id,
                    format!(
                        "{}, another agent run of your lead, sent you a message:",
                        my_name.clone().unwrap_or_default()
                    ),
                    me,
                )
            }
        };
        let (item, copy) = {
            let mut state = self.state.lock();
            match &recipient {
                // The lead steps in: its runs may exchange messages again.
                Recipient::Subagent(run) => state
                    .peer_counts
                    .retain(|(a, b, _)| *a != run.run_id && *b != run.run_id),
                Recipient::Peer(peer) => {
                    let pair = ordered(from, peer.run_id);
                    let count = match state
                        .peer_counts
                        .iter_mut()
                        .find(|(a, b, _)| (*a, *b) == pair)
                    {
                        Some((_, _, count)) => {
                            *count += 1;
                            *count
                        }
                        None => {
                            state.peer_counts.push((pair.0, pair.1, 1));
                            1
                        }
                    };
                    if count > config.max_peer_messages {
                        if let Some((_, _, count)) = state
                            .peer_counts
                            .iter_mut()
                            .find(|(a, b, _)| (*a, *b) == pair)
                        {
                            *count -= 1;
                        }
                        return Err(format!(
                            "you and {} have sent each other {} messages since your lead last wrote to either of you, the most allowed: tell your lead (message_agent to \"lead\") what you need instead",
                            peer.handle, config.max_peer_messages
                        ));
                    }
                }
                Recipient::Lead(_) => {}
            }
            let item = self.add_mail(
                &mut state,
                to_id,
                MailSender::Run(from),
                MailKind::Message,
                heading,
                text.to_string(),
            );
            let copy = match &recipient {
                Recipient::Peer(peer) if config.copy_peer_messages_to_lead => Some(self.add_mail(
                    &mut state,
                    peer.caller_run_id,
                    MailSender::Run(from),
                    MailKind::Copy,
                    format!(
                        "{} sent {} a message (a copy for you):",
                        my_name.clone().unwrap_or_default(),
                        peer.handle
                    ),
                    text.to_string(),
                )),
                Recipient::Subagent(_) => host.map(|(_, copy_heading)| {
                    self.add_mail(
                        &mut state,
                        from,
                        MailSender::Host,
                        MailKind::Copy,
                        copy_heading.to_string(),
                        text.to_string(),
                    )
                }),
                _ => None,
            };
            (item, copy)
        };
        self.emit(
            &about,
            SubagentEventKind::Message {
                mail_id: item.id,
                from,
                to: to_id,
            },
        );
        self.posted(ctx, &item);
        if let Some(copy) = &copy {
            self.posted(ctx, copy);
        }
        Ok(match &recipient {
            Recipient::Subagent(run) => {
                let now = self.record(run.run_id).map_or(run.state, |run| run.state);
                match (run.state, now) {
                    (_, SubagentState::Queued) if run.state != SubagentState::Queued => format!(
                        "Sent: {} goes on with your message once an agent run ends (the most are running).",
                        run.handle
                    ),
                    (SubagentState::Queued, _) => {
                        format!("Sent: {} reads it when it starts.", run.handle)
                    }
                    (SubagentState::Running, _) if run.asking => {
                        format!("Sent: {} gets it as the answer to its question.", run.handle)
                    }
                    (SubagentState::Running, _) => {
                        format!("Sent: {} reads it at its next step.", run.handle)
                    }
                    (SubagentState::Finished, _) => format!(
                        "Sent: {} continues its task with your message, with its context; its new report reaches you when it finishes.",
                        run.handle
                    ),
                    _ => format!(
                        "Sent: {} restarts where it stopped, with your message; its report reaches you when it finishes.",
                        run.handle
                    ),
                }
            }
            Recipient::Lead(_) => {
                "Sent. Your lead's answer, if any, reaches you at your next step (or through wait_for_agents).".to_string()
            }
            Recipient::Peer(peer) => format!("Sent to {}.", peer.handle),
        })
    }

    /// Whom `to` names for a message from run `from`.
    fn recipient(&self, from: Uuid, to: &str) -> Result<Recipient, String> {
        let to = to.trim();
        let state = self.state.lock();
        let me = state.runs.iter().find(|run| run.run_id == from);
        if to.eq_ignore_ascii_case("lead") {
            return me
                .map(|me| Recipient::Lead(me.caller_run_id))
                .ok_or_else(|| {
                    "you have no lead: you were not started as an agent run".to_string()
                });
        }
        if let Some(me) = me
            && to == me.caller_run_id.to_string()
        {
            return Ok(Recipient::Lead(me.caller_run_id));
        }
        let Some(run) = find(&state, to) else {
            return Err(unknown_run(&state, from, to));
        };
        if run.run_id == from {
            return Err("that is you".to_string());
        }
        if run.caller_run_id == from {
            return Ok(Recipient::Subagent(run.clone()));
        }
        match me {
            Some(me) if linked(me, run) => {
                if !self
                    .config
                    .as_ref()
                    .is_some_and(|config| config.peer_messages)
                {
                    return Err("messages between agent runs are off in this workflow".to_string());
                }
                Ok(Recipient::Peer(run.clone()))
            }
            Some(me) if me.caller_run_id == run.caller_run_id => Err(format!(
                "your lead hasn't linked you with {}: message your lead (to \"lead\") instead",
                run.handle
            )),
            _ => Err(unknown_run(&state, from, to)),
        }
    }

    /// Lets `runs`, all run `caller`'s, message each other.
    pub(crate) fn link(&self, caller: Uuid, runs: &[String]) -> Result<String, String> {
        if !self
            .config
            .as_ref()
            .is_some_and(|config| config.peer_messages)
        {
            return Err("messages between agent runs are off in this workflow".to_string());
        }
        let mut state = self.state.lock();
        let mut ids = Vec::new();
        for id in runs {
            let run = find(&state, id)
                .filter(|run| run.caller_run_id == caller)
                .ok_or_else(|| format!("{id} is not one of your agent runs"))?;
            if !ids.contains(&run.run_id) {
                ids.push(run.run_id);
            }
        }
        if ids.len() < 2 {
            return Err("name at least two of your agent runs".to_string());
        }
        for run in &mut state.runs {
            if ids.contains(&run.run_id) {
                for id in &ids {
                    if *id != run.run_id && !run.peers.contains(id) {
                        run.peers.push(*id);
                    }
                }
            }
        }
        // A new start for their exchanges.
        state
            .peer_counts
            .retain(|(a, b, _)| !(ids.contains(a) && ids.contains(b)));
        let handles = state
            .runs
            .iter()
            .filter(|run| ids.contains(&run.run_id))
            .map(|run| run.handle.clone())
            .collect::<Vec<_>>();
        drop(state);
        self.bump();
        Ok(format!(
            "{} may message each other now (message_agent with each other's ids). Tell them, if they don't know each other's ids.",
            handles.join(", ")
        ))
    }

    /// Waits for mail for run `me`, as `wait_for_agents` does: returns the
    /// tool's result. The call's `signals` end it early when it is stopped,
    /// moved to the background, or context arrives for it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn wait(
        &self,
        ctx: &Arc<SharedContext>,
        me: Uuid,
        agent_name: &str,
        runs: Option<Vec<String>>,
        all: bool,
        timeout: Duration,
        signals: &WaitSignals<'_>,
    ) -> Result<String, String> {
        let WaitSignals {
            stop,
            moved,
            context_arrived,
        } = signals;
        // The runs waited for, by id.
        let wanted = match &runs {
            None => None,
            Some(ids) => {
                let state = self.state.lock();
                let mut wanted = Vec::new();
                for id in ids {
                    let run = find(&state, id).ok_or_else(|| unknown_run(&state, me, id))?;
                    wanted.push(run.run_id);
                }
                Some(wanted)
            }
        };
        let lead = self.record(me).map(|record| record.caller_run_id);
        let mut version = self.version.subscribe();
        let mut notes = ctx.notes.watch();
        let deadline = tokio::time::Instant::now() + timeout;
        let mut collected: Vec<MailItem> = Vec::new();
        let mut stopping = ctx.run_control.agent_stopping(me);
        loop {
            version.borrow_and_update();
            notes.borrow_and_update();
            // What came for it, of what it waits for.
            let (taken, still) = {
                let mut state = self.state.lock();
                let now = Utc::now();
                let mut taken = Vec::new();
                for item in &mut state.mail {
                    if item.to != me || item.delivered_at.is_some() {
                        continue;
                    }
                    let from_wanted = match (&wanted, item.from) {
                        (None, _) => true,
                        (Some(wanted), MailSender::Run(from)) => {
                            wanted.contains(&from) || Some(from) == lead
                        }
                        (Some(_), MailSender::Host) => true,
                    };
                    if from_wanted {
                        item.delivered_at = Some(now);
                        taken.push(item.clone());
                    }
                }
                let still = state
                    .runs
                    .iter()
                    .filter(|run| {
                        run.caller_run_id == me
                            && run.state.is_active()
                            && wanted
                                .as_ref()
                                .is_none_or(|wanted| wanted.contains(&run.run_id))
                    })
                    .map(|run| run.handle.clone())
                    .collect::<Vec<_>>();
                (taken, still)
            };
            self.delivered(me, &taken);
            collected.extend(taken);
            let message = collected
                .iter()
                .any(|item| matches!(item.kind, MailKind::Message | MailKind::Question));
            let woken = collected.iter().any(|item| item.kind.wakes());
            let done = if all {
                still.is_empty() || message
            } else {
                woken
            };
            if done && !collected.is_empty() {
                return Ok(wait_result(&collected, &still));
            }
            if still.is_empty() && lead.is_none() {
                let state = self.state.lock();
                let which = match &wanted {
                    Some(wanted) => wanted
                        .iter()
                        .filter_map(|id| state.runs.iter().find(|run| run.run_id == *id))
                        .map(|run| format!("{} is {}", run.handle, run.state.describe()))
                        .collect::<Vec<_>>()
                        .join("; "),
                    None => "none of your agent runs is running".to_string(),
                };
                let mut result = format!(
                    "Nothing to wait for: {which}. Reports that came earlier reached you already."
                );
                if !collected.is_empty() {
                    result = format!("{}\n\n{result}", wait_result(&collected, &still));
                }
                return Ok(result);
            }
            if ctx.run_control.is_stopped() || ctx.run_control.is_wrapping_up() {
                return Ok(with_collected("Workflow stopping.", &collected, &still));
            }
            if *stopping.borrow_and_update() || stop.is_stopping() {
                return Ok(with_collected("Run stopping.", &collected, &still));
            }
            if ctx.notes.has_update(agent_name, me) {
                return Ok(with_collected(
                    "A note for you arrived while you waited: it follows. Wait again if you still need to.",
                    &collected,
                    &still,
                ));
            }
            if moved.is_moved() {
                return Ok(with_collected(WAIT_MOVED, &collected, &still));
            }
            if context_arrived.has_arrived() {
                return Ok(with_collected(CONTEXT_ARRIVED, &collected, &still));
            }
            tokio::select! {
                biased;
                _ = ctx.run_control.stopped() => {}
                _ = stop.stopping() => {}
                _ = stopping.changed() => {}
                () = moved.moved() => {}
                () = context_arrived.arrived() => {}
                _ = version.changed() => {}
                _ = notes.changed() => {}
                _ = tokio::time::sleep_until(deadline) => {
                    let state = self.state.lock();
                    let still = state
                        .runs
                        .iter()
                        .filter(|run| run.caller_run_id == me && run.state.is_active())
                        .map(|run| run.handle.clone())
                        .collect::<Vec<_>>();
                    drop(state);
                    let waited = format!(
                        "Nothing new in {} s.",
                        timeout.as_secs_f64().round() as u64
                    );
                    return Ok(with_collected(&waited, &collected, &still));
                }
            }
        }
    }

    /// Subagent `me`'s question goes to its lead, and its lead's next
    /// message is the answer.
    pub(crate) async fn ask(
        &self,
        ctx: &Arc<SharedContext>,
        me: Uuid,
        questions: &str,
    ) -> Result<String, WorkflowError> {
        let Some(record) = self.record(me) else {
            return Err(WorkflowError::Channel("not a subagent".to_string()));
        };
        let lead = record.caller_run_id;
        let item = {
            let mut state = self.state.lock();
            if let Some(run) = state.runs.iter_mut().find(|run| run.run_id == me) {
                run.asking = true;
            }
            self.add_mail(
                &mut state,
                lead,
                MailSender::Run(me),
                MailKind::Question,
                format!(
                    "{} (agent `{}`) asks you (it waits for your answer: message_agent with to \"{}\"):",
                    record.handle, record.agent, record.handle
                ),
                questions.trim().to_string(),
            )
        };
        self.emit(
            &record,
            SubagentEventKind::Message {
                mail_id: item.id,
                from: me,
                to: lead,
            },
        );
        self.posted(ctx, &item);
        struct Asking<'a>(&'a Board, Uuid);
        impl Drop for Asking<'_> {
            fn drop(&mut self) {
                if let Some(run) = self
                    .0
                    .state
                    .lock()
                    .runs
                    .iter_mut()
                    .find(|run| run.run_id == self.1)
                {
                    run.asking = false;
                }
            }
        }
        let _asking = Asking(self, me);
        let mut version = self.version.subscribe();
        let deadline = tokio::time::Instant::now() + QUESTION_TIMEOUT;
        loop {
            version.borrow_and_update();
            let answer = {
                let mut state = self.state.lock();
                let now = Utc::now();
                state
                    .mail
                    .iter_mut()
                    .find(|item| {
                        item.to == me
                            && item.delivered_at.is_none()
                            && item.from == MailSender::Run(lead)
                    })
                    .map(|item| {
                        item.delivered_at = Some(now);
                        item.clone()
                    })
            };
            if let Some(answer) = answer {
                self.delivered(me, std::slice::from_ref(&answer));
                return Ok(answer.text());
            }
            tokio::select! {
                biased;
                _ = ctx.run_control.stopped() => {
                    return Err(ctx.run_control.error().unwrap_or(WorkflowError::Cancelled));
                }
                _ = version.changed() => {}
                _ = tokio::time::sleep_until(deadline) => {
                    return Ok(format!(
                        "Your lead didn't answer within {} minutes. Decide yourself, and say what you decided in your report.",
                        QUESTION_TIMEOUT.as_secs() / 60
                    ));
                }
            }
        }
    }

    /// Applies a stop received while the driver was preparing this launch.
    /// Called just after `begin_run`, before the agent runtime is spawned.
    pub(crate) fn registered(&self, ctx: &Arc<SharedContext>, run_id: Uuid) {
        let mut starting = self.starting.lock();
        if let Some(Some((mut request, at))) = starting.remove(&run_id) {
            if let Some(wrap_up) = &mut request.wrap_up {
                wrap_up.within = wrap_up.within.saturating_sub(at.elapsed());
                if wrap_up.within.is_zero() {
                    request.wrap_up = None;
                }
            }
            crate::workflow::cancel::WorkflowCancelHandle::new(ctx).stop_run(
                run_id,
                request,
                RunStopIntent::Relaunch,
            );
        }
    }

    pub(crate) fn stop_subagent(
        &self,
        ctx: &Arc<SharedContext>,
        run_id: Uuid,
        request: StopRequest,
    ) -> bool {
        self.request_stop(ctx, run_id, request, StopOrigin::Host)
            .is_some()
    }

    /// Returns whether a changed stop belongs to a running (rather than
    /// queued) subagent. State selection and startup stops share the launch
    /// locks, so a queued stop cannot be overtaken by a driver starting it.
    /// A stop that changes anything reaches the subagent's own active
    /// subagents too, and theirs in turn.
    fn request_stop(
        &self,
        ctx: &Arc<SharedContext>,
        run_id: Uuid,
        request: StopRequest,
        origin: StopOrigin,
    ) -> Option<bool> {
        if ctx.run_control.is_stopped() {
            return None;
        }
        let reason = match &origin {
            StopOrigin::Caller(reason) => reason.clone(),
            StopOrigin::Host | StopOrigin::Lead(_) => request.reason.clone(),
        };
        let by_caller = matches!(origin, StopOrigin::Caller(_));
        let stop_record = StopRecord {
            kind: request.kind,
            reason: request.reason.clone(),
            timestamp: Utc::now(),
            intent: Some(RunStopIntent::Relaunch),
        };
        let for_workers = request.clone();
        let (record, running, item) = {
            let mut state = self.state.lock();
            let record = state.runs.iter_mut().find(|run| run.run_id == run_id)?;
            match record.state {
                SubagentState::Queued => {
                    record.state = SubagentState::Stopped;
                    record.since = Utc::now();
                    record.error = Some(reason.clone());
                    record.pending = None;
                    record.asking = false;
                    let record = record.clone();
                    let (who, why) = match &origin {
                        StopOrigin::Caller(_) => ("you stopped it".to_string(), reason.clone()),
                        StopOrigin::Host => {
                            ("the workflow's host stopped it".to_string(), reason.clone())
                        }
                        StopOrigin::Lead(words) => (words.clone(), String::new()),
                    };
                    let why = if why.is_empty() {
                        String::new()
                    } else {
                        format!(": {why}")
                    };
                    let item = self.add_mail(
                        &mut state,
                        record.caller_run_id,
                        MailSender::Run(run_id),
                        MailKind::Stopped,
                        format!(
                            "{} (agent `{}`) stopped before it started: {who}{why}.",
                            record.handle, record.agent
                        ),
                        "message_agent starts it later.".to_string(),
                    );
                    (record, false, Some(item))
                }
                SubagentState::Running => {
                    let mut starting = self.starting.lock();
                    if let Some(pending) = starting.get_mut(&run_id) {
                        match pending {
                            Some((first, _))
                                if first.wrap_up.is_some() && request.wrap_up.is_none() =>
                            {
                                first.wrap_up = None;
                            }
                            Some(_) => return None,
                            None => *pending = Some((request, std::time::Instant::now())),
                        }
                    } else if !crate::workflow::cancel::WorkflowCancelHandle::new(ctx).stop_run(
                        run_id,
                        request,
                        RunStopIntent::Relaunch,
                    ) {
                        return None;
                    }
                    match &origin {
                        StopOrigin::Caller(reason) if record.stopped_by_caller.is_none() => {
                            record.stopped_by_caller = Some(reason.clone());
                        }
                        StopOrigin::Lead(words) if record.stop_requested.is_none() => {
                            self.lead_stops.lock().insert(run_id, words.clone());
                        }
                        _ => {}
                    }
                    record.stop_requested.get_or_insert(stop_record);
                    (record.clone(), true, None)
                }
                _ => return None,
            }
        };
        self.bump();
        if running {
            self.save_soon(ctx);
        } else {
            self.emit(
                &record,
                SubagentEventKind::Stopped {
                    reason: reason.clone(),
                    by_caller,
                },
            );
            let news = item.expect("queued stop has news");
            match origin {
                // Its lead stopped too, and isn't told to start a turn for it.
                StopOrigin::Lead(_) => self.posted_quietly(ctx),
                _ => self.posted(ctx, &news),
            }
        }
        self.stop_workers_of(
            ctx,
            &record,
            &for_workers,
            &format!("its lead {} was stopped", record.handle),
        );
        Some(running)
    }

    /// The mailbox changed in a way that wakes no one.
    fn posted_quietly(&self, ctx: &Arc<SharedContext>) {
        self.bump();
        self.save_soon(ctx);
    }

    /// Stops run `lead`'s active subagents for `why` (`its lead r1 was
    /// stopped`), the way `request` stops the run, with its kind and its
    /// wrap-up if it has one. Their own subagents follow.
    fn stop_workers_of(
        &self,
        ctx: &Arc<SharedContext>,
        lead: &SubagentRecord,
        request: &StopRequest,
        why: &str,
    ) {
        let workers = self.active_workers_of(lead.run_id);
        for worker in workers {
            let request = StopRequest {
                kind: request.kind,
                reason: why.to_string(),
                wrap_up: request.wrap_up.as_ref().map(|wrap_up| WrapUp {
                    within: wrap_up.within,
                    message: None,
                }),
            };
            self.request_stop(
                ctx,
                worker.run_id,
                request,
                StopOrigin::Lead(why.to_string()),
            );
        }
    }

    /// The subagents that run, or wait to run, for run `lead`.
    fn active_workers_of(&self, lead: Uuid) -> Vec<SubagentRecord> {
        self.state
            .lock()
            .runs
            .iter()
            .filter(|run| run.caller_run_id == lead && run.state.is_active())
            .cloned()
            .collect()
    }

    /// The handles of the subagents of run `lead` that run or wait to run,
    /// and of theirs in turn.
    fn active_descendants(&self, lead: Uuid) -> Vec<String> {
        let state = self.state.lock();
        let mut found: Vec<&SubagentRecord> = Vec::new();
        let mut leads = vec![lead];
        while let Some(next) = leads.pop() {
            for run in &state.runs {
                if run.caller_run_id == next && run.state.is_active() {
                    found.push(run);
                    leads.push(run.run_id);
                }
            }
        }
        let mut handles = found
            .into_iter()
            .map(|run| (run.created, run.handle.clone()))
            .collect::<Vec<_>>();
        handles.sort();
        handles.into_iter().map(|(_, handle)| handle).collect()
    }

    /// Stops run `id` of caller `caller`, as `stop_agent` does: the tool's
    /// result.
    pub(crate) async fn stop(
        &self,
        ctx: &Arc<SharedContext>,
        caller: Uuid,
        id: &str,
        reason: &str,
        remove_workspace: bool,
    ) -> Result<String, String> {
        let run = {
            let state = self.state.lock();
            find(&state, id)
                .filter(|run| run.caller_run_id == caller)
                .cloned()
                .ok_or_else(|| unknown_run(&state, caller, id))?
        };
        let handle = run.handle.clone();
        if remove_workspace && run.isolated.is_none() {
            return Err(format!("{handle} has no isolated workspace to remove"));
        }
        let reason = reason.trim().to_string();
        let why = if reason.is_empty() {
            "your lead stopped you".to_string()
        } else {
            format!("your lead stopped you: {reason}")
        };
        let workers = self.active_descendants(run.run_id);
        let stopped = self.request_stop(
            ctx,
            run.run_id,
            StopRequest::operator(why).wrap_up(STOP_WRAP_UP),
            StopOrigin::Caller(reason),
        );
        if stopped.is_none() && !remove_workspace {
            return Ok(format!("{handle} is already stopped or stopping."));
        }
        let stopping = stopped == Some(true);
        // Its agent runs stop with it.
        let with_it = match workers.as_slice() {
            [] => String::new(),
            [one] => format!(" Its own agent run {one} is stopping with it."),
            many => format!(
                " Its own agent runs {} are stopping with it.",
                handles_in_words(many)
            ),
        };
        if !remove_workspace {
            return Ok(if stopping {
                format!(
                    "{handle} is stopping: it saves its progress, and its report reaches you.{with_it} message_agent restarts it later, where it stopped."
                )
            } else {
                format!(
                    "Stopped {handle} before it started.{with_it} message_agent starts it later."
                )
            });
        }
        // Removed once its run has ended.
        let ended = async {
            let mut version = self.version.subscribe();
            loop {
                version.borrow_and_update();
                if self
                    .record(run.run_id)
                    .is_none_or(|record| !record.state.is_active())
                    && !ctx.registry.is_running(run.run_id)
                {
                    return;
                }
                if version.changed().await.is_err() {
                    return;
                }
            }
        };
        let grace = ctx.run_control.grace_period() + STOP_WRAP_UP + Duration::from_secs(5);
        if tokio::time::timeout(grace, ended).await.is_err() {
            return Err(format!("{handle} is still running; try again in a moment"));
        }
        let provider = self
            .config
            .as_ref()
            .and_then(|config| config.workspace_provider.clone())
            .ok_or("isolated workspaces are not available in this workflow")?;
        let workspace = run.isolated.clone().expect("checked above");
        provider.remove(&workspace).await?;
        {
            let mut state = self.state.lock();
            if let Some(record) = state
                .runs
                .iter_mut()
                .find(|record| record.run_id == run.run_id)
            {
                record.state = SubagentState::Closed;
                record.since = Utc::now();
            }
        }
        self.bump();
        if let Some(record) = self.record(run.run_id) {
            self.emit(&record, SubagentEventKind::WorkspaceRemoved);
        }
        self.save_soon(ctx);
        let branch = workspace.branch.map_or_else(String::new, |branch| {
            format!(" Its branch `{branch}` stays.")
        });
        Ok(format!(
            "{}Removed {handle}'s workspace {}.{branch}",
            if stopping {
                format!("Stopped {handle}. ")
            } else {
                String::new()
            },
            workspace.root.unwrap_or(workspace.path).display()
        ))
    }

    /// `list_agents`'s result for run `me`.
    pub(crate) fn list(&self, ctx: &SharedContext, me: Uuid) -> String {
        let state = self.state.lock();
        let now = Utc::now();
        let mine = state
            .runs
            .iter()
            .filter(|run| run.caller_run_id == me)
            .collect::<Vec<_>>();
        if mine.is_empty() {
            return "You have started no agent runs.".to_string();
        }
        let queued = state
            .runs
            .iter()
            .filter(|run| run.state == SubagentState::Queued && run.pending.is_some())
            .map(|run| (run.since, run.run_id))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .map(|(_, id)| id)
            .collect::<Vec<_>>();
        let mut lines = Vec::new();
        for run in mine {
            let state_text = match run.state {
                SubagentState::Queued => match queued.iter().position(|id| *id == run.run_id) {
                    Some(place) => format!("queued (number {} in line)", place + 1),
                    None => "queued".to_string(),
                },
                SubagentState::Running if run.asking => {
                    "running, waiting for your answer to its question".to_string()
                }
                state => state.describe().to_string(),
            };
            let place = match &run.isolated {
                Some(isolated) => match &isolated.branch {
                    Some(branch) => {
                        format!("isolated in {} (branch {branch})", isolated.path.display())
                    }
                    None => format!("isolated in {}", isolated.path.display()),
                },
                None => format!("in {}", run.workspace.display()),
            };
            let mut line = format!(
                "{} · agent `{}` · {state_text} {} · {place}\n  task: {}",
                run.handle,
                run.agent,
                ago(run.since, now),
                cut(&run.task, 200)
            );
            if !run.state.is_active() {
                if let Some(report) = &run.report {
                    line.push_str(&format!("\n  report: {}", cut(report, 300)));
                }
                if let Some(error) = &run.error {
                    line.push_str(&format!("\n  why: {}", cut(error, 200)));
                }
            }
            if run.all_peers {
                line.push_str("\n  may message all your other agent runs");
            } else if !run.peers.is_empty() {
                let peers = run
                    .peers
                    .iter()
                    .filter_map(|id| state.runs.iter().find(|other| other.run_id == *id))
                    .map(|other| other.handle.clone())
                    .collect::<Vec<_>>();
                line.push_str(&format!("\n  may message {}", peers.join(", ")));
            }
            let unread = state
                .mail
                .iter()
                .filter(|item| item.to == run.run_id && item.delivered_at.is_none())
                .count();
            if unread > 0 {
                line.push_str(&format!(
                    "\n  {unread} message{} it hasn't read yet",
                    if unread == 1 { "" } else { "s" }
                ));
            }
            if let Some(result) = ctx
                .agent_results
                .read()
                .iter()
                .find(|result| result.agent_run_id == run.run_id)
            {
                let cost = result.total_cost();
                if cost > 0.0 {
                    line.push_str(&format!("\n  cost so far: ${cost:.4}"));
                }
            }
            lines.push(line);
        }
        lines.join("\n")
    }

    /// Whether the run's lead may start subagents at `depth`.
    pub(crate) fn may_start_at(&self, depth: u32) -> bool {
        self.config
            .as_ref()
            .is_some_and(|config| depth < config.max_depth)
    }
}

fn settle_record(
    record: &mut SubagentRecord,
    state: SubagentState,
    report: Option<String>,
    error: Option<String>,
) {
    record.state = state;
    record.since = Utc::now();
    if report.is_some() {
        record.report = report;
    }
    record.error = error;
    record.pending = None;
    record.stopped_by_caller = None;
    record.stop_requested = None;
    record.asking = false;
}

/// Drives subagent `run_id`'s run, launched as `launch` says, and records
/// how it ended.
async fn drive(ctx: Arc<SharedContext>, run_id: Uuid, launch: PendingLaunch) {
    let outcome = Box::pin(run(&ctx, run_id, launch)).await;
    ctx.subagents.ended(&ctx, run_id, outcome).await;
}

async fn run(
    ctx: &Arc<SharedContext>,
    run_id: Uuid,
    launch: PendingLaunch,
) -> Result<AgentResult, WorkflowError> {
    let engine = ctx
        .workflow_services
        .get()
        .map(|services| services.engine())
        .ok_or_else(|| WorkflowError::Channel("the workflow has no services".to_string()))?;
    let record = ctx
        .subagents
        .record(run_id)
        .ok_or_else(|| WorkflowError::Channel(format!("no subagent {run_id}")))?;
    let depth = record.depth;
    let launch = match (launch, ctx.subagents.config()) {
        // A run stopped before it started starts now, from its task.
        (PendingLaunch::Resume { .. }, Some(config)) if recorded_result(ctx, run_id).is_none() => {
            PendingLaunch::Start {
                prompt: subagent_prompt(config, ctx.workflow_run_id, &record),
            }
        }
        (launch, _) => launch,
    };
    match launch {
        PendingLaunch::Start { prompt } => {
            // Prepared at the start, or again for a start a resume relaunches.
            let prepared = ctx.subagents.prepared.lock().remove(&run_id);
            let config = match prepared {
                Some(config) => config,
                None => ctx
                    .subagents
                    .prepared_config(ctx, &record)
                    .map_err(WorkflowError::BuilderInvalid)?,
            };
            ctx.registry.start(&[(run_id, record.agent.as_str())]);
            let options = StartOptions {
                reserved: Some(run_id),
                depth,
                workspace: Some(record.workspace.clone()),
                subagent: true,
                config: Some(config),
                ..StartOptions::default()
            };
            let started = engine.start_agent(&record.agent, &prompt, ctx, options)?;
            drive_to_result(started, ctx, engine.persistence()).await
        }
        PendingLaunch::Resume { message } => {
            let previous = recorded_result(ctx, run_id).ok_or_else(|| {
                WorkflowError::BuilderInvalid(format!(
                    "{} has no recorded run to continue",
                    record.handle
                ))
            })?;
            match previous.status {
                AgentRunStatus::Completed | AgentRunStatus::WaitingForInput => {
                    let prompt = if message.trim().is_empty() {
                        MAIL_AFTER_REPORT.to_string()
                    } else {
                        message
                    };
                    engine
                        .continue_nested(&previous, &prompt, ctx, depth, None)
                        .await
                }
                AgentRunStatus::Interrupted => {
                    let recovery = Recovery {
                        depth,
                        message: Some(&message),
                        ..Recovery::default()
                    };
                    engine.recover_nested(&previous, ctx, recovery).await
                }
                AgentRunStatus::Cancelled if !is_superseded(&previous) => {
                    let message = reopened_message(&message);
                    let recovery = Recovery {
                        depth,
                        message: Some(&message),
                        reopen: true,
                        ..Recovery::default()
                    };
                    engine.recover_nested(&previous, ctx, recovery).await
                }
                AgentRunStatus::Paused
                    if matches!(
                        AgentPause::of(&previous.task.current_state),
                        Some(AgentPause::ErrorRecovery { .. })
                    ) =>
                {
                    let note = if message.trim().is_empty() {
                        "Continue your task.".to_string()
                    } else {
                        message
                    };
                    engine
                        .resume_nested(
                            &previous,
                            AgentResume::ContinueAfterError { note: Some(note) },
                            ctx,
                            depth,
                            None,
                        )
                        .await
                }
                status => Err(WorkflowError::BuilderInvalid(format!(
                    "{}'s run can't go on (it ended {status:?})",
                    record.handle
                ))),
            }
        }
    }
}

/// Whether a run that ended as `result` can go on with a message.
fn resumable(result: &AgentResult) -> bool {
    match result.status {
        AgentRunStatus::Completed
        | AgentRunStatus::WaitingForInput
        | AgentRunStatus::Interrupted => true,
        AgentRunStatus::Cancelled => {
            !is_superseded(result)
                && !result
                    .stopped_by
                    .as_ref()
                    .is_some_and(StopRecord::is_retired)
        }
        AgentRunStatus::Paused => matches!(
            AgentPause::of(&result.task.current_state),
            Some(AgentPause::ErrorRecovery { .. })
        ),
        AgentRunStatus::Failed => false,
    }
}

/// Whether a stop of the whole workflow ended run `run_id`, so a resume
/// relaunches it.
fn stopped_with_workflow(ctx: &SharedContext, run_id: Uuid) -> bool {
    recorded_result(ctx, run_id).is_some_and(|result| {
        result
            .stopped_by
            .as_ref()
            .is_some_and(|stop| stop.intent.is_none())
            && result.status != AgentRunStatus::Completed
    })
}

/// Settles the records a checkpoint saved running: by the outcome their
/// runs recorded.
fn settle_restored(state: &mut SubagentsSnapshot, results: &[AgentResult]) {
    let mut news = Vec::new();
    for record in &mut state.runs {
        if record.state != SubagentState::Running {
            continue;
        }
        let result = results
            .iter()
            .find(|result| result.agent_run_id == record.run_id);
        record.since = Utc::now();
        record.asking = false;
        if let Some(stop) = record.stop_requested.take() {
            record.state = SubagentState::Stopped;
            record.pending = None;
            record.error = Some(stop.reason.clone());
            news.push((
                record.clone(),
                MailKind::Stopped,
                format!(
                    "{} (agent `{}`) stopped: {}.",
                    record.handle, record.agent, stop.reason
                ),
                "message_agent restarts it where it stopped.".to_string(),
            ));
            continue;
        }
        match result.map(|result| (&result.status, result)) {
            Some((AgentRunStatus::Completed, result)) if result.stopped_by.is_none() => {
                record.state = SubagentState::Finished;
                record.report = Some(result.summary.clone());
                news.push((
                    record.clone(),
                    MailKind::Report,
                    format!(
                        "{} (agent `{}`) finished. Its report:",
                        record.handle, record.agent
                    ),
                    result.summary.clone(),
                ));
            }
            Some((AgentRunStatus::Cancelled | AgentRunStatus::Completed, result))
                if result.stopped_by.is_some() =>
            {
                record.state = SubagentState::Stopped;
                record.error = result.stopped_by.as_ref().map(|stop| stop.reason.clone());
            }
            Some((AgentRunStatus::Failed, result)) => {
                record.state = SubagentState::Failed;
                record.error = result.error.clone();
            }
            _ => record.state = SubagentState::Interrupted,
        }
    }
    // News the lead never got: it gets it now.
    for (record, kind, heading, body) in news {
        let delivered = state
            .mail
            .iter()
            .any(|item| item.from == MailSender::Run(record.run_id) && item.kind == kind);
        if delivered {
            continue;
        }
        state.last_mail += 1;
        state.mail.push(MailItem {
            id: state.last_mail,
            to: record.caller_run_id,
            from: MailSender::Run(record.run_id),
            kind,
            heading,
            body,
            at: Utc::now(),
            delivered_at: None,
            file: None,
            withheld: false,
        });
    }
}

/// The record a mailbox item to `to` from `from` concerns: the sending
/// subagent's, else the recipient's.
fn about_record(state: &SubagentsSnapshot, from: MailSender, to: Uuid) -> Option<SubagentRecord> {
    let by_id = |id: Uuid| state.runs.iter().find(|run| run.run_id == id).cloned();
    match from {
        MailSender::Run(from) => by_id(from).or_else(|| by_id(to)),
        MailSender::Host => by_id(to),
    }
}

/// The run `id` names: a short id (`r1`) or a run id.
fn find<'a>(state: &'a SubagentsSnapshot, id: &str) -> Option<&'a SubagentRecord> {
    let id = id.trim().trim_matches('`').trim_matches('"');
    state
        .runs
        .iter()
        .find(|run| run.handle.eq_ignore_ascii_case(id) || run.run_id.to_string() == id)
}

fn unknown_run(state: &SubagentsSnapshot, me: Uuid, id: &str) -> String {
    let mine = state
        .runs
        .iter()
        .filter(|run| run.caller_run_id == me)
        .map(|run| run.handle.clone())
        .collect::<Vec<_>>();
    if mine.is_empty() {
        format!("there is no agent run `{id}` of yours")
    } else {
        format!(
            "there is no agent run `{id}` of yours; yours are {}",
            mine.join(", ")
        )
    }
}

/// Whether two subagents of the same lead may message each other.
fn linked(a: &SubagentRecord, b: &SubagentRecord) -> bool {
    a.caller_run_id == b.caller_run_id
        && (a.all_peers
            || b.all_peers
            || a.peers.contains(&b.run_id)
            || b.peers.contains(&a.run_id))
}

fn ordered(a: Uuid, b: Uuid) -> (Uuid, Uuid) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Run ids, from a text with commas or spaces.
pub(crate) fn split_ids(text: &str) -> Vec<String> {
    text.split([',', ' ', '\n'])
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(str::to_string)
        .collect()
}

/// Whether `task` holds `item`: in its conversation (a context update, a
/// prompt or a tool result), in context it took and hasn't added, or given
/// before its latest handover (its report covers it).
/// Whether `messages` hold the result of the call that started subagent
/// `handle` (`start_agent`'s "Started r1 (agent ..." or "Queued r1
/// (agent ...").
fn holds_start(messages: &[arko_ai::Message], handle: &str) -> bool {
    let started = format!("Started {handle} (agent ");
    let queued = format!("Queued {handle} (agent ");
    messages.iter().any(|message| {
        message.content.iter().any(|content| match content {
            arko_ai::Content::ToolResult(result) => {
                result.output.contains(&started) || result.output.contains(&queued)
            }
            arko_ai::Content::Text(text) => text.contains(&started) || text.contains(&queued),
            arko_ai::Content::Image(_) => false,
        })
    })
}

fn task_holds(task: &Task, item: &MailItem) -> bool {
    if let (Some(handover), Some(delivered)) = (task.handovers.last(), item.delivered_at)
        && handover.at >= delivered
    {
        return true;
    }
    let marker = item.marker();
    task.pending_context_injections
        .iter()
        .any(|content| content.contains(&marker))
        || task.conversation.messages.iter().any(|message| {
            message.content.iter().any(|content| match content {
                arko_ai::Content::Text(text) => {
                    message.role == arko_ai::Role::User && text.contains(&marker)
                }
                arko_ai::Content::ToolResult(result) => result.output.contains(&marker),
                arko_ai::Content::Image(_) => false,
            })
        })
}

/// The request [`SubagentHook::prepare`](super::SubagentHook::prepare) and
/// [`StartContext`](super::StartContext) get for `record`.
fn start_request(workflow_run_id: Uuid, record: &SubagentRecord) -> StartRequest {
    StartRequest {
        workflow_run_id,
        run_id: record.run_id,
        handle: record.handle.clone(),
        agent: record.agent.clone(),
        caller_run_id: record.caller_run_id,
        caller_agent: record.caller_agent.clone(),
        task: record.task.clone(),
        parameters: record.parameters.clone(),
    }
}

/// A subagent's initial prompt: its task, then who it works for and where,
/// after the host's context when it gives one
/// ([`SubagentsConfig::start_context`]).
fn subagent_prompt(
    config: &SubagentsConfig,
    workflow_run_id: Uuid,
    record: &SubagentRecord,
) -> String {
    let context = config
        .start_context
        .as_ref()
        .and_then(|context| context(&start_request(workflow_run_id, record)))
        .filter(|context| !context.trim().is_empty());
    let mut text = match context {
        Some(context) => format!("{}\n\n# Your task\n\n{}", context.trim(), record.task),
        None => record.task.clone(),
    };
    text.push_str(&format!(
        "\n\n# Working as an agent run\nYou are agent run {}, started by agent `{}` (your lead) to carry out the task above while it goes on with its own work. Your report (your complete_task summary) goes to your lead when you finish. While you work, message_agent with to \"lead\" sends your lead a message (a finding it needs now, or a question only it can answer), and wait_for_agents waits for its answer. Its messages reach you at your next step.",
        record.handle, record.caller_agent
    ));
    if config.questions_to_lead {
        text.push_str(" A question you ask goes to your lead, which asks the user if it needs to.");
    }
    if record.all_peers {
        text.push_str(
            " You may also message your lead's other agent runs, by their ids (such as r2).",
        );
    } else if !record.peers.is_empty() {
        text.push_str(
            " You may also message the agent runs your lead linked you with, by their ids.",
        );
    }
    text.push_str("\n\n# Where you work\n");
    match record
        .isolated
        .as_ref()
        .and_then(|isolated| isolated.description.clone())
    {
        Some(description) => text.push_str(&description),
        None if record.isolated.is_some() => text.push_str(&format!(
            "You work in `{}`, a workspace of your own.",
            record.workspace.display()
        )),
        None => text.push_str(&format!(
            "You work in `{}`. Other agents may work there too: change only the files your task is about.",
            record.workspace.display()
        )),
    }
    text
}

/// What ends `wait_for_agents` early, from its call's
/// [`TaskContext`](arko_agents::TaskContext): a stop, a move to the
/// background, and context arriving.
pub(crate) struct WaitSignals<'a> {
    pub(crate) stop: &'a arko_agents::stop::StopSignal,
    pub(crate) moved: &'a arko_agents::MoveSignal,
    pub(crate) context_arrived: &'a arko_agents::ContextSignal,
}

/// How a wait that the user moved to the background ends.
pub(crate) const WAIT_MOVED: &str = "The user ended this wait so you can go on with other work. Reports and messages from your agent runs reach you as notes as they come.";

/// How a wait ends when context (a note, a terminal update) arrives for it.
pub(crate) const CONTEXT_ARRIVED: &str =
    "An update for you arrived while you waited. It follows. Wait again if you still need to.";

/// A wait's result: the items that came, whole, and the runs still active.
fn wait_result(items: &[MailItem], still: &[String]) -> String {
    let mut text = items
        .iter()
        .map(MailItem::text)
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");
    text.push_str("\n\n");
    text.push_str(&still_running(still));
    text
}

fn with_collected(lead: &str, items: &[MailItem], still: &[String]) -> String {
    if items.is_empty() {
        format!("{lead} {}", still_running(still))
    } else {
        format!("{}\n\n{lead}", wait_result(items, still))
    }
}

fn still_running(still: &[String]) -> String {
    match still {
        [] => "None of your agent runs is running now.".to_string(),
        [one] => format!("Still running: {one}."),
        many => format!("Still running: {}.", many.join(", ")),
    }
}

/// Handles in a sentence: `r3`, `r3 and r4`, `r3, r4 and r5`.
fn handles_in_words(handles: &[String]) -> String {
    match handles {
        [] => String::new(),
        [one] => one.clone(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

/// `for 3m`: how long a state has lasted.
fn ago(since: chrono::DateTime<Utc>, now: chrono::DateTime<Utc>) -> String {
    let seconds = (now - since).num_seconds().max(0);
    let span = if seconds < 60 {
        format!("{seconds}s")
    } else if seconds < 3600 {
        format!("{}m", seconds / 60)
    } else {
        format!("{}h {}m", seconds / 3600, seconds % 3600 / 60)
    };
    format!("for {span}")
}

/// `text`'s first line, cut to at most `max` bytes.
fn cut(text: &str, max: usize) -> String {
    let line = text.trim().lines().next().unwrap_or_default();
    if line.len() <= max {
        return line.to_string();
    }
    let mut end = max;
    while !line.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &line[..end])
}

#[cfg(test)]
mod stop_tests {
    use super::*;

    #[tokio::test]
    async fn a_selected_queued_run_stopped_before_launch_stays_stopped() {
        let workspace = crate::test_support::Workspace::new();
        let workflow = crate::Workflow::builder()
            .name("queued-stop")
            .workspace(&workspace.0)
            .llm_client(Arc::new(arko_ai::LLMClient::new()))
            .agent("lead", crate::test_support::agent_config("Lead"))
            .agent("worker", crate::test_support::agent_config("Worker"))
            .workflow_services(
                crate::services::WorkflowServicesConfig::default()
                    .allow_caller("lead")
                    .allow_target("worker"),
            )
            .subagents(SubagentsConfig::default().max_running(1))
            .build()
            .unwrap();
        let ctx = workflow.context();
        let caller = Uuid::new_v4();
        for task in ["First.", "Second."] {
            ctx.subagents
                .start(
                    ctx,
                    caller,
                    "lead",
                    0,
                    &workspace.0,
                    "worker",
                    task,
                    None,
                    false,
                    None,
                    BTreeMap::new(),
                )
                .await
                .unwrap();
        }
        let selected = workflow.subagents()[1].clone();
        assert!(workflow.stop_subagent(selected.run_id, StopRequest::operator("cancel queued")));
        // Simulate a place opening after start_queued selected its next run.
        ctx.subagents.state.lock().runs[0].state = SubagentState::Finished;
        ctx.subagents
            .launch_if(ctx, selected.run_id, selected.pending.unwrap(), true);
        assert_eq!(workflow.subagents()[1].state, SubagentState::Stopped);
        assert!(!ctx.subagents.starting.lock().contains_key(&selected.run_id));
    }
}
