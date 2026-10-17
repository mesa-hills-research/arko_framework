//! Operator notes: guidance for a workflow's agents, posted while the
//! workflow runs, without touching their configuration.
//!
//! [`Workflow::post_note`](crate::Workflow::post_note) (or a
//! [`NotesHandle`], from any task) posts a [`Note`]. A running agent gets it
//! at its next safe turn boundary, in its `# Workflow context update` (at
//! most [`PROMPT_MAX_NOTES`] and about [`PROMPT_MAX_BYTES`] of new notes per
//! update, oldest first; the rest at the next boundaries). A standing note
//! ([`NoteKind::Standing`], the default) is also shown in the initial prompt
//! of every later run of its audience, newest first and capped, in the
//! fresh conversation after a handover, and again after Claude Code compacts
//! the conversation. A one-time note ([`NoteKind::Once`])
//! reaches each run of its audience that has started when it is posted,
//! once.
//!
//! Each note is labelled with its source ([`NoteSource`]) and delivered in
//! the user role as a refinement of the task: it doesn't change a role's
//! rules. The audience ([`NoteAudience`]) names agents by registration name;
//! completion verifiers' reviews, and the agents set with
//! [`WorkflowBuilder::notes_by_name_only`](crate::WorkflowBuilder::notes_by_name_only),
//! get a note only when it names them.
//!
//! Notes are kept in checkpoints, so they survive a resume, and each is
//! logged once when it is posted. Agents can't post notes: the notes are
//! written only through this API, and through the opt-in
//! [control folder](control) the host enables.

pub mod control;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Weak};

use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::context::SharedContext;
use crate::error::WorkflowError;
use crate::persistence::PersistenceManager;
use arko_ai::image_store::ImageStore as _;

/// The longest note accepted, in bytes of its trimmed text: 16 KiB. A note
/// that is really a document belongs in a file or a memory tier.
pub const MAX_NOTE_BYTES: usize = 16 * 1024;

/// At most this many notes are shown in a prompt (the newest standing
/// notes; older ones are left out) or in one context update (the oldest new
/// notes; the rest come at the next turn boundary).
pub const PROMPT_MAX_NOTES: usize = 20;

/// About this many bytes of notes are shown in a prompt or a context update,
/// as [`PROMPT_MAX_NOTES`] says: no further note is added once it would pass
/// this. The first note is always shown in full.
pub const PROMPT_MAX_BYTES: usize = 16 * 1024;

/// Who receives a note. Names are registration names, as in
/// [`MessagingPolicy`](crate::messaging::MessagingPolicy). Completion
/// verifiers' reviews, and the agents set with
/// [`WorkflowBuilder::notes_by_name_only`](crate::WorkflowBuilder::notes_by_name_only),
/// receive a note only when [`Only`](Self::Only) names them. An agent never
/// receives a note it is the source of ([`NoteSource::Agent`]).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoteAudience {
    /// Every agent (the default).
    #[default]
    All,
    /// Only these agents.
    Only(BTreeSet<String>),
    /// Every agent except these.
    Except(BTreeSet<String>),
}

/// How long a note applies.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoteKind {
    /// Delivered to the running runs of its audience, and shown in the
    /// initial prompt of every later run (the default).
    #[default]
    Standing,
    /// Delivered once to each run of its audience that has started when it
    /// is posted: a running run at its next turn boundary, one that isn't
    /// running when it is continued, resumed or recovered. Later runs don't
    /// get it.
    Once,
}

/// Who wrote a note, as the agents are told. The host sets it; agents can't
/// post notes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoteSource {
    /// The person running the workflow (the default).
    #[default]
    Operator,
    /// The program running the workflow.
    Host,
    /// An agent, such as an observer reviewing the run; its note is shown as
    /// that agent's, and it doesn't receive the note itself. The name need
    /// not be a registered agent's.
    Agent(String),
}

/// A note for a workflow's agents. `Note::new(text)` is a standing note
/// from the operator to every agent.
///
/// In JSON, only `text` is required: `{"text": "…", "audience": "all" |
/// {"only": ["a"]} | {"except": ["b"]}, "kind": "standing" | "once",
/// "source": "operator" | "host" | {"agent": "observer"}}`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Note {
    /// The note, trimmed when posted; at most [`MAX_NOTE_BYTES`].
    pub text: String,
    #[serde(default)]
    pub audience: NoteAudience,
    #[serde(default)]
    pub kind: NoteKind,
    #[serde(default)]
    pub source: NoteSource,
    /// Images that go with the note, after the text of the context update
    /// that gives it, each after a label naming the note. Only a one-time
    /// note ([`NoteKind::Once`]) may carry images, and only to agents whose
    /// models take them ([`arko_ai::Model::image_refusal`]); with images, the
    /// text may be empty. A Claude Code agent gets them with its next input,
    /// not during a turn. Once the note is posted, the workflow keeps them
    /// apart from the notes: [`PostedNote::images`] names them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<arko_ai::Image>,
}

impl Note {
    /// A standing note from the operator to every agent.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Self::default()
        }
    }

    /// For these agents only.
    pub fn to<I, S>(mut self, agents: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.audience = NoteAudience::Only(agents.into_iter().map(Into::into).collect());
        self
    }

    /// For every agent except these.
    pub fn except<I, S>(mut self, agents: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.audience = NoteAudience::Except(agents.into_iter().map(Into::into).collect());
        self
    }

    /// A one-time note ([`NoteKind::Once`]).
    pub fn once(mut self) -> Self {
        self.kind = NoteKind::Once;
        self
    }

    /// Who wrote it (default [`NoteSource::Operator`]).
    pub fn source(mut self, source: NoteSource) -> Self {
        self.source = source;
        self
    }

    /// With `images`, as the note's `images` field describes.
    pub fn images(mut self, images: impl IntoIterator<Item = arko_ai::Image>) -> Self {
        self.images = images.into_iter().collect();
        self
    }
}

/// A note as posted: its identity, its place in the run's order, and when it
/// was posted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostedNote {
    /// Stable identity: assigned when posted, or the control folder entry's.
    pub id: Uuid,
    /// The note's number in the run, from 1, in the order notes were posted.
    pub seq: u64,
    /// When it was posted, or written to the control folder.
    pub posted_at: DateTime<Utc>,
    /// The note as posted, without its images (`note.images` is empty).
    pub note: Note,
    /// The note's images, kept apart from the notes, so that the notes stay
    /// small however many images they carry: checkpoints and the event log
    /// name them. [`Workflow::note_image`](crate::Workflow::note_image)
    /// reads one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<NoteImage>,
}

/// An image of a posted note ([`PostedNote::images`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoteImage {
    /// An image at a URL.
    Url(String),
    /// An image the workflow keeps, by name: `<SHA-256 of its bytes>.<type>`.
    /// A persisted run keeps it in a file of that name in its directory's
    /// `images` folder, written when the note is posted; a run that isn't
    /// persisted keeps it in memory.
    Stored { name: String, mime_type: String },
}

/// The notes as a checkpoint saves them:
/// [`WorkflowState::notes`](crate::persistence::state::WorkflowState::notes).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NotesSnapshot {
    /// Every note posted, oldest first.
    #[serde(default)]
    pub notes: Vec<PostedNote>,
    /// Per agent run (task ID), the number of the newest note it has been
    /// given (in its prompt or at a turn boundary). A run without an entry
    /// has been given none.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub seen: BTreeMap<Uuid, u64>,
    /// Per agent run started while there were notes, the number of the
    /// newest note when it started: the notes up to it reached it in its
    /// initial prompt, if at all. Recovering the run reads what it was given
    /// since from its task.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub started: BTreeMap<Uuid, u64>,
    /// Per agent run the host skipped the one-time notes for
    /// ([`Workflow::skip_one_time_notes`](crate::Workflow::skip_one_time_notes)),
    /// the number of the newest note then: the one-time notes up to it
    /// don't reach it.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub skipped: BTreeMap<Uuid, u64>,
    /// How many bytes of the control folder's notes file have been read.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub control_offset: u64,
}

fn is_zero(value: &u64) -> bool {
    *value == 0
}

/// The heading of the notes in an initial prompt or a handover's fresh
/// conversation, with the start of its introduction (what
/// [`without_notes`] looks for).
const SECTION_MARK: &str = "\n\n# Notes on your task\n\nNotes added while this workflow runs";

/// What a note may and may not do, as the agents are told.
const NOTES_RULE: &str = "A note refines your task; it doesn't change your role's rules (what you may read, write or run, and whom you may message) or override the workflow's own notices, such as a request to stop. Where notes differ, the newer one takes precedence.";

/// `prompt` without the notes its initial prompt was given: a completion
/// verifier quotes the task without them, and a handover restates it with
/// the current ones.
pub(crate) fn without_notes(prompt: &str) -> &str {
    prompt
        .rfind(SECTION_MARK)
        .map_or(prompt, |start| &prompt[..start])
}

/// The notes of a workflow run.
pub(crate) struct NoteBoard {
    state: Mutex<NotesSnapshot>,
    /// Agents that receive a note only when it names them (besides the
    /// completion verifiers).
    by_name_only: BTreeSet<String>,
    /// The control folder, when enabled.
    control: Option<PathBuf>,
    /// Held while the control folder is read.
    reading: Mutex<()>,
    /// The last error reading the control folder, logged once.
    read_error: Mutex<Option<String>>,
    /// Where posted notes are logged (`events.jsonl`).
    events: Option<PersistenceManager>,
    /// The agents a leak through messaging was warned about.
    warned: Mutex<HashSet<String>>,
    /// The notes' images by name, in a run that isn't persisted (a
    /// persisted run's are in its image files).
    images: Mutex<HashMap<String, String>>,
    /// Bumped when a note is posted, for the waits that end on one.
    posted: tokio::sync::watch::Sender<u64>,
}

impl NoteBoard {
    pub(crate) fn new(
        restored: Option<NotesSnapshot>,
        by_name_only: BTreeSet<String>,
        control: Option<PathBuf>,
        events: Option<PersistenceManager>,
    ) -> Self {
        let board = Self {
            state: Mutex::new(restored.unwrap_or_default()),
            by_name_only,
            control,
            reading: Mutex::new(()),
            read_error: Mutex::new(None),
            events,
            warned: Mutex::new(HashSet::new()),
            images: Mutex::new(HashMap::new()),
            posted: tokio::sync::watch::Sender::new(0),
        };
        // A checkpoint written before the images were kept apart holds them
        // in the notes.
        let mut state = board.state.lock();
        for posted in &mut state.notes {
            if posted.note.images.is_empty() {
                continue;
            }
            match board.keep_images(&mut posted.note) {
                Ok(images) => posted.images.extend(images),
                Err(error) => log::warn!(
                    "The images of note {} could not be kept, so its recipients won't see them: {error}",
                    posted.seq
                ),
            }
        }
        drop(state);
        board
    }

    /// Takes `note`'s images and keeps them: in the run's image files, or in
    /// memory for a run that isn't persisted.
    fn keep_images(&self, note: &mut Note) -> Result<Vec<NoteImage>, String> {
        std::mem::take(&mut note.images)
            .into_iter()
            .map(|image| match image {
                arko_ai::Image::Url(url) => Ok(NoteImage::Url(url)),
                arko_ai::Image::Base64 { data, mime_type } => {
                    let name = match &self.events {
                        Some(persistence) => persistence.images().save(&data, &mime_type)?,
                        None => {
                            let (name, _) =
                                crate::persistence::images::image_file(&data, &mime_type);
                            self.images.lock().insert(name.clone(), data);
                            name
                        }
                    };
                    Ok(NoteImage::Stored { name, mime_type })
                }
            })
            .collect()
    }

    /// Note image `image`.
    pub(crate) fn image(&self, image: &NoteImage) -> Result<arko_ai::Image, String> {
        let (name, mime_type) = match image {
            NoteImage::Url(url) => return Ok(arko_ai::Image::Url(url.clone())),
            NoteImage::Stored { name, mime_type } => (name, mime_type),
        };
        let data = match (self.images.lock().get(name).cloned(), &self.events) {
            (Some(data), _) => data,
            (None, Some(persistence)) => persistence.images().load(name, mime_type)?,
            (None, None) => return Err(format!("this run doesn't keep an image named {name}")),
        };
        Ok(arko_ai::Image::Base64 {
            data,
            mime_type: mime_type.clone(),
        })
    }

    /// The images of `notes` (each note's number and images), each labelled
    /// with its note, fitted to what `model` takes
    /// ([`arko_ai::Model::sendable_image`]). One that can't be read or
    /// sent is left out, with a warning.
    fn context_images(
        &self,
        notes: &[(u64, Vec<NoteImage>)],
        model: Option<&arko_ai::Model>,
    ) -> Vec<arko_agents::ContextImage> {
        let mut images = Vec::new();
        for (seq, note_images) in notes {
            for (index, image) in note_images.iter().enumerate() {
                let label = if note_images.len() == 1 {
                    format!("The image of note {seq}:")
                } else {
                    format!("Image {} of note {seq}:", index + 1)
                };
                let sendable = self.image(image).and_then(|image| match model {
                    Some(model) => Ok(model.sendable_image(&image)?.unwrap_or(image)),
                    None => Ok(image),
                });
                match sendable {
                    Ok(image) => images.push(arko_agents::ContextImage::new(label, image)),
                    Err(error) => log::warn!(
                        "An image of note {seq} can't be given, so it is left out: {error}"
                    ),
                }
            }
        }
        images
    }

    /// Changes when a note is posted.
    pub(crate) fn watch(&self) -> tokio::sync::watch::Receiver<u64> {
        self.posted.subscribe()
    }

    /// The number of the newest note run `task_id` has been given (in its
    /// prompt or at a turn boundary); 0 when it has been given none.
    pub(crate) fn given(&self, task_id: Uuid) -> u64 {
        self.state.lock().seen.get(&task_id).copied().unwrap_or(0)
    }

    /// Whether run `task_id` of `agent` (not a completion verifier's review)
    /// receives a note it hasn't been given.
    pub(crate) fn has_update(&self, agent: &str, task_id: Uuid) -> bool {
        let state = self.state.lock();
        let seen = state.seen.get(&task_id).copied().unwrap_or(0);
        state.notes.iter().any(|posted| {
            posted.seq > seen
                && self.receives(&posted.note, agent, false)
                && !skipped(&state, task_id, posted)
        })
    }

    /// The one-time notes posted so far don't reach run `task_id`; see
    /// [`Workflow::skip_one_time_notes`](crate::Workflow::skip_one_time_notes).
    pub(crate) fn skip_one_time(&self, ctx: &SharedContext, task_id: Uuid) {
        self.poll(ctx);
        let mut state = self.state.lock();
        if let Some(newest) = state.notes.last().map(|posted| posted.seq) {
            state.skipped.insert(task_id, newest);
        }
    }

    /// The state a checkpoint saves; `None` while there is nothing to save.
    pub(crate) fn snapshot(&self) -> Option<NotesSnapshot> {
        let state = self.state.lock();
        (*state != NotesSnapshot::default()).then(|| state.clone())
    }

    /// Every note posted, oldest first, after reading the control folder.
    pub(crate) fn notes(&self, ctx: &SharedContext) -> Vec<PostedNote> {
        self.poll(ctx);
        self.state.lock().notes.clone()
    }

    /// Posts `note`: checks it, numbers it and logs it.
    pub(crate) fn post(
        &self,
        ctx: &SharedContext,
        note: Note,
    ) -> Result<PostedNote, WorkflowError> {
        let (mut note, recipients) = self.check(ctx, note)?;
        let images = self.keep_images(&mut note).map_err(|error| {
            WorkflowError::Persistence(format!("could not keep the note's images: {error}"))
        })?;
        let posted = insert(&mut self.state.lock(), note, images, None, None);
        self.announce(ctx, &posted, &recipients);
        self.posted
            .send_modify(|count| *count = count.wrapping_add(1));
        Ok(posted)
    }

    /// The notes section of the initial prompt of run `task_id` of `agent`
    /// (`review` for a completion verifier's review): the standing notes it
    /// receives, newest first and capped; empty when there are none. The
    /// run has then been given every note so far.
    pub(crate) fn initial_section(
        &self,
        ctx: &SharedContext,
        agent: &str,
        task_id: Uuid,
        review: bool,
    ) -> String {
        self.poll(ctx);
        let mut state = self.state.lock();
        let Some(newest) = state.notes.last().map(|note| note.seq) else {
            return String::new();
        };
        state.seen.insert(task_id, newest);
        state.started.insert(task_id, newest);
        let notes = state
            .notes
            .iter()
            .rev()
            .filter(|posted| {
                posted.note.kind == NoteKind::Standing && self.receives(&posted.note, agent, review)
            })
            .collect::<Vec<_>>();
        section(&notes)
    }

    /// The notes for the fresh conversation of run `task_id` after a
    /// handover, and for its conversation after Claude Code compacted it:
    /// the standing notes it receives that it had been given,
    /// newest first and capped, as an initial prompt shows them. The notes
    /// it hasn't been given (such as one posted while it wrote its report)
    /// come in the context update before its first request, so none is lost
    /// to the cap or given twice; the one-time notes it had been given
    /// aren't repeated (its report covers them).
    pub(crate) fn handover_section(&self, agent: &str, task_id: Uuid, review: bool) -> String {
        let state = self.state.lock();
        let seen = state.seen.get(&task_id).copied().unwrap_or(0);
        let notes = state
            .notes
            .iter()
            .rev()
            .filter(|posted| {
                posted.note.kind == NoteKind::Standing
                    && posted.seq <= seen
                    && self.receives(&posted.note, agent, review)
            })
            .collect::<Vec<_>>();
        section(&notes)
    }

    /// The notes run `task_id` of `agent` (`review` for a completion
    /// verifier's review) receives that it hasn't been given, for its next
    /// turn boundary, oldest first, as many as fit ([`PROMPT_MAX_NOTES`],
    /// [`PROMPT_MAX_BYTES`]), with the numbers of the notes shown and their
    /// images, each labelled with its note; `None` when there are none. Only
    /// the notes shown count as given: the rest come at the next boundary.
    pub(crate) fn update_with_seqs(
        &self,
        ctx: &SharedContext,
        agent: &str,
        task_id: Uuid,
        review: bool,
    ) -> Option<(String, Vec<u64>, Vec<arko_agents::ContextImage>)> {
        self.poll(ctx);
        let mut state = self.state.lock();
        let newest = state.notes.last()?.seq;
        let seen = state.seen.get(&task_id).copied().unwrap_or(0);
        if seen >= newest {
            return None;
        }
        let notes = state
            .notes
            .iter()
            .filter(|posted| {
                posted.seq > seen
                    && self.receives(&posted.note, agent, review)
                    && !skipped(&state, task_id, posted)
            })
            .collect::<Vec<_>>();
        let shown = fitting(&notes);
        let given = match notes.get(shown) {
            // The newest shown: the notes after it wait for the next update.
            Some(_) => notes[shown - 1].seq,
            None => newest,
        };
        let text = (!notes.is_empty()).then(|| {
            (
                update(&notes[..shown], notes.len() - shown),
                notes[..shown].iter().map(|posted| posted.seq).collect(),
                notes[..shown]
                    .iter()
                    .filter(|posted| !posted.images.is_empty())
                    .map(|posted| (posted.seq, posted.images.clone()))
                    .collect::<Vec<_>>(),
            )
        });
        state.seen.insert(task_id, given);
        drop(state);
        // Read outside the lock: an image may come from its file.
        let model = ctx
            .agent_configs
            .get(agent)
            .map(|config| &config.llm_config.model);
        text.map(|(text, seqs, images)| (text, seqs, self.context_images(&images, model)))
    }

    /// Run `task` of `agent` is recovered from its task as last saved (after
    /// a crash, or aborted after the grace period), which may lack notes it
    /// was given after that save. Like a message, every note it receives
    /// that the task doesn't hold is given again, at its next turn boundary.
    /// `review` as the run is launched with (see [`WithNotes::review`]).
    pub(crate) fn recovered(&self, agent: &str, task: &arko_agents::Task, review: bool) {
        let (started, given) = {
            let state = self.state.lock();
            let started = state.started.get(&task.id).copied().unwrap_or(0);
            let given = state
                .notes
                .iter()
                .rev()
                .filter(|posted| posted.seq > started && self.receives(&posted.note, agent, review))
                .cloned()
                .collect::<Vec<_>>();
            (started, given)
        };
        // Notes are given oldest first, each update starting after the last
        // one shown, so the newest one the task holds tells what it was
        // given. The conversation is searched outside the lock (it can be
        // long).
        let held = given
            .iter()
            .find(|posted| holds(task, &item(posted)))
            .map_or(started, |posted| posted.seq);
        let mut state = self.state.lock();
        if state.seen.get(&task.id).is_some_and(|seen| *seen > held) {
            state.seen.insert(task.id, held);
        }
    }

    /// Run `task` of `agent` was cut back (`Task::cut_back`): the notes it
    /// was given in the messages the cut removed are given again at its next
    /// turn boundary, as for a recovered run, except the one-time notes,
    /// which went with the messages around them (such as what the user wrote
    /// while it worked). One-time notes it hasn't been given still reach it.
    pub(crate) fn cut_back(&self, agent: &str, task: &arko_agents::Task, review: bool) {
        let given = self.given(task.id);
        self.recovered(agent, task, review);
        if given > 0 {
            let mut state = self.state.lock();
            let skipped = state.skipped.entry(task.id).or_insert(0);
            *skipped = (*skipped).max(given);
        }
    }

    /// Run `task_id` joined the workflow with a conversation of its own (a
    /// branch, `Workflow::import_agent_run`): it counts as given every note
    /// posted so far, as a run started now does, and gets the later ones.
    pub(crate) fn adopted(&self, ctx: &SharedContext, task_id: Uuid) {
        self.poll(ctx);
        let mut state = self.state.lock();
        if let Some(newest) = state.notes.last().map(|posted| posted.seq) {
            state.seen.insert(task_id, newest);
            state.started.insert(task_id, newest);
        }
    }

    /// Whether a run of `agent` receives `note`; `review` for a completion
    /// verifier's review.
    fn receives(&self, note: &Note, agent: &str, review: bool) -> bool {
        if matches!(&note.source, NoteSource::Agent(source) if source == agent) {
            return false;
        }
        match &note.audience {
            NoteAudience::Only(agents) => agents.contains(agent),
            NoteAudience::All => !self.named_only(agent, review),
            NoteAudience::Except(agents) => {
                !agents.contains(agent) && !self.named_only(agent, review)
            }
        }
    }

    /// Whether a run of `agent` receives a note only when it names it: a
    /// completion verifier's `review` (the same agent run as an ordinary
    /// delegation target is not one), or a run of an agent set with
    /// `notes_by_name_only`.
    fn named_only(&self, agent: &str, review: bool) -> bool {
        review || self.by_name_only.contains(agent)
    }

    /// Whether `agent` receives `note` in the runs it is posted for, as
    /// posting checks and logs it: a completion verifier counts as
    /// reviewing (it runs as one).
    fn receives_as_posted(&self, ctx: &SharedContext, note: &Note, agent: &str) -> bool {
        self.receives(
            note,
            agent,
            crate::workflow::engine::is_verifier_agent(ctx, agent),
        )
    }

    /// `note` with its text trimmed, and the agents that receive it, sorted;
    /// an error for a note that can't be posted.
    fn check(
        &self,
        ctx: &SharedContext,
        mut note: Note,
    ) -> Result<(Note, Vec<String>), WorkflowError> {
        note.text = if note.images.is_empty() || !note.text.trim().is_empty() {
            checked_text(&note.text).map_err(WorkflowError::BuilderInvalid)?
        } else {
            String::new()
        };
        if !note.images.is_empty() && note.kind != NoteKind::Once {
            return Err(WorkflowError::BuilderInvalid(
                "only a one-time note (Note::once) may carry images: a standing note is shown in every later run's prompt".to_string(),
            ));
        }
        if let NoteSource::Agent(name) = &note.source {
            if name.trim().is_empty() {
                return Err(WorkflowError::BuilderInvalid(
                    "a note's source agent has no name".to_string(),
                ));
            }
            if name.chars().any(is_break_or_control) {
                return Err(WorkflowError::BuilderInvalid(format!(
                    "a note's source agent name {name:?} has a line break or a control character"
                )));
            }
        }
        let named = match &note.audience {
            NoteAudience::All => None,
            NoteAudience::Only(agents) if agents.is_empty() => {
                return Err(WorkflowError::BuilderInvalid(
                    "a note's audience names no agent".to_string(),
                ));
            }
            NoteAudience::Only(agents) | NoteAudience::Except(agents) => Some(agents),
        };
        if let Some(unknown) = named
            .into_iter()
            .flatten()
            .find(|agent| !ctx.agent_configs.contains_key(*agent))
        {
            return Err(WorkflowError::AgentNotFound(unknown.clone()));
        }
        let recipients = ctx
            .messages
            .roster()
            .into_iter()
            .filter(|agent| self.receives_as_posted(ctx, &note, agent))
            .collect::<Vec<_>>();
        if recipients.is_empty() {
            return Err(WorkflowError::BuilderInvalid(
                "no agent receives this note: completion verifiers and agents set with notes_by_name_only get a note only when it names them".to_string(),
            ));
        }
        for agent in &recipients {
            if let Some(config) = ctx.agent_configs.get(agent) {
                // Checked for every recipient here, fitted to each when given.
                crate::workflow::engine::sendable_images(agent, config, note.images.clone(), None)?;
            }
        }
        Ok((note, recipients))
    }

    /// Logs a posted note (once: notes restored from a checkpoint aren't
    /// posted again), in the log and the event log, and warns about the
    /// agents it can reach through messages although it excludes them.
    fn announce(&self, ctx: &SharedContext, posted: &PostedNote, recipients: &[String]) {
        let first_line = posted.note.text.lines().next().unwrap_or_default();
        log::info!(
            "Workflow '{}': note {} posted from {}, for {}: {}",
            ctx.workflow_name,
            posted.seq,
            source_label(&posted.note.source),
            recipients.join(", "),
            cut(first_line, 160)
        );
        if let Some(events) = &self.events {
            events.append_note(posted);
        }
        for (agent, senders) in self.leaks(ctx, recipients) {
            if !self.warned.lock().insert(agent.clone()) {
                continue;
            }
            let message = format!(
                "Workflow '{}': agent '{agent}' gets a note only when it names it, but {} may message it (MessagingPolicy), so the content of note {} can reach it in a message. If it must stay independent, restrict whom it can hear from.",
                ctx.workflow_name,
                senders
                    .iter()
                    .map(|sender| format!("'{sender}'"))
                    .collect::<Vec<_>>()
                    .join(", "),
                posted.seq
            );
            if self.leak_warns(ctx, &agent) {
                log::warn!("{message}");
            } else {
                log::debug!("{message}");
            }
        }
    }

    /// Whether a note's content reaching `agent` in a message (see
    /// [`leaks`](Self::leaks)) is worth a warning, not just a debug line: it
    /// is for an agent the host marked independent (`notes_by_name_only`).
    /// A completion verifier runs only while it reviews, inside its target's
    /// call, and a message to it is refused at other times unless messages
    /// are kept for finished agents, so it is warned about only then.
    fn leak_warns(&self, ctx: &SharedContext, agent: &str) -> bool {
        self.by_name_only.contains(agent) || ctx.keeps_messages_for_finished()
    }

    /// The agents that receive a note only when it names them and don't
    /// receive this one, each with the note's `recipients` that the
    /// messaging policy lets message it.
    fn leaks(&self, ctx: &SharedContext, recipients: &[String]) -> Vec<(String, Vec<String>)> {
        ctx.messages
            .roster()
            .into_iter()
            .filter(|agent| {
                self.named_only(
                    agent,
                    crate::workflow::engine::is_verifier_agent(ctx, agent),
                ) && !recipients.contains(agent)
            })
            .filter_map(|agent| {
                let senders = recipients
                    .iter()
                    .filter(|sender| ctx.message_recipients(sender).contains(&agent))
                    .cloned()
                    .collect::<Vec<_>>();
                (!senders.is_empty()).then_some((agent, senders))
            })
            .collect()
    }

    /// Posts the notes written to the control folder since it was last
    /// read. Another caller reading it already, or a writer holding its
    /// lock, leaves them for the next call.
    fn poll(&self, ctx: &SharedContext) {
        let Some(folder) = &self.control else {
            return;
        };
        let Some(_reading) = self.reading.try_lock() else {
            return;
        };
        let offset = self.state.lock().control_offset;
        let batch = match control::read_from(folder, offset) {
            Ok(Some(batch)) => batch,
            Ok(None) => return,
            Err(error) => {
                let error = error.to_string();
                let mut last = self.read_error.lock();
                if last.as_deref() != Some(error.as_str()) {
                    log::warn!(
                        "Workflow '{}': can't read the notes in {}: {error}. They are read again at the next turn.",
                        ctx.workflow_name,
                        folder.display()
                    );
                    *last = Some(error);
                }
                return;
            }
        };
        *self.read_error.lock() = None;
        let mut accepted = Vec::new();
        for entry in batch.entries {
            let checked = entry
                .map_err(|error| format!("it isn't a note ({error})"))
                .and_then(|entry| {
                    let (mut note, recipients) = self
                        .check(ctx, entry.note)
                        .map_err(|error| error.to_string())?;
                    let images = self
                        .keep_images(&mut note)
                        .map_err(|error| format!("its images could not be kept ({error})"))?;
                    Ok((entry.id, entry.at, note, images, recipients))
                });
            match checked {
                Ok(checked) => accepted.push(checked),
                Err(reason) => log::warn!(
                    "Workflow '{}': skipped an entry of {}: {reason}",
                    ctx.workflow_name,
                    folder.join(control::NOTES_FILE).display()
                ),
            }
        }
        // The notes and how much of the file they came from change together,
        // so a checkpoint never holds one without the other.
        let posted = {
            let mut state = self.state.lock();
            let mut posted = Vec::new();
            for (id, at, note, images, recipients) in accepted {
                if id.is_some_and(|id| state.notes.iter().any(|posted| posted.id == id)) {
                    continue;
                }
                posted.push((insert(&mut state, note, images, id, at), recipients));
            }
            state.control_offset = batch.end;
            posted
        };
        let any = !posted.is_empty();
        for (posted, recipients) in posted {
            self.announce(ctx, &posted, &recipients);
        }
        if any {
            self.posted
                .send_modify(|count| *count = count.wrapping_add(1));
        }
    }
}

/// Adds `note`, with its kept `images`, as the next note.
fn insert(
    state: &mut NotesSnapshot,
    note: Note,
    images: Vec<NoteImage>,
    id: Option<Uuid>,
    at: Option<DateTime<Utc>>,
) -> PostedNote {
    let posted = PostedNote {
        id: id.unwrap_or_else(Uuid::new_v4),
        seq: state.notes.last().map_or(0, |note| note.seq) + 1,
        posted_at: at.unwrap_or_else(Utc::now),
        note,
        images,
    };
    state.notes.push(posted.clone());
    posted
}

/// Whether `task` holds `text` in a user message of its conversation, or in
/// context it took and hasn't added yet.
fn holds(task: &arko_agents::Task, text: &str) -> bool {
    task.pending_context_injections
        .iter()
        .any(|content| content.contains(text))
        || task.conversation.messages.iter().any(|message| {
            message.role == arko_ai::Role::User
                && message
                    .content
                    .iter()
                    .filter_map(arko_ai::Content::as_text)
                    .any(|part| part.contains(text))
        })
}

/// Whether `posted` is a one-time note the host skipped for run `task_id`
/// ([`NoteBoard::skip_one_time`]).
fn skipped(state: &NotesSnapshot, task_id: Uuid, posted: &PostedNote) -> bool {
    posted.note.kind == NoteKind::Once
        && state
            .skipped
            .get(&task_id)
            .is_some_and(|newest| posted.seq <= *newest)
}

/// Whether `c` is a line break or a control character: what a source
/// agent's name may not contain.
fn is_break_or_control(c: char) -> bool {
    c.is_control() || matches!(c, '\u{2028}' | '\u{2029}')
}

/// `text` trimmed, with every line break (CR LF, a lone CR, NEL, U+2028 and
/// U+2029) made a LF, so every line of a note is indented when it is shown;
/// or why it can't be a note.
pub(crate) fn checked_text(text: &str) -> Result<String, String> {
    let text = text
        .replace("\r\n", "\n")
        .replace(['\r', '\u{85}', '\u{2028}', '\u{2029}'], "\n");
    let text = text.trim();
    if text.is_empty() {
        return Err("a note's text is empty".to_string());
    }
    if text.len() > MAX_NOTE_BYTES {
        return Err(format!(
            "a note may be at most {MAX_NOTE_BYTES} bytes; this one is {} bytes (a document belongs in a file or a memory tier)",
            text.len()
        ));
    }
    Ok(text.to_string())
}

/// How a note's source is named to the agents.
fn source_label(source: &NoteSource) -> String {
    match source {
        NoteSource::Operator => "the operator (the person running this workflow)".to_string(),
        NoteSource::Host => "the workflow (the program running the agents)".to_string(),
        NoteSource::Agent(name) => format!("agent `{name}`"),
    }
}

/// `text` cut to at most `max` bytes, at a character boundary.
fn cut(text: &str, max: usize) -> &str {
    if text.len() <= max {
        return text;
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// One note as a list item. Every line of its text is indented, so no line
/// of a note can pass for a heading of the prompt.
fn item(posted: &PostedNote) -> String {
    let mut text = posted.note.text.replace('\n', "\n  ");
    let with_images = match posted.images.len() {
        0 => None,
        1 => Some("(With an image, shown after this update.)".to_string()),
        count => Some(format!("(With {count} images, shown after this update.)")),
    };
    if let Some(line) = with_images {
        if !text.is_empty() {
            text.push_str("\n  ");
        }
        text.push_str(&line);
    }
    format!(
        "\n- Note {} from {}, {}:\n  {}",
        posted.seq,
        source_label(&posted.note.source),
        posted.posted_at.format("%Y-%m-%d %H:%M UTC"),
        text
    )
}

/// How many of `notes`, from the first, fit in a prompt or an update: at
/// most [`PROMPT_MAX_NOTES`], and no further note once it would pass
/// [`PROMPT_MAX_BYTES`]; the first always (at least one, when there are
/// any).
fn fitting(notes: &[&PostedNote]) -> usize {
    let mut bytes = 0;
    let mut shown = 0;
    for posted in notes.iter().take(PROMPT_MAX_NOTES) {
        let len = item(posted).len();
        if shown > 0 && bytes + len > PROMPT_MAX_BYTES {
            break;
        }
        bytes += len;
        shown += 1;
    }
    shown
}

fn items(notes: &[&PostedNote]) -> String {
    notes.iter().map(|posted| item(posted)).collect()
}

/// The notes section of a prompt: `notes` (newest first) as many as fit,
/// with a line counting the older ones left out; empty without notes.
fn section(notes: &[&PostedNote]) -> String {
    if notes.is_empty() {
        return String::new();
    }
    let shown = fitting(notes);
    let left = match notes.len() - shown {
        0 => String::new(),
        1 => "\n- (1 earlier note is not shown.)".to_string(),
        left => format!("\n- ({left} earlier notes are not shown.)"),
    };
    format!(
        "{SECTION_MARK}, newest first, each labelled with who wrote it. {NOTES_RULE}\n{}{left}",
        items(&notes[..shown])
    )
}

/// New notes for a running agent's context update, oldest first, and how
/// many more come at its next turn boundary.
fn update(notes: &[&PostedNote], more: usize) -> String {
    let intro = if notes.len() == 1 {
        "A new note on your task."
    } else {
        "New notes on your task, oldest first."
    };
    let more = match more {
        0 => String::new(),
        1 => "\n- (1 more note follows at your next turn.)".to_string(),
        more => format!("\n- ({more} more notes follow at your next turn.)"),
    };
    format!("{intro} {NOTES_RULE}{}{more}", items(notes))
}

/// Posts notes to a running workflow from any task or thread.
///
/// Obtained from [`Workflow::notes_handle`](crate::Workflow::notes_handle);
/// clones post to the same run. It holds only a weak reference to the
/// workflow: once the workflow is dropped, `post` returns an error and
/// `notes` returns none.
#[derive(Clone)]
pub struct NotesHandle {
    context: Weak<SharedContext>,
}

impl NotesHandle {
    pub(crate) fn new(context: &Arc<SharedContext>) -> Self {
        Self {
            context: Arc::downgrade(context),
        }
    }

    /// Posts `note`; see [`Workflow::post_note`](crate::Workflow::post_note).
    pub fn post(&self, note: Note) -> Result<PostedNote, WorkflowError> {
        let context = self
            .context
            .upgrade()
            .ok_or_else(|| WorkflowError::BuilderInvalid("the workflow was dropped".to_string()))?;
        let posted = context.notes.post(&context, note)?;
        // Saved at once while agents run, as a budget change is.
        if let Some(persistence) = context.notes.events.clone()
            && persistence.has_live_runs()
            && let Ok(runtime) = tokio::runtime::Handle::try_current()
        {
            let context = context.clone();
            runtime.spawn(async move {
                if let Err(error) = persistence.write_live_checkpoint(&context, false).await {
                    log::warn!(
                        "Workflow '{}': could not save the posted note: {error}",
                        context.workflow_name
                    );
                }
            });
        }
        Ok(posted)
    }

    /// The number of the newest note agent run `agent_run_id` has been
    /// given, in its prompt or at a turn boundary (notes are given oldest
    /// first, so it has been given every earlier note it receives); 0 when
    /// it has been given none. A run gets its notes in the context update
    /// [`AgentEvent::ContextInjected`](arko_agents::AgentEvent::ContextInjected)
    /// reports, whose ids name them (`note:<number>`).
    pub fn given(&self, agent_run_id: Uuid) -> u64 {
        self.context
            .upgrade()
            .map_or(0, |context| context.notes.given(agent_run_id))
    }

    /// Every note posted, oldest first; see
    /// [`Workflow::notes`](crate::Workflow::notes).
    pub fn notes(&self) -> Vec<PostedNote> {
        self.context
            .upgrade()
            .map(|context| context.notes.notes(&context))
            .unwrap_or_default()
    }

    /// A posted note's image; see
    /// [`Workflow::note_image`](crate::Workflow::note_image).
    pub fn image(&self, image: &NoteImage) -> Result<arko_ai::Image, WorkflowError> {
        let context = self
            .context
            .upgrade()
            .ok_or_else(|| WorkflowError::BuilderInvalid("the workflow was dropped".to_string()))?;
        context
            .notes
            .image(image)
            .map_err(WorkflowError::Persistence)
    }
}

impl std::fmt::Debug for NotesHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NotesHandle")
            .field("live", &(self.context.strong_count() > 0))
            .finish()
    }
}

/// A turn context source that adds the notes an agent run hasn't been given
/// to `inner`'s context, and holds requests as `inner` does.
pub(crate) struct WithNotes<S> {
    pub inner: S,
    pub context: Arc<SharedContext>,
    pub agent_name: String,
    /// The run is a completion verifier's review: it gets only the notes
    /// that name it.
    pub review: bool,
}

impl<S: arko_agents::TurnContextSource> arko_agents::TurnContextSource for WithNotes<S> {
    fn turn_context(&self, task_id: Uuid) -> Option<String> {
        self.context_update(task_id).map(|update| update.content)
    }

    /// The inner source's context, the notes the run hasn't been given
    /// (`note:<number>`), and its subagent mail (`mail:<id>`).
    fn context_update(&self, task_id: Uuid) -> Option<arko_agents::TurnContext> {
        let mut parts = Vec::new();
        let mut ids = Vec::new();
        let mut images = Vec::new();
        if let Some(inner) = self.inner.context_update(task_id) {
            parts.push(inner.content);
            ids.extend(inner.ids);
            images.extend(inner.images);
        }
        if let Some((notes, seqs, note_images)) = self.context.notes.update_with_seqs(
            &self.context,
            &self.agent_name,
            task_id,
            self.review,
        ) {
            parts.push(notes);
            ids.extend(seqs.into_iter().map(|seq| format!("note:{seq}")));
            images.extend(note_images);
        }
        if !self.review
            && let Some((mail, mail_ids)) = self.context.subagents.update(task_id)
        {
            parts.push(mail);
            ids.extend(mail_ids);
        }
        (!parts.is_empty()).then(|| arko_agents::TurnContext {
            content: parts.join("\n\n"),
            ids,
            images,
        })
    }

    /// An agent message's delivery: `message:<id>`.
    fn identify(&self, content: &str) -> Option<String> {
        let rest = content.strip_prefix("Message ")?;
        let (id, rest) = rest.split_once(' ')?;
        let id = Uuid::parse_str(id).ok()?;
        rest.starts_with("from agent '")
            .then(|| format!("message:{id}"))
    }

    fn hold_request(&self, task_id: Uuid) -> Option<String> {
        self.inner.hold_request(task_id)
    }
}

#[cfg(test)]
mod tests;
