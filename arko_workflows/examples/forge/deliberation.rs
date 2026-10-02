//! Deliberation: one-way-door decisions are proposed independently,
//! criticised, answered, tested and ruled on, instead of being made in one
//! turn of the architect. See `README.md` ("Deliberation").
//!
//! The architect frames a decision with `request_deliberation` and ends its
//! turn. The forge then runs each stage as agent runs: proposers on different
//! models, a critic per proposal, the proposers' rebuttals (in their own
//! conversations), a second critique round, spikes on contested claims, and
//! a ruling by judges or by the architect itself. Every agent submits its
//! part through a tool into the ledger (`deliberations.json` in the state
//! directory) the moment it has it, so a resumed forge continues where it
//! stopped and never deliberates twice. The decision record goes to the
//! crate's `decisions/` folder and back to the architect.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::Poll;

use arko_agents::runtime::event::AgentEvent;
use arko_agents::{
    ErrorRecoveryCause, TaskContext, TaskState, Tool, ToolDefinition, ToolError, ToolRecovery,
    ToolTypeDetails,
};
use arko_ai::ErrorKind;
use arko_workflows::prelude::*;
use async_trait::async_trait;
use clap::ValueEnum;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{ModelSpec, prompts, within_budget};

pub const PROPOSER: &str = "proposer";
pub const CRITIC: &str = "critic";
pub const SPIKER: &str = "spiker";
pub const JUDGE: &str = "judge";

/// The registered name of slot `index` (from 0) of `role`: `proposer-1`.
pub fn agent(role: &str, index: usize) -> String {
    format!("{role}-{}", index + 1)
}

// ---------------------------------------------------------------------------
// Settings

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub enum Level {
    /// Two proposers, one critique round, no spikes; the architect rules;
    /// only the top-level architecture.
    Light,
    /// Four proposers, two critique rounds, up to three spikes per decision;
    /// a separate judge rules; every one-way door.
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub enum Gate {
    /// Only the top-level architecture, framed as one decision.
    Top,
    /// Every one-way door, each as its own decision.
    OneWay,
}

/// The deliberation flags, as given. `--deliberate` turns it on and picks
/// the preset; the others override the preset's values.
#[derive(Debug, Clone, Default, clap::Args, Serialize, Deserialize)]
pub struct Flags {
    /// Deliberate on one-way-door decisions: `light` or `full`. Off when
    /// omitted.
    #[arg(long, value_enum)]
    #[serde(default)]
    pub deliberate: Option<Level>,

    /// How many proposers draft a design for each decision.
    #[arg(long, requires = "deliberate", value_parser = clap::value_parser!(u32).range(1..=8))]
    #[serde(default)]
    pub proposers: Option<u32>,

    /// The proposers' models, comma-separated, each `id` or `id:effort`
    /// (`gpt-6.1-sol:xhigh`); used in turn when there are more proposers.
    #[arg(
        long,
        requires = "deliberate",
        value_delimiter = ',',
        value_name = "MODELS"
    )]
    #[serde(default)]
    pub proposer_models: Vec<String>,

    /// The critics' models, in proposal order; by default each proposal's
    /// critic runs on the next proposer's model.
    #[arg(
        long,
        requires = "deliberate",
        value_delimiter = ',',
        value_name = "MODELS"
    )]
    #[serde(default)]
    pub critic_models: Vec<String>,

    /// 1: critique and rebuttal. 2: then a focused re-critique of what the
    /// rebuttals contested.
    #[arg(long, requires = "deliberate", value_parser = clap::value_parser!(u32).range(1..=2))]
    #[serde(default)]
    pub critique_rounds: Option<u32>,

    /// The most spikes (small experiments on contested claims) per decision.
    #[arg(long, requires = "deliberate", value_parser = clap::value_parser!(u32).range(0..=8))]
    #[serde(default)]
    pub spikes: Option<u32>,

    /// The model that runs spikes; the architect's when omitted.
    #[arg(long, requires = "deliberate", value_name = "MODEL")]
    #[serde(default)]
    pub spike_model: Option<String>,

    /// Who rules: `lead` (the architect), or one or more models,
    /// comma-separated (several vote as a panel).
    #[arg(
        long,
        requires = "deliberate",
        value_delimiter = ',',
        value_name = "lead|MODELS"
    )]
    #[serde(default)]
    pub judges: Vec<String>,

    /// Which decisions are deliberated.
    #[arg(long, requires = "deliberate", value_enum)]
    #[serde(default)]
    pub gate: Option<Gate>,

    /// The most decisions deliberated in a run (reopening one doesn't count).
    #[arg(long, requires = "deliberate", value_parser = clap::value_parser!(u32).range(1..=16))]
    #[serde(default)]
    pub max_decisions: Option<u32>,
}

/// A run's deliberation settings, resolved from the flags when it starts
/// and kept in its manifest, so a resume uses the same.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub level: Level,
    pub proposers: Vec<ModelSpec>,
    /// One per proposer: the critic of proposal `i`.
    pub critics: Vec<ModelSpec>,
    pub rounds: u32,
    pub spikes: u32,
    pub spiker: ModelSpec,
    /// Empty: the architect rules.
    pub judges: Vec<ModelSpec>,
    pub gate: Gate,
    pub max_decisions: u32,
}

impl Flags {
    /// The settings for a new run, or `None` without `--deliberate`.
    /// `openai`: whether OpenAI models can be reached (a ChatGPT login or a
    /// key); the presets use Claude models only without it.
    pub fn resolve(&self, architect: &ModelSpec, openai: bool) -> Result<Option<Config>, String> {
        let Some(level) = self.deliberate else {
            return Ok(None);
        };
        let full = level == Level::Full;
        let preset: &[&str] = match (full, openai) {
            (false, true) => &["claude-opus-5-5:xhigh", "gpt-6.1-sol:xhigh"],
            (true, true) => &[
                "claude-opus-5-5:xhigh",
                "gpt-6.1-sol:xhigh",
                "claude-fable-5-1:xhigh",
                "gpt-6-astra:xhigh",
            ],
            (_, false) => &["claude-opus-5-5:xhigh", "claude-fable-5-1:xhigh"],
        };
        let specs = |given: &[String]| -> Result<Vec<ModelSpec>, String> {
            given.iter().map(|text| text.parse()).collect()
        };
        let models = if self.proposer_models.is_empty() {
            specs(
                &preset
                    .iter()
                    .map(|text| text.to_string())
                    .collect::<Vec<_>>(),
            )?
        } else {
            specs(&self.proposer_models)?
        };
        let count = match self.proposers {
            Some(count) => count as usize,
            None if !self.proposer_models.is_empty() => models.len(),
            None => {
                if full {
                    4
                } else {
                    2
                }
            }
        };
        let proposers: Vec<ModelSpec> = (0..count)
            .map(|index| models[index % models.len()].clone())
            .collect();
        let critics = if self.critic_models.is_empty() {
            // Each proposal is reviewed on another proposer's model.
            (0..count)
                .map(|index| proposers[(index + 1) % count].clone())
                .collect()
        } else {
            let models = specs(&self.critic_models)?;
            (0..count)
                .map(|index| models[index % models.len()].clone())
                .collect()
        };
        let judges = match self.judges.as_slice() {
            [] if full => vec!["claude-fable-5-1:xhigh".parse()?],
            [] => Vec::new(),
            [lead] if lead.eq_ignore_ascii_case("lead") => Vec::new(),
            given if given.iter().any(|judge| judge.eq_ignore_ascii_case("lead")) => {
                return Err(
                    "`--judges lead` means the architect rules alone; give it alone, or give models"
                        .into(),
                );
            }
            given => specs(given)?,
        };
        let gate = self
            .gate
            .unwrap_or(if full { Gate::OneWay } else { Gate::Top });
        Ok(Some(Config {
            level,
            proposers,
            critics,
            rounds: self.critique_rounds.unwrap_or(if full { 2 } else { 1 }),
            spikes: self.spikes.unwrap_or(if full { 3 } else { 0 }),
            spiker: match &self.spike_model {
                Some(model) => model.parse()?,
                None => architect.clone(),
            },
            judges,
            gate,
            max_decisions: self.max_decisions.unwrap_or(match gate {
                Gate::Top => 1,
                Gate::OneWay => 4,
            }),
        }))
    }
}

impl Config {
    /// The architect's instructions about deliberation.
    pub fn architect_section(&self, lessons: bool) -> String {
        let gate = match self.gate {
            Gate::Top => prompts::GATE_TOP,
            Gate::OneWay => prompts::GATE_ONE_WAY,
        };
        let mut process = format!(
            "The forge has {} proposers draft designs independently, on different models where it can; a critic reviews each design; the proposers answer the findings",
            self.proposers.len()
        );
        if self.rounds > 1 {
            process.push_str(", and the critics reply to what they contested");
        }
        if self.spikes > 0 {
            process.push_str("; contested claims that matter are tested with small experiments");
        }
        process.push_str(match self.judges.len() {
            0 => "; then you rule, with `submit_ruling`, on the findings and the design",
            1 => "; then a judge rules on the findings and picks a design",
            _ => "; then a panel of judges votes on the findings and the design",
        });
        process.push('.');
        prompts::ARCHITECT_DELIBERATION
            .replace("{gate}", gate)
            .replace("{process}", &process)
            .replace("{max_decisions}", &self.max_decisions.to_string())
            .replace(
                "{lessons}",
                if lessons {
                    prompts::DELIBERATION_LESSONS
                } else {
                    ""
                },
            )
    }

    /// The agents to register: `(name, role, slot, model)`.
    pub fn agents(&self) -> Vec<(String, &'static str, usize, &ModelSpec)> {
        let spikers = vec![&self.spiker; self.spikes as usize];
        [
            (PROPOSER, self.proposers.iter().collect::<Vec<_>>()),
            (CRITIC, self.critics.iter().collect()),
            (SPIKER, spikers),
            (JUDGE, self.judges.iter().collect()),
        ]
        .into_iter()
        .flat_map(|(role, models)| {
            models
                .into_iter()
                .enumerate()
                .map(move |(slot, model)| (agent(role, slot), role, slot, model))
        })
        .collect()
    }

    /// What each role's agents submit.
    pub fn parts(role: &str) -> &'static [Part] {
        match role {
            PROPOSER => &[Part::Proposal, Part::Rebuttal],
            CRITIC => &[Part::Findings, Part::Reply],
            SPIKER => &[Part::SpikeResult],
            _ => &[Part::Ruling],
        }
    }

    fn judged_by(&self) -> String {
        match self.judges.as_slice() {
            [] => "the architect".to_string(),
            [judge] => format!("a judge ({judge})"),
            judges => format!(
                "a panel of {} ({})",
                judges.len(),
                judges
                    .iter()
                    .map(ModelSpec::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// The ledger

/// Every deliberation of the run, saved in `deliberations.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Ledger {
    #[serde(default)]
    pub decisions: Vec<Decision>,
    /// The decision whose stages are running: the one the agents' submit
    /// tools record into.
    #[serde(default)]
    pub active: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    /// `D1`, `D2`, …
    pub id: String,
    pub frame: Frame,
    /// The decision this one reopens.
    #[serde(default)]
    pub reopens: Option<String>,
    /// The decision that reopened this one.
    #[serde(default)]
    pub reopened_by: Option<String>,
    /// By proposer slot.
    #[serde(default)]
    pub proposals: Vec<Option<Proposal>>,
    /// The critic's findings on each proposal, by slot.
    #[serde(default)]
    pub findings: Vec<Option<Vec<Finding>>>,
    #[serde(default)]
    pub rebuttals: Vec<Option<Rebuttal>>,
    /// The critic's replies to what the rebuttal contested, by slot.
    #[serde(default)]
    pub replies: Vec<Option<Vec<Reply>>>,
    /// `None` until the contested claims to test are chosen.
    #[serde(default)]
    pub spikes: Option<Vec<Spike>>,
    /// By judge slot; the architect's ruling is slot 0.
    #[serde(default)]
    pub rulings: Vec<Option<Ruling>>,
    #[serde(default)]
    pub outcome: Option<Outcome>,
    /// The architect has been given the outcome.
    #[serde(default)]
    pub delivered: bool,
    /// The architect's run and its model requests when the forge continued
    /// it with the outcome. If the forge stopped before that continuation
    /// was saved, the run is still finished with no more requests than these,
    /// and the outcome is given again.
    #[serde(default)]
    pub delivered_at: Option<(uuid::Uuid, usize)>,
}

/// The decision as the architect framed it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    /// A short name for it (records, file names, log lines); see
    /// [`name`](Self::name).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub question: String,
    #[serde(default)]
    pub context: String,
    #[serde(default)]
    pub constraints: Vec<String>,
    #[serde(default)]
    pub evidence: Vec<String>,
    #[serde(default)]
    pub criteria: Vec<String>,
}

impl Frame {
    /// The decision's short name: its title, or else (a frame without one,
    /// such as an older ledger's) its question's first sentence, cut to 100
    /// characters. The whole question stays in the record's "The decision".
    pub fn name(&self) -> String {
        let title = self.title.as_deref().map(str::trim).unwrap_or_default();
        if !title.is_empty() {
            return crate::cut(title, 100);
        }
        let question = self.question.trim();
        let line = question.lines().next().unwrap_or_default();
        let end = line
            .char_indices()
            .find(|&(at, c)| {
                matches!(c, '.' | '?' | '!')
                    && line[at + c.len_utf8()..].starts_with(char::is_whitespace)
            })
            .map_or(line.len(), |(at, c)| at + c.len_utf8());
        crate::cut(&line[..end], 100)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    /// The model that drafted it (the record names it; critics and judges
    /// don't see it).
    #[serde(default)]
    pub model: String,
    pub title: String,
    pub design: String,
    #[serde(default)]
    pub assumptions: Vec<String>,
    #[serde(default)]
    pub risks: Vec<String>,
    /// The evidence that would prove the design wrong.
    #[serde(default)]
    pub falsifiers: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Critical,
    Major,
    Minor,
}

impl Severity {
    fn label(self) -> &'static str {
        match self {
            Self::Critical => "critical",
            Self::Major => "major",
            Self::Minor => "minor",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    /// `P1-F1`: proposal 1, finding 1.
    pub id: String,
    pub severity: Severity,
    pub claim: String,
    pub evidence: String,
    /// The concrete circumstance in which the problem occurs.
    pub trigger: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Answer {
    Accept,
    /// Right in part: the design is revised for that part, and the rest is
    /// contested (the argument says which).
    AcceptInPart,
    Incorrect,
    NotApplicable,
    OutOfScope,
    Implausible,
}

impl Answer {
    fn label(self) -> &'static str {
        match self {
            Self::Accept => "accepted",
            Self::AcceptInPart => "accepted in part, the rest contested",
            Self::Incorrect => "contested as incorrect",
            Self::NotApplicable => "contested as not applicable",
            Self::OutOfScope => "contested as out of scope",
            Self::Implausible => "contested as implausible",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub finding: String,
    pub answer: Answer,
    pub argument: String,
    /// An experiment that would settle the point.
    #[serde(default)]
    pub settle_by: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rebuttal {
    pub responses: Vec<Response>,
    /// The whole design, revised for the findings it accepts.
    #[serde(default)]
    pub revised_design: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stance {
    Maintain,
    Withdraw,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reply {
    pub finding: String,
    pub stance: Stance,
    pub argument: String,
    #[serde(default)]
    pub settle_by: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spike {
    /// The finding it tests.
    pub finding: String,
    #[serde(default)]
    pub result: Option<SpikeResult>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpikeOutcome {
    FindingHolds,
    FindingRefuted,
    Inconclusive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeResult {
    pub outcome: SpikeOutcome,
    pub what_ran: String,
    pub evidence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graft {
    pub from: String,
    pub change: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alternative {
    pub proposal: String,
    pub why_not: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingRuling {
    pub finding: String,
    pub upheld: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ruling {
    /// The proposal chosen: `P2`.
    pub chosen: String,
    pub rationale: String,
    #[serde(default)]
    pub grafts: Vec<Graft>,
    #[serde(default)]
    pub alternatives: Vec<Alternative>,
    /// On the contested findings.
    #[serde(default)]
    pub rulings: Vec<FindingRuling>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Outcome {
    Decided {
        /// The record, relative to the crate.
        record: String,
        /// When it was decided (`2026-09-30`).
        #[serde(default)]
        date: String,
        /// The ruling (a panel's, merged).
        ruling: Ruling,
        /// A panel's votes for each proposal.
        #[serde(default)]
        votes: BTreeMap<String, u32>,
    },
    Failed {
        reason: String,
    },
}

/// Where a finding stands after the rebuttal and the critic's reply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Standing {
    /// The proposer accepted it and revised the design.
    Conceded,
    /// The critic withdrew it.
    Withdrawn,
    /// Still disputed: the judges rule on it.
    Contested,
}

impl Decision {
    fn new(id: String, frame: Frame, reopens: Option<String>, proposers: usize) -> Self {
        Self {
            id,
            frame,
            reopens,
            reopened_by: None,
            proposals: vec![None; proposers],
            findings: vec![None; proposers],
            rebuttals: vec![None; proposers],
            replies: vec![None; proposers],
            spikes: None,
            rulings: Vec::new(),
            outcome: None,
            delivered: false,
            delivered_at: None,
        }
    }

    pub fn proposal_id(slot: usize) -> String {
        format!("P{}", slot + 1)
    }

    /// The proposal slot of a proposal id (`P2`) or a finding id (`P2-F1`).
    fn slot_of(id: &str) -> Option<usize> {
        let digits: String = id
            .trim()
            .strip_prefix(['P', 'p'])?
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        digits.parse::<usize>().ok()?.checked_sub(1)
    }

    /// The proposals that were submitted, with their slots.
    pub fn submitted(&self) -> impl Iterator<Item = (usize, &Proposal)> {
        self.proposals
            .iter()
            .enumerate()
            .filter_map(|(slot, proposal)| proposal.as_ref().map(|proposal| (slot, proposal)))
    }

    fn findings_of(&self, slot: usize) -> &[Finding] {
        self.findings
            .get(slot)
            .and_then(Option::as_ref)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    fn response(&self, finding: &Finding) -> Option<&Response> {
        let slot = Self::slot_of(&finding.id)?;
        self.rebuttals
            .get(slot)?
            .as_ref()?
            .responses
            .iter()
            .find(|response| response.finding == finding.id)
    }

    fn reply(&self, finding: &Finding) -> Option<&Reply> {
        let slot = Self::slot_of(&finding.id)?;
        self.replies
            .get(slot)?
            .as_ref()?
            .iter()
            .find(|reply| reply.finding == finding.id)
    }

    pub fn standing(&self, finding: &Finding) -> Standing {
        if self
            .response(finding)
            .is_some_and(|response| response.answer == Answer::Accept)
        {
            Standing::Conceded
        } else if self
            .reply(finding)
            .is_some_and(|reply| reply.stance == Stance::Withdraw)
        {
            Standing::Withdrawn
        } else {
            Standing::Contested
        }
    }

    /// Every finding on a submitted proposal.
    pub fn all_findings(&self) -> impl Iterator<Item = &Finding> {
        self.submitted()
            .flat_map(|(slot, _)| self.findings_of(slot).iter())
    }

    pub fn contested(&self) -> Vec<&Finding> {
        self.all_findings()
            .filter(|finding| self.standing(finding) == Standing::Contested)
            .collect()
    }

    /// The design as it stands after the rebuttal.
    fn final_design<'a>(&'a self, slot: usize, proposal: &'a Proposal) -> &'a str {
        self.rebuttals
            .get(slot)
            .and_then(Option::as_ref)
            .and_then(|rebuttal| rebuttal.revised_design.as_deref())
            .unwrap_or(&proposal.design)
    }

    /// Up to `budget` contested findings that matter (critical or major),
    /// the most severe first.
    pub fn choose_spikes(&self, budget: usize) -> Vec<Spike> {
        let mut candidates: Vec<&Finding> = self
            .contested()
            .into_iter()
            .filter(|finding| finding.severity != Severity::Minor)
            .collect();
        candidates.sort_by_key(|finding| finding.severity);
        candidates
            .into_iter()
            .take(budget)
            .map(|finding| Spike {
                finding: finding.id.clone(),
                result: None,
            })
            .collect()
    }

    fn finding(&self, id: &str) -> Option<&Finding> {
        self.all_findings().find(|finding| finding.id == id)
    }
}

/// Merges the judges' rulings: the design with the most votes (a tie goes
/// to the judge listed first), and on each contested finding the majority
/// (a tie upholds it). The rationale, grafts and reasons for the losers
/// come from the judges who chose the winner.
pub fn merge(rulings: &[Ruling], contested: &[String]) -> (Ruling, BTreeMap<String, u32>) {
    let mut votes: BTreeMap<String, u32> = BTreeMap::new();
    for ruling in rulings {
        *votes.entry(ruling.chosen.clone()).or_default() += 1;
    }
    let most = votes.values().copied().max().unwrap_or(0);
    let chosen = rulings
        .iter()
        .map(|ruling| ruling.chosen.clone())
        .find(|chosen| votes[chosen] == most)
        .unwrap_or_default();
    if rulings.len() == 1 {
        return (rulings[0].clone(), votes);
    }
    let winners: Vec<&Ruling> = rulings
        .iter()
        .filter(|ruling| ruling.chosen == chosen)
        .collect();
    let rationale = winners
        .iter()
        .map(|ruling| ruling.rationale.trim())
        .collect::<Vec<_>>()
        .join("\n\n");
    let grafts = winners
        .iter()
        .flat_map(|ruling| ruling.grafts.iter().cloned())
        .collect();
    let mut alternatives: Vec<Alternative> = Vec::new();
    for ruling in &winners {
        for alternative in &ruling.alternatives {
            match alternatives
                .iter_mut()
                .find(|known| known.proposal == alternative.proposal)
            {
                Some(known) => {
                    known.why_not = format!("{}\n\n{}", known.why_not, alternative.why_not)
                }
                None => alternatives.push(alternative.clone()),
            }
        }
    }
    let rulings_on = contested
        .iter()
        .map(|finding| {
            let said: Vec<&FindingRuling> = rulings
                .iter()
                .filter_map(|ruling| ruling.rulings.iter().find(|r| &r.finding == finding))
                .collect();
            let upheld = said.iter().filter(|ruling| ruling.upheld).count();
            let upheld = upheld * 2 >= said.len();
            FindingRuling {
                finding: finding.clone(),
                upheld,
                reason: said
                    .iter()
                    .filter(|ruling| ruling.upheld == upheld)
                    .map(|ruling| ruling.reason.trim())
                    .collect::<Vec<_>>()
                    .join(" / "),
            }
        })
        .collect();
    (
        Ruling {
            chosen,
            rationale,
            grafts,
            alternatives,
            rulings: rulings_on,
        },
        votes,
    )
}

/// The ledger and the file it's kept in.
#[derive(Debug)]
pub struct Store {
    path: PathBuf,
    /// The workflow's crate: records go to its `decisions/` folder.
    crate_dir: PathBuf,
    ledger: Mutex<Ledger>,
}

impl Store {
    pub fn open(state_dir: &Path, crate_dir: &Path) -> io::Result<Self> {
        let path = state_dir.join("deliberations.json");
        let ledger = match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(io::Error::other)?,
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ledger::default(),
            Err(error) => return Err(error),
        };
        Ok(Self {
            path,
            crate_dir: crate_dir.to_path_buf(),
            ledger: Mutex::new(ledger),
        })
    }

    pub fn get(&self) -> Ledger {
        self.ledger.lock().unwrap().clone()
    }

    pub fn decision(&self, id: &str) -> Option<Decision> {
        self.get()
            .decisions
            .into_iter()
            .find(|decision| decision.id == id)
    }

    /// Applies `change` and saves the result before anyone sees it.
    pub fn update<R>(&self, change: impl FnOnce(&mut Ledger) -> R) -> io::Result<R> {
        let mut ledger = self.ledger.lock().unwrap();
        let mut next = ledger.clone();
        let result = change(&mut next);
        write_atomically(
            &self.path,
            &serde_json::to_vec_pretty(&next).map_err(io::Error::other)?,
        )?;
        *ledger = next;
        Ok(result)
    }

    /// Records a part into decision `id` (the active one when `None`),
    /// checked by `apply`. A settled decision takes no more parts.
    fn record_into(
        &self,
        id: Option<&str>,
        apply: impl FnOnce(&mut Decision) -> Result<(), String>,
    ) -> Result<String, ToolError> {
        let outcome = self.update(|ledger| {
            let id = id.map(str::to_string).or_else(|| ledger.active.clone());
            let decision = ledger
                .decisions
                .iter_mut()
                .find(|decision| Some(&decision.id) == id.as_ref())
                .ok_or_else(|| "no deliberation is running".to_string())?;
            if decision.outcome.is_some() {
                return Err(format!(
                    "{} is already settled; there's nothing more to submit",
                    decision.id
                ));
            }
            apply(decision).map(|()| decision.id.clone())
        });
        match outcome {
            Ok(Ok(id)) => Ok(id),
            Ok(Err(message)) => Err(ToolError::ExecutionFailed(message)),
            Err(error) => Err(ToolError::ExecutionFailed(format!(
                "couldn't save it: {error}"
            ))),
        }
    }

    /// Settles decision `id` from its rulings: merges them, writes the
    /// record, and records the outcome. Returns the record's path relative
    /// to the crate.
    pub fn decide(&self, id: &str, config: &Config) -> io::Result<String> {
        let ledger = self.get();
        let decision = ledger
            .decisions
            .iter()
            .find(|decision| decision.id == id)
            .ok_or_else(|| io::Error::other(format!("no decision {id}")))?;
        let rulings: Vec<Ruling> = decision.rulings.iter().flatten().cloned().collect();
        let contested: Vec<String> = decision
            .contested()
            .into_iter()
            .map(|finding| finding.id.clone())
            .collect();
        let (ruling, votes) = merge(&rulings, &contested);
        let name = format!("{id}-{}.md", slug(&decision.frame.name()));
        let record = format!("decisions/{name}");
        let mut decided = decision.clone();
        decided.outcome = Some(Outcome::Decided {
            record: record.clone(),
            date: chrono::Utc::now().format("%Y-%m-%d").to_string(),
            ruling,
            votes,
        });
        std::fs::create_dir_all(self.crate_dir.join("decisions"))?;
        std::fs::write(
            self.crate_dir.join(&record),
            render_record(&decided, config),
        )?;
        // A reopened decision's record points to the one that replaces it.
        if let Some(earlier) = decided
            .reopens
            .as_ref()
            .and_then(|earlier| ledger.decisions.iter().find(|d| &d.id == earlier))
            && let Some(Outcome::Decided { record: path, .. }) = &earlier.outcome
        {
            let mut earlier = earlier.clone();
            earlier.reopened_by = Some(id.to_string());
            let _ = std::fs::write(self.crate_dir.join(path), render_record(&earlier, config));
        }
        let outcome = decided.outcome.clone();
        let reopens = decided.reopens.clone();
        self.update(|ledger| {
            for decision in &mut ledger.decisions {
                if decision.id == id {
                    decision.outcome = outcome.clone();
                } else if Some(&decision.id) == reopens.as_ref() {
                    decision.reopened_by = Some(id.to_string());
                }
            }
        })?;
        Ok(record)
    }

    fn fail(&self, id: &str, reason: String) -> io::Result<()> {
        self.update(|ledger| {
            if let Some(decision) = ledger.decisions.iter_mut().find(|d| d.id == id) {
                decision.outcome = Some(Outcome::Failed { reason });
            }
        })
    }
}

/// Writes `bytes` to a temporary file beside `path`, syncs it, and renames
/// it over `path`, so a crash leaves the old content or the new.
fn write_atomically(path: &Path, bytes: &[u8]) -> io::Result<()> {
    use std::io::Write as _;
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(format!(".{}.tmp", std::process::id()));
    let temporary = PathBuf::from(temporary);
    {
        let mut file = std::fs::File::create(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    std::fs::rename(&temporary, path)
}

/// A file name from a question: its first words, lowercase, with dashes.
fn slug(text: &str) -> String {
    let words: Vec<String> = text
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .take(6)
        .map(str::to_ascii_lowercase)
        .collect();
    if words.is_empty() {
        "decision".into()
    } else {
        words.join("-")
    }
}

// ---------------------------------------------------------------------------
// Rendering

fn list(items: &[String]) -> String {
    if items.is_empty() {
        return "- (none given)\n".into();
    }
    items
        .iter()
        .map(|item| format!("- {}\n", item.trim()))
        .collect()
}

pub fn render_frame(frame: &Frame) -> String {
    let mut text = format!("**Question:** {}\n\n", frame.question.trim());
    if !frame.context.trim().is_empty() {
        let _ = write!(text, "**Context:** {}\n\n", frame.context.trim());
    }
    let _ = write!(
        text,
        "**Constraints:**\n{}\n**Evidence so far:**\n{}\n**Criteria for a good answer:**\n{}",
        list(&frame.constraints),
        list(&frame.evidence),
        list(&frame.criteria)
    );
    text
}

fn render_design(title: &str, design: &str, proposal: &Proposal) -> String {
    format!(
        "### {title}: {}\n\n{}\n\n**Assumptions:**\n{}\n**Risks:**\n{}\n**What would prove it wrong:**\n{}",
        proposal.title.trim(),
        design.trim(),
        list(&proposal.assumptions),
        list(&proposal.risks),
        list(&proposal.falsifiers)
    )
}

fn render_finding(finding: &Finding) -> String {
    format!(
        "- **{}** ({}): {}\n  - Evidence: {}\n  - Trigger: {}\n",
        finding.id,
        finding.severity.label(),
        finding.claim.trim(),
        finding.evidence.trim(),
        finding.trigger.trim()
    )
}

pub fn render_findings(findings: &[Finding]) -> String {
    findings.iter().map(render_finding).collect()
}

fn render_response(response: &Response) -> String {
    let mut text = format!(
        "- **{}** {}: {}\n",
        response.finding,
        response.answer.label(),
        response.argument.trim()
    );
    if let Some(settle_by) = response
        .settle_by
        .as_deref()
        .filter(|s| !s.trim().is_empty())
    {
        let _ = writeln!(text, "  - Would be settled by: {}", settle_by.trim());
    }
    text
}

fn render_reply(reply: &Reply) -> String {
    let stance = match reply.stance {
        Stance::Maintain => "maintained",
        Stance::Withdraw => "withdrawn",
    };
    let mut text = format!(
        "- **{}** {stance} by the critic: {}\n",
        reply.finding,
        reply.argument.trim()
    );
    if let Some(settle_by) = reply.settle_by.as_deref().filter(|s| !s.trim().is_empty()) {
        let _ = writeln!(text, "  - Would be settled by: {}", settle_by.trim());
    }
    text
}

fn render_spike(spike: &Spike) -> String {
    match &spike.result {
        Some(result) => format!(
            "- Spike on **{}**: {}.\n  - Ran: {}\n  - Evidence: {}\n",
            spike.finding,
            match result.outcome {
                SpikeOutcome::FindingHolds => "the finding holds",
                SpikeOutcome::FindingRefuted => "the finding is refuted",
                SpikeOutcome::Inconclusive => "inconclusive",
            },
            result.what_ran.trim(),
            result.evidence.trim()
        ),
        None => format!("- Spike on **{}**: didn't finish.\n", spike.finding),
    }
}

/// Everything a judge needs, with the proposals by id only.
pub fn render_dossier(decision: &Decision) -> String {
    let mut text = format!(
        "## The decision\n\n{}\n\n## Proposals\n\n",
        render_frame(&decision.frame)
    );
    for (slot, proposal) in decision.submitted() {
        let _ = write!(
            text,
            "{}\n\n",
            render_design(
                &Decision::proposal_id(slot),
                decision.final_design(slot, proposal),
                proposal
            )
        );
    }
    text.push_str("## Findings and answers\n\n");
    for (slot, _) in decision.submitted() {
        let findings = decision.findings_of(slot);
        let _ = writeln!(text, "### On {}\n", Decision::proposal_id(slot));
        if decision
            .findings
            .get(slot)
            .and_then(Option::as_ref)
            .is_none()
        {
            text.push_str("(Its critique didn't finish.)\n\n");
            continue;
        }
        if findings.is_empty() {
            text.push_str("(No findings.)\n\n");
            continue;
        }
        for finding in findings {
            text.push_str(&render_finding(finding));
            match decision.response(finding) {
                Some(response) => text.push_str(&indent(&render_response(response))),
                None => text.push_str("  - (No answer from the proposer.)\n"),
            }
            if let Some(reply) = decision.reply(finding) {
                text.push_str(&indent(&render_reply(reply)));
            }
            let _ = writeln!(
                text,
                "  - Standing: {}",
                match decision.standing(finding) {
                    Standing::Conceded => "conceded (the design was revised)",
                    Standing::Withdrawn => "withdrawn",
                    Standing::Contested => "contested: rule on it",
                }
            );
        }
        text.push('\n');
    }
    if let Some(spikes) = decision.spikes.as_ref().filter(|spikes| !spikes.is_empty()) {
        text.push_str("## Spikes\n\n");
        for spike in spikes {
            text.push_str(&render_spike(spike));
        }
        text.push('\n');
    }
    let contested: Vec<String> = decision
        .contested()
        .into_iter()
        .map(|finding| finding.id.clone())
        .collect();
    if contested.is_empty() {
        text.push_str(
            "## To rule on\n\nNo finding is contested: each was conceded (and the design revised) or withdrawn. Rule only on the design: choose one, graft from the others, and say why each other lost.\n",
        );
    } else {
        let _ = writeln!(
            text,
            "## To rule on\n\nContested findings, each needing a ruling: {}.",
            contested.join(", ")
        );
    }
    text
}

fn indent(text: &str) -> String {
    text.lines().map(|line| format!("  {line}\n")).collect()
}

/// The decision record: what was decided, why, and what lost.
pub fn render_record(decision: &Decision, config: &Config) -> String {
    let mut text = format!("# {}: {}\n\n", decision.id, decision.frame.name());
    if let Some(later) = &decision.reopened_by {
        let _ = write!(
            text,
            "> Reopened and replaced by {later}: see its record in this folder.\n\n"
        );
    }
    if let Some(earlier) = &decision.reopens {
        let _ = write!(text, "Reopens {earlier}, with new evidence.\n\n");
    }
    let Some(Outcome::Decided {
        ruling,
        votes,
        date,
        ..
    }) = &decision.outcome
    else {
        text.push_str("Not decided.\n");
        return text;
    };
    let spikes = decision.spikes.as_ref().map(Vec::len).unwrap_or(0);
    let _ = write!(
        text,
        "Settled by deliberation on {}: {} proposals, {} critique round{}, {spikes} spike{}; ruled by {}.\n\n",
        date,
        decision.submitted().count(),
        config.rounds,
        if config.rounds == 1 { "" } else { "s" },
        if spikes == 1 { "" } else { "s" },
        config.judged_by()
    );
    if votes.len() > 1 || votes.values().any(|count| *count > 1) {
        let _ = write!(
            text,
            "Votes: {}.\n\n",
            votes
                .iter()
                .map(|(proposal, count)| format!("{proposal} {count}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let _ = write!(
        text,
        "## The decision\n\n{}\n",
        render_frame(&decision.frame)
    );
    let chosen = Decision::slot_of(&ruling.chosen)
        .and_then(|slot| Some((slot, decision.proposals.get(slot)?.as_ref()?)));
    match chosen {
        Some((slot, proposal)) => {
            let _ = write!(
                text,
                "## Chosen: {} ({}), \"{}\"\n\n{}\n\n",
                ruling.chosen,
                proposal.model,
                proposal.title.trim(),
                ruling.rationale.trim()
            );
            if !ruling.grafts.is_empty() {
                text.push_str("### Changes grafted from other proposals\n\n");
                for graft in &ruling.grafts {
                    let _ = writeln!(text, "- From {}: {}", graft.from, graft.change.trim());
                }
                text.push('\n');
            }
            let _ = write!(
                text,
                "### The design\n\n{}\n\n**Assumptions:**\n{}\n**Risks:**\n{}\n**What would prove it wrong:**\n{}\n",
                decision.final_design(slot, proposal).trim(),
                list(&proposal.assumptions),
                list(&proposal.risks),
                list(&proposal.falsifiers)
            );
        }
        None => {
            let _ = write!(
                text,
                "## Chosen: {}\n\n{}\n\n",
                ruling.chosen,
                ruling.rationale.trim()
            );
        }
    }
    text.push_str("## Alternatives and why they lost\n\n");
    for (slot, proposal) in decision.submitted() {
        let id = Decision::proposal_id(slot);
        if id == ruling.chosen {
            continue;
        }
        let why = ruling
            .alternatives
            .iter()
            .find(|alternative| alternative.proposal == id)
            .map(|alternative| alternative.why_not.trim().to_string())
            .unwrap_or_else(|| "(no reason given)".into());
        let _ = writeln!(
            text,
            "- **{id}** ({}), \"{}\": {why}",
            proposal.model,
            proposal.title.trim()
        );
    }
    text.push_str("\n## Findings\n\n");
    let mut upheld = String::new();
    let mut dismissed = String::new();
    for finding in decision.all_findings() {
        let (keep, reason) = match decision.standing(finding) {
            Standing::Conceded => (
                true,
                "conceded by the proposer; the design was revised".into(),
            ),
            Standing::Withdrawn => (false, "withdrawn by the critic".into()),
            Standing::Contested => match ruling.rulings.iter().find(|r| r.finding == finding.id) {
                Some(ruling) => (ruling.upheld, ruling.reason.trim().to_string()),
                None => (true, "not ruled on; kept".into()),
            },
        };
        let line = format!(
            "- **{}** ({}) {}: {}\n  - Trigger: {}\n  - {}{}\n",
            finding.id,
            finding.severity.label(),
            if keep { "upheld" } else { "dismissed" },
            finding.claim.trim(),
            finding.trigger.trim(),
            decision
                .response(finding)
                .map(|response| format!("Proposer {}. ", response.answer.label()))
                .unwrap_or_default(),
            reason
        );
        if keep {
            upheld.push_str(&line);
        } else {
            dismissed.push_str(&line);
        }
    }
    let _ = write!(
        text,
        "### Upheld\n\n{}\n### Dismissed\n\n{}\n",
        if upheld.is_empty() {
            "(none)\n"
        } else {
            &upheld
        },
        if dismissed.is_empty() {
            "(none)\n"
        } else {
            &dismissed
        }
    );
    if let Some(spikes) = decision.spikes.as_ref().filter(|spikes| !spikes.is_empty()) {
        text.push_str("## Spikes\n\n");
        for spike in spikes {
            text.push_str(&render_spike(spike));
        }
    }
    text
}

// ---------------------------------------------------------------------------
// Tools

fn tool_definition(name: &str, description: &str, schema: Value) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: description.to_string(),
        output_based_example: None,
        parameters: Vec::new(),
        parallel_safe: false,
        tool_type_details: ToolTypeDetails::DeveloperDefined,
        // Arrays and objects, which `parameters` can't describe.
        raw_json_schema_override: Some(schema),
    }
}

/// The parameters as `T`. Some models send an array or object as its JSON
/// text; that is accepted too.
fn parse<T: DeserializeOwned>(params: HashMap<String, Value>) -> Result<T, ToolError> {
    let object = params
        .into_iter()
        .map(|(name, value)| {
            let value = match value {
                Value::String(text) if text.trim_start().starts_with(['[', '{']) => {
                    serde_json::from_str(&text).unwrap_or(Value::String(text))
                }
                value => value,
            };
            (name, value)
        })
        .collect();
    serde_json::from_value(Value::Object(object)).map_err(|error| ToolError::ParameterValidation {
        param_name: "parameters".into(),
        message: error.to_string(),
    })
}

fn strings(description: &str) -> Value {
    json!({"type": "array", "items": {"type": "string"}, "description": description})
}

fn text(description: &str) -> Value {
    json!({"type": "string", "description": description})
}

const SUBMITTED: &str =
    "Recorded. Now call complete_task with a one-line summary; that ends your part.";

/// Records a decision for the forge to deliberate, and ends the architect's
/// turn.
#[derive(Debug)]
pub struct RequestDeliberation {
    pub store: Arc<Store>,
    pub config: Config,
}

#[derive(Deserialize)]
struct Request {
    #[serde(default)]
    title: Option<String>,
    question: String,
    #[serde(default)]
    context: String,
    #[serde(default)]
    constraints: Vec<String>,
    #[serde(default)]
    evidence: Vec<String>,
    #[serde(default)]
    criteria: Vec<String>,
    #[serde(default)]
    reopens: Option<String>,
}

#[async_trait]
impl Tool for RequestDeliberation {
    fn definition(&self) -> ToolDefinition {
        tool_definition(
            "request_deliberation",
            "Have the forge deliberate a one-way-door decision: proposers on different models design it independently, critics review, the proposers answer, and it is ruled on. Frame it neutrally (the facts and criteria, not your preferred answer), then end your turn with complete_task; the decision record comes back in this conversation. To reopen a recorded decision once, give its id in `reopens` and the new evidence.",
            json!({
                "type": "object",
                "properties": {
                    "title": text("A short name for the decision, under 80 characters."),
                    "question": text("The decision, as a question with real alternatives."),
                    "context": text("Why it matters and what depends on it."),
                    "constraints": strings("What every answer must respect: from the use case, the user's answers, the framework."),
                    "evidence": strings("What you have confirmed that bears on it, with file references: facts, not preferences. When reopening: what went wrong."),
                    "criteria": strings("What a good answer achieves, most important first."),
                    "reopens": text("Optional: the id of a recorded decision to reopen (once per decision).")
                },
                "required": ["question", "constraints", "evidence", "criteria"]
            }),
        )
    }

    fn recovery_mode(&self) -> ToolRecovery {
        // A request already recorded is found again, not added twice.
        ToolRecovery::Rerun
    }

    async fn execute(
        &self,
        _context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let request: Request = parse(params)?;
        if request.question.trim().is_empty() {
            return Err(ToolError::MissingParameter {
                param_name: "question".into(),
            });
        }
        let reopens = request
            .reopens
            .map(|id| id.trim().to_ascii_uppercase())
            .filter(|id| !id.is_empty());
        let frame = Frame {
            title: request
                .title
                .map(|title| title.trim().to_string())
                .filter(|title| !title.is_empty()),
            question: request.question.trim().to_string(),
            context: request.context,
            constraints: request.constraints,
            evidence: request.evidence,
            criteria: request.criteria,
        };
        let proposers = self.config.proposers.len();
        let max = self.config.max_decisions as usize;
        let outcome = self.store.update(|ledger| {
            if let Some(known) = ledger
                .decisions
                .iter()
                .find(|d| d.frame.question == frame.question && !d.delivered)
            {
                return Ok(known.id.clone());
            }
            match &reopens {
                Some(earlier) => {
                    let decision = ledger
                        .decisions
                        .iter()
                        .find(|d| &d.id == earlier)
                        .ok_or_else(|| format!("there is no decision {earlier}"))?;
                    if !matches!(decision.outcome, Some(Outcome::Decided { .. })) {
                        return Err(format!("{earlier} wasn't decided, so it can't be reopened"));
                    }
                    if decision.reopens.is_some()
                        || ledger
                            .decisions
                            .iter()
                            .any(|d| d.reopens.as_ref() == Some(earlier))
                    {
                        return Err(format!(
                            "{earlier} was already reopened once; decide any further change yourself and record why in DESIGN.md"
                        ));
                    }
                }
                None => {
                    let new = ledger
                        .decisions
                        .iter()
                        .filter(|d| d.reopens.is_none())
                        .count();
                    if new >= max {
                        return Err(format!(
                            "this run's {max} deliberation{} used; decide this one yourself and record your reasoning in DESIGN.md",
                            if max == 1 { " is" } else { "s are" }
                        ));
                    }
                }
            }
            let id = format!("D{}", ledger.decisions.len() + 1);
            ledger.decisions.push(Decision::new(
                id.clone(),
                frame.clone(),
                reopens.clone(),
                proposers,
            ));
            Ok(id)
        });
        match outcome {
            Ok(Ok(id)) => Ok(format!(
                "Recorded as {id}. Frame any other decision to deliberate now with its own call; then end your turn with complete_task and a one-line summary. The forge deliberates them in the order you framed them and continues this conversation once they're all settled."
            )),
            Ok(Err(message)) => Err(ToolError::ExecutionFailed(message)),
            Err(error) => Err(ToolError::ExecutionFailed(format!(
                "couldn't save the request: {error}"
            ))),
        }
    }
}

/// What a submit tool records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    Proposal,
    Findings,
    Rebuttal,
    Reply,
    SpikeResult,
    Ruling,
}

/// A deliberation agent's submission, recorded into the active decision.
#[derive(Debug)]
pub struct Submit {
    pub part: Part,
    /// The agent's slot: the proposal it wrote or reviews, its spike, its
    /// seat on the panel.
    pub slot: usize,
    pub store: Arc<Store>,
    pub config: Config,
    /// The architect ruling in the lead's seat: the tool settles the
    /// decision and returns the record.
    pub lead: bool,
}

#[derive(Deserialize)]
struct NewFinding {
    severity: Severity,
    claim: String,
    evidence: String,
    trigger: String,
}

#[derive(Deserialize)]
struct Findings {
    findings: Vec<NewFinding>,
}

#[derive(Deserialize)]
struct Replies {
    responses: Vec<Reply>,
}

impl Submit {
    fn name(&self) -> &'static str {
        match self.part {
            Part::Proposal => "submit_proposal",
            Part::Findings => "submit_findings",
            Part::Rebuttal => "submit_rebuttal",
            Part::Reply => "submit_reply",
            Part::SpikeResult => "submit_spike_result",
            Part::Ruling => "submit_ruling",
        }
    }

    fn schema(&self) -> (&'static str, Value) {
        let severity = json!({"type": "string", "enum": ["critical", "major", "minor"]});
        match self.part {
            Part::Proposal => (
                "Submit your design for the decision. Submitting again replaces it.",
                json!({
                    "type": "object",
                    "properties": {
                        "title": text("A short name for the design."),
                        "design": text("The design in Markdown, concrete enough to build from: the arko primitives it uses (with file references), the agents and their roles, the state, the flow, how it recovers."),
                        "assumptions": strings("What the design rests on."),
                        "risks": strings("Where it could go wrong."),
                        "falsifiers": strings("The evidence that would prove it wrong.")
                    },
                    "required": ["title", "design", "assumptions", "risks", "falsifiers"]
                }),
            ),
            Part::Findings => (
                "Submit your review's findings (an empty list when there are none). Submitting again replaces them.",
                json!({
                    "type": "object",
                    "properties": {
                        "findings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "severity": severity,
                                    "claim": text("The problem, specifically."),
                                    "evidence": text("What shows it: code or docs with file:line, a known behaviour, a test."),
                                    "trigger": text("The concrete circumstance in which it actually occurs.")
                                },
                                "required": ["severity", "claim", "evidence", "trigger"]
                            }
                        }
                    },
                    "required": ["findings"]
                }),
            ),
            Part::Rebuttal => (
                "Submit your answer to every finding on your design. Submitting again replaces it.",
                json!({
                    "type": "object",
                    "properties": {
                        "responses": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "finding": text("The finding's id, e.g. P1-F2."),
                                    "answer": {"type": "string", "enum": ["accept", "accept_in_part", "incorrect", "not_applicable", "out_of_scope", "implausible"]},
                                    "argument": text("Why, with evidence."),
                                    "settle_by": text("Optional: a small experiment that would settle it.")
                                },
                                "required": ["finding", "answer", "argument"]
                            }
                        },
                        "revised_design": text("The whole design, revised for the findings you accept, in whole or in part. Required when you accept any.")
                    },
                    "required": ["responses"]
                }),
            ),
            Part::Reply => (
                "Submit your reply to each finding the proposer contested. Submitting again replaces it.",
                json!({
                    "type": "object",
                    "properties": {
                        "responses": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "finding": text("The finding's id."),
                                    "stance": {"type": "string", "enum": ["maintain", "withdraw"]},
                                    "argument": text("Why, answering the rebuttal, with evidence."),
                                    "settle_by": text("Optional: a small experiment that would settle it.")
                                },
                                "required": ["finding", "stance", "argument"]
                            }
                        }
                    },
                    "required": ["responses"]
                }),
            ),
            Part::SpikeResult => (
                "Submit what your experiment showed. Submitting again replaces it.",
                json!({
                    "type": "object",
                    "properties": {
                        "outcome": {"type": "string", "enum": ["finding_holds", "finding_refuted", "inconclusive"]},
                        "what_ran": text("The experiment: what you built and ran."),
                        "evidence": text("What it showed: output, file:line.")
                    },
                    "required": ["outcome", "what_ran", "evidence"]
                }),
            ),
            Part::Ruling => (
                "Submit your ruling: each contested finding upheld or dismissed, the chosen proposal, specific grafts from the others, and why each other proposal lost.",
                json!({
                    "type": "object",
                    "properties": {
                        "chosen": text("The chosen proposal's id, e.g. P2."),
                        "rationale": text("Why this design, citing the findings, answers and spikes."),
                        "grafts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "from": text("The proposal it comes from."),
                                    "change": text("The specific change to the chosen design.")
                                },
                                "required": ["from", "change"]
                            }
                        },
                        "alternatives": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "proposal": text("A proposal that wasn't chosen."),
                                    "why_not": text("Why it lost.")
                                },
                                "required": ["proposal", "why_not"]
                            }
                        },
                        "rulings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "finding": text("A contested finding's id."),
                                    "upheld": {"type": "boolean"},
                                    "reason": text("Why, and for a dismissal which: incorrect, not applicable, out of scope, or implausible.")
                                },
                                "required": ["finding", "upheld", "reason"]
                            }
                        }
                    },
                    "required": ["chosen", "rationale", "alternatives", "rulings"]
                }),
            ),
        }
    }

    /// Records the submission of a run whose first prompt is `prompt`: into
    /// the decision its marker names, or for the architect, the active one.
    fn record(&self, prompt: &str, params: HashMap<String, Value>) -> Result<String, ToolError> {
        let slot = self.slot;
        let target = decision_of(prompt);
        let target = target.as_deref();
        match self.part {
            Part::Proposal => {
                let mut proposal: Proposal = parse(params)?;
                proposal.model = self
                    .config
                    .proposers
                    .get(slot)
                    .map(ModelSpec::to_string)
                    .unwrap_or_default();
                // Each part is taken in its own stage only: a design replaced
                // after its review would leave the findings, and the critic's
                // reply, about a design the ledger no longer has.
                self.store.record_into(target, |decision| {
                    if decision.findings.get(slot).is_some_and(Option::is_some) {
                        return Err("your design has been reviewed, so it can't be replaced now: answer the findings with submit_rebuttal, with the revised design in its `revised_design`".into());
                    }
                    set(&mut decision.proposals, slot, proposal);
                    Ok(())
                })
            }
            Part::Findings => {
                let findings: Findings = parse(params)?;
                self.store.record_into(target, |decision| {
                    if decision.rebuttals.get(slot).is_some_and(Option::is_some) {
                        return Err("the proposer has answered your findings, so they can't be replaced now: reply to the ones it contested with submit_reply".into());
                    }
                    let prefix = Decision::proposal_id(slot);
                    let findings = findings
                        .findings
                        .into_iter()
                        .enumerate()
                        .map(|(index, finding)| Finding {
                            id: format!("{prefix}-F{}", index + 1),
                            severity: finding.severity,
                            claim: finding.claim,
                            evidence: finding.evidence,
                            trigger: finding.trigger,
                        })
                        .collect();
                    set(&mut decision.findings, slot, findings);
                    Ok(())
                })
            }
            Part::Rebuttal => {
                let rebuttal: Rebuttal = parse(params)?;
                self.store.record_into(target, |decision| {
                    if decision.findings.get(slot).is_none_or(Option::is_none) {
                        return Err("your design hasn't been reviewed yet: there are no findings to answer".into());
                    }
                    if decision.replies.get(slot).is_some_and(Option::is_some) {
                        return Err("the critic has replied to your answers, so they can't be changed now".into());
                    }
                    let expected: Vec<String> = decision
                        .findings_of(slot)
                        .iter()
                        .map(|finding| finding.id.clone())
                        .collect();
                    covers(
                        &expected,
                        rebuttal.responses.iter().map(|r| r.finding.as_str()),
                    )?;
                    if rebuttal
                        .responses
                        .iter()
                        .any(|r| matches!(r.answer, Answer::Accept | Answer::AcceptInPart))
                        && rebuttal
                            .revised_design
                            .as_deref()
                            .is_none_or(|design| design.trim().is_empty())
                    {
                        return Err("you accepted a finding, in whole or in part: include the whole revised design in `revised_design`".into());
                    }
                    set(&mut decision.rebuttals, slot, rebuttal);
                    Ok(())
                })
            }
            Part::Reply => {
                let replies: Replies = parse(params)?;
                self.store.record_into(target, |decision| {
                    if decision.rebuttals.get(slot).is_none_or(Option::is_none) {
                        return Err("the proposer hasn't answered your findings yet".into());
                    }
                    let expected: Vec<String> = decision
                        .findings_of(slot)
                        .iter()
                        .filter(|finding| {
                            decision.response(finding).is_none_or(|r| r.answer != Answer::Accept)
                        })
                        .map(|finding| finding.id.clone())
                        .collect();
                    covers(
                        &expected,
                        replies.responses.iter().map(|r| r.finding.as_str()),
                    )?;
                    set(&mut decision.replies, slot, replies.responses);
                    Ok(())
                })
            }
            Part::SpikeResult => {
                let result: SpikeResult = parse(params)?;
                self.store.record_into(target, |decision| {
                    let spike = decision
                        .spikes
                        .as_mut()
                        .and_then(|spikes| spikes.get_mut(slot))
                        .ok_or("there is no spike for you to report on")?;
                    spike.result = Some(result);
                    Ok(())
                })
            }
            Part::Ruling => {
                let mut ruling: Ruling = parse(params)?;
                // A ruling recorded before a restart, run again.
                if self.lead
                    && let Some(decision) = self.store.get().active.and_then(|id| self.store.decision(&id))
                    && let Some(Outcome::Decided { record, .. }) = &decision.outcome
                {
                    return Ok(format!(
                        "{} is settled; its record is `{record}` in your workspace. Continue the build on it.",
                        decision.id
                    ));
                }
                let id = self.store.record_into(target, |decision| {
                    if Decision::slot_of(&ruling.chosen)
                        .and_then(|slot| decision.proposals.get(slot)?.as_ref())
                        .is_none()
                    {
                        return Err(format!(
                            "`chosen` must be one of the proposals: {}",
                            decision
                                .submitted()
                                .map(|(slot, _)| Decision::proposal_id(slot))
                                .collect::<Vec<_>>()
                                .join(", ")
                        ));
                    }
                    // Every contested finding needs a ruling; a ruling on one
                    // already conceded or withdrawn is settled already, and
                    // dropped. Only ids that aren't findings are refused.
                    let known: Vec<String> = decision
                        .all_findings()
                        .map(|finding| finding.id.clone())
                        .collect();
                    let unknown: Vec<&str> = ruling
                        .rulings
                        .iter()
                        .map(|r| r.finding.trim())
                        .filter(|id| !known.iter().any(|known| known == id))
                        .collect();
                    if !unknown.is_empty() {
                        return Err(format!(
                            "these aren't findings of this decision: {}; the findings are: {}",
                            unknown.join(", "),
                            if known.is_empty() {
                                "none".to_string()
                            } else {
                                known.join(", ")
                            }
                        ));
                    }
                    let contested: Vec<String> = decision
                        .contested()
                        .into_iter()
                        .map(|finding| finding.id.clone())
                        .collect();
                    let missing: Vec<&str> = contested
                        .iter()
                        .map(String::as_str)
                        .filter(|id| !ruling.rulings.iter().any(|r| r.finding.trim() == *id))
                        .collect();
                    if !missing.is_empty() {
                        return Err(format!(
                            "rule on every contested finding; missing: {}",
                            missing.join(", ")
                        ));
                    }
                    ruling
                        .rulings
                        .retain(|r| contested.iter().any(|id| id == r.finding.trim()));
                    set(&mut decision.rulings, slot, ruling);
                    Ok(())
                })?;
                if !self.lead {
                    return Ok(SUBMITTED.into());
                }
                let record = self.store.decide(&id, &self.config).map_err(|error| {
                    ToolError::ExecutionFailed(format!("couldn't write the record: {error}"))
                })?;
                self.store
                    .update(|ledger| {
                        if let Some(decision) = ledger.decisions.iter_mut().find(|d| d.id == id) {
                            decision.delivered = true;
                        }
                    })
                    .map_err(|error| ToolError::ExecutionFailed(error.to_string()))?;
                eprintln!("forge: {id} decided by the architect; record in {record}");
                let waiting: Vec<String> = self
                    .store
                    .get()
                    .decisions
                    .into_iter()
                    .filter(|decision| !decision.delivered)
                    .map(|decision| decision.id)
                    .collect();
                return Ok(if waiting.is_empty() {
                    format!(
                        "Recorded. {id} is settled; its record is `{record}` in your workspace. Build on it, summarise it in DESIGN.md and link the record, and continue the build."
                    )
                } else {
                    format!(
                        "Recorded. {id} is settled; its record is `{record}` in your workspace. {} still to be settled: end your turn now with complete_task and a one-line summary, without building yet. The forge deliberates the next decision and gives you its dossier.",
                        waiting.join(", ")
                    )
                });
            }
        }
        .map(|_| SUBMITTED.to_string())
    }
}

fn set<T>(slots: &mut Vec<Option<T>>, slot: usize, value: T) {
    if slots.len() <= slot {
        slots.resize_with(slot + 1, || None);
    }
    slots[slot] = Some(value);
}

/// Checks that `answered` names each of `expected` (and nothing else).
fn covers<'a>(expected: &[String], answered: impl Iterator<Item = &'a str>) -> Result<(), String> {
    let answered: HashSet<&str> = answered.map(str::trim).collect();
    let missing: Vec<&str> = expected
        .iter()
        .map(String::as_str)
        .filter(|id| !answered.contains(id))
        .collect();
    let unknown: Vec<&str> = answered
        .iter()
        .copied()
        .filter(|id| !expected.iter().any(|known| known == id))
        .collect();
    match (missing.is_empty(), unknown.is_empty()) {
        (true, true) => Ok(()),
        (false, _) => Err(format!(
            "answer every finding; missing: {}",
            missing.join(", ")
        )),
        (true, false) if expected.is_empty() => Err(format!(
            "there are no findings to answer here, so leave out: {}",
            unknown.join(", ")
        )),
        (true, false) => Err(format!(
            "these aren't findings to answer here: {}; the findings are: {}",
            unknown.join(", "),
            expected.join(", ")
        )),
    }
}

#[async_trait]
impl Tool for Submit {
    fn definition(&self) -> ToolDefinition {
        let (description, schema) = self.schema();
        tool_definition(self.name(), description, schema)
    }

    fn recovery_mode(&self) -> ToolRecovery {
        // Recording the same submission again changes nothing.
        ToolRecovery::Rerun
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        self.record(context.initial_task_prompt, params)
    }
}

// ---------------------------------------------------------------------------
// The host side: running the stages

/// Runs the requested deliberations and hands their records to the
/// architect.
pub struct Deliberator<'a> {
    pub workflow: &'a Workflow,
    pub state_dir: &'a Path,
    pub store: Arc<Store>,
    pub config: &'a Config,
    pub arko: &'a Path,
    pub target_dir: &'a Path,
    /// What this process tried once: error pauses resumed, by run, and jobs
    /// started afresh after their run failed for a credential or the
    /// configuration, by a key from the job ([`Job::fresh_start`]).
    pub resumed: Mutex<HashSet<uuid::Uuid>>,
    /// The architect's run calls, started afresh after a login failure.
    pub architect: crate::Architect<'a>,
}

/// One agent's part in a stage.
struct Job {
    agent: String,
    /// The first line of the run's prompt: the run to start, or the earlier
    /// stage's run to continue.
    marker: String,
    /// Continue the earlier stage's run with `prompt`, instead of starting
    /// a run with it.
    continues: bool,
    prompt: String,
    /// The prompt of a fresh run with the same marker, when the run failed
    /// for a credential or the configuration: the start prompt, or for a
    /// continuation, one that carries its context.
    fresh: String,
    /// Whether the agent's part is in.
    done: Box<dyn Fn(&Decision) -> bool + Send + Sync>,
}

impl Job {
    fn start(
        role: &str,
        stage: &str,
        id: &str,
        slot: usize,
        prompt: String,
        done: impl Fn(&Decision) -> bool + Send + Sync + 'static,
    ) -> Self {
        let prompt = format!("{}\n{prompt}", marker(id, stage, slot));
        Self {
            agent: agent(role, slot),
            marker: marker(id, stage, slot),
            continues: false,
            fresh: prompt.clone(),
            prompt,
            done: Box::new(done),
        }
    }

    /// Continues the run `role` started in `stage` with `prompt`; `fresh`
    /// is the same request with its context, for a fresh run.
    fn next(
        role: &str,
        stage: &str,
        id: &str,
        slot: usize,
        prompt: String,
        fresh: String,
        done: impl Fn(&Decision) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            continues: true,
            prompt,
            ..Self::start(role, stage, id, slot, fresh, done)
        }
    }

    /// The job's key among what this process tried once
    /// ([`Deliberator::resumed`]), from its agent and its prompt's first line
    /// (its stage: a rebuttal continues the proposal's run, under the same
    /// marker).
    fn fresh_start(&self) -> uuid::Uuid {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (&self.agent, self.prompt.lines().next()).hash(&mut hasher);
        uuid::Uuid::from_u128(u128::from(hasher.finish()) << 64 | 0x6a6f_625f_6672_6573)
    }
}

/// Why the forge stops when `agent` (on `model`, when known) can't reach its
/// model, also when started afresh: the request needs a new credential or a
/// configuration change.
pub(crate) fn login_reason(agent: &str, model: Option<&ModelSpec>, error: &str) -> String {
    let openai = model
        .and_then(|model| arko_ai::Model::find(&model.id))
        .is_some_and(|model| model.provider == arko_ai::Provider::OpenAI);
    let (via, fix) = if openai {
        (
            "OpenAI",
            "sign in to ChatGPT again (`codex login`), or check OPENAI_API_KEY or the OPENAI_OAUTH_* variables",
        )
    } else {
        (
            "Anthropic",
            "check the Claude Code CLI and its login (run `claude`), or ANTHROPIC_API_KEY with `--backend api`",
        )
    };
    format!(
        "{agent} ({}, {via}) can't reach its model, also when started again: the request needs a new credential or a configuration change ({}). To continue, {fix}",
        model.map_or_else(|| "its model".to_string(), ModelSpec::to_string),
        crate::cut(error.trim(), 400)
    )
}

/// Stops the forge, saved, with exit status 3: an agent's model request
/// needs a new credential or a configuration change (`reason`).
pub(crate) async fn login_stop(workflow: &Workflow, state_dir: &Path, reason: &str) -> ! {
    if let Err(error) = workflow.checkpoint().await {
        eprintln!("forge: couldn't save a checkpoint ({error}); the latest one is kept");
    }
    eprintln!(
        "\nforge: stopped: {reason}.\nforge: the run is saved. To continue, sign in again (or fix the configuration), then resume:\n\n    forge --resume {}\n",
        state_dir.display()
    );
    std::process::exit(3);
}

/// A proposer's fresh run for its rebuttal, after its conversation failed.
const AFRESH_PROPOSER: &str = "You proposed the design below for the decision below, and a critic reviewed it. Your earlier conversation ended when its model request failed, so this one starts from your design as you submitted it.";

/// A critic's fresh run for its reply, after its conversation failed.
const AFRESH_CRITIC: &str = "You reviewed the proposal below for the decision below; your findings follow it. Your earlier conversation ended when its model request failed, so this one starts from them.";

/// The error, when `run` failed because its model request needs a new
/// credential or a configuration change (arko's
/// `ErrorRecoveryCause::NonRetryable`: an expired or invalidated login, a
/// bad key), as the run's failure records it (`AgentResult::failure_cause`).
/// A run saved before arko recorded that is read from its state changes in
/// the event log (`events.jsonl` in the state directory): the run went from
/// that pause straight to its failure.
pub(crate) fn login_failure(state_dir: &Path, run: &AgentResult) -> Option<String> {
    #[derive(Deserialize)]
    struct Logged {
        agent_run_id: uuid::Uuid,
        event: AgentEvent,
    }
    if run.status != AgentRunStatus::Failed {
        return None;
    }
    // A request too large for the model fails that run alone: no credential
    // or configuration change is needed.
    if matches!(
        run.failure_kind(),
        Some(ErrorKind::ContextOverflow | ErrorKind::RequestTooLarge)
    ) {
        return None;
    }
    if let Some(cause) = run.failure_cause() {
        return (*cause == ErrorRecoveryCause::NonRetryable)
            .then(|| run.error.clone().unwrap_or_default());
    }
    let log = std::fs::File::open(state_dir.join("events.jsonl")).ok()?;
    let id = run.agent_run_id.to_string();
    let mut before_failure = None;
    for line in io::BufReader::new(log).lines().map_while(Result::ok) {
        if !line.contains(&id) || !line.contains("TaskStateChanged") {
            continue;
        }
        let Ok(logged) = serde_json::from_str::<Logged>(&line) else {
            continue;
        };
        match logged.event {
            AgentEvent::TaskStateChanged(TaskState::Failed { .. }) => {}
            AgentEvent::TaskStateChanged(state) if logged.agent_run_id == run.agent_run_id => {
                before_failure = Some(state)
            }
            _ => {}
        }
    }
    match before_failure? {
        TaskState::AwaitingErrorRecoveryApproval {
            cause: ErrorRecoveryCause::NonRetryable,
            error_message,
            ..
        } => Some(error_message),
        _ => None,
    }
}

enum Call {
    Run(String),
    Continue(AgentResult, String),
    Recover(AgentResult),
    Reopen(AgentResult),
    Resume(AgentResult),
}

/// Actions on one run before its stage gives up on it.
const MAX_ACTIONS: usize = 8;
/// Continuations of a finished run that didn't submit its part.
const MAX_NUDGES: usize = 2;

impl Deliberator<'_> {
    /// Runs the requested deliberations that aren't settled, one after
    /// another in the order the architect framed them (so each can build on
    /// the records before it), and continues the architect once with all
    /// their outcomes. When the architect rules, it rules on each in turn,
    /// before the next is deliberated. Returns the architect's latest result.
    pub async fn settle(&self, mut architect: AgentResult) -> Result<AgentResult, WorkflowError> {
        let mut lead_turns: HashMap<String, usize> = HashMap::new();
        loop {
            if architect.status != AgentRunStatus::Completed {
                return Ok(architect);
            }
            let pending = to_deliver(&self.store.get(), &architect);
            if pending.is_empty() {
                return Ok(architect);
            }
            let mut to_rule = None;
            for decision in &pending {
                if decision.outcome.is_none() {
                    match self.deliberate(&decision.id).await {
                        // Only `drive` interrupts a deliberation: an agent's
                        // login or configuration needs the user.
                        Err(WorkflowError::AgentInterrupted { reason, .. }) => {
                            login_stop(self.workflow, self.state_dir, &reason).await
                        }
                        result => result?,
                    }
                }
                let decision = self.decision(&decision.id);
                if decision.outcome.is_none() {
                    if self.config.judges.is_empty() {
                        to_rule = Some(decision);
                        break;
                    }
                    self.fail(
                        &decision.id,
                        "the judges' ruling couldn't be recorded".into(),
                    )?;
                }
            }
            let prompt = match to_rule {
                // The architect rules: it gets the dossier, and its
                // `submit_ruling` settles the decision.
                Some(decision) => {
                    let id = decision.id.clone();
                    let turns = lead_turns.entry(id.clone()).or_default();
                    *turns += 1;
                    if *turns > 1 + MAX_NUDGES {
                        self.fail(&id, "the architect didn't submit a ruling".into())?;
                        continue;
                    }
                    self.activate(&id)?;
                    let dossier = render_dossier(&decision);
                    if *turns == 1 {
                        eprintln!("forge: {id} is ready for the architect's ruling");
                        format!("[forge {id} ruling]\n{}\n\n{dossier}", prompts::LEAD_RULING)
                    } else {
                        format!(
                            "[forge {id} ruling]\nYou haven't ruled on {id} with `submit_ruling` yet. Rule now; the dossier is repeated below.\n\n{dossier}"
                        )
                    }
                }
                // Every pending decision is settled: the architect gets all
                // their outcomes at once.
                None => {
                    let ids: Vec<String> = pending.iter().map(|d| d.id.clone()).collect();
                    let at = (architect.agent_run_id, architect.task.usage_history.len());
                    self.store
                        .update(|ledger| {
                            for decision in &mut ledger.decisions {
                                if ids.contains(&decision.id) {
                                    decision.delivered = true;
                                    decision.delivered_at = Some(at);
                                }
                            }
                        })
                        .map_err(WorkflowError::Io)?;
                    let mut prompt = format!(
                        "[forge decisions {}]\n{}",
                        ids.join(" "),
                        prompts::DECISIONS_DELIVERED.replace("{ids}", &ids.join(", "))
                    );
                    for id in &ids {
                        match self.decision(id).outcome {
                            Some(Outcome::Decided { record, .. }) => {
                                let text =
                                    std::fs::read_to_string(self.store.crate_dir.join(&record))
                                        .unwrap_or_default();
                                let _ = write!(
                                    prompt,
                                    "\n\n---\n\n{id}'s record is `{record}` in your workspace:\n\n{text}"
                                );
                            }
                            Some(Outcome::Failed { reason }) => {
                                let _ = write!(
                                    prompt,
                                    "\n\n---\n\nThe deliberation on {id} couldn't finish: {reason}. Decide it yourself and record your reasoning in DESIGN.md."
                                );
                            }
                            None => {}
                        }
                    }
                    prompt
                }
            };
            architect = self
                .architect
                .call(self.workflow.continue_agent(&architect, &prompt))
                .await?;
        }
    }

    /// Why the forge stops when `agent`'s model request needs a new
    /// credential or a configuration change, also when started afresh.
    fn login_error(&self, agent: &str, error: &str) -> WorkflowError {
        let model = self
            .config
            .agents()
            .into_iter()
            .find(|(name, ..)| name == agent)
            .map(|(.., model)| model.clone());
        WorkflowError::AgentInterrupted {
            agent: agent.to_string(),
            reason: login_reason(agent, model.as_ref(), error),
        }
    }

    fn activate(&self, id: &str) -> Result<(), WorkflowError> {
        self.store
            .update(|ledger| ledger.active = Some(id.to_string()))
            .map_err(WorkflowError::Io)
    }

    fn fail(&self, id: &str, reason: String) -> Result<(), WorkflowError> {
        eprintln!("forge: {id} couldn't be deliberated: {reason}");
        self.store.fail(id, reason).map_err(WorkflowError::Io)
    }

    fn decision(&self, id: &str) -> Decision {
        self.store.decision(id).expect("recorded")
    }

    /// Runs decision `id`'s stages that haven't run, up to its ruling (with
    /// judges, its record).
    async fn deliberate(&self, id: &str) -> Result<(), WorkflowError> {
        self.activate(id)?;
        let config = self.config;
        let decision = self.decision(id);
        eprintln!("forge: deliberating {id}: {}", decision.frame.name());
        let frame = render_frame(&decision.frame);

        // Proposals, independently.
        let reopening = decision
            .reopens
            .as_ref()
            .and_then(|earlier| self.store.decision(earlier))
            .and_then(|earlier| match earlier.outcome {
                Some(Outcome::Decided { record, .. }) => Some(
                    prompts::REOPENING
                        .replace("{earlier}", &earlier.id)
                        .replace("{record}", &record),
                ),
                _ => None,
            })
            .unwrap_or_default();
        self.run_stage(
            id,
            (0..config.proposers.len())
                .map(|slot| {
                    Job::start(
                        PROPOSER,
                        "proposal",
                        id,
                        slot,
                        format!(
                            "{}{reopening}\n\n# The decision ({id})\n\n{frame}",
                            prompts::PROPOSE
                        ),
                        move |d| d.proposals.get(slot).is_some_and(Option::is_some),
                    )
                })
                .collect(),
        )
        .await?;
        let decision = self.decision(id);
        let submitted: Vec<usize> = decision.submitted().map(|(slot, _)| slot).collect();
        eprintln!(
            "forge: {id}: {} of {} proposals in",
            submitted.len(),
            config.proposers.len()
        );
        if submitted.is_empty() {
            return self.fail(id, "no proposer submitted a design".into());
        }

        // A critic per proposal.
        self.run_stage(
            id,
            decision
                .submitted()
                .map(|(slot, proposal)| {
                    Job::start(
                        CRITIC,
                        "critique",
                        id,
                        slot,
                        format!(
                            "{}\n\n# The decision ({id})\n\n{frame}\n\n# The proposal\n\n{}",
                            prompts::CRITIQUE,
                            render_design("The proposal", &proposal.design, proposal)
                        ),
                        move |d| d.findings.get(slot).is_some_and(Option::is_some),
                    )
                })
                .collect(),
        )
        .await?;

        // Each proposer answers the findings on its design, in its own
        // conversation.
        let decision = self.decision(id);
        self.run_stage(
            id,
            submitted
                .iter()
                .filter(|&&slot| !decision.findings_of(slot).is_empty())
                .map(|&slot| {
                    let findings = render_findings(decision.findings_of(slot));
                    let proposal = decision.proposals[slot].as_ref().expect("submitted");
                    Job::next(
                        PROPOSER,
                        "proposal",
                        id,
                        slot,
                        format!(
                            "[forge {id} rebuttal {}]\n{}\n\n{findings}",
                            slot + 1,
                            prompts::REBUT
                        ),
                        format!(
                            "{AFRESH_PROPOSER}\n\n# The decision ({id})\n\n{frame}\n\n# Your design\n\n{}\n\n{}\n\n{findings}",
                            render_design(&Decision::proposal_id(slot), &proposal.design, proposal),
                            prompts::REBUT
                        ),
                        move |d| d.rebuttals.get(slot).is_some_and(Option::is_some),
                    )
                })
                .collect(),
        )
        .await?;

        // The critics reply to what was contested.
        if config.rounds > 1 {
            let decision = self.decision(id);
            self.run_stage(
                id,
                submitted
                    .iter()
                    .filter_map(|&slot| {
                        let contested: String = decision
                            .rebuttals
                            .get(slot)?
                            .as_ref()?
                            .responses
                            .iter()
                            .filter(|response| response.answer != Answer::Accept)
                            .map(render_response)
                            .collect();
                        let proposal = decision.proposals.get(slot)?.as_ref()?;
                        (!contested.is_empty()).then(|| {
                            Job::next(
                                CRITIC,
                                "critique",
                                id,
                                slot,
                                format!(
                                    "[forge {id} reply {}]\n{}\n\n{contested}",
                                    slot + 1,
                                    prompts::REPLY
                                ),
                                format!(
                                    "{AFRESH_CRITIC}\n\n# The decision ({id})\n\n{frame}\n\n# The proposal\n\n{}\n\n# Your findings\n\n{}\n\n{}\n\n{contested}",
                                    render_design("The proposal", &proposal.design, proposal),
                                    render_findings(decision.findings_of(slot)),
                                    prompts::REPLY
                                ),
                                move |d| d.replies.get(slot).is_some_and(Option::is_some),
                            )
                        })
                    })
                    .collect(),
            )
            .await?;
        }

        // Spikes on the contested claims that matter.
        if config.spikes > 0 {
            if self.decision(id).spikes.is_none() {
                self.store
                    .update(|ledger| {
                        if let Some(d) = ledger.decisions.iter_mut().find(|d| d.id == id) {
                            d.spikes = Some(d.choose_spikes(config.spikes as usize));
                        }
                    })
                    .map_err(WorkflowError::Io)?;
            }
            let decision = self.decision(id);
            let spikes = decision.spikes.clone().unwrap_or_default();
            if !spikes.is_empty() {
                eprintln!("forge: {id}: {} spike(s)", spikes.len());
            }
            let mut jobs = Vec::new();
            for (slot, spike) in spikes.iter().enumerate() {
                let scratch = self
                    .state_dir
                    .join("deliberations")
                    .join(id)
                    .join(format!("spike-{}", slot + 1));
                if !scratch.join("Cargo.toml").is_file() {
                    crate::scaffold::scaffold(
                        &scratch,
                        &format!("spike-{}-{}", id.to_ascii_lowercase(), slot + 1),
                        self.arko,
                        &format!("Spike on {} for deliberation {id}.", spike.finding),
                    )
                    .map_err(WorkflowError::Io)?;
                }
                jobs.push(Job::start(
                    SPIKER,
                    "spike",
                    id,
                    slot,
                    format!(
                        "{}\n\n{}",
                        prompts::SPIKE
                            .replace("{scratch}", &scratch.display().to_string())
                            .replace("{target_dir}", &self.target_dir.display().to_string()),
                        spike_brief(&decision, spike)
                    ),
                    move |d| {
                        d.spikes
                            .as_ref()
                            .and_then(|spikes| spikes.get(slot))
                            .is_some_and(|spike| spike.result.is_some())
                    },
                ));
            }
            self.run_stage(id, jobs).await?;
        }

        // The ruling: the architect's (in `settle`), or the judges'.
        if config.judges.is_empty() {
            return Ok(());
        }
        let dossier = render_dossier(&self.decision(id));
        self.run_stage(
            id,
            (0..config.judges.len())
                .map(|slot| {
                    Job::start(
                        JUDGE,
                        "ruling",
                        id,
                        slot,
                        format!("{}\n\n{dossier}", prompts::RULE),
                        move |d| d.rulings.get(slot).is_some_and(Option::is_some),
                    )
                })
                .collect(),
        )
        .await?;
        if self.decision(id).rulings.iter().all(Option::is_none) {
            return self.fail(id, "no judge submitted a ruling".into());
        }
        let record = self.store.decide(id, config).map_err(WorkflowError::Io)?;
        eprintln!("forge: {id} decided; record in {record}");
        Ok(())
    }

    /// Runs a stage's jobs at once, each until its part is in or its run
    /// can't get there.
    async fn run_stage(&self, id: &str, jobs: Vec<Job>) -> Result<(), WorkflowError> {
        join_all(jobs.iter().map(|job| self.drive(id, job)))
            .await
            .into_iter()
            .collect()
    }

    /// Drives `job`'s run until its part is in, or the run can't get there.
    /// A run whose model request needs a new credential or a configuration
    /// change (an expired login) isn't the end of the job: a failed one
    /// starts again afresh once per process, and when that fails the same
    /// way (or a paused one was resumed once already), the forge stops,
    /// saved ([`login_stop`](Self::login_stop)), so the user can sign in
    /// again and resume; the resumed forge starts the job afresh again.
    async fn drive(&self, id: &str, job: &Job) -> Result<(), WorkflowError> {
        let agent = job.agent.as_str();
        let mut nudges = 0;
        let mut last_error = None;
        for _ in 0..MAX_ACTIONS {
            if (job.done)(&self.decision(id)) {
                return Ok(());
            }
            let call = match find_run(self.workflow, agent, &job.marker) {
                None if job.continues => return Ok(()),
                None => Call::Run(job.prompt.clone()),
                Some(run) => match run.status {
                    AgentRunStatus::Completed => {
                        // It ended its turn without submitting (or, for a
                        // later stage, hasn't been given it yet).
                        let limit = MAX_NUDGES + usize::from(job.continues);
                        if nudges >= limit {
                            eprintln!("forge: {agent} ended without submitting its part");
                            return Ok(());
                        }
                        let prompt = match (job.continues, nudges) {
                            (true, 0) => job.prompt.clone(),
                            (true, _) => format!("{}\n\n{}", prompts::NUDGE, job.prompt),
                            (false, _) => prompts::NUDGE.to_string(),
                        };
                        nudges += 1;
                        Call::Continue(run, prompt)
                    }
                    AgentRunStatus::Failed => match login_failure(self.state_dir, &run) {
                        Some(error) => {
                            if !self.resumed.lock().unwrap().insert(job.fresh_start()) {
                                return Err(self.login_error(agent, &error));
                            }
                            eprintln!(
                                "forge: {agent} couldn't reach its model ({}); starting its part again",
                                crate::cut(error.trim(), 200)
                            );
                            nudges = 0;
                            Call::Run(job.fresh.clone())
                        }
                        None => {
                            eprintln!(
                                "forge: {agent} failed: {}",
                                crate::cut(run.error.as_deref().unwrap_or("no reason given"), 300)
                            );
                            return Ok(());
                        }
                    },
                    _ => match self.next_call(&run) {
                        Some(call) => call,
                        // A credential or the configuration, resumed once
                        // already in this process (agents in Claude Code
                        // pause for it rather than fail).
                        None => match AgentPause::of(&run.task.current_state) {
                            Some(AgentPause::ErrorRecovery {
                                cause: ErrorRecoveryCause::NonRetryable,
                                error,
                                ..
                            }) => return Err(self.login_error(agent, &error)),
                            _ => return Ok(()),
                        },
                    },
                },
            };
            match self.call(agent, call).await {
                Ok(()) => {}
                Err(error) if is_fatal(&error) => return Err(error),
                // The run's recorded status says what happened.
                Err(error) => last_error = Some(error.to_string()),
            }
        }
        if !(job.done)(&self.decision(id)) {
            eprintln!(
                "forge: {agent} couldn't be settled (last error: {})",
                last_error.unwrap_or_else(|| "none".into())
            );
        }
        Ok(())
    }

    /// The call that moves on a run that neither completed nor failed, or
    /// `None` when it can't go on.
    fn next_call(&self, run: &AgentResult) -> Option<Call> {
        Some(match run.status {
            AgentRunStatus::Interrupted => Call::Recover(run.clone()),
            AgentRunStatus::Cancelled => Call::Reopen(run.clone()),
            AgentRunStatus::WaitingForInput => {
                Call::Continue(run.clone(), prompts::NO_USER.to_string())
            }
            AgentRunStatus::Paused => match AgentPause::of(&run.task.current_state) {
                Some(AgentPause::ErrorRecovery {
                    cause: ErrorRecoveryCause::Refusal,
                    ..
                })
                | Some(AgentPause::ToolApproval { .. }) => return None,
                // A credential, the CLI, the configuration: once per
                // process, in case it was fixed before a resume.
                Some(AgentPause::ErrorRecovery {
                    cause: ErrorRecoveryCause::NonRetryable,
                    ..
                }) => {
                    if !self.resumed.lock().unwrap().insert(run.agent_run_id) {
                        return None;
                    }
                    Call::Resume(run.clone())
                }
                // The budget, an outage, a usage limit: go on.
                Some(AgentPause::ErrorRecovery { .. }) => Call::Resume(run.clone()),
                None => Call::Recover(run.clone()),
            },
            AgentRunStatus::Completed | AgentRunStatus::Failed => return None,
        })
    }

    /// Makes one workflow call, unless the budget stops the forge first.
    async fn call(&self, agent: &str, call: Call) -> Result<(), WorkflowError> {
        let workflow = self.workflow;
        let state_dir = self.state_dir;
        match call {
            Call::Run(prompt) => {
                within_budget(workflow, state_dir, workflow.run_agent(agent, &prompt)).await
            }
            Call::Continue(run, prompt) => {
                within_budget(workflow, state_dir, workflow.continue_agent(&run, &prompt)).await
            }
            Call::Recover(run) => {
                within_budget(workflow, state_dir, workflow.recover_agent(&run)).await
            }
            Call::Reopen(run) => {
                within_budget(workflow, state_dir, workflow.reopen_agent(&run, "")).await
            }
            Call::Resume(run) => {
                let resume = AgentResume::ContinueAfterError { note: None };
                within_budget(workflow, state_dir, workflow.resume_agent(&run, resume)).await
            }
        }
        .map(|_| ())
    }
}

/// The decisions the finished `architect` run is still to be given, in the
/// order it framed them: those it hasn't been given, and those the forge
/// stopped giving it before the continuation was saved (the run has made
/// no request since).
fn to_deliver(ledger: &Ledger, architect: &AgentResult) -> Vec<Decision> {
    let requests = architect.task.usage_history.len();
    ledger
        .decisions
        .iter()
        .filter(|decision| {
            !decision.delivered
                || decision
                    .delivered_at
                    .is_some_and(|(run, at)| run == architect.agent_run_id && requests <= at)
        })
        .cloned()
        .collect()
}

/// The first line of a stage's prompt: `[forge D1 proposal 2]`.
fn marker(id: &str, stage: &str, slot: usize) -> String {
    format!("[forge {id} {stage} {}]", slot + 1)
}

/// The decision a stage's run belongs to, from its prompt's marker.
fn decision_of(prompt: &str) -> Option<String> {
    let rest = prompt.strip_prefix("[forge ")?;
    let id = rest.split_whitespace().next()?;
    (id.starts_with('D') && id[1..].chars().all(|c| c.is_ascii_digit()) && id.len() > 1)
        .then(|| id.to_string())
}

/// Whether `agent` is one of the deliberation's stage agents, whose runs
/// only the deliberation moves on.
pub fn is_stage_agent(agent: &str) -> bool {
    [PROPOSER, CRITIC, SPIKER, JUDGE].iter().any(|role| {
        agent
            .strip_prefix(role)
            .is_some_and(|rest| rest.starts_with('-'))
    })
}

/// The latest run of `agent` whose prompt starts with `marker`.
fn find_run(workflow: &Workflow, agent: &str, marker: &str) -> Option<AgentResult> {
    workflow
        .context()
        .agent_results
        .read()
        .iter()
        .rev()
        .find(|run| run.agent_name == agent && run.task.initial_task_prompt.starts_with(marker))
        .cloned()
}

/// Errors after which the forge can't go on. Others (a call refused
/// because of the run's state, a failed run) are settled by the run's
/// recorded status.
fn is_fatal(error: &WorkflowError) -> bool {
    matches!(
        error,
        WorkflowError::Cancelled
            | WorkflowError::AbortedByHook { .. }
            | WorkflowError::Persistence(_)
            | WorkflowError::AgentNotFound(_)
            | WorkflowError::BuilderMissing { .. }
    )
}

/// What a spike tests: the finding, the design it's about, and both sides.
fn spike_brief(decision: &Decision, spike: &Spike) -> String {
    let Some(finding) = decision.finding(&spike.finding) else {
        return format!("Finding {} (no longer recorded).", spike.finding);
    };
    let mut text = format!("# The contested finding\n\n{}", render_finding(finding));
    if let Some(response) = decision.response(finding) {
        let _ = write!(
            text,
            "\n## The proposer's answer\n\n{}",
            render_response(response)
        );
    }
    if let Some(reply) = decision.reply(finding) {
        let _ = write!(text, "\n## The critic's reply\n\n{}", render_reply(reply));
    }
    if let Some((slot, proposal)) = Decision::slot_of(&finding.id)
        .and_then(|slot| Some((slot, decision.proposals.get(slot)?.as_ref()?)))
    {
        let _ = write!(
            text,
            "\n# The design it is about\n\n{}",
            render_design(
                &Decision::proposal_id(slot),
                decision.final_design(slot, proposal),
                proposal
            )
        );
    }
    let _ = write!(
        text,
        "\n\n# The decision\n\n{}",
        render_frame(&decision.frame)
    );
    text
}

/// Runs `futures` concurrently and returns their outputs in order.
async fn join_all<F: Future>(futures: impl IntoIterator<Item = F>) -> Vec<F::Output> {
    let mut futures: Vec<Pin<Box<F>>> = futures.into_iter().map(Box::pin).collect();
    let mut outputs: Vec<Option<F::Output>> = futures.iter().map(|_| None).collect();
    std::future::poll_fn(|cx| {
        let mut pending = false;
        for (future, output) in futures.iter_mut().zip(outputs.iter_mut()) {
            if output.is_none() {
                match future.as_mut().poll(cx) {
                    Poll::Ready(value) => *output = Some(value),
                    Poll::Pending => pending = true,
                }
            }
        }
        if pending {
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    })
    .await;
    outputs
        .into_iter()
        .map(|output| output.expect("ready"))
        .collect()
}

/// A line for the forge's report.
pub fn summary_line(decision: &Decision) -> String {
    let question = decision.frame.name();
    match &decision.outcome {
        Some(Outcome::Decided { record, ruling, .. }) => {
            let chosen = Decision::slot_of(&ruling.chosen)
                .and_then(|slot| decision.proposals.get(slot)?.as_ref())
                .map(|proposal| format!(" \"{}\" ({})", proposal.title.trim(), proposal.model))
                .unwrap_or_default();
            let findings = decision.all_findings().count();
            let spikes = decision.spikes.as_ref().map(Vec::len).unwrap_or(0);
            format!(
                "- **{}** {question}: {}{chosen}; {} proposals, {findings} findings, {spikes} spikes. Record: `{record}`.",
                decision.id,
                ruling.chosen,
                decision.submitted().count()
            )
        }
        Some(Outcome::Failed { reason }) => {
            format!("- **{}** {question}: not settled ({reason}).", decision.id)
        }
        None => format!("- **{}** {question}: not finished.", decision.id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finding(id: &str, severity: Severity) -> Finding {
        Finding {
            id: id.into(),
            severity,
            claim: format!("claim {id}"),
            evidence: "evidence".into(),
            trigger: "trigger".into(),
        }
    }

    fn proposal(title: &str) -> Proposal {
        Proposal {
            model: "claude-opus-5-5:xhigh".into(),
            title: title.into(),
            design: format!("design {title}"),
            assumptions: vec![],
            risks: vec![],
            falsifiers: vec![],
        }
    }

    fn response(finding: &str, answer: Answer) -> Response {
        Response {
            finding: finding.into(),
            answer,
            argument: "because".into(),
            settle_by: None,
        }
    }

    fn decision() -> Decision {
        let frame = Frame {
            title: None,
            question: "How should the workflow recover after a crash?".into(),
            context: String::new(),
            constraints: vec![],
            evidence: vec![],
            criteria: vec![],
        };
        let mut decision = Decision::new("D1".into(), frame, None, 2);
        decision.proposals = vec![Some(proposal("host loop")), Some(proposal("one agent"))];
        decision.findings = vec![
            Some(vec![
                finding("P1-F1", Severity::Minor),
                finding("P1-F2", Severity::Major),
            ]),
            Some(vec![
                finding("P2-F1", Severity::Critical),
                finding("P2-F2", Severity::Major),
            ]),
        ];
        decision.rebuttals = vec![
            Some(Rebuttal {
                responses: vec![
                    response("P1-F1", Answer::Accept),
                    response("P1-F2", Answer::Implausible),
                ],
                revised_design: Some("revised host loop".into()),
            }),
            Some(Rebuttal {
                responses: vec![
                    response("P2-F1", Answer::Incorrect),
                    response("P2-F2", Answer::OutOfScope),
                ],
                revised_design: None,
            }),
        ];
        decision.replies = vec![
            Some(vec![Reply {
                finding: "P1-F2".into(),
                stance: Stance::Maintain,
                argument: "it happens".into(),
                settle_by: None,
            }]),
            Some(vec![
                Reply {
                    finding: "P2-F1".into(),
                    stance: Stance::Maintain,
                    argument: "see the code".into(),
                    settle_by: Some("run it".into()),
                },
                Reply {
                    finding: "P2-F2".into(),
                    stance: Stance::Withdraw,
                    argument: "fair".into(),
                    settle_by: None,
                },
            ]),
        ];
        decision
    }

    #[test]
    fn findings_stand_by_the_rebuttal_and_the_reply() {
        let decision = decision();
        let standing = |id: &str| decision.standing(decision.finding(id).unwrap());
        assert_eq!(standing("P1-F1"), Standing::Conceded);
        assert_eq!(standing("P1-F2"), Standing::Contested);
        assert_eq!(standing("P2-F1"), Standing::Contested);
        assert_eq!(standing("P2-F2"), Standing::Withdrawn);
        assert_eq!(
            decision.final_design(0, decision.proposals[0].as_ref().unwrap()),
            "revised host loop"
        );
    }

    #[test]
    fn spikes_test_the_most_severe_contested_findings() {
        let decision = decision();
        let chosen: Vec<String> = decision
            .choose_spikes(1)
            .into_iter()
            .map(|spike| spike.finding)
            .collect();
        assert_eq!(chosen, ["P2-F1"]);
        let chosen: Vec<String> = decision
            .choose_spikes(5)
            .into_iter()
            .map(|spike| spike.finding)
            .collect();
        // Minor findings aren't tested; withdrawn and conceded ones neither.
        assert_eq!(chosen, ["P2-F1", "P1-F2"]);
    }

    fn ruling(chosen: &str, upheld: &[(&str, bool)]) -> Ruling {
        Ruling {
            chosen: chosen.into(),
            rationale: format!("{chosen} is better"),
            grafts: vec![],
            alternatives: vec![],
            rulings: upheld
                .iter()
                .map(|(finding, upheld)| FindingRuling {
                    finding: finding.to_string(),
                    upheld: *upheld,
                    reason: "because".into(),
                })
                .collect(),
        }
    }

    #[test]
    fn a_panel_goes_by_majority() {
        let contested = ["P1-F2".to_string(), "P2-F1".to_string()];
        let (merged, votes) = merge(
            &[
                ruling("P2", &[("P1-F2", true), ("P2-F1", false)]),
                ruling("P1", &[("P1-F2", true), ("P2-F1", true)]),
                ruling("P2", &[("P1-F2", false), ("P2-F1", false)]),
            ],
            &contested,
        );
        assert_eq!(merged.chosen, "P2");
        assert_eq!(votes["P2"], 2);
        assert!(merged.rulings[0].upheld);
        assert!(!merged.rulings[1].upheld);
        // A tie goes to the judge listed first, and upholds the finding.
        let (merged, _) = merge(
            &[
                ruling("P1", &[("P1-F2", false)]),
                ruling("P2", &[("P1-F2", true)]),
            ],
            &contested[..1],
        );
        assert_eq!(merged.chosen, "P1");
        assert!(merged.rulings[0].upheld);
    }

    #[test]
    fn a_submission_goes_to_its_runs_decision() {
        assert_eq!(
            decision_of("[forge D12 critique 3]\nReview…").as_deref(),
            Some("D12")
        );
        assert_eq!(decision_of("Build the workflow the user describes."), None);
        assert_eq!(decision_of("[forge decisions]"), None);
        assert!(is_stage_agent("proposer-2") && is_stage_agent("judge-1"));
        assert!(!is_stage_agent("architect") && !is_stage_agent("explorer"));
    }

    #[test]
    fn a_record_the_architect_never_got_is_given_again() {
        let task = arko_agents::Task::new(
            arko_agents::AgentConfig::builder(
                "architect".into(),
                "Architect".into(),
                arko_agents::agent_config::AgentLLMConfig {
                    model: arko_ai::Model::find("claude-opus-5-5").unwrap(),
                    generation_config: arko_ai::GenerationConfig::default(),
                },
            )
            .role_description("Architect".into())
            .instructions("Build it.".into())
            .build()
            .unwrap(),
            "Build it.".into(),
            std::env::temp_dir(),
            None,
        );
        let mut architect = AgentResult {
            agent_run_id: uuid::Uuid::new_v4(),
            agent_name: "architect".into(),
            status: AgentRunStatus::Completed,
            summary: String::new(),
            error: None,
            task,
        };
        architect.task.usage_history = vec![Default::default(); 7];
        let mut decision = decision();
        decision.delivered = true;
        decision.delivered_at = Some((architect.agent_run_id, 7));
        let ledger = Ledger {
            decisions: vec![decision],
            active: None,
        };
        // Stopped before the continuation was saved: given again.
        assert_eq!(to_deliver(&ledger, &architect).len(), 1);
        // The continuation ran (it made requests): delivered.
        architect.task.usage_history.push(Default::default());
        assert!(to_deliver(&ledger, &architect).is_empty());
        // Another architect run (a fresh start reads `decisions/` itself).
        architect.task.usage_history.truncate(3);
        architect.agent_run_id = uuid::Uuid::new_v4();
        assert!(to_deliver(&ledger, &architect).is_empty());
    }

    #[test]
    fn the_architect_rules_on_each_waiting_decision_before_it_builds() {
        let dir = std::env::temp_dir().join(format!("forge-lead-{}", uuid::Uuid::new_v4()));
        let crate_dir = dir.join("crate");
        std::fs::create_dir_all(&crate_dir).unwrap();
        let store = Arc::new(Store::open(&dir, &crate_dir).unwrap());
        let mut second = decision();
        second.id = "D2".into();
        second.frame.question = "Where does the run keep its state?".into();
        store
            .update(|ledger| {
                ledger.decisions = vec![decision(), second];
                ledger.active = Some("D1".into());
            })
            .unwrap();
        let ruling = Submit {
            part: Part::Ruling,
            slot: 0,
            store: store.clone(),
            config: config(Level::Light, true),
            lead: true,
        };
        let params = |chosen: &str| {
            HashMap::from([
                ("chosen".to_string(), json!(chosen)),
                ("rationale".to_string(), json!("it resumes")),
                (
                    "alternatives".to_string(),
                    json!([{"proposal": "P2", "why_not": "it can't resume"}]),
                ),
                (
                    "rulings".to_string(),
                    json!([
                        {"finding": "P1-F2", "upheld": false, "reason": "implausible"},
                        {"finding": "P2-F1", "upheld": true, "reason": "the spike showed it"}
                    ]),
                ),
            ])
        };
        // A ruling on an unknown finding is refused; one on a finding that
        // was conceded (P1-F1) is accepted and dropped.
        let mut unknown = params("P1");
        unknown.insert(
            "rulings".into(),
            json!([{"finding": "P7-F1", "upheld": true, "reason": "?"}]),
        );
        assert!(
            ruling
                .record("Build it.", unknown)
                .unwrap_err()
                .to_string()
                .contains("aren't findings of this decision")
        );
        let mut settled = params("P1");
        settled.insert(
            "rulings".into(),
            json!([
                {"finding": "P1-F1", "upheld": true, "reason": "conceded"},
                {"finding": "P1-F2", "upheld": false, "reason": "implausible"},
                {"finding": "P2-F1", "upheld": true, "reason": "the spike showed it"}
            ]),
        );
        // D2 is still to be deliberated: rule, then end the turn.
        let reply = ruling.record("Build it.", settled).unwrap();
        let recorded = store.decision("D1").unwrap().rulings[0].clone().unwrap();
        assert_eq!(recorded.rulings.len(), 2);
        assert!(reply.contains("D2 still to be settled") && reply.contains("end your turn"));
        assert!(
            crate_dir
                .join("decisions/D1-how-should-the-workflow-recover-after.md")
                .is_file()
        );
        // The last ruling: build.
        store
            .update(|ledger| ledger.active = Some("D2".into()))
            .unwrap();
        let reply = ruling.record("Build it.", params("P2")).unwrap();
        assert!(reply.contains("continue the build"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_rebuttal_must_answer_every_finding() {
        let expected = ["P1-F1".to_string(), "P1-F2".to_string()];
        assert!(covers(&expected, ["P1-F1", "P1-F2"].into_iter()).is_ok());
        assert!(covers(&expected, ["P1-F1"].into_iter()).is_err());
        assert!(covers(&expected, ["P1-F1", "P1-F2", "P9-F1"].into_iter()).is_err());
    }

    #[test]
    fn each_part_is_taken_in_its_own_stage_only() {
        let dir = std::env::temp_dir().join(format!("forge-stages-{}", uuid::Uuid::new_v4()));
        let crate_dir = dir.join("crate");
        std::fs::create_dir_all(&crate_dir).unwrap();
        let store = Arc::new(Store::open(&dir, &crate_dir).unwrap());
        let mut proposed = decision();
        proposed.id = "D2".into();
        proposed.findings = vec![None, None];
        proposed.rebuttals = vec![None, None];
        proposed.replies = vec![None, None];
        store
            .update(|ledger| ledger.decisions = vec![decision(), proposed])
            .unwrap();
        let submit = |part| Submit {
            part,
            slot: 0,
            store: store.clone(),
            config: config(Level::Full, true),
            lead: false,
        };
        let proposal = || {
            HashMap::from([
                ("title".to_string(), json!("host loop, revised")),
                ("design".to_string(), json!("a new design")),
            ])
        };
        let findings = || {
            HashMap::from([(
                "findings".to_string(),
                json!([{"severity": "major", "claim": "c", "evidence": "e", "trigger": "t"}]),
            )])
        };
        let rebuttal = || {
            HashMap::from([(
                "responses".to_string(),
                json!([
                    {"finding": "P1-F1", "answer": "incorrect", "argument": "a"},
                    {"finding": "P1-F2", "answer": "implausible", "argument": "a"}
                ]),
            )])
        };
        let reply = || {
            HashMap::from([(
                "responses".to_string(),
                json!([{"finding": "P1-F2", "stance": "withdraw", "argument": "a"}]),
            )])
        };
        let refusal = |part, params, prompt: &str| {
            submit(part).record(prompt, params).unwrap_err().to_string()
        };

        // D1 has been through every stage: none of its parts can be replaced.
        let d1 = "[forge D1 proposal 1]\n";
        assert!(refusal(Part::Proposal, proposal(), d1).contains("can't be replaced now"));
        assert!(refusal(Part::Findings, findings(), d1).contains("can't be replaced now"));
        assert!(refusal(Part::Rebuttal, rebuttal(), d1).contains("can't be changed now"));
        let d1 = store.decision("D1").unwrap();
        assert_eq!(d1.proposals[0].as_ref().unwrap().title, "host loop");
        assert_eq!(d1.findings_of(0).len(), 2);

        // D2 has only its proposals: nothing to answer or reply to yet, and
        // the design may still be replaced.
        let d2 = "[forge D2 proposal 1]\n";
        assert!(refusal(Part::Rebuttal, rebuttal(), d2).contains("hasn't been reviewed yet"));
        assert!(refusal(Part::Reply, reply(), d2).contains("hasn't answered"));
        submit(Part::Proposal).record(d2, proposal()).unwrap();
        let d2 = store.decision("D2").unwrap();
        assert_eq!(
            d2.proposals[0].as_ref().unwrap().title,
            "host loop, revised"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_finding_accepted_in_part_revises_the_design_and_stays_contested() {
        let dir = std::env::temp_dir().join(format!("forge-in-part-{}", uuid::Uuid::new_v4()));
        let crate_dir = dir.join("crate");
        std::fs::create_dir_all(&crate_dir).unwrap();
        let store = Arc::new(Store::open(&dir, &crate_dir).unwrap());
        let mut reviewed = decision();
        reviewed.rebuttals[0] = None;
        reviewed.replies = vec![None, None];
        store
            .update(|ledger| ledger.decisions = vec![reviewed])
            .unwrap();
        let submit = |part| Submit {
            part,
            slot: 0,
            store: store.clone(),
            config: config(Level::Full, true),
            lead: false,
        };
        let rebuttal = |revised: Option<&str>| {
            let mut params = HashMap::from([(
                "responses".to_string(),
                json!([
                    {"finding": "P1-F1", "answer": "accept_in_part", "argument": "half of it"},
                    {"finding": "P1-F2", "answer": "implausible", "argument": "rare"}
                ]),
            )]);
            if let Some(revised) = revised {
                params.insert("revised_design".into(), json!(revised));
            }
            params
        };
        let prompt = "[forge D1 rebuttal 1]\n";
        assert!(
            submit(Part::Rebuttal)
                .record(prompt, rebuttal(None))
                .unwrap_err()
                .to_string()
                .contains("in whole or in part")
        );
        submit(Part::Rebuttal)
            .record(prompt, rebuttal(Some("host loop, half revised")))
            .unwrap();

        // The critic replies to it, as to any contested finding.
        let reply = |findings: &[&str]| {
            let responses: Vec<Value> = findings
                .iter()
                .map(|finding| json!({"finding": finding, "stance": "maintain", "argument": "a"}))
                .collect();
            HashMap::from([("responses".to_string(), json!(responses))])
        };
        assert!(
            submit(Part::Reply)
                .record(prompt, reply(&["P1-F2"]))
                .is_err()
        );
        submit(Part::Reply)
            .record(prompt, reply(&["P1-F1", "P1-F2"]))
            .unwrap();
        let decision = store.decision("D1").unwrap();
        let finding = decision.finding("P1-F1").unwrap();
        assert_eq!(decision.standing(finding), Standing::Contested);
        let proposal = decision.proposals[0].as_ref().unwrap();
        assert_eq!(
            decision.final_design(0, proposal),
            "host loop, half revised"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn arrays_sent_as_text_are_accepted() {
        let params = HashMap::from([(
            "findings".to_string(),
            Value::String(
                r#"[{"severity":"major","claim":"c","evidence":"e","trigger":"t"}]"#.into(),
            ),
        )]);
        let findings: Findings = parse(params).unwrap();
        assert_eq!(findings.findings[0].severity, Severity::Major);
    }

    fn config(level: Level, openai: bool) -> Config {
        Flags {
            deliberate: Some(level),
            ..Flags::default()
        }
        .resolve(&"claude-opus-5-5:xhigh".parse().unwrap(), openai)
        .unwrap()
        .unwrap()
    }

    #[test]
    fn presets_mix_providers_when_they_can() {
        let light = config(Level::Light, true);
        let ids: Vec<&str> = light.proposers.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["claude-opus-5-5", "gpt-6.1-sol"]);
        // Each proposal is reviewed on the other proposer's model.
        assert_eq!(light.critics[0].id, "gpt-6.1-sol");
        assert!(light.judges.is_empty());
        assert_eq!((light.rounds, light.spikes, light.max_decisions), (1, 0, 1));
        let full = config(Level::Full, true);
        assert_eq!(full.proposers.len(), 4);
        assert_eq!(full.judges[0].id, "claude-fable-5-1");
        assert_eq!((full.rounds, full.spikes, full.gate), (2, 3, Gate::OneWay));
        let claude_only = config(Level::Full, false);
        assert!(
            claude_only
                .proposers
                .iter()
                .all(|m| m.id.starts_with("claude-"))
        );
        let lead = Flags {
            deliberate: Some(Level::Full),
            judges: vec!["lead".into()],
            proposer_models: vec!["gpt-6.1-sol".into(), "claude-fable-5-1:high".into()],
            ..Flags::default()
        }
        .resolve(&"claude-opus-5-5:xhigh".parse().unwrap(), true)
        .unwrap()
        .unwrap();
        assert!(lead.judges.is_empty());
        assert_eq!(lead.proposers.len(), 2);
        let mixed = Flags {
            deliberate: Some(Level::Light),
            judges: vec!["lead".into(), "claude-fable-5-1".into()],
            ..Flags::default()
        }
        .resolve(&"claude-opus-5-5:xhigh".parse().unwrap(), true);
        assert!(mixed.unwrap_err().contains("alone"));
    }

    /// A local stand-in for OpenAI that answers every request with 401, as
    /// an invalidated ChatGPT login does, and counts the requests. No real
    /// provider is reached.
    async fn invalid_login_server() -> (String, Arc<std::sync::atomic::AtomicUsize>) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(AtomicUsize::new(0));
        let counted = requests.clone();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                counted.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(async move {
                    let mut request = Vec::new();
                    let mut buffer = [0_u8; 8192];
                    loop {
                        let Ok(read) = socket.read(&mut buffer).await else {
                            return;
                        };
                        if read == 0 {
                            break;
                        }
                        request.extend_from_slice(&buffer[..read]);
                        let text = String::from_utf8_lossy(&request).to_ascii_lowercase();
                        if let Some(end) = text.find("\r\n\r\n") {
                            let length: usize = text[..end]
                                .lines()
                                .find_map(|line| line.strip_prefix("content-length:"))
                                .map_or(0, |value| value.trim().parse().unwrap_or(0));
                            if request.len() >= end + 4 + length {
                                break;
                            }
                        }
                    }
                    let body = r#"{"error":{"message":"Your refresh token has been invalidated. Please sign in again.","type":"invalid_request_error","code":"refresh_token_invalidated"}}"#;
                    let response = format!(
                        "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                });
            }
        });
        (url, requests)
    }

    /// A one-proposer deliberation on GPT-6.1 Sol whose OpenAI client is
    /// `url`, persisted in `state_dir`.
    struct Fixture {
        root: PathBuf,
        state_dir: PathBuf,
        config: Config,
        agent: arko_agents::AgentConfig,
        client: arko_ai::LLMClient,
        store: Arc<Store>,
    }

    impl Fixture {
        fn new(name: &str, url: &str) -> Self {
            let root = std::env::temp_dir().join(format!("forge-{name}-{}", uuid::Uuid::new_v4()));
            let state_dir = root.join("state");
            std::fs::create_dir_all(root.join("crate")).unwrap();
            std::fs::create_dir_all(&state_dir).unwrap();
            let config = Flags {
                deliberate: Some(Level::Light),
                proposers: Some(1),
                proposer_models: vec!["gpt-6.1-sol:low".into()],
                ..Flags::default()
            }
            .resolve(&"claude-opus-5-5:xhigh".parse().unwrap(), true)
            .unwrap()
            .unwrap();
            let agent = crate::Agents {
                backend: crate::Backend::ClaudeCode,
                target_dir: root.join("target"),
                metered: false,
                spike_root: state_dir.join("deliberations"),
            }
            .config(
                "proposer-1",
                "Design proposer",
                &config.proposers[0],
                "Role.",
                "Propose a design.",
                crate::Access::Read,
                &[],
            )
            .unwrap();
            let client = arko_ai::LLMClient::new().with_openai_client(
                arko_ai::providers::openai::OpenAIClient::new("test").with_base_url(url),
            );
            let store = Arc::new(Store::open(&state_dir, &root.join("crate")).unwrap());
            store
                .update(|ledger| {
                    ledger
                        .decisions
                        .push(Decision::new("D1".into(), decision().frame, None, 1))
                })
                .unwrap();
            Self {
                root,
                state_dir,
                config,
                agent,
                client,
                store,
            }
        }

        fn workflow(&self, checkpoint: Option<PathBuf>) -> Workflow {
            let mut builder = Workflow::builder()
                .name("login")
                .workspace(self.root.join("crate"))
                .llm_client(Arc::new(self.client.clone()))
                .agent("proposer-1", self.agent.clone())
                .disable_messaging()
                .persistence(PersistenceConfig::new(&self.state_dir));
            if let Some(checkpoint) = checkpoint {
                builder = builder.resume_from(checkpoint);
            }
            builder.build().unwrap()
        }

        /// The deliberator of a new process: nothing tried yet.
        fn deliberator<'a>(&'a self, workflow: &'a Workflow) -> Deliberator<'a> {
            Deliberator {
                workflow,
                state_dir: &self.state_dir,
                store: self.store.clone(),
                config: &self.config,
                arko: &self.root,
                target_dir: &self.root,
                resumed: Mutex::default(),
                architect: crate::Architect {
                    workflow,
                    state_dir: &self.state_dir,
                    model: "claude-opus-5-5:xhigh".parse().unwrap(),
                    restart: "Continue from the crate.".into(),
                    restarted: Arc::default(),
                },
            }
        }
    }

    fn proposal_job() -> Job {
        Job::start(PROPOSER, "proposal", "D1", 0, "Design it.".into(), |d| {
            d.proposals.first().is_some_and(Option::is_some)
        })
    }

    fn runs(workflow: &Workflow, job: &Job) -> Vec<AgentResult> {
        workflow
            .context()
            .agent_results
            .read()
            .iter()
            .filter(|run| run.task.initial_task_prompt.starts_with(&job.marker))
            .cloned()
            .collect()
    }

    fn failed_run(fixture: &Fixture, job: &Job, state: TaskState) -> AgentResult {
        let mut task = arko_agents::Task::new(
            fixture.agent.clone(),
            job.prompt.clone(),
            fixture.root.clone(),
            None,
        );
        let paused = matches!(state, TaskState::AwaitingErrorRecoveryApproval { .. });
        task.current_state = state;
        AgentResult {
            agent_run_id: uuid::Uuid::new_v4(),
            agent_name: "proposer-1".into(),
            status: if paused {
                AgentRunStatus::Paused
            } else {
                AgentRunStatus::Failed
            },
            summary: String::new(),
            error: Some("failed".into()),
            task,
        }
    }

    #[tokio::test]
    async fn an_invalid_login_restarts_the_job_once_then_stops_the_forge_and_a_resume_reruns_it() {
        use std::sync::atomic::Ordering;
        let (url, requests) = invalid_login_server().await;
        let fixture = Fixture::new("login", &url);
        let job = proposal_job();

        let workflow = fixture.workflow(None);
        let deliberator = fixture.deliberator(&workflow);
        let stopped = deliberator.drive("D1", &job).await.unwrap_err();
        // The run failed for its login, started afresh once, failed again,
        // and the forge stops (`settle` saves it and exits with status 3).
        let first = runs(&workflow, &job);
        assert_eq!(first.len(), 2);
        assert!(first.iter().all(|run| run.status == AgentRunStatus::Failed));
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        match stopped {
            WorkflowError::AgentInterrupted { agent, reason } => {
                assert_eq!(agent, "proposer-1");
                assert!(reason.contains("gpt-6.1-sol:low, OpenAI"), "{reason}");
                assert!(reason.contains("`codex login`"), "{reason}");
                assert!(
                    reason.contains("refresh token has been invalidated"),
                    "{reason}"
                );
            }
            other => panic!("{other:?}"),
        }
        workflow.checkpoint().await.unwrap();
        drop(deliberator);
        drop(workflow);

        // Resumed: the job starts afresh rather than taking the failed run as
        // final.
        let pointer = std::fs::read_to_string(fixture.state_dir.join("latest-checkpoint")).unwrap();
        let resumed = fixture.workflow(Some(fixture.state_dir.join(pointer.trim())));
        let deliberator = fixture.deliberator(&resumed);
        assert!(deliberator.drive("D1", &job).await.is_err());
        assert_eq!(runs(&resumed, &job).len(), 3);
        assert_eq!(requests.load(Ordering::SeqCst), 3);
        std::fs::remove_dir_all(&fixture.root).unwrap();
    }

    #[tokio::test]
    async fn another_failure_still_just_fails_the_job_and_a_repeated_login_pause_stops() {
        use std::sync::atomic::Ordering;
        let (url, requests) = invalid_login_server().await;
        let fixture = Fixture::new("other-failure", &url);
        let job = proposal_job();
        let workflow = fixture.workflow(None);
        // A run that failed for another reason (no credential pause before).
        workflow.context().agent_results.write().push(failed_run(
            &fixture,
            &job,
            TaskState::Failed {
                reason: "the model refused".into(),
                cause: Some(ErrorRecoveryCause::Refusal),
                error_kind: None,
            },
        ));
        let deliberator = fixture.deliberator(&workflow);
        deliberator.drive("D1", &job).await.unwrap();
        assert_eq!(runs(&workflow, &job).len(), 1);
        assert_eq!(requests.load(Ordering::SeqCst), 0);
        // An agent in Claude Code pauses for its login instead; one this
        // process resumed already stops the forge, as a failed one does.
        let paused = failed_run(
            &fixture,
            &job,
            TaskState::AwaitingErrorRecoveryApproval {
                error_message: "Claude Code could not start".into(),
                cause: ErrorRecoveryCause::NonRetryable,
                error_kind: None,
            },
        );
        deliberator
            .resumed
            .lock()
            .unwrap()
            .insert(paused.agent_run_id);
        workflow.context().agent_results.write().push(paused);
        assert!(matches!(
            deliberator.drive("D1", &job).await,
            Err(WorkflowError::AgentInterrupted { .. })
        ));
        assert_eq!(requests.load(Ordering::SeqCst), 0);
        std::fs::remove_dir_all(&fixture.root).unwrap();
    }

    #[tokio::test]
    async fn an_architect_that_cant_reach_its_model_starts_afresh_once_then_stops() {
        use std::sync::atomic::Ordering;
        let (url, requests) = invalid_login_server().await;
        let fixture = Fixture::new("architect-login", &url);
        let workflow = Workflow::builder()
            .name("login")
            .workspace(fixture.root.join("crate"))
            .llm_client(Arc::new(fixture.client.clone()))
            .agent(crate::ARCHITECT, fixture.agent.clone())
            .disable_messaging()
            .persistence(PersistenceConfig::new(&fixture.state_dir))
            .build()
            .unwrap();
        let architect = |restarted: bool| crate::Architect {
            workflow: &workflow,
            state_dir: &fixture.state_dir,
            model: "gpt-6.1-sol:low".parse().unwrap(),
            restart: "Continue from the crate.".into(),
            restarted: Arc::new(std::sync::atomic::AtomicBool::new(restarted)),
        };
        let architect_runs = || {
            workflow
                .context()
                .agent_results
                .read()
                .iter()
                .filter(|run| run.agent_name == crate::ARCHITECT)
                .count()
        };
        // Failed for its login (the recorded cause), started afresh once,
        // failed again: the forge stops (`call` saves and exits with 3).
        let first = architect(false);
        match first
            .attempt(workflow.run_agent(crate::ARCHITECT, "Build it."))
            .await
        {
            Err(reason) => {
                assert!(
                    reason.starts_with("architect (gpt-6.1-sol:low, OpenAI)"),
                    "{reason}"
                );
                assert!(reason.contains("`codex login`"), "{reason}");
            }
            Ok(outcome) => panic!("{outcome:?}"),
        }
        assert_eq!(architect_runs(), 2);
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        // A resumed forge whose fresh start is its resume run stops at once.
        let resumed = architect(true);
        let stopped = resumed
            .attempt(workflow.run_agent(crate::ARCHITECT, "Continue."))
            .await;
        assert!(stopped.is_err());
        assert_eq!(architect_runs(), 3);
        // A failure for another reason is the call's own outcome.
        let other = architect(false);
        let injected = {
            let mut task = arko_agents::Task::new(
                fixture.agent.clone(),
                "Build it.".into(),
                fixture.root.clone(),
                None,
            );
            task.current_state = TaskState::Failed {
                reason: "the model refused".into(),
                cause: Some(ErrorRecoveryCause::Refusal),
                error_kind: None,
            };
            AgentResult {
                agent_run_id: task.id,
                agent_name: crate::ARCHITECT.into(),
                status: AgentRunStatus::Failed,
                summary: String::new(),
                error: Some("the model refused".into()),
                task,
            }
        };
        let outcome = other
            .attempt(async {
                workflow.context().agent_results.write().push(injected);
                Err(WorkflowError::Cancelled)
            })
            .await;
        assert!(matches!(outcome, Ok(Err(WorkflowError::Cancelled))));
        assert_eq!(requests.load(Ordering::SeqCst), 3);
        std::fs::remove_dir_all(&fixture.root).unwrap();
    }

    #[test]
    fn a_login_failure_is_read_from_the_runs_recorded_states() {
        let fixture = Fixture::new("events", "http://127.0.0.1:9");
        let job = proposal_job();
        let mut run = failed_run(
            &fixture,
            &job,
            TaskState::Failed {
                reason: "failed".into(),
                cause: None,
                error_kind: None,
            },
        );
        let (run_id, other) = (run.agent_run_id, uuid::Uuid::new_v4());
        let line = |id: uuid::Uuid, state: Value| {
            json!({"timestamp": "2026-10-01T00:00:00Z", "workflow_run_id": uuid::Uuid::nil(), "agent_name": "proposer-1", "agent_run_id": id, "event": {"TaskStateChanged": state}})
                .to_string()
        };
        let pause = |cause: &str| json!({"AwaitingErrorRecoveryApproval": {"error_message": "401 refresh_token_invalidated", "cause": cause}});
        let failed = json!({"Failed": {"reason": "The model request can't succeed"}});
        let log = |lines: Vec<String>| {
            std::fs::write(
                fixture.state_dir.join("events.jsonl"),
                lines.join("\n") + "\n",
            )
            .unwrap()
        };
        let classify = |run: &AgentResult| login_failure(&fixture.state_dir, run);
        assert_eq!(classify(&run), None);
        log(vec![
            line(run_id, json!("InProgress")),
            line(run_id, pause("non_retryable")),
            line(run_id, failed.clone()),
            line(other, json!("InProgress")),
        ]);
        assert_eq!(
            classify(&run).as_deref(),
            Some("401 refresh_token_invalidated")
        );
        log(vec![
            line(other, pause("non_retryable")),
            line(run_id, failed.clone()),
        ]);
        assert_eq!(classify(&run), None);
        log(vec![
            line(run_id, pause("refusal")),
            line(run_id, failed.clone()),
        ]);
        assert_eq!(classify(&run), None);
        log(vec![
            line(run_id, pause("non_retryable")),
            line(run_id, json!("InProgress")),
            line(run_id, failed),
        ]);
        assert_eq!(classify(&run), None);
        log(vec![line(run_id, pause("non_retryable"))]);
        run.status = AgentRunStatus::Completed;
        assert_eq!(classify(&run), None);
        // The cause recorded with the failure comes first; the log is only
        // for runs saved before arko recorded it.
        run.status = AgentRunStatus::Failed;
        run.error = Some("can't succeed: 401".into());
        run.task.current_state = TaskState::Failed {
            reason: "can't succeed: 401".into(),
            cause: Some(ErrorRecoveryCause::NonRetryable),
            error_kind: None,
        };
        std::fs::remove_file(fixture.state_dir.join("events.jsonl")).unwrap();
        assert_eq!(classify(&run).as_deref(), Some("can't succeed: 401"));
        // A recorded kind tells a lost login from a request too large for
        // the model, which needs neither a sign-in nor a new configuration.
        for (kind, login) in [
            (ErrorKind::Authentication, true),
            (ErrorKind::ContextOverflow, false),
            (ErrorKind::RequestTooLarge, false),
        ] {
            run.task.current_state = TaskState::Failed {
                reason: "can't succeed: 401".into(),
                cause: Some(ErrorRecoveryCause::NonRetryable),
                error_kind: Some(kind),
            };
            assert_eq!(classify(&run).is_some(), login, "{kind:?}");
        }
        run.task.current_state = TaskState::Failed {
            reason: "refused".into(),
            cause: Some(ErrorRecoveryCause::Refusal),
            error_kind: None,
        };
        log(vec![
            line(run_id, pause("non_retryable")),
            line(run_id, json!({"Failed": {"reason": "x"}})),
        ]);
        assert_eq!(classify(&run), None);
        // A continuation starts afresh with its context, under its marker.
        let rebuttal = Job::next(
            PROPOSER,
            "proposal",
            "D1",
            0,
            "[forge D1 rebuttal 1]\nAnswer.".into(),
            "The context.".into(),
            |_| false,
        );
        assert_eq!(rebuttal.fresh, "[forge D1 proposal 1]\nThe context.");
        assert_ne!(rebuttal.fresh_start(), proposal_job().fresh_start());
        assert_eq!(job.fresh_start(), proposal_job().fresh_start());
        std::fs::remove_dir_all(&fixture.root).unwrap();
    }

    #[test]
    fn a_decision_is_named_by_its_title_or_its_questions_first_sentence() {
        let long = "What is the top-level architecture of the workflow, taken as one design? (1) the topology; (2) recovery; (3) state; (4) models.";
        let mut decision = decision();
        decision.frame.question = long.into();
        // An older ledger's frame has no title: the first sentence.
        assert_eq!(
            decision.frame.name(),
            "What is the top-level architecture of the workflow, taken as one design?"
        );
        decision.frame.title = Some("  Workflow architecture ".into());
        assert_eq!(decision.frame.name(), "Workflow architecture");
        // The record is headed and named by it, and keeps the whole question.
        let dir = std::env::temp_dir().join(format!("forge-title-{}", uuid::Uuid::new_v4()));
        let crate_dir = dir.join("crate");
        std::fs::create_dir_all(&crate_dir).unwrap();
        let store = Store::open(&dir, &crate_dir).unwrap();
        decision.rulings = vec![Some(ruling("P1", &[("P1-F2", false), ("P2-F1", true)]))];
        store
            .update(|ledger| ledger.decisions = vec![decision])
            .unwrap();
        let record = store.decide("D1", &config(Level::Light, true)).unwrap();
        assert_eq!(record, "decisions/D1-workflow-architecture.md");
        let text = std::fs::read_to_string(crate_dir.join(&record)).unwrap();
        assert!(text.starts_with("# D1: Workflow architecture\n"));
        assert!(text.contains(&format!("**Question:** {long}")));
        assert!(
            summary_line(&store.decision("D1").unwrap())
                .starts_with("- **D1** Workflow architecture: P1")
        );
        let old: Frame = serde_json::from_value(json!({"question": "Q?"})).unwrap();
        assert_eq!((old.name(), old.title), ("Q?".to_string(), None));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn the_record_says_what_won_and_what_lost() {
        let mut decision = decision();
        let config = config(Level::Light, true);
        let (ruling, votes) = merge(
            &[Ruling {
                alternatives: vec![Alternative {
                    proposal: "P2".into(),
                    why_not: "it can't resume".into(),
                }],
                ..ruling("P1", &[("P1-F2", false), ("P2-F1", true)])
            }],
            &["P1-F2".to_string(), "P2-F1".to_string()],
        );
        decision.outcome = Some(Outcome::Decided {
            record: "decisions/D1-x.md".into(),
            date: "2026-09-30".into(),
            ruling,
            votes,
        });
        let record = render_record(&decision, &config);
        assert!(record.starts_with("# D1: How should the workflow recover"));
        assert!(record.contains("Settled by deliberation on 2026-09-30"));
        assert!(record.contains("## Chosen: P1"));
        assert!(record.contains("revised host loop"));
        assert!(record.contains("**P2** (claude-opus-5-5:xhigh), \"one agent\": it can't resume"));
        let (upheld, dismissed) = record.split_once("### Dismissed").unwrap();
        assert!(upheld.contains("P1-F1") && upheld.contains("P2-F1"));
        assert!(dismissed.contains("P1-F2") && dismissed.contains("P2-F2"));
    }
}
