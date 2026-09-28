//! Structured ballots for code-defined deliberation workflows.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use arko_agents::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};

pub const CAST_VOTE_TOOL_NAME: &str = "cast_vote";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteOption {
    pub id: String,
    pub description: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TieBreak {
    RejectTie,
    Lexicographic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BallotConfig {
    pub question: String,
    pub options: Vec<VoteOption>,
    pub eligible_voters: HashSet<String>,
    pub quorum: usize,
    pub consensus_threshold: f64,
    pub weights: HashMap<String, f64>,
    pub tie_break: TieBreak,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CastVote {
    pub voter: String,
    pub task_id: Uuid,
    pub option: String,
    pub reasoning: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BallotResult {
    pub ballot_id: Uuid,
    pub winner: String,
    pub consensus: f64,
    pub votes: Vec<CastVote>,
}

struct Ballot {
    config: BallotConfig,
    votes: HashMap<String, CastVote>,
}

#[derive(Default)]
pub struct BallotStore {
    ballots: Mutex<HashMap<Uuid, Ballot>>,
}

impl BallotStore {
    pub fn begin(&self, config: BallotConfig) -> Result<Uuid, String> {
        if config.options.len() < 2 {
            return Err("a ballot requires at least two options".to_string());
        }
        if config.quorum == 0 {
            return Err("ballot quorum must be positive".to_string());
        }
        if !(0.0..=1.0).contains(&config.consensus_threshold)
            || !config.consensus_threshold.is_finite()
        {
            return Err("consensus threshold must be finite and between 0 and 1".to_string());
        }
        let mut ids = HashSet::new();
        if config
            .options
            .iter()
            .any(|option| option.id.trim().is_empty() || !ids.insert(option.id.clone()))
        {
            return Err("ballot option IDs must be non-empty and unique".to_string());
        }
        let id = Uuid::new_v4();
        self.ballots.lock().insert(
            id,
            Ballot {
                config,
                votes: HashMap::new(),
            },
        );
        Ok(id)
    }

    fn cast(&self, ballot_id: Uuid, vote: CastVote) -> Result<(), String> {
        let mut ballots = self.ballots.lock();
        let ballot = ballots
            .get_mut(&ballot_id)
            .ok_or_else(|| "unknown ballot".to_string())?;
        if !ballot.config.eligible_voters.is_empty()
            && !ballot.config.eligible_voters.contains(&vote.voter)
        {
            return Err(format!(
                "agent '{}' is not eligible for this ballot",
                vote.voter
            ));
        }
        if !ballot
            .config
            .options
            .iter()
            .any(|option| option.id == vote.option)
        {
            return Err(format!(
                "unknown option '{}'; valid options: {}",
                vote.option,
                ballot
                    .config
                    .options
                    .iter()
                    .map(|option| option.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if ballot.votes.contains_key(&vote.voter) {
            return Err("this agent already voted in the ballot".to_string());
        }
        ballot.votes.insert(vote.voter.clone(), vote);
        Ok(())
    }

    pub fn finish(&self, ballot_id: Uuid) -> Result<BallotResult, String> {
        let ballots = self.ballots.lock();
        let ballot = ballots
            .get(&ballot_id)
            .ok_or_else(|| "unknown ballot".to_string())?;
        if ballot.votes.len() < ballot.config.quorum {
            return Err(format!(
                "quorum not reached: {} of {} votes",
                ballot.votes.len(),
                ballot.config.quorum
            ));
        }
        let mut totals = HashMap::<String, f64>::new();
        let mut total_weight = 0.0;
        for vote in ballot.votes.values() {
            let weight = ballot
                .config
                .weights
                .get(&vote.voter)
                .copied()
                .unwrap_or(1.0);
            if !weight.is_finite() || weight <= 0.0 {
                return Err(format!("invalid weight for '{}'", vote.voter));
            }
            *totals.entry(vote.option.clone()).or_default() += weight;
            total_weight += weight;
        }
        let best = totals.values().copied().fold(0.0_f64, f64::max);
        let mut winners = totals
            .iter()
            .filter(|(_, value)| **value == best)
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        winners.sort();
        if winners.len() > 1 && ballot.config.tie_break == TieBreak::RejectTie {
            return Err(format!("ballot tied between {}", winners.join(", ")));
        }
        let winner = winners
            .into_iter()
            .next()
            .ok_or_else(|| "ballot has no votes".to_string())?;
        let consensus = best / total_weight;
        if consensus < ballot.config.consensus_threshold {
            return Err(format!(
                "consensus {:.3} is below required {:.3}",
                consensus, ballot.config.consensus_threshold
            ));
        }
        Ok(BallotResult {
            ballot_id,
            winner,
            consensus,
            votes: ballot.votes.values().cloned().collect(),
        })
    }
}

pub(crate) struct CastVoteTool {
    store: Arc<BallotStore>,
    agent: String,
}
impl CastVoteTool {
    pub fn new(store: Arc<BallotStore>, agent: String) -> Self {
        Self { store, agent }
    }
}
impl std::fmt::Debug for CastVoteTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CastVoteTool")
            .field("agent", &self.agent)
            .finish()
    }
}

#[async_trait]
impl Tool for CastVoteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: CAST_VOTE_TOOL_NAME.to_string(),
            description: "Cast one structured vote in an active workflow ballot.".to_string(),
            output_based_example: None,
            parameters: vec![
                ToolParameterDefinition {
                    name: "ballot_id".to_string(),
                    description: "Ballot UUID supplied by the workflow.".to_string(),
                    param_type: ToolParameterType::String,
                    required: true,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
                ToolParameterDefinition {
                    name: "option".to_string(),
                    description: "Exact option ID.".to_string(),
                    param_type: ToolParameterType::String,
                    required: true,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
                ToolParameterDefinition {
                    name: "reasoning".to_string(),
                    description: "Independent rationale for the vote.".to_string(),
                    param_type: ToolParameterType::String,
                    required: true,
                    is_path: false,
                    flag: None,
                    default_value: None,
                },
            ],
            parallel_safe: false,
            tool_type_details: ToolTypeDetails::DeveloperDefined,
            raw_json_schema_override: None,
        }
    }
    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let ballot_id = params
            .get("ballot_id")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::MissingParameter {
                param_name: "ballot_id".to_string(),
            })?;
        let ballot_id =
            Uuid::parse_str(ballot_id).map_err(|error| ToolError::ParameterValidation {
                param_name: "ballot_id".to_string(),
                message: error.to_string(),
            })?;
        let option = params
            .get("option")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::MissingParameter {
                param_name: "option".to_string(),
            })?;
        let reasoning = params
            .get("reasoning")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::MissingParameter {
                param_name: "reasoning".to_string(),
            })?;
        self.store
            .cast(
                ballot_id,
                CastVote {
                    voter: self.agent.clone(),
                    task_id: context.task_id,
                    option: option.to_string(),
                    reasoning: reasoning.to_string(),
                    timestamp: chrono::Utc::now(),
                },
            )
            .map_err(ToolError::ExecutionFailed)?;
        Ok(format!("Vote recorded for option '{option}'."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(tie_break: TieBreak) -> BallotConfig {
        BallotConfig {
            question: "Choose".to_string(),
            options: vec![
                VoteOption {
                    id: "a".to_string(),
                    description: "A".to_string(),
                },
                VoteOption {
                    id: "b".to_string(),
                    description: "B".to_string(),
                },
            ],
            eligible_voters: HashSet::from(["one".to_string(), "two".to_string()]),
            quorum: 2,
            consensus_threshold: 0.5,
            weights: HashMap::new(),
            tie_break,
        }
    }

    fn vote(voter: &str, option: &str) -> CastVote {
        CastVote {
            voter: voter.to_string(),
            task_id: Uuid::new_v4(),
            option: option.to_string(),
            reasoning: "reason".to_string(),
            timestamp: chrono::Utc::now(),
        }
    }

    #[test]
    fn ballot_enforces_quorum_and_explicit_ties() {
        let store = BallotStore::default();
        let id = store.begin(config(TieBreak::RejectTie)).unwrap();
        store.cast(id, vote("one", "a")).unwrap();
        assert!(store.finish(id).unwrap_err().contains("quorum"));
        store.cast(id, vote("two", "b")).unwrap();
        assert!(store.finish(id).unwrap_err().contains("tied"));
    }

    #[test]
    fn deterministic_tie_break_is_reproducible() {
        let store = BallotStore::default();
        let id = store.begin(config(TieBreak::Lexicographic)).unwrap();
        store.cast(id, vote("one", "b")).unwrap();
        store.cast(id, vote("two", "a")).unwrap();
        assert_eq!(store.finish(id).unwrap().winner, "a");
    }
}
