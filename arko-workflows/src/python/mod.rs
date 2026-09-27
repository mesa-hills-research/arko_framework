//! Python bindings via PyO3 (feature-gated behind `python`).

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn arko_workflows(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWorkflowBuilder>()?;
    m.add_class::<PyWorkflow>()?;
    m.add_class::<PyAgentResult>()?;
    Ok(())
}

#[cfg(feature = "python")]
#[pyclass]
pub struct PyWorkflowBuilder {
    name: Option<String>,
    workspace: Option<String>,
    agents: Vec<(String, String)>,
    budget: Option<f64>,
    persistence_dir: Option<String>,
    auto_checkpoint: bool,
    resume_from: Option<String>,
    memory_tiers: Vec<(String, Option<usize>)>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyWorkflowBuilder {
    #[new]
    fn new() -> Self {
        Self {
            name: None,
            workspace: None,
            agents: Vec::new(),
            budget: None,
            persistence_dir: None,
            auto_checkpoint: false,
            resume_from: None,
            memory_tiers: Vec::new(),
        }
    }

    fn name(&mut self, name: String) -> PyResult<()> {
        self.name = Some(name);
        Ok(())
    }

    fn workspace(&mut self, path: String) -> PyResult<()> {
        self.workspace = Some(path);
        Ok(())
    }

    fn agent(&mut self, name: String, config_path: String) -> PyResult<()> {
        self.agents.push((name, config_path));
        Ok(())
    }

    fn budget(&mut self, limit: f64) -> PyResult<()> {
        self.budget = Some(limit);
        Ok(())
    }

    #[pyo3(signature = (checkpoint_dir, auto_checkpoint=false))]
    fn persistence(&mut self, checkpoint_dir: String, auto_checkpoint: bool) -> PyResult<()> {
        self.persistence_dir = Some(checkpoint_dir);
        self.auto_checkpoint = auto_checkpoint;
        Ok(())
    }

    fn resume_from(&mut self, checkpoint: String) -> PyResult<()> {
        self.resume_from = Some(checkpoint);
        Ok(())
    }

    #[pyo3(signature = (name, max_bytes=None))]
    fn global_memory_tier(&mut self, name: String, max_bytes: Option<usize>) -> PyResult<()> {
        self.memory_tiers.push((name, max_bytes));
        Ok(())
    }

    fn build(&self) -> PyResult<PyWorkflow> {
        let mut builder = crate::Workflow::builder();

        if let Some(ref name) = self.name {
            builder = builder.name(name.clone());
        }
        if let Some(ref ws) = self.workspace {
            builder = builder.workspace(ws.clone());
        }
        if let Some(b) = self.budget {
            builder = builder.budget(b);
        }

        if let Some(ref directory) = self.persistence_dir {
            builder = builder.persistence(crate::persistence::PersistenceConfig {
                checkpoint_dir: std::path::PathBuf::from(directory),
                auto_checkpoint: self.auto_checkpoint,
            });
        }
        if let Some(ref checkpoint) = self.resume_from {
            builder = builder.resume_from(checkpoint);
        }
        if !self.memory_tiers.is_empty() {
            let mut memory = crate::memory::MemoryConfig::new();
            for (name, max_bytes) in &self.memory_tiers {
                memory = match max_bytes {
                    Some(limit) => memory.global_tier_with_limit(name, *limit),
                    None => memory.global_tier(name),
                };
            }
            builder = builder.memory(memory);
        }

        let llm_client = std::sync::Arc::new(arko_ai::LLMClient::from_env());
        builder = builder.llm_client(llm_client);

        for (name, config_path) in &self.agents {
            let config =
                arko_agents::AgentConfig::load_from_path(std::path::Path::new(config_path))
                    .map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Failed to load agent config: {}",
                            e
                        ))
                    })?;
            builder = builder.agent(name.clone(), config);
        }

        let workflow = builder.build().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to build workflow: {}", e))
        })?;

        Ok(PyWorkflow {
            inner: std::sync::Arc::new(workflow),
        })
    }
}

#[cfg(feature = "python")]
#[pyclass]
pub struct PyWorkflow {
    inner: std::sync::Arc<crate::Workflow>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyWorkflow {
    fn run_agent<'py>(
        &self,
        py: Python<'py>,
        agent_name: String,
        prompt: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let workflow = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = workflow
                .run_agent(&agent_name, &prompt)
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("Workflow error: {}", e))
                })?;
            Ok(PyAgentResult::from(result))
        })
    }

    fn run_agents_parallel<'py>(
        &self,
        py: Python<'py>,
        agents: Vec<(String, String)>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let workflow = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let borrowed = agents
                .iter()
                .map(|(name, prompt)| (name.as_str(), prompt.clone()))
                .collect();
            workflow
                .run_agents_parallel(borrowed)
                .await
                .map(|results| {
                    results
                        .into_iter()
                        .map(PyAgentResult::from)
                        .collect::<Vec<_>>()
                })
                .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
        })
    }

    #[pyo3(signature = (amount, description, metadata=None))]
    fn add_manual_cost(
        &self,
        amount: f64,
        description: String,
        metadata: Option<String>,
    ) -> PyResult<f64> {
        self.inner
            .add_manual_cost(amount, &description, metadata)
            .map(|result| result.new_total)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    fn cost_ledger_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.cost_ledger())
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    fn checkpoint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let workflow = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            workflow
                .checkpoint()
                .await
                .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
        })
    }

    #[pyo3(signature = (agent_name, tier, max_chars=None))]
    fn read_memory(
        &self,
        agent_name: String,
        tier: String,
        max_chars: Option<usize>,
    ) -> PyResult<String> {
        self.inner
            .context()
            .memory
            .read_window(&agent_name, &tier, max_chars)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    fn write_memory(&self, agent_name: String, tier: String, content: String) -> PyResult<()> {
        self.inner
            .context()
            .memory
            .write(&agent_name, &tier, &content)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }
}

#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone)]
pub struct PyAgentResult {
    #[pyo3(get)]
    pub agent_name: String,
    #[pyo3(get)]
    pub summary: String,
    #[pyo3(get)]
    pub total_cost: f64,
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub agent_run_id: String,
}

#[cfg(feature = "python")]
impl From<crate::AgentResult> for PyAgentResult {
    fn from(result: crate::AgentResult) -> Self {
        let total_cost = result.total_cost();
        Self {
            agent_name: result.agent_name,
            summary: result.summary,
            total_cost,
            status: format!("{:?}", result.status),
            agent_run_id: result.agent_run_id.to_string(),
        }
    }
}
