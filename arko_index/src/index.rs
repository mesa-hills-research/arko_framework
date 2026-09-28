use crate::{
    backend::{Backend, ModelRequest},
    budget::{self, Budget},
    chunker::{self, Cuts, Window},
    packing::{self, Payload},
    scheduler::{Control, Operation, Reply, Scheduler},
    source::{Snapshot, Structure},
    storage::{self, Db, add_job_usage, decode, encode, job_usage, required},
    *,
};
use futures::{StreamExt, stream::FuturesUnordered};
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    path::Path,
    sync::{Arc, atomic::Ordering},
};
use tokio::sync::Mutex;

type DispatchResult = (
    Vec<(usize, usize)>,
    Option<chunker::Repair>,
    Result<backend::ModelReply>,
);
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobStatus {
    pub job: String,
    pub index: String,
    pub phase: String,
    pub problem: Option<String>,
    pub files: usize,
    pub validated: usize,
    pub bypassed: usize,
    pub unchecked: usize,
    /// Files left out because they exceed the chunk request budget under
    /// [`OversizePolicy::Skip`]; the job's `excluded` list names them.
    #[serde(default)]
    pub skipped: usize,
    /// Chunk-model requests in flight, across all jobs of this index service.
    #[serde(alias = "active_luna")]
    pub active_chunking: usize,
    pub active_embeddings: usize,
    /// Token usage and cost of the provider responses this job received so
    /// far. Saved costs that aren't numbers read as zero; see [`JobUsage`].
    #[serde(default)]
    pub usage: JobUsage,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Event {
    pub sequence: i64,
    pub kind: String,
    pub detail: Value,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexInfo {
    pub id: String,
    pub name: String,
    pub generation: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobHandle {
    pub index: String,
    pub job: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct FileState {
    pub windows: Vec<Window>,
    pub structure: Structure,
    pub bypass: bool,
    pub complete: bool,
    /// Why the file is left out of the generation (it has no windows).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skipped: Option<String>,
}
#[derive(Clone)]
pub struct ArkoIndex {
    pub(crate) inner: Arc<Inner>,
}
pub(crate) struct Inner {
    pub db: Db,
    /// The backend's embedding request limits, lowered by the configuration.
    pub limits: backend::Limits,
    pub backend: Arc<dyn Backend>,
    pub config: Config,
    pub chunking: Scheduler,
    pub embeddings: Scheduler,
    controls: Mutex<HashMap<String, Control>>,
    pub matrix: Arc<std::sync::Mutex<Option<crate::search::MatrixCache>>>,
    pub queries: Mutex<HashMap<String, Arc<tokio::sync::OnceCell<Vec<f32>>>>>,
}
impl ArkoIndex {
    pub async fn open(
        path: impl AsRef<Path>,
        backend: Arc<dyn Backend>,
        config: Config,
    ) -> Result<Self> {
        config.validate()?;
        if !config.embedding.same_embeddings(backend.profile()) {
            return Err(error::invalid(
                "backend embedding profile differs from index configuration",
            ));
        }
        let limits = embedding_request_limits(&config, &*backend)?;
        let db = Db::open(path.as_ref(), config.synchronous).await?;
        Ok(Self {
            inner: Arc::new(Inner {
                db,
                limits,
                chunking: Scheduler::new(
                    backend.clone(),
                    config.concurrency,
                    retry_policy(&config),
                ),
                embeddings: Scheduler::new(
                    backend.clone(),
                    config.embedding_concurrency,
                    retry_policy(&config),
                ),
                backend,
                config,
                controls: Mutex::new(HashMap::new()),
                matrix: Arc::new(std::sync::Mutex::new(None)),
                queries: Mutex::new(HashMap::new()),
            }),
        })
    }
    pub async fn start_build(&self, mut spec: BuildSpec) -> Result<JobHandle> {
        spec.validate()?;
        spec.root = std::fs::canonicalize(&spec.root)?;
        let index = uuid::Uuid::new_v4().to_string();
        let job = uuid::Uuid::new_v4().to_string();
        let (i, j, config) = (index.clone(), job.clone(), encode(&self.inner.config)?);
        self.inner
            .db
            .call(move |c| {
                let tx = c.transaction()?;
                tx.execute(
                    "INSERT INTO indexes VALUES(?1,?2,?3,NULL,?4)",
                    params![i, spec.name, encode(&spec)?, config],
                )?;
                tx.execute(
                    "INSERT INTO jobs(id,index_id,phase,config) VALUES(?1,?2,'snapshotting',?3)",
                    params![j, i, config],
                )?;
                tx.commit()?;
                Ok(())
            })
            .await?;
        self.launch(&job).await?;
        Ok(JobHandle { index, job })
    }
    pub async fn start_update(&self, index: &str) -> Result<JobHandle> {
        let (i, j, config) = (
            index.to_owned(),
            uuid::Uuid::new_v4().to_string(),
            encode(&self.inner.config)?,
        );
        let handle = JobHandle {
            index: i.clone(),
            job: j.clone(),
        };
        self.inner.db.call(move|c|{let tx=c.transaction()?;
            required(tx.query_row("SELECT id FROM indexes WHERE id=?1",[&i],|r|r.get::<_,String>(0)).optional()?,"index")?;
            let pending:i64=tx.query_row("SELECT count(*) FROM jobs WHERE index_id=?1 AND phase NOT IN ('ready','cancelled','failed','pruned')",[&i],|r|r.get(0))?;
            if pending>0{return Err(error::invalid("resume or cancel the existing update before starting another"));}
            tx.execute("INSERT INTO jobs(id,index_id,phase,config) VALUES(?1,?2,'snapshotting',?3)",params![j,i,config])?;tx.commit()?;Ok(())}).await?;
        self.launch(&handle.job).await?;
        Ok(handle)
    }
    async fn launch(&self, job: &str) -> Result<()> {
        let mut controls = self.inner.controls.lock().await;
        if controls.contains_key(job) {
            return Err(error::invalid("job is still running/draining"));
        }
        let control = Control::new();
        let key = job.to_owned();
        self.inner.db.call(move|c|{c.execute("UPDATE jobs SET phase=CASE WHEN manifest IS NULL THEN 'snapshotting' ELSE 'chunking' END,problem=NULL WHERE id=?1",[key])?;Ok(())}).await?;
        controls.insert(job.into(), control.clone());
        let this = self.clone();
        let job = job.to_owned();
        tokio::spawn(async move {
            // On success, publish records "ready" and removes the control.
            if let Err(error) = this.run(&job, &control).await {
                let error = error::usage_limit(&error).unwrap_or(error);
                let phase = match error {
                    Error::Paused | Error::UsageLimit(_) => "paused",
                    Error::Cancelled => "cancelled",
                    Error::Blocked(_) | Error::Invalid(_) | Error::Provider { .. } => "blocked",
                    _ => "failed",
                };
                let retryable = matches!(
                    error.retry_class(),
                    Some(arko_ai::RetryClass::Transient | arko_ai::RetryClass::Capacity)
                );
                // The final phase and the control's removal are one step
                // under the controls lock: once a caller sees the phase,
                // `resume` can relaunch, and no relaunch can come between
                // them (where this late write would overwrite its phase).
                let mut controls = this.inner.controls.lock().await;
                if let Err(e) = this
                    .inner
                    .db
                    .stopped(&job, phase, Some(error.to_string()), retryable)
                    .await
                {
                    log::error!("failed to persist the status of job {job}: {e}");
                }
                controls.remove(&job);
            }
        });
        Ok(())
    }
    /// Drains a running job to a checkpoint. A job being cancelled stays
    /// cancelled: pausing it is an error.
    pub async fn pause(&self, job: &str) -> Result<()> {
        let controls = self.inner.controls.lock().await;
        if required(controls.get(job), "running job")?.pause() == 2 {
            return Err(error::invalid("job is being cancelled"));
        }
        drop(controls);
        self.inner
            .db
            .event(job, "pause_requested", json!({"draining":true}))
            .await
    }
    pub async fn cancel(&self, job: &str) -> Result<()> {
        if let Some(control) = self.inner.controls.lock().await.get(job) {
            control.set(2);
            return Ok(());
        }
        let status = self.status(job).await?;
        if matches!(status.phase.as_str(), "ready" | "pruned") {
            return Err(error::invalid("completed generation cannot be cancelled"));
        }
        self.inner.db.phase(job, "cancelled", None).await
    }
    pub async fn resume(&self, job: &str) -> Result<()> {
        let status = self.status(job).await?;
        if !matches!(status.phase.as_str(), "paused" | "blocked" | "failed") {
            return Err(error::invalid("job is not resumable"));
        }
        let job_key = job.to_owned();
        let expected = self.inner.config.clone();
        let changes = self
            .inner
            .db
            .call(move |c| {
                let config: String =
                    c.query_row("SELECT config FROM jobs WHERE id=?1", [&job_key], |r| {
                        r.get(0)
                    })?;
                // Publishing it would replace newer content with older.
                let superseded: bool = c.query_row(
                    "SELECT j.rowid < a.rowid FROM jobs j JOIN indexes i ON i.id=j.index_id JOIN jobs a ON a.id=i.active WHERE j.id=?1",
                    [&job_key],
                    |r| r.get(0),
                ).optional()?.unwrap_or(false);
                if superseded {
                    return Err(error::invalid(
                        "a newer generation of the index was published since this job started; start an update instead",
                    ));
                }
                let saved: Config = decode(&config)?;
                // Only the embedding profile shapes stored output; the other
                // settings are runtime settings, taken from the current
                // configuration.
                if !saved.embedding.same_embeddings(&expected.embedding) {
                    return Err(error::invalid(format!(
                        "the job embeds with {} at {} dimensions (tokenizer revision {}), but the index is configured with {} at {} dimensions (tokenizer revision {}); resume it with its embedding profile, or cancel it and start an update",
                        saved.embedding.model,
                        saved.embedding.dimensions,
                        saved.embedding.revision_label(),
                        expected.embedding.model,
                        expected.embedding.dimensions,
                        expected.embedding.revision_label()
                    )));
                }
                let changes = runtime_changes(&saved, &expected)?;
                if !changes.is_empty() {
                    c.execute(
                        "UPDATE jobs SET config=?2 WHERE id=?1",
                        params![job_key, encode(&expected)?],
                    )?;
                    c.execute(
                        "INSERT INTO events(job,kind,detail) VALUES(?1,'settings_changed',?2)",
                        params![job_key, encode(&json!({"changes": changes}))?],
                    )?;
                }
                let policy:Option<String>=c.query_row("SELECT json_extract(detail,'$.chunking_policy') FROM events WHERE job=?1 AND kind='snapshot' ORDER BY sequence DESC LIMIT 1",[&job_key],|r|r.get(0)).optional()?.flatten();
                let has_manifest:bool=c.query_row("SELECT manifest IS NOT NULL FROM jobs WHERE id=?1",[&job_key],|r|r.get(0))?;
                if has_manifest && policy.as_deref()!=Some(chunker::policy_hash().as_str()) { return Err(error::invalid("chunking policy changed; start an update to re-evaluate source with the current policy")); }
                let states: Vec<(String, String)> = c
                    .prepare("SELECT path,state FROM files WHERE job=?1")?
                    .query_map([&job_key], |r| Ok((r.get(0)?, r.get(1)?)))?
                    .collect::<std::result::Result<_, _>>()?;
                for (path, state) in states {
                    let mut state: FileState = decode(&state)?;
                    for w in &mut state.windows {
                        if w.problem.is_some() {
                            w.repair_rounds = 0;
                            w.problem = None;
                        }
                    }
                    c.execute(
                        "UPDATE files SET state=?3 WHERE job=?1 AND path=?2",
                        params![job_key, path, encode(&state)?],
                    )?;
                }
                Ok(changes)
            })
            .await?;
        if !changes.is_empty() {
            log::info!(
                "job {job}: resuming with changed runtime settings: {}",
                changes
                    .iter()
                    .map(|c| format!(
                        "{} {} -> {}",
                        c["setting"].as_str().unwrap_or(""),
                        c["from"],
                        c["to"]
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        self.launch(job).await
    }
    pub async fn status(&self, job: &str) -> Result<JobStatus> {
        let j = job.to_owned();
        let mut status = self
            .inner
            .db
            .call(move |c| {
                let (index, phase, problem) = required(
                    c.query_row(
                        "SELECT index_id,phase,problem FROM jobs WHERE id=?1",
                        [&j],
                        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                    )
                    .optional()?,
                    "job",
                )?;
                let states: Vec<String> = c
                    .prepare("SELECT state FROM files WHERE job=?1")?
                    .query_map([&j], |r| r.get(0))?
                    .collect::<std::result::Result<_, _>>()?;
                let usage = job_usage(c, &j)?;
                let mut result = JobStatus {
                    job: j,
                    index,
                    phase,
                    problem,
                    files: states.len(),
                    validated: 0,
                    bypassed: 0,
                    unchecked: 0,
                    skipped: 0,
                    active_chunking: 0,
                    active_embeddings: 0,
                    usage,
                };
                for s in states {
                    let s: FileState = decode(&s)?;
                    result.validated += usize::from(s.complete && s.skipped.is_none());
                    result.bypassed += usize::from(s.bypass);
                    result.unchecked += usize::from(s.structure.status == "unchecked");
                    result.skipped += usize::from(s.skipped.is_some());
                }
                Ok(result)
            })
            .await?;
        status.active_chunking = self.inner.chunking.active.load(Ordering::SeqCst);
        status.active_embeddings = self.inner.embeddings.active.load(Ordering::SeqCst);
        Ok(status)
    }
    pub async fn events(&self, job: &str, after: i64, limit: usize) -> Result<Vec<Event>> {
        let job = job.to_owned();
        self.inner.db.call(move|c|{
            let rows:Vec<(i64,String,String)>=c.prepare("SELECT sequence,kind,detail FROM events WHERE job=?1 AND sequence>?2 ORDER BY sequence LIMIT ?3")?.query_map(params![job,after,i64::try_from(limit).unwrap_or(i64::MAX)],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?)))?.collect::<std::result::Result<_,_>>()?;
            rows.into_iter().map(|(sequence,kind,detail)|Ok(Event{sequence,kind,detail:decode(&detail)?})).collect()
        }).await
    }
    pub async fn list(&self) -> Result<Vec<IndexInfo>> {
        self.inner
            .db
            .call(|c| {
                Ok(
                    c.prepare("SELECT id,name,active FROM indexes ORDER BY name,id")?
                        .query_map([], |r| {
                            Ok(IndexInfo {
                                id: r.get(0)?,
                                name: r.get(1)?,
                                generation: r.get(2)?,
                            })
                        })?
                        .collect::<std::result::Result<_, _>>()?,
                )
            })
            .await
    }
    /// Detailed saved model request/response; never included in routine status/search.
    /// `attempts` lists the request's attempt events in order (starts,
    /// failures with the usage of rejected responses, retry waits,
    /// completion), for failed requests too.
    pub async fn request_artifact(&self, job: &str, id: i64) -> Result<Value> {
        let job = job.to_owned();
        self.inner.db.call(move|c|{let(request,response,error):(String,Option<String>,Option<String>)=required(c.query_row("SELECT request,response,error FROM requests WHERE job=?1 AND id=?2",params![job,id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()?,"request")?;
            let attempts:Vec<String>=c.prepare("SELECT json_extract(detail,'$.event') FROM events WHERE job=?1 AND kind='request_attempt' AND json_extract(detail,'$.artifact')=?2 ORDER BY sequence")?.query_map(params![job,id],|r|r.get(0))?.collect::<std::result::Result<_,_>>()?;
            Ok(json!({"request":decode::<Value>(&request)?,"response":response.map(|v|decode::<Value>(&v)).transpose()?,"error":error,"attempts":attempts.iter().map(|a|decode::<Value>(a)).collect::<Result<Vec<_>>>()?}))}).await
    }
    /// Coalesced additions, modifications, deletions and unambiguous moves.
    pub async fn changes(&self, job: &str) -> Result<Vec<crate::FileChange>> {
        let job = job.to_owned();
        self.inner
            .db
            .call(move |c| {
                let value: String =
                    c.query_row("SELECT changes FROM jobs WHERE id=?1", [job], |r| r.get(0))?;
                decode(&value)
            })
            .await
    }
    pub async fn delete(&self, index: &str) -> Result<()> {
        let index = index.to_owned();
        self.inner.db.call(move|c|{
            let tx=c.transaction()?;
            let running:usize=tx.query_row("SELECT count(*) FROM jobs WHERE index_id=?1 AND phase IN ('snapshotting','chunking','embedding')",[&index],|r|r.get(0))?;
            if running>0{return Err(error::invalid("cancel or pause active work before deleting the index"));}
            tx.execute("DELETE FROM indexes WHERE id=?1",[&index])?;
            let collected=storage::collect_garbage(&tx)?;
            tx.commit()?;
            log::info!("index {index} deleted; removed unreferenced data: {collected}");
            Ok(())
        }).await
    }
    async fn prepare(&self, job: &str) -> Result<()> {
        let j = job.to_owned();
        let(spec,manifest,previous):(BuildSpec,Option<String>,Option<String>)=self.inner.db.call(move|c|{
            let(spec,manifest,previous):(String,Option<String>,Option<String>)=c.query_row("SELECT i.spec,j.manifest,i.active FROM jobs j JOIN indexes i ON i.id=j.index_id WHERE j.id=?1",[j],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?)))?;Ok((decode(&spec)?,manifest,previous))}).await?;
        if manifest.is_some() {
            return Ok(());
        }
        let scan_spec = spec.clone();
        let manifest = tokio::task::spawn_blocking(move || source::scan(&scan_spec)).await??;
        let budget = Budget::resolve(&spec.chunking, &self.inner.backend.chunk_input().await);
        if !budget.max_input_known {
            log::warn!(
                "job {job}: the chunk model's input limit is unknown; chunk requests are limited to {} ({}% of an assumed {} tokens)",
                budget.describe(),
                crate::budget::DEFAULT_BUDGET_PERCENT,
                crate::budget::FALLBACK_MAX_INPUT_TOKENS
            );
        }
        let config = self.inner.config.clone();
        let this_job = job.to_owned();
        let (old, reuse, policy_changed) = self.inner.db.call(move|c|{
            let mut old=BTreeMap::<String,String>::new();let mut reuse=HashMap::<(String,String),FileState>::new();
            let policy=chunker::policy_hash();
            let job_policy=|job:&str|->Result<Option<String>>{Ok(c.query_row("SELECT json_extract(detail,'$.chunking_policy') FROM events WHERE job=?1 AND kind='snapshot' ORDER BY sequence DESC LIMIT 1",[job],|r|r.get(0)).optional()?.flatten())};
            // Complete boundaries of `job`, reusable when it ran with the same
            // chunking policy and embedding profile; also its file hashes.
            let mut add=|job:&str,old:Option<&mut BTreeMap<String,String>>|->Result<()>{
                let job_config:String=c.query_row("SELECT config FROM jobs WHERE id=?1",[job],|r|r.get(0))?;
                let usable=job_policy(job)?.as_deref()==Some(policy.as_str()) && decode::<Config>(&job_config)?.embedding.same_embeddings(&config.embedding);
                let rows:Vec<(String,String,String)>=c.prepare("SELECT path,hash,state FROM files WHERE job=?1")?.query_map([job],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?)))?.collect::<std::result::Result<_,_>>()?;
                let mut old=old;
                for(path,hash,state)in rows{
                    if let Some(old)=old.as_deref_mut(){old.insert(path.clone(),hash.clone());}
                    if usable{let state:FileState=decode(&state)?;if state.complete&&state.skipped.is_none(){reuse.entry((hash,extension(&path))).or_insert(state);}}
                }
                Ok(())
            };
            let mut policy_changed=true;
            if let Some(previous)=&previous {
                policy_changed=job_policy(previous)?.as_deref()!=Some(policy.as_str());
                add(previous,Some(&mut old))?;
            }
            // Finished chunking of the newest cancelled or failed job, such as
            // a build that was blocked and then cancelled.
            let unfinished:Option<String>=c.query_row("SELECT id FROM jobs WHERE index_id=(SELECT index_id FROM jobs WHERE id=?1) AND id<>?1 AND phase IN ('cancelled','failed') AND manifest IS NOT NULL ORDER BY rowid DESC LIMIT 1",[&this_job],|r|r.get(0)).optional()?;
            if let Some(unfinished)=unfinished {
                add(&unfinished,None)?;
            }
            Ok((old,reuse,policy_changed))
        }).await?;
        let plan_budget = budget.clone();
        let (files, oversize) =
            tokio::task::spawn_blocking(move || plan_files(manifest.files, &reuse, &plan_budget))
                .await?;
        if budget.oversize == OversizePolicy::Block && !oversize.is_empty() {
            let paths: Vec<_> = oversize.iter().filter_map(|o| o["path"].as_str()).collect();
            return Err(Error::Blocked(format!(
                "{} file(s) exceed the chunk request budget of {} and the oversize policy is block: {}",
                paths.len(),
                budget.describe(),
                paths.join(", ")
            )));
        }
        let mut excluded = manifest.excluded;
        excluded.extend(files.iter().filter_map(|(s, state)| {
            state.skipped.as_ref().map(|reason| source::Exclusion {
                path: s.path.clone(),
                reason: reason.clone(),
            })
        }));
        excluded.sort_by(|a, b| a.path.cmp(&b.path));
        let j = job.to_owned();
        let chunk_model = self.inner.backend.chunk_model_info();
        let manifest_hash = manifest.hash;
        self.inner.db.call(move|c|{
            let policy=chunker::policy_hash();
            let current:BTreeMap<_,_>=files.iter().map(|(s,_)|(s.path.clone(),s.hash.clone())).collect();
            let affected:BTreeSet<_>=old.keys().chain(current.keys()).filter(|path|policy_changed || old.get(*path)!=current.get(*path)).cloned().collect();
            let tx=c.transaction()?;
            for (s,state) in &files {
                tx.execute("INSERT OR IGNORE INTO snapshots VALUES(?1,?2)",params![s.hash,s.text])?;
                tx.execute("INSERT INTO files VALUES(?1,?2,?3,?4,?5)",params![j,s.path,s.hash,s.original_hash,encode(state)?])?;
            }
            let skipped=files.iter().filter(|(_,state)|state.skipped.is_some()).count();
            tx.execute("UPDATE jobs SET manifest=?2,excluded=?3,affected=?4,changes=?5 WHERE id=?1",params![j,manifest_hash,encode(&excluded)?,encode(&affected)?,encode(&crate::coalesce(&old,&current))?])?;
            tx.execute("INSERT INTO events(job,kind,detail) VALUES(?1,'snapshot',?2)",params![j,encode(&json!({"manifest":manifest_hash,"files":files.len(),"excluded":excluded.len(),"affected":affected.len(),"prompt_hash":source::hash(chunker::initial_prompt()),"chunking_policy":policy,"policy_changed":policy_changed,"source_format":"lf-v1","extractors":"syn-2/pulldown-cmark-0.13","model":chunk_model.as_ref().map(|m|&m.model),"reasoning":chunk_model.as_ref().map(|m|&m.reasoning),"chunk_budget":budget,"oversize_files":oversize.len(),"skipped":skipped}))?])?;
            for detail in &oversize {
                tx.execute("INSERT INTO events(job,kind,detail) VALUES(?1,'oversize',?2)",params![j,encode(detail)?])?;
            }
            tx.commit()?;Ok(())}).await
    }
    async fn files(&self, job: &str) -> Result<Vec<(Snapshot, FileState)>> {
        let job = job.to_owned();
        self.inner.db.call(move|c|{
            let rows:Vec<(String,String,String,String,String)>=c.prepare("SELECT f.path,s.content,f.hash,f.original_hash,f.state FROM files f JOIN snapshots s ON s.hash=f.hash WHERE f.job=?1 ORDER BY f.path")?.query_map([job],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?)))?.collect::<std::result::Result<_,_>>()?;
            // Snapshots are stored normalized under their hash; nothing to
            // normalize or hash again.
            rows.into_iter().map(|(path,text,hash,original,state)|Ok((Snapshot::stored(path,text,hash,original),decode(&state)?))).collect()
        }).await
    }
    fn dispatch(
        &self,
        job: &str,
        mut request: ModelRequest,
        keys: Vec<(usize, usize)>,
        repair: Option<chunker::Repair>,
        control: Control,
    ) -> futures::future::BoxFuture<'static, DispatchResult> {
        let this = self.clone();
        let job = job.to_owned();
        // One prompt-cache key per job: its initial requests and repairs
        // share the instructions prefix, before and after a resume.
        request.cache_key = Some(job.clone());
        request.stream_idle_timeout_ms = self.inner.config.stream_idle_timeout_ms;
        Box::pin(async move {
            let outcome = async {
                let (j, r) = (job.clone(), encode(&request)?);
                let id = this
                    .inner
                    .db
                    .call(move |c| {
                        c.execute(
                            "INSERT INTO requests(job,request) VALUES(?1,?2)",
                            params![j, r],
                        )?;
                        Ok(c.last_insert_rowid())
                    })
                    .await?;
                let (progress_tx,mut progress_rx)=tokio::sync::mpsc::unbounded_channel();
                let call=this.inner.chunking.call_tracked(Operation::Chunk(request),repair.is_none(),&control,Some(progress_tx));
                tokio::pin!(call);
                let outcome=loop {
                    tokio::select! {
                        outcome=&mut call => break outcome,
                        Some(event)=progress_rx.recv() => {
                            this.record_attempt(&job,json!({"artifact":id,"repair":repair.is_some(),"event":event})).await?;
                        }
                    }
                };
                while let Ok(event)=progress_rx.try_recv() {
                    this.record_attempt(&job,json!({"artifact":id,"repair":repair.is_some(),"event":event})).await?;
                }
                // Rejected responses were counted with their attempt events.
                let result = outcome.result;
                let (response, usage) = match &result {
                    Ok(Reply::Chunk(r)) => (Some(encode(r)?), Some(r.usage_info.clone())),
                    _ => (None, None),
                };
                let error = result.as_ref().err().map(ToString::to_string);
                let finished = encode(&json!({"artifact":id,"repair":repair.is_some(),"usage":usage.clone().flatten()}))?;
                let j = job.clone();
                // The saved reply, its share of the job's usage and the event
                // are recorded together.
                this.inner
                    .db
                    .call(move |c| {
                        let tx = c.transaction()?;
                        tx.execute(
                            "UPDATE requests SET response=?2,error=?3 WHERE id=?1",
                            params![id, response, error],
                        )?;
                        if let Some(usage) = &usage {
                            add_job_usage(&tx, &j, |total| total.record_chunk(usage.as_ref()))?;
                        }
                        tx.execute(
                            "INSERT INTO events(job,kind,detail) VALUES(?1,'request_finished',?2)",
                            params![j, finished],
                        )?;
                        tx.commit()?;
                        Ok(())
                    })
                    .await?;
                match result? {
                    Reply::Chunk(r) => Ok(r),
                    _ => Err(error::invalid("unexpected provider result")),
                }
            }
            .await;
            (keys, repair, outcome)
        })
    }
    /// Saves a chunk request's attempt event. A failed attempt's
    /// rejected-response usage is added to the job's totals in the same
    /// transaction, so it is counted once, even if the process stops during
    /// the retry wait that follows.
    async fn record_attempt(&self, job: &str, detail: Value) -> Result<()> {
        let event = &detail["event"];
        let rejected: Option<arko_ai::UsageInfo> = match &event["detail"]["usage"] {
            usage if event["phase"] == "failed" && !usage.is_null() => {
                Some(serde_json::from_value(usage.clone())?)
            }
            _ => None,
        };
        let (job, detail) = (job.to_owned(), encode(&detail)?);
        self.inner
            .db
            .call(move |c| {
                let tx = c.transaction()?;
                tx.execute(
                    "INSERT INTO events(job,kind,detail) VALUES(?1,'request_attempt',?2)",
                    params![job, detail],
                )?;
                if let Some(usage) = &rejected {
                    add_job_usage(&tx, &job, |total| total.record_rejected_chunk(usage))?;
                }
                tx.commit()?;
                Ok(())
            })
            .await
    }
    async fn canonicalize_cuts(
        &self,
        job: &str,
        s: &Snapshot,
        guard: &Structure,
        window: &mut Window,
    ) -> Result<()> {
        if window.forced {
            return Ok(());
        }
        if let Some(cuts) = &window.cuts {
            let fixed = chunker::remove_impossible_cuts(s, guard, window, cuts)?;
            if fixed.len() != cuts.len() {
                self.inner.db.event(job, "deterministic_repair", json!({"path":s.path,"snapshot":s.hash,"window_start":window.start,"removed_cuts":cuts.len()-fixed.len(),"reason":"no legal internal boundary in entire fixed-neighbor interval"})).await?;
                window.cuts = Some(fixed);
                window.problem = None;
            }
        }
        Ok(())
    }
    /// Applies the repair limit to `window` if its cuts are still invalid.
    async fn repair_limit(
        &self,
        job: &str,
        s: &Snapshot,
        guard: &Structure,
        window: &mut Window,
        chunking: &ChunkingOptions,
    ) -> Result<()> {
        let Some(cuts) = &window.cuts else {
            return Ok(());
        };
        if window.forced || chunker::failures(s, guard, window, cuts).is_empty() {
            return Ok(());
        }
        let rounds = window.repair_rounds;
        match chunking.repair_limit {
            RepairLimitPolicy::Merge => {
                let merged = chunker::merge_invalid_cuts(s, guard, window, cuts);
                self.inner.db.event(job, "deterministic_repair", json!({"path":s.path,"snapshot":s.hash,"window_start":window.start,"removed_cuts":cuts.len()-merged.len(),"reason":format!("repair limit reached after {rounds} round(s); disputed regions merged")})).await?;
                window.cuts = Some(merged);
                window.problem = None;
            }
            RepairLimitPolicy::Block => {
                window.problem = Some(format!(
                    "cuts still invalid after {rounds} repair round(s); the repair limit blocks the file"
                ));
            }
        }
        Ok(())
    }
    async fn run(&self, job: &str, control: &Control) -> Result<()> {
        control.check()?;
        self.prepare(job).await?;
        control.check()?;
        self.inner.db.phase(job, "chunking", None).await?;
        let mut files = self.files(job).await?;
        for (s, state) in &mut files {
            for window in &mut state.windows {
                self.canonicalize_cuts(job, s, &state.structure, window)
                    .await?;
            }
            self.inner.db.file(job, &s.path, state).await?;
        }
        let j = job.to_owned();
        let (budget, chunking): (Option<Budget>, ChunkingOptions) = self.inner.db.call(move |c| {
            let saved: Option<String> = c.query_row("SELECT json_extract(detail,'$.chunk_budget') FROM events WHERE job=?1 AND kind='snapshot' ORDER BY sequence DESC LIMIT 1", [&j], |r| r.get(0)).optional()?.flatten();
            let spec: String = c.query_row("SELECT i.spec FROM jobs j JOIN indexes i ON i.id=j.index_id WHERE j.id=?1", [&j], |r| r.get(0))?;
            Ok((saved.map(|saved| decode(&saved)).transpose()?, decode::<BuildSpec>(&spec)?.chunking))
        }).await?;
        // Whether a window's repairs have run out.
        let exhausted = |window: &Window| {
            chunking
                .max_repair_rounds
                .is_some_and(|max| window.repair_rounds >= max)
        };
        let mut work = FuturesUnordered::new();
        let mut group = vec![];
        let mut group_size = budget::Size::default();
        // Windows whose last failure was a transient provider error.
        let mut transient = HashSet::new();
        // The usage limit that paused the job.
        let mut usage_limited = None;
        for (s, state) in &mut files {
            for window in &mut state.windows {
                if window.cuts.is_some() && exhausted(window) {
                    self.repair_limit(job, s, &state.structure, window, &chunking)
                        .await?;
                }
            }
        }
        for (i, (s, state)) in files.iter().enumerate() {
            for (w, window) in state.windows.iter().enumerate() {
                if let Some(cuts) = &window.cuts {
                    if !window.forced
                        && window.problem.is_none()
                        && !chunker::failures(s, &state.structure, window, cuts).is_empty()
                    {
                        let repair = chunker::repair(s, &state.structure, window, cuts)?;
                        work.push(self.dispatch(
                            job,
                            repair.request.clone(),
                            vec![(i, w)],
                            Some(repair),
                            control.clone(),
                        ));
                    }
                } else {
                    let build = |keys: &Vec<(usize, usize)>| {
                        chunker::request(
                            &keys
                                .iter()
                                .enumerate()
                                .map(|(n, &(i, w))| {
                                    (format!("f{n}"), &files[i].0, &files[i].1.windows[w])
                                })
                                .collect::<Vec<_>>(),
                        )
                    };
                    // Initial requests hold up to batch_count windows
                    // within the request budget.
                    let size = budget::window_size(s, window);
                    let full = group.len() == self.inner.config.batch_count
                        || budget.as_ref().is_some_and(|budget| {
                            !group.is_empty() && !budget.fits(group_size + size)
                        });
                    if full {
                        let request = build(&group);
                        work.push(self.dispatch(
                            job,
                            request,
                            std::mem::take(&mut group),
                            None,
                            control.clone(),
                        ));
                        group_size = budget::Size::default();
                    }
                    group.push((i, w));
                    group_size = if group.len() == 1 {
                        size
                    } else {
                        group_size + size
                    };
                }
            }
        }
        if !group.is_empty() {
            let request = chunker::request(
                &group
                    .iter()
                    .enumerate()
                    .map(|(n, &(i, w))| (format!("f{n}"), &files[i].0, &files[i].1.windows[w]))
                    .collect::<Vec<_>>(),
            );
            work.push(self.dispatch(job, request, group, None, control.clone()));
        }
        while let Some((keys, repair, result)) = work.next().await {
            for (n, (i, w)) in keys.iter().copied().enumerate() {
                let (s, state) = &mut files[i];
                let window = &mut state.windows[w];
                match &result {
                    Ok(reply) => {
                        transient.remove(&(i, w));
                        if let Some(repair) = &repair {
                            window.repair_rounds = window.repair_rounds.saturating_add(1);
                            match chunker::apply_repair(
                                s,
                                &state.structure,
                                window,
                                window.cuts.as_ref().unwrap(),
                                repair,
                                &reply.value,
                            ) {
                                Ok(cuts) => window.cuts = Some(cuts),
                                Err(e) => window.problem = Some(e.to_string()),
                            }
                        } else {
                            match reply
                                .value
                                .get(format!("f{n}"))
                                .cloned()
                                .ok_or_else(|| error::invalid("missing file result"))
                                .and_then(|v| {
                                    serde_json::from_value::<Cuts>(v).map_err(Error::from)
                                }) {
                                Ok(cuts) => window.cuts = Some(cuts.cuts),
                                Err(e) => window.problem = Some(e.to_string()),
                            }
                        }
                    }
                    Err(Error::Paused | Error::Cancelled) => {}
                    Err(e) if error::usage_limit(e).is_some() => {
                        // The whole job pauses: queued requests stop and the
                        // window is sent again when the job is resumed.
                        if usage_limited.is_none() {
                            usage_limited = error::usage_limit(e);
                            control.pause();
                        }
                    }
                    Err(e) => {
                        window.problem = Some(e.to_string());
                        if matches!(
                            e.retry_class(),
                            Some(arko_ai::RetryClass::Transient | arko_ai::RetryClass::Capacity)
                        ) {
                            transient.insert((i, w));
                        }
                    }
                }
                self.canonicalize_cuts(job, s, &state.structure, window)
                    .await?;
                if window.cuts.is_some() && result.is_ok() && exhausted(window) {
                    self.repair_limit(job, s, &state.structure, window, &chunking)
                        .await?;
                }
                if let Some(cuts) = &window.cuts {
                    let failed = chunker::failures(s, &state.structure, window, cuts);
                    if !failed.is_empty()
                        && result.is_ok()
                        && !exhausted(window)
                        && control.check().is_ok()
                    {
                        let repair = chunker::repair(s, &state.structure, window, cuts)?;
                        window.problem = None;
                        work.push(self.dispatch(
                            job,
                            repair.request.clone(),
                            vec![(i, w)],
                            Some(repair),
                            control.clone(),
                        ));
                    }
                }
                state.complete = state.windows.iter().all(|w| {
                    w.cuts.as_ref().is_some_and(|cuts| {
                        w.problem.is_none()
                            && (w.forced
                                || chunker::failures(s, &state.structure, w, cuts).is_empty())
                    })
                });
                self.inner.db.file(job, &s.path, state).await?;
            }
        }
        if let Some(error) = usage_limited
            && !matches!(control.check(), Err(Error::Cancelled))
        {
            return Err(error);
        }
        control.check()?;
        // Includes resumed forced/bypass windows, even when there was no provider work.
        for (s, state) in &mut files {
            state.complete = state.windows.iter().all(|w| {
                w.cuts.as_ref().is_some_and(|cuts| {
                    w.problem.is_none()
                        && (w.forced || chunker::failures(s, &state.structure, w, cuts).is_empty())
                })
            });
            self.inner.db.file(job, &s.path, state).await?;
        }
        let blocked: Vec<_> = files
            .iter()
            .filter(|(_, s)| !s.complete)
            .map(|(s, _)| s.path.clone())
            .collect();
        // Files held up only by transient provider failures (retries ran
        // out) may succeed when the job is resumed unchanged.
        let only_transient = files.iter().enumerate().all(|(i, (_, state))| {
            state.complete
                || state
                    .windows
                    .iter()
                    .enumerate()
                    .all(|(w, window)| window.problem.is_none() || transient.contains(&(i, w)))
        });
        if !blocked.is_empty() && only_transient && !transient.is_empty() {
            return Err(Error::Provider {
                message: format!(
                    "{} file(s) could not be chunked after transient provider failures; first: {}",
                    blocked.len(),
                    blocked[0]
                ),
                transient: true,
                retry_after: None,
                source: None,
                usage: None,
            });
        }
        if !blocked.is_empty() {
            return Err(Error::Blocked(format!(
                "{} file(s) need repair; first: {}",
                blocked.len(),
                blocked[0]
            )));
        }
        self.inner.db.phase(job, "embedding", None).await?;
        let backend = self.inner.backend.clone();
        let prepared =
            tokio::task::spawn_blocking(move || prepare_chunks(files, &*backend)).await??;
        let all_payloads = prepared.1;
        let ids: Vec<_> = all_payloads.keys().cloned().collect();
        let existing = self
            .inner
            .db
            .call(move |c| {
                let mut found = BTreeSet::new();
                let mut statement = c.prepare("SELECT id FROM vectors WHERE id=?1")?;
                for id in ids {
                    if statement
                        .query_row([&id], |r| r.get::<_, String>(0))
                        .optional()?
                        .is_some()
                    {
                        found.insert(id);
                    }
                }
                Ok(found)
            })
            .await?;
        let missing = all_payloads
            .values()
            .filter(|p| !existing.contains(&p.id))
            .cloned()
            .collect();
        let bins = packing::pack(missing, self.inner.limits)?;
        self.inner
            .db
            .event(
                job,
                "embedding_plan",
                json!({"payloads":all_payloads.len(),"reused":existing.len(),"bins":bins.len()}),
            )
            .await?;
        let mut requests = FuturesUnordered::new();
        for bin in bins {
            let (scheduler, control) = (self.inner.embeddings.clone(), control.clone());
            requests.push(async move {
                let outcome = scheduler
                    .call(
                        Operation::Embed(bin.iter().map(|p| p.text.clone()).collect(), false),
                        true,
                        &control,
                    )
                    .await;
                (bin, outcome)
            });
        }
        let mut failure = None;
        while let Some((bin, outcome)) = requests.next().await {
            let rejected = outcome.rejected;
            let result = match outcome.result {
                Ok(Reply::Embed(result)) => Ok(result),
                Ok(_) => Err(error::invalid("unexpected embedding response")),
                Err(e) => Err(e),
            };
            match result {
                Ok(result) => {
                    let vectors = if result.vectors.len() != bin.len() {
                        Err(error::invalid("embedding count mismatch"))
                    } else {
                        result
                            .vectors
                            .into_iter()
                            .map(|v| packing::normalize(v, self.inner.config.embedding.dimensions))
                            .collect::<Result<Vec<_>>>()
                    };
                    let profile = encode(&self.inner.config.embedding)?;
                    let estimated: usize = bin.iter().map(|p| p.tokens).sum();
                    // The request was billed even when its vectors are unusable.
                    let (kind, detail) = match &vectors {
                        Ok(_) => (
                            "embedded",
                            json!({"billed_tokens":result.billed_tokens,"reserved_tokens":estimated,"usage":result.usage_info,"rejected_usage":rejected}),
                        ),
                        Err(e) => (
                            "embedding_rejected",
                            json!({"error":e.to_string(),"billed_tokens":result.billed_tokens,"usage":result.usage_info,"rejected_usage":rejected}),
                        ),
                    };
                    let (j, usage, detail) = (job.to_owned(), result.usage_info, encode(&detail)?);
                    let vectors = match vectors {
                        Ok(v) => v,
                        Err(e) => {
                            failure = Some(e);
                            vec![]
                        }
                    };
                    // Vectors, the request's share of the job's usage and the
                    // event are recorded together.
                    self.inner
                        .db
                        .call(move |c| {
                            let tx = c.transaction()?;
                            for (p, v) in bin.iter().zip(vectors) {
                                let blob: Vec<u8> =
                                    v.iter().flat_map(|v| v.to_le_bytes()).collect();
                                tx.execute(
                                    "INSERT OR IGNORE INTO vectors VALUES(?1,?2,?3,?4)",
                                    params![p.id, profile, p.tokens, blob],
                                )?;
                            }
                            add_job_usage(&tx, &j, |total| {
                                for usage in &rejected {
                                    total.record_rejected_embedding(usage);
                                }
                                total.record_embedding(usage.as_ref());
                            })?;
                            tx.execute(
                                "INSERT INTO events(job,kind,detail) VALUES(?1,?2,?3)",
                                params![j, kind, detail],
                            )?;
                            tx.commit()?;
                            Ok(())
                        })
                        .await?;
                }
                Err(e) => {
                    // Responses the backend rejected before the request
                    // failed are counted with the event that lists them.
                    if !rejected.is_empty() || !matches!(e, Error::Paused | Error::Cancelled) {
                        let j = job.to_owned();
                        let detail =
                            encode(&json!({"error":e.to_string(),"rejected_usage":rejected}))?;
                        self.inner
                            .db
                            .call(move |c| {
                                let tx = c.transaction()?;
                                if !rejected.is_empty() {
                                    add_job_usage(&tx, &j, |total| {
                                        for usage in &rejected {
                                            total.record_rejected_embedding(usage);
                                        }
                                    })?;
                                }
                                tx.execute(
                                    "INSERT INTO events(job,kind,detail) VALUES(?1,'embedding_failed',?2)",
                                    params![j, detail],
                                )?;
                                tx.commit()?;
                                Ok(())
                            })
                            .await?;
                    }
                    failure = Some(e);
                }
            }
        }
        control.check()?;
        if let Some(error) = failure {
            return Err(error);
        }
        self.publish(job, prepared.0, control.clone()).await?;
        Ok(())
    }
    async fn publish(&self, job: &str, chunks: Vec<PreparedChunk>, control: Control) -> Result<()> {
        let j = job.to_owned();
        let spec: BuildSpec = self
            .inner
            .db
            .call(move |c| {
                let s: String = c.query_row(
                    "SELECT i.spec FROM indexes i JOIN jobs j ON j.index_id=i.id WHERE j.id=?1",
                    [j],
                    |r| r.get(0),
                )?;
                decode(&s)
            })
            .await?;
        // Whether the source still matches the snapshot; a scan that fails
        // (the root was moved, say) says it may not, and publishes anyway.
        let scanned = match tokio::task::spawn_blocking(move || source::scan(&spec)).await? {
            Ok(manifest) => Some(manifest.hash),
            Err(error) => {
                log::warn!("job {job}: could not rescan the source at publish: {error}");
                None
            }
        };
        let j = job.to_owned();
        // "ready" becomes visible together with the control's removal, as the
        // other final phases do in `launch`; pause and cancel wait for both.
        let mut controls = self.inner.controls.lock().await;
        let keep_generations = self.inner.config.keep_generations;
        self.inner.db.call(move|c|{control.check()?;let mut tx=c.transaction()?;
            let old_hash:String=tx.query_row("SELECT manifest FROM jobs WHERE id=?1",[&j],|r|r.get(0))?;
            tx.execute("DELETE FROM chunks WHERE job=?1",[&j])?;
            for mut chunk in chunks {
                // Generations that keep a chunk share its text row.
                let text=storage::chunk_text(&tx,&chunk.id,&chunk.path,&chunk.text)?;
                chunk.id=source::hash(serde_json::to_vec(&(&j,&chunk.id))?);
                tx.execute("INSERT INTO chunks(id,job,path,hash,start,end,first_line,last_line,payload,structure,text_id) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",params![chunk.id,j,chunk.path,chunk.hash,chunk.start,chunk.end,chunk.first_line,chunk.last_line,chunk.payload,chunk.structure,text])?;
            }
            tx.execute("UPDATE jobs SET phase='ready',fresh=?2,problem=NULL WHERE id=?1",params![j,scanned.as_ref()==Some(&old_hash)])?;
            tx.execute("UPDATE indexes SET active=?1,config=(SELECT config FROM jobs WHERE id=?1) WHERE id=(SELECT index_id FROM jobs WHERE id=?1)",[&j])?;
            tx.execute("INSERT INTO events(job,kind,detail) VALUES(?1,'phase','{\"phase\":\"ready\"}')",[&j])?;
            // Pruning is part of publishing: once the job is `ready`, older
            // generations are pruned. A failure keeps them and is logged.
            if let Some(keep)=keep_generations {
                let savepoint=tx.savepoint()?;
                match prune_generations(&savepoint,&j,keep) {
                    Ok(())=>savepoint.commit()?,
                    Err(error)=>log::warn!("job {j}: could not prune older generations: {error}"),
                }
            }
            control.check()?;tx.commit()?;Ok(())}).await?;
        controls.remove(job);
        Ok(())
    }
}
/// Prunes the complete generations of `job`'s index older than the newest
/// `keep` (never the active one), then collects data nothing references; in
/// the caller's transaction.
fn prune_generations(c: &rusqlite::Connection, job: &str, keep: usize) -> Result<()> {
    let index: String =
        c.query_row("SELECT index_id FROM jobs WHERE id=?1", [job], |r| r.get(0))?;
    let old: Vec<String> = c
        .prepare("SELECT id FROM jobs WHERE index_id=?1 AND phase='ready' AND id<>(SELECT active FROM indexes WHERE id=?1) ORDER BY rowid DESC")?
        .query_map([&index], |r| r.get(0))?
        .collect::<std::result::Result<_, _>>()?;
    let old: Vec<String> = old.into_iter().skip(keep.saturating_sub(1)).collect();
    if old.is_empty() {
        return Ok(());
    }
    for generation in &old {
        for table in ["chunks", "files", "requests"] {
            c.execute(&format!("DELETE FROM {table} WHERE job=?1"), [generation])?;
        }
        c.execute("UPDATE jobs SET phase='pruned' WHERE id=?1", [generation])?;
        c.execute(
            "INSERT INTO events(job,kind,detail) VALUES(?1,'phase','{\"phase\":\"pruned\"}')",
            [generation],
        )?;
    }
    let collected = storage::collect_garbage(c)?;
    c.execute(
        "INSERT INTO events(job,kind,detail) VALUES(?1,'pruned',?2)",
        params![
            job,
            encode(&json!({"generations": old, "removed": collected}))?
        ],
    )?;
    Ok(())
}
/// The runtime settings that differ between a job's saved configuration and
/// the current one, as `{setting, from, to}`.
fn runtime_changes(saved: &Config, current: &Config) -> Result<Vec<Value>> {
    let (Value::Object(saved), Value::Object(current)) =
        (serde_json::to_value(saved)?, serde_json::to_value(current)?)
    else {
        return Ok(vec![]);
    };
    Ok(current
        .iter()
        .filter(|(setting, value)| *setting != "embedding" && saved.get(*setting) != Some(value))
        .map(
            |(setting, value)| json!({"setting": setting, "from": saved.get(setting), "to": value}),
        )
        .collect())
}
/// The backend's embedding request limits, lowered where the configuration
/// asks; a setting may not raise them.
fn embedding_request_limits(config: &Config, backend: &dyn Backend) -> Result<backend::Limits> {
    let mut limits = backend.limits();
    if let Some(tokens) = config.embedding_batch_tokens {
        if tokens > limits.batch_tokens || tokens < limits.item_tokens {
            return Err(error::invalid(format!(
                "embedding_batch_tokens must be between the model's per-input limit ({}) and its per-request limit ({})",
                limits.item_tokens, limits.batch_tokens
            )));
        }
        limits.batch_tokens = tokens;
    }
    if let Some(items) = config.embedding_batch_items {
        if items == 0 || items > limits.batch_items {
            return Err(error::invalid(format!(
                "embedding_batch_items must be between 1 and the model's limit ({})",
                limits.batch_items
            )));
        }
        limits.batch_items = items;
    }
    Ok(limits)
}
fn retry_policy(config: &Config) -> crate::scheduler::RetryPolicy {
    crate::scheduler::RetryPolicy {
        max_transient_attempts: config.max_transient_attempts,
        max_capacity_wait: config
            .max_capacity_wait_ms
            .map(std::time::Duration::from_millis),
    }
}
fn extension(path: &str) -> String {
    Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .into()
}
/// The state of every scanned file: reused from an earlier job when its
/// content and extension match a complete one, otherwise planned under
/// `budget`. Also returns the `oversize` event of each file over the budget.
fn plan_files(
    files: Vec<Snapshot>,
    reuse: &HashMap<(String, String), FileState>,
    budget: &Budget,
) -> (Vec<(Snapshot, FileState)>, Vec<Value>) {
    let mut oversize = vec![];
    let planned = files
        .into_iter()
        .map(|s| {
            if let Some(state) = reuse.get(&(s.hash.clone(), extension(&s.path))) {
                return (s, state.clone());
            }
            let structure = source::structure(&s);
            let empty = s.text.trim().is_empty();
            let bypass = empty
                || !(1..s.rows()).any(|row| {
                    s.line_end(row).is_ok_and(|end| structure.legal(end))
                        && s.line(row).is_ok_and(|line| !line.trim().is_empty())
                });
            let mut state = FileState {
                windows: vec![],
                structure,
                bypass,
                complete: bypass,
                skipped: None,
            };
            if empty {
                return (s, state);
            }
            // A file without a legal internal boundary needs no model
            // request; it is measured only for the skip and block policies.
            let plan = (!bypass || budget.oversize != OversizePolicy::Split)
                .then(|| budget::plan(&s, &state.structure, budget));
            let Some(plan) = plan else {
                let mut window = Window::new(0, s.text.len());
                window.cuts = Some(vec![]);
                state.windows.push(window);
                return (s, state);
            };
            if plan.oversize {
                let action = match budget.oversize {
                    OversizePolicy::Split => "split",
                    OversizePolicy::Skip => "skipped",
                    OversizePolicy::Block => "blocked",
                };
                oversize.push(json!({"path":s.path,"bytes":plan.size.bytes,"tokens":plan.size.tokens,"budget":budget.describe(),"action":action,"windows":plan.windows.len(),"forced_rows":plan.forced_rows}));
                if budget.oversize == OversizePolicy::Skip {
                    state.skipped = Some(source::TOO_LARGE_FOR_A_REQUEST.into());
                    state.complete = true;
                }
            }
            if bypass && state.skipped.is_none() {
                // One chunk without a model request.
                let mut window = Window::new(0, s.text.len());
                window.cuts = Some(vec![]);
                state.windows = vec![window];
            } else {
                state.windows = plan.windows;
            }
            (s, state)
        })
        .collect();
    (planned, oversize)
}
/// Bytes per token above which text is certain to exceed an embedding input
/// in practice: splitting considers at most this many bytes per token of
/// the input limit at a time.
const PROBE_BYTES_PER_TOKEN: usize = 16;
struct PreparedChunk {
    id: String,
    path: String,
    hash: String,
    start: usize,
    end: usize,
    first_line: usize,
    last_line: usize,
    payload: String,
    structure: String,
    text: String,
}
fn prepare_chunks(
    files: Vec<(Snapshot, FileState)>,
    backend: &dyn Backend,
) -> Result<(Vec<PreparedChunk>, BTreeMap<String, Payload>)> {
    let mut chunks = vec![];
    let mut payloads = BTreeMap::new();
    for (s, state) in files {
        if state.skipped.is_some() {
            continue;
        }
        let item_tokens = backend.limits().item_tokens;
        // Text longer than this is split without tokenizing all of it, so
        // splitting a huge chunk stays linear in its size.
        let probe_bytes = item_tokens.saturating_mul(PROBE_BYTES_PER_TOKEN);
        let embedding_structure = Structure {
            status: state.structure.status.clone(),
            protected: state
                .structure
                .protected
                .iter()
                .filter_map(|range| {
                    if range.len() > probe_bytes {
                        return None;
                    }
                    match backend.input_tokens(&s.text[range.clone()], false) {
                        Ok(tokens) if tokens <= item_tokens => Some(Ok(range.clone())),
                        Ok(_) => None,
                        Err(e) => Some(Err(e)),
                    }
                })
                .collect::<Result<Vec<_>>>()?,
        };
        let fits = |start: usize, end: usize| -> Result<bool> {
            Ok(backend.input_tokens(&s.text[start..end], false)? <= item_tokens)
        };
        let mut covered = 0;
        for w in &state.windows {
            let ranges = if w.forced {
                std::iter::once(w.start..w.end).collect()
            } else {
                chunker::spans(&s, &state.structure, w, w.cuts.as_ref().unwrap())?
            };
            for range in ranges {
                let mut start = range.start;
                while start < range.end {
                    let mut end = range.end;
                    let mut forced = w.forced;
                    // Far more text than one input can hold: consider only a
                    // prefix of it.
                    if end - start > probe_bytes {
                        forced = true;
                        end = start + probe_bytes;
                        while !s.text.is_char_boundary(end) {
                            end -= 1;
                        }
                    }
                    while !fits(start, end)? {
                        forced = true;
                        // Prefer the last whole-component boundary that actually fits
                        // (a binary search: longer prefixes have more tokens). Byte
                        // midpoints must not force a split inside a component that fits.
                        let (from, to) = (
                            s.line_starts.partition_point(|&c| c <= start),
                            s.line_starts.partition_point(|&c| c < end),
                        );
                        let candidates: Vec<usize> = s.line_starts[from..to]
                            .iter()
                            .copied()
                            .filter(|&c| embedding_structure.legal(c))
                            .collect();
                        let (mut low, mut high) = (0, candidates.len());
                        while low < high {
                            let middle = (low + high) / 2;
                            if fits(start, candidates[middle])? {
                                low = middle + 1;
                            } else {
                                high = middle;
                            }
                        }
                        end = match low.checked_sub(1) {
                            Some(index) => candidates[index],
                            None => start + (end - start) / 2,
                        };
                        while end > start && !s.text.is_char_boundary(end) {
                            end -= 1;
                        }
                        if end == start {
                            return Err(error::invalid(
                                "one Unicode scalar exceeds embedding limit",
                            ));
                        }
                    }
                    if start != covered {
                        return Err(error::invalid("source reconstruction gap/overlap"));
                    }
                    covered = end;
                    let text = s.text[start..end].to_owned();
                    let payload = packing::payload(backend, text.clone())?;
                    let id = storage::chunk_text_key(&s.path, &s.hash, start, end, &payload.id)?;
                    chunks.push(PreparedChunk {
                        id,
                        path: s.path.clone(),
                        hash: s.hash.clone(),
                        start,
                        end,
                        first_line: s.row_at(start),
                        last_line: s.row_at(end - 1),
                        payload: payload.id.clone(),
                        structure: if forced {
                            "forced_split".into()
                        } else {
                            state.structure.status.clone()
                        },
                        text,
                    });
                    payloads.entry(payload.id.clone()).or_insert(payload);
                    start = end;
                }
            }
        }
        if !s.text.trim().is_empty() && covered != s.text.len() {
            return Err(error::invalid("incomplete source reconstruction"));
        }
    }
    Ok((chunks, payloads))
}
