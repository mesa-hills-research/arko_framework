use crate::{
    backend::{Backend, ModelRequest},
    chunker::{self, Cuts, Window},
    packing::{self, Payload},
    scheduler::{Control, Operation, Reply, Scheduler},
    source::{Snapshot, Structure},
    storage::{Db, decode, encode, required},
    *,
};
use futures::{StreamExt, stream::FuturesUnordered};
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
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
    pub active_luna: usize,
    pub active_embeddings: usize,
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
}
#[derive(Clone)]
pub struct ArkoIndex {
    pub(crate) inner: Arc<Inner>,
}
pub(crate) struct Inner {
    pub db: Db,
    pub backend: Arc<dyn Backend>,
    pub config: Config,
    pub luna: Scheduler,
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
        if &config.embedding != backend.profile() {
            return Err(error::invalid(
                "backend embedding profile differs from index configuration",
            ));
        }
        let db = Db::open(path.as_ref()).await?;
        Ok(Self {
            inner: Arc::new(Inner {
                db,
                luna: Scheduler::new(
                    backend.clone(),
                    config.concurrency,
                    config.max_transient_attempts,
                ),
                embeddings: Scheduler::new(
                    backend.clone(),
                    config.embedding_concurrency,
                    config.max_transient_attempts,
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
            let pending:i64=tx.query_row("SELECT count(*) FROM jobs WHERE index_id=?1 AND phase NOT IN ('ready','cancelled','failed')",[&i],|r|r.get(0))?;
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
            let outcome = this.run(&job, &control).await;
            if let Err(error) = outcome {
                let phase = match error {
                    Error::Paused => "paused",
                    Error::Cancelled => "cancelled",
                    Error::Blocked(_) | Error::Invalid(_) | Error::Provider { .. } => "blocked",
                    _ => "failed",
                };
                if let Err(e) = this
                    .inner
                    .db
                    .phase(&job, phase, Some(error.to_string()))
                    .await
                {
                    log::error!("failed to persist the status of job {job}: {e}");
                }
            }
            this.inner.controls.lock().await.remove(&job);
        });
        Ok(())
    }
    pub async fn pause(&self, job: &str) -> Result<()> {
        let controls = self.inner.controls.lock().await;
        required(controls.get(job), "running job")?.set(1);
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
        if status.phase == "ready" {
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
        self.inner
            .db
            .call(move |c| {
                let config: String =
                    c.query_row("SELECT config FROM jobs WHERE id=?1", [&job_key], |r| {
                        r.get(0)
                    })?;
                if decode::<Config>(&config)? != expected {
                    return Err(error::invalid(
                        "resume requires the original resolved configuration",
                    ));
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
                Ok(())
            })
            .await?;
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
                let mut result = JobStatus {
                    job: j,
                    index,
                    phase,
                    problem,
                    files: states.len(),
                    validated: 0,
                    bypassed: 0,
                    unchecked: 0,
                    active_luna: 0,
                    active_embeddings: 0,
                };
                for s in states {
                    let s: FileState = decode(&s)?;
                    result.validated += usize::from(s.complete);
                    result.bypassed += usize::from(s.bypass);
                    result.unchecked += usize::from(s.structure.status == "unchecked");
                }
                Ok(result)
            })
            .await?;
        status.active_luna = self.inner.luna.active.load(Ordering::SeqCst);
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
    pub async fn request_artifact(&self, job: &str, id: i64) -> Result<Value> {
        let job = job.to_owned();
        self.inner.db.call(move|c|{let(request,response,error):(String,Option<String>,Option<String>)=required(c.query_row("SELECT request,response,error FROM requests WHERE job=?1 AND id=?2",params![job,id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()?,"request")?;Ok(json!({"request":decode::<Value>(&request)?,"response":response.map(|v|decode::<Value>(&v)).transpose()?,"error":error}))}).await
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
            tx.execute("DELETE FROM chunk_fts WHERE job IN (SELECT id FROM jobs WHERE index_id=?1)",[&index])?;
            tx.execute("DELETE FROM indexes WHERE id=?1",[index])?;tx.commit()?;Ok(())
        }).await
    }
    async fn prepare(&self, job: &str) -> Result<()> {
        let j = job.to_owned();
        let(spec,manifest,previous):(BuildSpec,Option<String>,Option<String>)=self.inner.db.call(move|c|{
            let(spec,manifest,previous):(String,Option<String>,Option<String>)=c.query_row("SELECT i.spec,j.manifest,i.active FROM jobs j JOIN indexes i ON i.id=j.index_id WHERE j.id=?1",[j],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?)))?;Ok((decode(&spec)?,manifest,previous))}).await?;
        if manifest.is_some() {
            return Ok(());
        }
        let manifest = tokio::task::spawn_blocking(move || source::scan(&spec)).await??;
        let (j, config) = (job.to_owned(), self.inner.config.clone());
        self.inner.db.call(move|c|{
            let mut old=BTreeMap::<String,String>::new();let mut reuse=HashMap::<(String,String),FileState>::new();
            let policy=chunker::policy_hash();
            let mut policy_changed=true;
            if let Some(previous)=previous {
                let previous_policy:Option<String>=c.query_row("SELECT json_extract(detail,'$.chunking_policy') FROM events WHERE job=?1 AND kind='snapshot' ORDER BY sequence DESC LIMIT 1",[&previous],|r|r.get(0)).optional()?.flatten();
                policy_changed=previous_policy.as_deref()!=Some(policy.as_str());
                let old_config:String=c.query_row("SELECT config FROM jobs WHERE id=?1",[&previous],|r|r.get(0))?;
                let rows:Vec<(String,String,String)>=c.prepare("SELECT path,hash,state FROM files WHERE job=?1")?.query_map([previous],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?)))?.collect::<std::result::Result<_,_>>()?;
                for(path,hash,state)in rows{old.insert(path.clone(),hash.clone());if !policy_changed && decode::<Config>(&old_config)?.embedding==config.embedding{let state:FileState=decode(&state)?;if state.complete{reuse.insert((hash,Path::new(&path).extension().and_then(|e|e.to_str()).unwrap_or("").into()),state);}}}
            }
            let current:BTreeMap<_,_>=manifest.files.iter().map(|s|(s.path.clone(),s.hash.clone())).collect();
            let affected:BTreeSet<_>=old.keys().chain(current.keys()).filter(|path|policy_changed || old.get(*path)!=current.get(*path)).cloned().collect();
            let tx=c.transaction()?;
            for s in &manifest.files {
                let state=if let Some(state)=reuse.get(&(s.hash.clone(),Path::new(&s.path).extension().and_then(|e|e.to_str()).unwrap_or("").into())){state.clone()}else{
                    let structure=source::structure(s);
                    let bypass=s.text.trim().is_empty() || !(1..s.rows()).any(|row|s.line_end(row).is_ok_and(|end|structure.legal(end)) && s.line(row).is_ok_and(|line|!line.trim().is_empty()));
                    let windows=if s.text.trim().is_empty(){vec![]}else if bypass{let mut w=Window::new(0,s.text.len());w.cuts=Some(vec![]);vec![w]}else{chunker::windows(s)};
                    FileState{windows,structure,bypass,complete:bypass}
                };
                tx.execute("INSERT OR IGNORE INTO snapshots VALUES(?1,?2)",params![s.hash,s.text])?;
                tx.execute("INSERT INTO files VALUES(?1,?2,?3,?4,?5)",params![j,s.path,s.hash,s.original_hash,encode(&state)?])?;
            }
            tx.execute("UPDATE jobs SET manifest=?2,excluded=?3,affected=?4,changes=?5 WHERE id=?1",params![j,manifest.hash,encode(&manifest.excluded)?,encode(&affected)?,encode(&crate::coalesce(&old,&current))?])?;
            tx.execute("INSERT INTO events(job,kind,detail) VALUES(?1,'snapshot',?2)",params![j,encode(&json!({"manifest":manifest.hash,"files":manifest.files.len(),"excluded":manifest.excluded.len(),"affected":affected.len(),"prompt_hash":source::hash(chunker::initial_prompt()),"chunking_policy":policy,"policy_changed":policy_changed,"source_format":"lf-v1","extractors":"syn-2/pulldown-cmark-0.13","model":"gpt-5.6-luna","reasoning":"high"}))?])?;
            tx.commit()?;Ok(())}).await
    }
    async fn files(&self, job: &str) -> Result<Vec<(Snapshot, FileState)>> {
        let job = job.to_owned();
        self.inner.db.call(move|c|{
            let rows:Vec<(String,String,String,String)>=c.prepare("SELECT f.path,s.content,f.original_hash,f.state FROM files f JOIN snapshots s ON s.hash=f.hash WHERE f.job=?1 ORDER BY f.path")?.query_map([job],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?)))?.collect::<std::result::Result<_,_>>()?;
            rows.into_iter().map(|(path,text,original,state)|{let mut s=Snapshot::new(path,text.as_bytes())?;s.original_hash=original;Ok((s,decode(&state)?))}).collect()
        }).await
    }
    fn dispatch(
        &self,
        job: &str,
        request: ModelRequest,
        keys: Vec<(usize, usize)>,
        repair: Option<chunker::Repair>,
        control: Control,
    ) -> futures::future::BoxFuture<'static, DispatchResult> {
        let this = self.clone();
        let job = job.to_owned();
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
                let call=this.inner.luna.call_tracked(Operation::Chunk(request),repair.is_none(),&control,Some(progress_tx));
                tokio::pin!(call);
                let result=loop {
                    tokio::select! {
                        result=&mut call => break result,
                        Some(event)=progress_rx.recv() => {
                            this.inner.db.event(&job,"request_attempt",json!({"artifact":id,"repair":repair.is_some(),"event":event})).await?;
                        }
                    }
                };
                while let Ok(event)=progress_rx.try_recv() {
                    this.inner.db.event(&job,"request_attempt",json!({"artifact":id,"repair":repair.is_some(),"event":event})).await?;
                }
                let response = match &result {
                    Ok(Reply::Chunk(r)) => Some(encode(r)?),
                    _ => None,
                };
                let error = result.as_ref().err().map(ToString::to_string);
                this.inner
                    .db
                    .call(move |c| {
                        c.execute(
                            "UPDATE requests SET response=?2,error=?3 WHERE id=?1",
                            params![id, response, error],
                        )?;
                        Ok(())
                    })
                    .await?;
                this.inner
                    .db
                    .event(
                        &job,
                        "request_finished",
                        json!({"artifact":id,"repair":repair.is_some()}),
                    )
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
        let mut work = FuturesUnordered::new();
        let mut group = vec![];
        for (i, (s, state)) in files.iter().enumerate() {
            for (w, window) in state.windows.iter().enumerate() {
                if let Some(cuts) = &window.cuts {
                    if !window.forced
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
                    if group.len() == self.inner.config.batch_count {
                        let request = build(&group);
                        work.push(self.dispatch(
                            job,
                            request,
                            std::mem::take(&mut group),
                            None,
                            control.clone(),
                        ));
                    }
                    group.push((i, w));
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
                    Err(e) => window.problem = Some(e.to_string()),
                }
                self.canonicalize_cuts(job, s, &state.structure, window)
                    .await?;
                if let Some(cuts) = &window.cuts {
                    let failed = chunker::failures(s, &state.structure, window, cuts);
                    if !failed.is_empty() && result.is_ok() && control.check().is_ok() {
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
        let bins = packing::pack(missing, self.inner.backend.limits())?;
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
                let result = scheduler
                    .call(
                        Operation::Embed(bin.iter().map(|p| p.text.clone()).collect(), false),
                        true,
                        &control,
                    )
                    .await;
                (bin, result)
            });
        }
        let mut failure = None;
        while let Some((bin, result)) = requests.next().await {
            let result = match result {
                Ok(Reply::Embed(result)) => Ok(result),
                Ok(_) => Err(error::invalid("unexpected embedding response")),
                Err(e) => Err(e),
            };
            match result {
                Ok(result) => {
                    if result.vectors.len() != bin.len() {
                        failure = Some(error::invalid("embedding count mismatch"));
                        continue;
                    }
                    let normalized = result
                        .vectors
                        .into_iter()
                        .map(|v| packing::normalize(v, self.inner.config.embedding.dimensions))
                        .collect::<Result<Vec<_>>>();
                    let vectors = match normalized {
                        Ok(v) => v,
                        Err(e) => {
                            failure = Some(e);
                            continue;
                        }
                    };
                    let profile = encode(&self.inner.config.embedding)?;
                    let usage = result.billed_tokens;
                    let estimated: usize = bin.iter().map(|p| p.tokens).sum();
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
                            tx.commit()?;
                            Ok(())
                        })
                        .await?;
                    self.inner
                        .db
                        .event(
                            job,
                            "embedded",
                            json!({"billed_tokens":usage,"reserved_tokens":estimated}),
                        )
                        .await?;
                }
                Err(e) => failure = Some(e),
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
        let manifest = tokio::task::spawn_blocking(move || source::scan(&spec)).await??;
        let j = job.to_owned();
        self.inner.db.call(move|c|{control.check()?;let tx=c.transaction()?;
            let old_hash:String=tx.query_row("SELECT manifest FROM jobs WHERE id=?1",[&j],|r|r.get(0))?;
            tx.execute("DELETE FROM chunk_fts WHERE job=?1",[&j])?;tx.execute("DELETE FROM chunks WHERE job=?1",[&j])?;
            for mut chunk in chunks {
                chunk.id=source::hash(serde_json::to_vec(&(&j,&chunk.id))?);
                tx.execute("INSERT INTO chunks VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",params![chunk.id,j,chunk.path,chunk.hash,chunk.start,chunk.end,chunk.first_line,chunk.last_line,chunk.payload,chunk.structure])?;
                tx.execute("INSERT INTO chunk_fts(id,job,path,terms,body) VALUES(?1,?2,?3,?4,?5)",params![chunk.id,j,chunk.path,crate::search::identifier_terms(&chunk.text),chunk.text])?;
            }
            tx.execute("UPDATE jobs SET phase='ready',fresh=?2,problem=NULL WHERE id=?1",params![j,old_hash==manifest.hash])?;
            tx.execute("UPDATE indexes SET active=?1,config=(SELECT config FROM jobs WHERE id=?1) WHERE id=(SELECT index_id FROM jobs WHERE id=?1)",[&j])?;
            tx.execute("INSERT INTO events(job,kind,detail) VALUES(?1,'phase','{\"phase\":\"ready\"}')",[&j])?;
            control.check()?;tx.commit()?;Ok(())}).await
    }
}
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
        let embedding_structure = Structure {
            status: state.structure.status.clone(),
            protected: state
                .structure
                .protected
                .iter()
                .filter_map(
                    |range| match backend.input_tokens(&s.text[range.clone()], false) {
                        Ok(tokens) if tokens <= backend.limits().item_tokens => {
                            Some(Ok(range.clone()))
                        }
                        Ok(_) => None,
                        Err(e) => Some(Err(e)),
                    },
                )
                .collect::<Result<Vec<_>>>()?,
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
                    while backend.input_tokens(&s.text[start..end], false)?
                        > backend.limits().item_tokens
                    {
                        forced = true;
                        // Prefer the last whole-component boundary that actually fits.
                        // Byte midpoints must not force a split inside a component that fits.
                        let mut legal_end = None;
                        for &candidate in s.line_starts.iter().rev() {
                            if start < candidate
                                && candidate < end
                                && embedding_structure.legal(candidate)
                                && backend.input_tokens(&s.text[start..candidate], false)?
                                    <= backend.limits().item_tokens
                            {
                                legal_end = Some(candidate);
                                break;
                            }
                        }
                        end = legal_end.unwrap_or(start + (end - start) / 2);
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
                    let id = source::hash(serde_json::to_vec(&(
                        &s.path,
                        &s.hash,
                        start,
                        end,
                        &payload.id,
                    ))?);
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
