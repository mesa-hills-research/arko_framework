use crate::{ArkoIndex, BuildSpec, Result, error::invalid, source, storage::decode};
use notify::Watcher;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, time::Duration};
use tokio::sync::watch;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum FileChange {
    Added { path: String },
    Deleted { path: String },
    Modified { path: String },
    Moved { from: String, to: String },
}
/// Reconcile final states rather than replaying raw filesystem events. Rename
/// inference requires a unique deleted/added content match; duplicates stay explicit.
pub fn coalesce(old: &BTreeMap<String, String>, new: &BTreeMap<String, String>) -> Vec<FileChange> {
    let mut deleted: BTreeMap<_, _> = old
        .iter()
        .filter(|(p, _)| !new.contains_key(*p))
        .map(|(p, h)| (p.clone(), h.clone()))
        .collect();
    let mut added: BTreeMap<_, _> = new
        .iter()
        .filter(|(p, _)| !old.contains_key(*p))
        .map(|(p, h)| (p.clone(), h.clone()))
        .collect();
    let mut out = vec![];
    for (path, hash) in old {
        if let Some(current) = new.get(path)
            && current != hash
        {
            out.push(FileChange::Modified { path: path.clone() });
        }
    }
    for (from, hash) in deleted.clone() {
        let from_count = deleted.values().filter(|h| **h == hash).count();
        let matches: Vec<_> = added
            .iter()
            .filter(|(_, h)| **h == hash)
            .map(|(p, _)| p.clone())
            .collect();
        if from_count == 1 && matches.len() == 1 {
            let to = matches[0].clone();
            deleted.remove(&from);
            added.remove(&to);
            out.push(FileChange::Moved { from, to });
        }
    }
    out.extend(deleted.into_keys().map(|path| FileChange::Deleted { path }));
    out.extend(added.into_keys().map(|path| FileChange::Added { path }));
    out
}
pub struct WatchHandle {
    stop: watch::Sender<bool>,
    task: tokio::task::JoinHandle<()>,
}
impl WatchHandle {
    pub async fn stop(self) -> Result<()> {
        self.stop.send_replace(true);
        self.task
            .await
            .map_err(|e| crate::Error::Worker(e.to_string()))
    }
}
impl ArkoIndex {
    /// Watches portable notify events; a periodic reconciliation scan also recovers
    /// missed events. Bursts reset the latest due time and updates stay serialized.
    pub async fn watch(
        &self,
        index: &str,
        debounce: Duration,
        reconcile: Duration,
    ) -> Result<WatchHandle> {
        if debounce.is_zero() || reconcile.is_zero() {
            return Err(invalid("watch intervals must be positive"));
        }
        let index = index.to_owned();
        let key = index.clone();
        let spec: BuildSpec = self
            .inner
            .db
            .call(move |c| {
                let s: String =
                    c.query_row("SELECT spec FROM indexes WHERE id=?1", [key], |r| r.get(0))?;
                decode(&s)
            })
            .await?;
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let mut watcher =
            notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
                if event.as_ref().is_ok_and(|event| event.kind.is_access()) {
                    return;
                }
                let _ = tx.try_send(());
            })
            .map_err(|e| invalid(e.to_string()))?;
        watcher
            .watch(&spec.root, notify::RecursiveMode::Recursive)
            .map_err(|e| invalid(e.to_string()))?;
        let (stop, mut stopped) = watch::channel(false);
        let this = self.clone();
        let task = tokio::spawn(async move {
            let _watcher = watcher;
            let mut interval = tokio::time::interval(reconcile);
            let mut due = None;
            loop {
                let timer = async {
                    match due {
                        Some(t) => tokio::time::sleep_until(t).await,
                        None => std::future::pending().await,
                    }
                };
                tokio::select! {
                    _=stopped.changed()=>break,
                    event=rx.recv()=>{if event.is_none(){break;}due=Some(tokio::time::Instant::now()+debounce);},
                    _=interval.tick()=>{if due.is_none(){due=Some(tokio::time::Instant::now()+debounce);}},
                    _=timer=>{
                        due=None;
                        let job_index=index.clone();let status=this.inner.db.call(move|c|{
                            let pending:usize=c.query_row("SELECT count(*) FROM jobs WHERE index_id=?1 AND phase IN ('snapshotting','chunking','embedding','paused','blocked')",[&job_index],|r|r.get(0))?;
                            let mut map=BTreeMap::new();let mut stmt=c.prepare("SELECT path,hash FROM files WHERE job=(SELECT active FROM indexes WHERE id=?1)")?;
                            for row in stmt.query_map([job_index],|r|Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?)))?{let(p,h)=row?;map.insert(p,h);}Ok((pending,map))
                        }).await;
                        let (pending,old)=match status {Ok(value)=>value,Err(error)=>{eprintln!("arko-index watcher status: {error}");continue;}};if pending>0{due=Some(tokio::time::Instant::now()+debounce);continue;}
                        let scan_spec=spec.clone();let scan=tokio::task::spawn_blocking(move||source::scan(&scan_spec)).await;
                        let manifest=match scan {Ok(Ok(value))=>value,Ok(Err(error))=>{eprintln!("arko-index watcher scan: {error}");continue;},Err(error)=>{eprintln!("arko-index watcher worker: {error}");continue;}};let new=manifest.files.into_iter().map(|s|(s.path,s.hash)).collect();
                        if !coalesce(&old,&new).is_empty()&& let Err(error)=this.start_update(&index).await{eprintln!("arko-index watcher update: {error}");}
                    }
                }
            }
        });
        Ok(WatchHandle { stop, task })
    }
}
