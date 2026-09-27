use crate::{Error, JobUsage, Result, error::invalid};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    path::Path,
    sync::{Arc, Mutex},
};
#[derive(Clone)]
pub(crate) struct Db(Arc<Database>);
struct Database {
    connection: Mutex<Connection>,
    _lock: std::fs::File,
}
impl Db {
    pub async fn open(path: &Path) -> Result<Self> {
        let path = path.to_owned();
        tokio::task::spawn_blocking(move || {
            if let Some(parent)=path.parent().filter(|p|!p.as_os_str().is_empty()) {std::fs::create_dir_all(parent)?;}
            let mut lock_path=path.as_os_str().to_owned();lock_path.push(".lock");
            let lock=std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(lock_path)?;
            // A just-dropped index can still be releasing the lock from a background
            // task in this process, so wait briefly before reporting a conflict.
            let deadline=std::time::Instant::now()+std::time::Duration::from_secs(1);
            loop {
                match lock.try_lock() {
                    Ok(())=>break,
                    Err(std::fs::TryLockError::WouldBlock) if std::time::Instant::now()<deadline=>std::thread::sleep(std::time::Duration::from_millis(10)),
                    Err(e)=>return Err(invalid(format!("index database is already open, or filesystem locking is unsupported: {e}"))),
                }
            }
            let c=Connection::open(path)?;
            c.busy_timeout(std::time::Duration::from_secs(5))?;
            c.execute_batch("PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL;")?;
            let version:i64=c.query_row("PRAGMA user_version",[],|r|r.get(0))?;
            let tables:i64=c.query_row("SELECT count(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",[],|r|r.get(0))?;
            if version!=1 && (version!=0 || tables!=0) {return Err(invalid("unsupported/legacy index database; use a new database path"));}
            c.execute_batch(SCHEMA)?;
            // A terminated process cannot leave a generation apparently active forever.
            c.execute("UPDATE jobs SET phase='paused', problem='process interrupted; resume persisted work' WHERE phase IN ('snapshotting','chunking','embedding')",[])?;
            Ok(Self(Arc::new(Database{connection:Mutex::new(c),_lock:lock})))
        }).await?
    }
    pub async fn call<T, F>(&self, f: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut Connection) -> Result<T> + Send + 'static,
    {
        let c = self.0.clone();
        tokio::task::spawn_blocking(move || {
            f(&mut *c
                .connection
                .lock()
                .map_err(|_| Error::Worker("database lock poisoned".into()))?)
        })
        .await?
    }
    pub async fn event(&self, job: &str, kind: &str, detail: serde_json::Value) -> Result<()> {
        let (job, kind) = (job.to_owned(), kind.to_owned());
        self.call(move |c| {
            c.execute(
                "INSERT INTO events(job,kind,detail) VALUES(?1,?2,?3)",
                params![job, kind, encode(&detail)?],
            )?;
            Ok(())
        })
        .await
    }
    pub async fn phase(&self, job: &str, phase: &str, problem: Option<String>) -> Result<()> {
        let (job, phase) = (job.to_owned(), phase.to_owned());
        self.call(move |c| {
            let tx = c.transaction()?;
            tx.execute(
                "UPDATE jobs SET phase=?2,problem=?3 WHERE id=?1",
                params![job, phase, problem],
            )?;
            tx.execute(
                "INSERT INTO events(job,kind,detail) VALUES(?1,'phase',?2)",
                params![
                    job,
                    encode(&serde_json::json!({"phase":phase,"problem":problem}))?
                ],
            )?;
            tx.commit()?;
            Ok(())
        })
        .await
    }
    pub async fn file(&self, job: &str, path: &str, state: &impl Serialize) -> Result<()> {
        let (job, path, state) = (job.to_owned(), path.to_owned(), encode(state)?);
        self.call(move |c| {
            c.execute(
                "UPDATE files SET state=?3 WHERE job=?1 AND path=?2",
                params![job, path, state],
            )?;
            Ok(())
        })
        .await
    }
}
pub(crate) fn encode(value: &impl Serialize) -> Result<String> {
    Ok(serde_json::to_string(value)?)
}
pub(crate) fn decode<T: DeserializeOwned>(value: &str) -> Result<T> {
    Ok(serde_json::from_str(value)?)
}
pub(crate) fn required<T>(v: Option<T>, what: &str) -> Result<T> {
    v.ok_or_else(|| invalid(format!("unknown {what}")))
}
/// A job's usage totals. Jobs recorded before usage was tracked have none.
/// Unreadable totals (such as a non-finite cost saved as `null` before costs
/// were sanitized) are logged and restart from zero rather than failing the
/// job's status and every later update.
pub(crate) fn job_usage(c: &Connection, job: &str) -> Result<JobUsage> {
    let saved: Option<String> = c
        .query_row("SELECT usage FROM job_usage WHERE job=?1", [job], |r| {
            r.get(0)
        })
        .optional()?;
    Ok(saved.map_or_else(JobUsage::default, |usage| {
        decode(&usage).unwrap_or_else(|error| {
            log::warn!("job {job}: ignoring unreadable usage totals: {error}");
            JobUsage::default()
        })
    }))
}
/// Adds to a job's usage totals, in the caller's transaction.
pub(crate) fn add_job_usage(
    c: &Connection,
    job: &str,
    record: impl FnOnce(&mut JobUsage),
) -> Result<()> {
    let mut usage = job_usage(c, job)?;
    record(&mut usage);
    c.execute(
        "INSERT INTO job_usage(job,usage) VALUES(?1,?2) ON CONFLICT(job) DO UPDATE SET usage=excluded.usage",
        params![job, encode(&usage)?],
    )?;
    Ok(())
}
const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS indexes(id TEXT PRIMARY KEY,name TEXT NOT NULL,spec TEXT NOT NULL,active TEXT,config TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,index_id TEXT NOT NULL REFERENCES indexes(id) ON DELETE CASCADE,phase TEXT NOT NULL,config TEXT NOT NULL,manifest TEXT,excluded TEXT NOT NULL DEFAULT '[]',affected TEXT NOT NULL DEFAULT '[]',changes TEXT NOT NULL DEFAULT '[]',fresh INTEGER NOT NULL DEFAULT 1,problem TEXT);
CREATE TABLE IF NOT EXISTS snapshots(hash TEXT PRIMARY KEY,content TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files(job TEXT REFERENCES jobs(id) ON DELETE CASCADE,path TEXT,hash TEXT NOT NULL REFERENCES snapshots(hash),original_hash TEXT NOT NULL,state TEXT NOT NULL,PRIMARY KEY(job,path));
CREATE TABLE IF NOT EXISTS vectors(id TEXT PRIMARY KEY,profile TEXT NOT NULL,tokens INTEGER NOT NULL,vector BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY,job TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,path TEXT NOT NULL,hash TEXT NOT NULL REFERENCES snapshots(hash),start INTEGER NOT NULL,end INTEGER NOT NULL,first_line INTEGER NOT NULL,last_line INTEGER NOT NULL,payload TEXT NOT NULL REFERENCES vectors(id),structure TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS chunks_job_payload ON chunks(job,payload);
CREATE INDEX IF NOT EXISTS files_hash ON files(hash);
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(id UNINDEXED,job UNINDEXED,path,terms,body,tokenize='unicode61');
CREATE TABLE IF NOT EXISTS events(sequence INTEGER PRIMARY KEY AUTOINCREMENT,job TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,kind TEXT NOT NULL,detail TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS events_job ON events(job,sequence);
CREATE TABLE IF NOT EXISTS requests(id INTEGER PRIMARY KEY AUTOINCREMENT,job TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,request TEXT NOT NULL,response TEXT,error TEXT);
CREATE TABLE IF NOT EXISTS job_usage(job TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,usage TEXT NOT NULL);
PRAGMA user_version=1;
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unreadable_or_non_finite_usage_never_blocks_status_or_updates() {
        let c = Connection::open_in_memory().unwrap();
        c.execute_batch(SCHEMA).unwrap();
        c.execute_batch(
            "INSERT INTO indexes VALUES('i','test','{}',NULL,'{}');
             INSERT INTO jobs(id,index_id,phase,config) VALUES('j','i','chunking','{}');",
        )
        .unwrap();
        // How a NaN cost used to be saved: serde_json writes it as null.
        let broken = r#"{"chunk_calls":1,"chunking":{"input_tokens":1,"output_tokens":1,"input_cost":null,"output_cost":0.0,"total_cost":null}}"#;
        c.execute("INSERT INTO job_usage VALUES('j',?1)", [broken])
            .unwrap();
        assert_eq!(job_usage(&c, "j").unwrap(), JobUsage::default());
        let nan = arko_ai::UsageInfo {
            input_cost: f64::NAN,
            total_cost: f64::INFINITY,
            ..arko_ai::UsageInfo::new(100, 10)
        };
        for _ in 0..2 {
            add_job_usage(&c, "j", |usage| usage.record_chunk(Some(&nan))).unwrap();
        }
        let usage = job_usage(&c, "j").unwrap();
        assert_eq!(usage.chunk_calls, 2);
        let chunking = usage.chunking.unwrap();
        assert_eq!(chunking.input_tokens, 200);
        assert_eq!((chunking.input_cost, chunking.total_cost), (0.0, 0.0));
        assert_eq!(job_usage(&c, "unknown").unwrap(), JobUsage::default());
    }
}
