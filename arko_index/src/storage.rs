use crate::{Error, JobUsage, Result, error::invalid};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    path::Path,
    sync::{Arc, Mutex},
};
/// The problem of a job whose process stopped while it ran, set when the
/// database is next opened.
pub(crate) const INTERRUPTED: &str = "process interrupted; resume persisted work";
#[derive(Clone)]
pub(crate) struct Db(Arc<Database>);
struct Database {
    connection: Mutex<Connection>,
    _lock: std::fs::File,
}
impl Db {
    pub async fn open(path: &Path, synchronous: crate::Synchronous) -> Result<Self> {
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
            let mut c=Connection::open(path)?;
            c.busy_timeout(std::time::Duration::from_secs(5))?;
            c.execute_batch("PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL;")?;
            if synchronous==crate::Synchronous::Normal {c.execute_batch("PRAGMA synchronous=NORMAL;")?;}
            let version:i64=c.query_row("PRAGMA user_version",[],|r|r.get(0))?;
            let tables:i64=c.query_row("SELECT count(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",[],|r|r.get(0))?;
            match version {
                0 if tables==0 => {}
                1 => migrate_v1(&mut c)?,
                SCHEMA_VERSION => {}
                _ => return Err(invalid("unsupported/legacy index database; use a new database path")),
            }
            c.execute_batch(SCHEMA)?;
            // A terminated process cannot leave a generation apparently active
            // forever. A job the host asked to pause (its last pause request is
            // newer than its last phase change) stays paused as asked; the others
            // are marked interrupted, which a watcher resumes.
            let tx = c.transaction()?;
            let mut jobs = tx.prepare("SELECT id, CASE WHEN (SELECT kind FROM events WHERE job=jobs.id AND kind IN ('pause_requested','phase') ORDER BY sequence DESC LIMIT 1)='pause_requested' THEN 'paused' ELSE ?1 END FROM jobs WHERE phase IN ('snapshotting','chunking','embedding')")?;
            let interrupted = jobs
                .query_map([INTERRUPTED], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            drop(jobs);
            for (job, problem) in interrupted {
                tx.execute(
                    "UPDATE jobs SET phase='paused',problem=?2 WHERE id=?1",
                    rusqlite::params![job, problem],
                )?;
                let detail = serde_json::json!({"phase":"paused","problem":problem});
                tx.execute(
                    "INSERT INTO events(job,kind,detail) VALUES(?1,'phase',?2)",
                    rusqlite::params![job, encode(&detail)?],
                )?;
            }
            tx.commit()?;
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
        self.stopped(job, phase, problem, false).await
    }
    /// Records a phase; `retryable` marks a stop caused by transient
    /// provider failures, which may succeed when resumed unchanged.
    pub async fn stopped(
        &self,
        job: &str,
        phase: &str,
        problem: Option<String>,
        retryable: bool,
    ) -> Result<()> {
        let (job, phase) = (job.to_owned(), phase.to_owned());
        self.call(move |c| {
            let tx = c.transaction()?;
            tx.execute(
                "UPDATE jobs SET phase=?2,problem=?3 WHERE id=?1",
                params![job, phase, problem],
            )?;
            let mut detail = serde_json::json!({"phase":phase,"problem":problem});
            if retryable {
                detail["retryable"] = true.into();
            }
            tx.execute(
                "INSERT INTO events(job,kind,detail) VALUES(?1,'phase',?2)",
                params![job, encode(&detail)?],
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
/// Totals saved with a cost that isn't a number (a non-finite cost was once
/// saved as `null`) load with that cost as zero, keeping every count; totals
/// that still can't be read are logged and restart from zero rather than
/// failing the job's status and every later update.
pub(crate) fn job_usage(c: &Connection, job: &str) -> Result<JobUsage> {
    let saved: Option<String> = c
        .query_row("SELECT usage FROM job_usage WHERE job=?1", [job], |r| {
            r.get(0)
        })
        .optional()?;
    let Some(saved) = saved else {
        return Ok(JobUsage::default());
    };
    Ok(decode(&saved)
        .or_else(|error| {
            let usage = salvage_usage(decode(&saved)?)?;
            log::warn!("job {job}: usage totals had unreadable costs, now zero: {error}");
            Ok::<_, Error>(usage)
        })
        .unwrap_or_else(|error| {
            log::warn!("job {job}: ignoring unreadable usage totals: {error}");
            JobUsage::default()
        }))
}
/// Saved totals with every cost that isn't a number set to zero. Optional
/// costs may be `null` (absent).
fn salvage_usage(mut saved: serde_json::Value) -> Result<JobUsage> {
    const REQUIRED: [&str; 3] = ["input_cost", "output_cost", "total_cost"];
    const OPTIONAL: [&str; 5] = [
        "cache_read_cost",
        "cache_write_cost",
        "reported_cost",
        "upstream_cost",
        "billed_cost",
    ];
    for kind in ["chunking", "embedding"] {
        let Some(usage) = saved.get_mut(kind).and_then(|u| u.as_object_mut()) else {
            continue;
        };
        for field in REQUIRED.into_iter().chain(OPTIONAL) {
            let Some(cost) = usage.get_mut(field) else {
                continue;
            };
            if !cost.is_number() && (REQUIRED.contains(&field) || !cost.is_null()) {
                *cost = 0.0.into();
            }
        }
    }
    Ok(serde_json::from_value(saved)?)
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
/// Removes the texts, snapshots and vectors nothing references any more:
/// chunk texts no chunk uses, and snapshots neither a file state nor a chunk
/// uses. Vectors no chunk uses are removed only while no job is unfinished,
/// since an unfinished job may rely on vectors it found or stored before
/// publishing. Returns what was removed.
pub(crate) fn collect_garbage(c: &Connection) -> Result<serde_json::Value> {
    c.execute("DELETE FROM chunk_text_fts WHERE rowid IN (SELECT t.id FROM chunk_texts t WHERE NOT EXISTS (SELECT 1 FROM chunks ch WHERE ch.text_id=t.id))",[])?;
    let texts = c.execute("DELETE FROM chunk_texts WHERE NOT EXISTS (SELECT 1 FROM chunks ch WHERE ch.text_id=chunk_texts.id)",[])?;
    let snapshots = c.execute("DELETE FROM snapshots WHERE NOT EXISTS (SELECT 1 FROM files f WHERE f.hash=snapshots.hash) AND NOT EXISTS (SELECT 1 FROM chunks ch WHERE ch.hash=snapshots.hash)",[])?;
    let unfinished: i64 = c.query_row("SELECT count(*) FROM jobs WHERE phase IN ('snapshotting','chunking','embedding','paused','blocked')",[],|r|r.get(0))?;
    let vectors = if unfinished == 0 {
        c.execute("DELETE FROM vectors WHERE NOT EXISTS (SELECT 1 FROM chunks ch WHERE ch.payload=vectors.id)",[])?
    } else {
        0
    };
    Ok(
        serde_json::json!({"chunk_texts":texts,"snapshots":snapshots,"vectors":vectors,"vectors_kept":unfinished>0}),
    )
}
/// The shared text row of a chunk, keyed by its location and content:
/// generations that keep a chunk share one row and one full-text entry.
pub(crate) fn chunk_text(c: &Connection, key: &str, path: &str, text: &str) -> Result<i64> {
    if let Some(id) = c
        .query_row("SELECT id FROM chunk_texts WHERE key=?1", [key], |r| {
            r.get(0)
        })
        .optional()?
    {
        return Ok(id);
    }
    c.execute("INSERT INTO chunk_texts(key) VALUES(?1)", [key])?;
    let id = c.last_insert_rowid();
    c.execute(
        "INSERT INTO chunk_text_fts(rowid,path,terms,body) VALUES(?1,?2,?3,?4)",
        params![id, path, crate::search::identifier_terms(text), text],
    )?;
    Ok(id)
}
/// The key of a chunk text: its path, snapshot, byte range and payload.
pub(crate) fn chunk_text_key(
    path: &str,
    hash: &str,
    start: usize,
    end: usize,
    payload: &str,
) -> Result<String> {
    Ok(crate::source::hash(serde_json::to_vec(&(
        path, hash, start, end, payload,
    ))?))
}
/// Moves a version 1 database to version 2 in one transaction: every
/// generation kept its own full-text copy of each chunk; now chunks with the
/// same location and content share one text row. Nothing else changes.
fn migrate_v1(c: &mut Connection) -> Result<()> {
    let tx = c.transaction()?;
    tx.execute_batch(
        "CREATE TABLE chunk_texts(id INTEGER PRIMARY KEY,key TEXT NOT NULL UNIQUE);
         CREATE VIRTUAL TABLE chunk_text_fts USING fts5(path,terms,body,tokenize='unicode61');
         ALTER TABLE chunks ADD COLUMN text_id INTEGER REFERENCES chunk_texts(id);",
    )?;
    // Chunks with their full-text copy (the full-text table scanned once,
    // chunks looked up by id), then any chunk without one, from its snapshot.
    for query in [
        "SELECT ch.rowid,ch.path,ch.hash,ch.start,ch.end,ch.payload,f.body FROM chunk_fts f CROSS JOIN chunks ch ON ch.id=f.id",
        "SELECT ch.rowid,ch.path,ch.hash,ch.start,ch.end,ch.payload,CAST(substr(CAST(s.content AS BLOB),ch.start+1,ch.end-ch.start) AS TEXT) FROM chunks ch JOIN snapshots s ON s.hash=ch.hash WHERE ch.text_id IS NULL",
    ] {
        let mut links = vec![];
        {
            let mut statement = tx.prepare(query)?;
            let mut rows = statement.query([])?;
            while let Some(row) = rows.next()? {
                let (rowid, path, hash, start, end, payload, text): (
                    i64,
                    String,
                    String,
                    usize,
                    usize,
                    String,
                    String,
                ) = (
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                );
                let key = chunk_text_key(&path, &hash, start, end, &payload)?;
                links.push((rowid, chunk_text(&tx, &key, &path, &text)?));
            }
        }
        for (rowid, text) in links {
            tx.execute(
                "UPDATE chunks SET text_id=?2 WHERE rowid=?1",
                params![rowid, text],
            )?;
        }
    }
    tx.execute_batch("DROP TABLE chunk_fts; PRAGMA user_version=2;")?;
    tx.commit()?;
    Ok(())
}
const SCHEMA_VERSION: i64 = 2;
const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS indexes(id TEXT PRIMARY KEY,name TEXT NOT NULL,spec TEXT NOT NULL,active TEXT,config TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,index_id TEXT NOT NULL REFERENCES indexes(id) ON DELETE CASCADE,phase TEXT NOT NULL,config TEXT NOT NULL,manifest TEXT,excluded TEXT NOT NULL DEFAULT '[]',affected TEXT NOT NULL DEFAULT '[]',changes TEXT NOT NULL DEFAULT '[]',fresh INTEGER NOT NULL DEFAULT 1,problem TEXT);
CREATE TABLE IF NOT EXISTS snapshots(hash TEXT PRIMARY KEY,content TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files(job TEXT REFERENCES jobs(id) ON DELETE CASCADE,path TEXT,hash TEXT NOT NULL REFERENCES snapshots(hash),original_hash TEXT NOT NULL,state TEXT NOT NULL,PRIMARY KEY(job,path));
CREATE TABLE IF NOT EXISTS vectors(id TEXT PRIMARY KEY,profile TEXT NOT NULL,tokens INTEGER NOT NULL,vector BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS chunk_texts(id INTEGER PRIMARY KEY,key TEXT NOT NULL UNIQUE);
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_text_fts USING fts5(path,terms,body,tokenize='unicode61');
CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY,job TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,path TEXT NOT NULL,hash TEXT NOT NULL REFERENCES snapshots(hash),start INTEGER NOT NULL,end INTEGER NOT NULL,first_line INTEGER NOT NULL,last_line INTEGER NOT NULL,payload TEXT NOT NULL REFERENCES vectors(id),structure TEXT NOT NULL,text_id INTEGER REFERENCES chunk_texts(id));
CREATE INDEX IF NOT EXISTS chunks_job_payload ON chunks(job,payload);
CREATE INDEX IF NOT EXISTS chunks_text ON chunks(text_id,job);
CREATE INDEX IF NOT EXISTS chunks_hash ON chunks(hash);
CREATE INDEX IF NOT EXISTS chunks_payload ON chunks(payload);
CREATE INDEX IF NOT EXISTS files_hash ON files(hash);
CREATE TABLE IF NOT EXISTS events(sequence INTEGER PRIMARY KEY AUTOINCREMENT,job TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,kind TEXT NOT NULL,detail TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS events_job ON events(job,sequence);
CREATE TABLE IF NOT EXISTS requests(id INTEGER PRIMARY KEY AUTOINCREMENT,job TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,request TEXT NOT NULL,response TEXT,error TEXT);
CREATE TABLE IF NOT EXISTS job_usage(job TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,usage TEXT NOT NULL);
PRAGMA user_version=2;
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn synchronous_normal_is_opt_in() {
        let dir = tempfile::tempdir().unwrap();
        for (setting, expected) in [
            (crate::Synchronous::Full, 2),
            (crate::Synchronous::Normal, 1),
        ] {
            let db = Db::open(&dir.path().join("i.db"), setting).await.unwrap();
            let value: i64 = db
                .call(|c| Ok(c.query_row("PRAGMA synchronous", [], |r| r.get(0))?))
                .await
                .unwrap();
            assert_eq!(value, expected);
            drop(db);
        }
    }

    #[tokio::test]
    async fn reopening_active_jobs_records_the_recovery_phase_event_atomically() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.db");
        let db = Db::open(&path, crate::Synchronous::Full).await.unwrap();
        db.call(|c| {
            c.execute(
                "INSERT INTO indexes VALUES('i','test','{}',NULL,'{}')",
                [],
            )?;
            c.execute(
                "INSERT INTO jobs(id,index_id,phase,config) VALUES('interrupted','i','chunking','{}'),('requested','i','embedding','{}')",
                [],
            )?;
            c.execute(
                "INSERT INTO events(job,kind,detail) VALUES('requested','pause_requested','{}')",
                [],
            )?;
            Ok(())
        })
        .await
        .unwrap();
        drop(db);

        let db = Db::open(&path, crate::Synchronous::Full).await.unwrap();
        for (job, problem) in [("interrupted", INTERRUPTED), ("requested", "paused")] {
            let row = db
                .call({
                    let job = job.to_string();
                    move |c| {
                        Ok(c.query_row(
                            "SELECT phase,problem FROM jobs WHERE id=?1",
                            [&job],
                            |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
                        )?)
                    }
                })
                .await
                .unwrap();
            assert_eq!(row, ("paused".to_string(), Some(problem.to_string())));
            let detail = db
                .call({
                    let job = job.to_string();
                    move |c| {
                        Ok(c.query_row(
                            "SELECT detail FROM events WHERE job=?1 AND kind='phase' ORDER BY sequence DESC LIMIT 1",
                            [&job],
                            |row| row.get::<_, String>(0),
                        )?)
                    }
                })
                .await
                .unwrap();
            assert_eq!(
                decode::<serde_json::Value>(&detail).unwrap()["phase"],
                "paused"
            );
        }
        drop(db);

        let db = Db::open(&path, crate::Synchronous::Full).await.unwrap();
        let phase_events = db
            .call(|c| {
                Ok(c.query_row(
                    "SELECT count(*) FROM events WHERE kind='phase'",
                    [],
                    |row| row.get::<_, i64>(0),
                )?)
            })
            .await
            .unwrap();
        assert_eq!(phase_events, 2);
    }

    #[test]
    fn unreadable_or_non_finite_usage_never_blocks_status_or_updates() {
        let c = Connection::open_in_memory().unwrap();
        c.execute_batch(SCHEMA).unwrap();
        c.execute_batch(
            "INSERT INTO indexes VALUES('i','test','{}',NULL,'{}');
             INSERT INTO jobs(id,index_id,phase,config) VALUES('j','i','chunking','{}');
             INSERT INTO jobs(id,index_id,phase,config) VALUES('k','i','chunking','{}');",
        )
        .unwrap();
        // How a NaN cost used to be saved: serde_json writes it as null.
        let broken = r#"{"chunk_calls":4,"rejected_calls":2,"chunking":{"input_tokens":600,"output_tokens":60,"input_cost":null,"output_cost":0.5,"total_cost":null,"cache_read_cost":null,"billed_cost":"NaN"},"embedding":{"input_tokens":1000,"output_tokens":0,"input_cost":0.25,"output_cost":0.0,"total_cost":0.25}}"#;
        c.execute("INSERT INTO job_usage VALUES('j',?1)", [broken])
            .unwrap();
        // Counts and readable costs survive; unreadable costs become zero.
        let salvaged = job_usage(&c, "j").unwrap();
        assert_eq!((salvaged.chunk_calls, salvaged.rejected_calls), (4, 2));
        let chunking = salvaged.chunking.as_ref().unwrap();
        assert_eq!((chunking.input_tokens, chunking.output_tokens), (600, 60));
        assert_eq!((chunking.input_cost, chunking.output_cost), (0.0, 0.5));
        assert_eq!(chunking.total_cost, 0.0);
        assert_eq!(chunking.cache_read_cost, None);
        assert_eq!(chunking.billed_cost, Some(0.0));
        assert_eq!(salvaged.embedding.as_ref().unwrap().total_cost, 0.25);
        // Later updates keep adding to the salvaged totals, with non-finite
        // costs counted as zero.
        let nan = arko_ai::UsageInfo {
            input_cost: f64::NAN,
            total_cost: f64::INFINITY,
            ..arko_ai::UsageInfo::new(100, 10)
        };
        for _ in 0..2 {
            add_job_usage(&c, "j", |usage| usage.record_chunk(Some(&nan))).unwrap();
        }
        let usage = job_usage(&c, "j").unwrap();
        assert_eq!((usage.chunk_calls, usage.rejected_calls), (6, 2));
        let chunking = usage.chunking.unwrap();
        assert_eq!(chunking.input_tokens, 800);
        assert_eq!((chunking.input_cost, chunking.total_cost), (0.0, 0.0));
        assert_eq!(usage.embedding.unwrap().input_tokens, 1000);
        // Totals that can't be salvaged restart from zero.
        c.execute(
            "INSERT INTO job_usage VALUES('k','{\"chunk_calls\":\"many\"}')",
            [],
        )
        .unwrap();
        assert_eq!(job_usage(&c, "k").unwrap(), JobUsage::default());
        add_job_usage(&c, "k", |usage| usage.record_chunk(None)).unwrap();
        assert_eq!(job_usage(&c, "k").unwrap().chunk_calls, 1);
        assert_eq!(job_usage(&c, "unknown").unwrap(), JobUsage::default());
    }
}
