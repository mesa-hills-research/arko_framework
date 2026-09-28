use crate::{
    ArkoIndex, Error, Result,
    error::invalid,
    packing,
    scheduler::{Control, Operation, Reply},
    storage::{decode, required},
};
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    sync::Arc,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    Lexical,
    Semantic,
    Hybrid,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchOptions {
    pub mode: SearchMode,
    /// Complete results per page: 10 by default. `None` returns every
    /// eligible result.
    pub top_k: Option<usize>,
    /// Exclude source chunks shorter than this many UTF-8 bytes before
    /// ranking: 40 by default. Zero keeps every chunk. Source reads are
    /// unaffected.
    pub min_chunk_bytes: usize,
    pub path_prefix: Option<String>,
    pub literal: Option<String>,
    pub cursor: Option<SearchCursor>,
    /// The most affected paths a rebuild notice lists: 100 by default.
    /// `None` lists every one.
    pub rebuild_file_limit: Option<usize>,
}
impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            mode: SearchMode::Hybrid,
            top_k: Some(10),
            min_chunk_bytes: 40,
            path_prefix: None,
            literal: None,
            cursor: None,
            rebuild_file_limit: Some(100),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchCursor {
    pub generation: String,
    pub query_hash: String,
    pub offset: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RebuildNotice {
    pub message: String,
    pub job: String,
    pub phase: String,
    pub affected_files: Vec<String>,
    pub omitted_files: usize,
    pub determining_changes: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchHit {
    pub id: String,
    pub path: String,
    pub snapshot: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub first_line: usize,
    pub last_line: usize,
    pub structure: String,
    pub score: f64,
    pub score_type: String,
    pub text: String,
    pub other_locations: usize,
    /// The file may have changed since this chunk was indexed: an unfinished
    /// update lists it as affected, or the file on disk no longer matches
    /// the chunk's snapshot (it changed, was removed or can't be read).
    #[serde(default)]
    pub may_be_stale: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchPage {
    pub generation: String,
    pub mode: SearchMode,
    pub hits: Vec<SearchHit>,
    pub next: Option<SearchCursor>,
    pub omitted_hits: usize,
    pub candidate_count: usize,
    pub fresh_at_publication: bool,
    pub rebuild: Option<RebuildNotice>,
    /// Informational source token count for all returned complete chunks.
    pub search_tokens: usize,
    pub rebuild_tokens: usize,
    pub tokenizer: String,
    /// Present only for explicitly judged searches. Scores remain original
    /// retrieval scores; result order is the judge's selected order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge: Option<crate::JudgeReport>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkRead {
    pub id: String,
    pub path: String,
    pub snapshot: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub first_line: usize,
    pub last_line: usize,
    pub text: String,
}
#[derive(Clone, Debug)]
struct Rank {
    score: f64,
    id: String,
}
impl PartialEq for Rank {
    fn eq(&self, o: &Self) -> bool {
        self.score == o.score && self.id == o.id
    }
}
impl Eq for Rank {}
impl PartialOrd for Rank {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Rank {
    fn cmp(&self, o: &Self) -> Ordering {
        self.score
            .total_cmp(&o.score)
            .then_with(|| o.id.cmp(&self.id))
    }
}

/// One generation/filter matrix, bounded by the configured budget. Oversized
/// matrices use streamed vector-only rows; no source text is held in this cache.
#[derive(Clone)]
pub(crate) struct MatrixCache {
    key: String,
    ids: Vec<String>,
    payloads: Vec<String>,
    vectors: Vec<f32>,
}
pub(crate) fn identifier_terms(text: &str) -> String {
    let mut result = String::new();
    let mut previous = false;
    for c in text.chars() {
        if c.is_alphanumeric() {
            if c.is_uppercase() && previous {
                result.push(' ');
            }
            result.extend(c.to_lowercase());
            previous = c.is_lowercase() || c.is_numeric();
        } else {
            result.push(' ');
            previous = false;
        }
    }
    result
}
fn fts_query(query: &str) -> String {
    let mut terms: Vec<_> = query
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect();
    terms.extend(
        identifier_terms(query)
            .split_whitespace()
            .map(str::to_owned),
    );
    terms.sort();
    terms.dedup();
    terms
        .into_iter()
        .map(|s| format!("\"{}\"", s.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" OR ")
}
/// The paths of `hits` (path, snapshot) whose file under the index root no
/// longer matches the snapshot: changed, removed, unreadable, or now over
/// the size ceiling. None when the root itself is not accessible (an index
/// used away from its source).
fn changed_on_disk(spec: &crate::BuildSpec, hits: Vec<(String, String)>) -> HashSet<String> {
    if !spec.root.is_dir() {
        return HashSet::new();
    }
    let mut checked = HashMap::new();
    for (path, snapshot) in hits {
        checked.entry(path).or_insert_with_key(|path| {
            let file = spec.root.join(path);
            let current = std::fs::metadata(&file)
                .ok()
                .filter(|metadata| metadata.len() <= spec.max_file_bytes)
                .and_then(|_| std::fs::read(&file).ok())
                .and_then(|bytes| crate::source::Snapshot::new(path.clone(), &bytes).ok());
            current.is_none_or(|current| current.hash != snapshot)
        });
    }
    checked
        .into_iter()
        .filter(|(_, changed)| *changed)
        .map(|(path, _)| path)
        .collect()
}
impl ArkoIndex {
    pub async fn search(
        &self,
        index: &str,
        query: &str,
        options: SearchOptions,
    ) -> Result<SearchPage> {
        if options.top_k == Some(0) {
            return Err(invalid("top_k must be positive when supplied"));
        }
        let idx = index.to_owned();
        let requested = options.cursor.as_ref().map(|c| c.generation.clone());
        let(generation,fresh,rebuild,spec)=self.inner.db.call(move|c|{
            let (active,spec):(Option<String>,String)=required(c.query_row("SELECT active,spec FROM indexes WHERE id=?1",[&idx],|r|Ok((r.get(0)?,r.get(1)?))).optional()?,"index")?;
            let spec:crate::BuildSpec=decode(&spec)?;
            let from_cursor=requested.is_some();
            let generation=requested.or(active).ok_or_else(||Error::Blocked("initial index build has no complete searchable generation yet".into()))?;
            let fresh:Option<bool>=c.query_row("SELECT fresh FROM jobs WHERE id=?1 AND index_id=?2 AND phase='ready'",params![generation,idx],|r|r.get(0)).optional()?;
            let fresh=match fresh {
                Some(fresh)=>fresh,
                // A pruned generation, or one of another index.
                None if from_cursor=>return Err(Error::Stale("the cursor's generation is no longer available".into())),
                None=>return Err(invalid("unknown complete generation")),
            };
            let pending:Option<(String,String,String,Option<String>)>=c.query_row("SELECT id,phase,affected,manifest FROM jobs WHERE index_id=?1 AND phase IN ('snapshotting','chunking','embedding','paused','blocked') ORDER BY rowid DESC LIMIT 1",[idx],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))).optional()?;
            let rebuild=pending.map(|(job,phase,affected,manifest)|Ok::<_,Error>(RebuildNotice{message:"Index is being rebuilt; results use the last complete generation.".into(),job,phase,affected_files:decode(&affected)?,omitted_files:0,determining_changes:manifest.is_none()})).transpose()?;
            Ok((generation,fresh,rebuild,spec))
        }).await?;
        // Files an unfinished update found changed.
        let affected: HashSet<String> = rebuild
            .as_ref()
            .map(|notice| notice.affected_files.iter().cloned().collect())
            .unwrap_or_default();
        let query_hash = crate::source::hash(serde_json::to_vec(&(
            query,
            &options.mode,
            &options.path_prefix,
            &options.literal,
            options.top_k,
            options.min_chunk_bytes,
        ))?);
        let offset = if let Some(cursor) = &options.cursor {
            if cursor.query_hash != query_hash {
                return Err(Error::Stale("cursor query/options changed".into()));
            }
            cursor.offset
        } else {
            0
        };
        let vector = if options.mode != SearchMode::Lexical {
            let key =
                crate::source::hash(serde_json::to_vec(&(query, &self.inner.config.embedding))?);
            let cell = {
                let mut cache = self.inner.queries.lock().await;
                // Keep completed query embeddings bounded; retain in-flight singleflight cells.
                if cache.len() >= 128
                    && let Some(key) = cache
                        .iter()
                        .find(|(_, c)| c.get().is_some())
                        .map(|(k, _)| k.clone())
                {
                    cache.remove(&key);
                }
                cache
                    .entry(key)
                    .or_insert_with(|| Arc::new(tokio::sync::OnceCell::new()))
                    .clone()
            };
            Some(
                cell.get_or_try_init(|| async {
                    let count = self.inner.backend.input_tokens(query, true)?;
                    if count > self.inner.backend.limits().item_tokens {
                        return Err(invalid("query exceeds embedding limit"));
                    }
                    match self
                        .inner
                        .embeddings
                        .call(
                            Operation::Embed(vec![query.into()], true),
                            false,
                            &Control::new(),
                        )
                        .await
                        .result?
                    {
                        Reply::Embed(mut response) if response.vectors.len() == 1 => {
                            packing::normalize(
                                response.vectors.remove(0),
                                self.inner.config.embedding.dimensions,
                            )
                        }
                        _ => Err(invalid("invalid query embedding response")),
                    }
                })
                .await?
                .clone(),
            )
        } else {
            None
        };
        let (g, prefix, literal, mode, fts, min_chunk_bytes) = (
            generation.clone(),
            options.path_prefix.clone().unwrap_or_default(),
            options.literal.clone().unwrap_or_default(),
            options.mode.clone(),
            fts_query(query),
            options.min_chunk_bytes,
        );
        let matrix = self.inner.matrix.clone();
        let cache_budget = self.inner.config.vector_cache_bytes;
        let (page_offset, page_size) = (offset, options.top_k.unwrap_or(usize::MAX));
        // Ranks every eligible chunk from the ranking rows alone, then loads
        // text and locations only for the page returned.
        let (available, page_hits)=self.inner.db.call(move|c|{
            // (chunk id, score, payload)
            let mut lexical:Vec<(String,f64,String)>=vec![];let mut semantic:Vec<(String,f64,String)>=vec![];
            if mode!=SearchMode::Semantic&&!fts.is_empty(){
                let mut statement=c.prepare("SELECT ch.id,bm25(chunk_text_fts),ch.payload FROM chunk_text_fts f JOIN chunks ch ON ch.text_id=f.rowid WHERE chunk_text_fts MATCH ?1 AND ch.job=?2 AND substr(ch.path,1,length(?3))=?3 AND (?4='' OR instr(f.body,?4)>0) AND ch.end-ch.start>=?5 ORDER BY bm25(chunk_text_fts),ch.id")?;
                lexical=statement.query_map(params![fts,g,prefix,literal,min_chunk_bytes],|r|Ok((r.get::<_,String>(0)?,r.get::<_,f64>(1)?,r.get::<_,String>(2)?)))?.collect::<std::result::Result<_,_>>()?;
            }
            if let Some(vector)=vector{
                // SQLite streams vector-only rows; source bodies are fetched only for finalists.
                // Score every matching vector before ranking; the cache never limits candidates.
                let cache_key=crate::source::hash(serde_json::to_vec(&(&g,&prefix,&literal,min_chunk_bytes))?);
                let mut cache=matrix.lock().map_err(|_|invalid("vector cache lock poisoned"))?;
                let mut ranks=Vec::new();
                let mut score_row=|id:String,payload:String,values:&[f32]|->Result<()> {
                    if values.len()!=vector.len(){return Err(invalid("stored vector dimension mismatch"));}
                    let score:f64=values.iter().zip(&vector).map(|(&v,&q)|v as f64*q as f64).sum();
                    if !score.is_finite(){return Err(invalid("nonfinite stored vector score"));}
                    ranks.push((Rank{score,id},payload));Ok(())
                };
                if let Some(cached)=cache.as_ref().filter(|c|c.key==cache_key){
                    for((id,payload),values)in cached.ids.iter().zip(&cached.payloads).zip(cached.vectors.chunks_exact(vector.len())){score_row(id.clone(),payload.clone(),values)?;}
                }else{
                    let mut statement=c.prepare("SELECT min(ch.id),v.vector,ch.payload FROM chunks ch JOIN vectors v ON v.id=ch.payload JOIN snapshots s ON s.hash=ch.hash WHERE ch.job=?1 AND substr(ch.path,1,length(?2))=?2 AND (?3='' OR instr(CAST(substr(CAST(s.content AS BLOB),ch.start+1,ch.end-ch.start) AS TEXT),?3)>0) AND ch.end-ch.start>=?4 GROUP BY ch.payload")?;
                    let mut rows=statement.query(params![g,prefix,literal,min_chunk_bytes])?;
                    let mut candidate=Some(MatrixCache{key:cache_key,ids:vec![],payloads:vec![],vectors:vec![]});let mut bytes_used=0usize;
                    while let Some(row)=rows.next()?{
                        let id:String=row.get(0)?;let bytes:Vec<u8>=row.get(1)?;let payload:String=row.get(2)?;
                        if bytes.len()!=vector.len()*4{return Err(invalid("stored vector dimension mismatch"));}
                        let values:Vec<_>=bytes.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();score_row(id.clone(),payload.clone(),&values)?;
                        bytes_used=bytes_used.saturating_add(bytes.len()+id.len()+payload.len()+2*std::mem::size_of::<String>());
                        if bytes_used>cache_budget{candidate=None;}
                        if let Some(candidate)=&mut candidate{candidate.ids.push(id);candidate.payloads.push(payload);candidate.vectors.extend(values);}
                    }
                    *cache=candidate;
                }
                ranks.sort_by(|a,b|b.0.cmp(&a.0));semantic=ranks.into_iter().map(|(r,payload)|(r.id,r.score,payload)).collect();
            }
            let mut payloads=HashMap::<String,String>::new();
            for (id,_,payload) in lexical.iter().chain(&semantic) {payloads.insert(id.clone(),payload.clone());}
            let mut scores=HashMap::<String,f64>::new();
            match mode{
                SearchMode::Lexical=>{for(id,s,_)in lexical{scores.insert(id,-s);}},
                SearchMode::Semantic=>{scores.extend(semantic.into_iter().map(|(id,s,_)|(id,s)));},
                SearchMode::Hybrid=>{for list in [&lexical,&semantic]{for(rank,(id,_,_))in list.iter().enumerate(){*scores.entry(id.clone()).or_default()+=1.0/(61.0+rank as f64);}}},
            }
            let mut ranks:Vec<_>=scores.into_iter().map(|(id,score)|Rank{id,score}).collect();ranks.sort_by(|a,b|b.cmp(a));
            // A text found at several locations is one result, at its best rank.
            let mut seen=HashSet::new();
            let ranked:Vec<(Rank,String)>=ranks.into_iter().filter_map(|rank|{let payload=payloads.get(&rank.id)?.clone();seen.insert(payload.clone()).then_some((rank,payload))}).collect();
            let available=ranked.len();
            let mut hits=vec![];
            for (rank,payload) in ranked.into_iter().skip(page_offset).take(page_size){
                let (mut hit,text):(SearchHit,String)=c.query_row("SELECT ch.id,ch.path,ch.hash,ch.start,ch.end,ch.first_line,ch.last_line,ch.structure,CAST(substr(CAST(s.content AS BLOB),ch.start+1,ch.end-ch.start) AS TEXT) FROM chunks ch JOIN snapshots s ON s.hash=ch.hash WHERE ch.id=?1 AND ch.job=?2",params![rank.id,g],|r|Ok((SearchHit{id:r.get(0)?,path:r.get(1)?,snapshot:r.get(2)?,start_byte:r.get(3)?,end_byte:r.get(4)?,first_line:r.get(5)?,last_line:r.get(6)?,structure:r.get(7)?,score:rank.score,score_type:match mode{SearchMode::Hybrid=>"rrf",SearchMode::Semantic=>"cosine",SearchMode::Lexical=>"negative_bm25"}.into(),text:String::new(),other_locations:0,may_be_stale:false},r.get(8)?)))?;
                let count:usize=c.query_row("SELECT count(*) FROM chunks WHERE job=?1 AND payload=?2",params![g,payload],|r|r.get(0))?;hit.other_locations=count.saturating_sub(1);hits.push((hit,text));
            }
            Ok((available,hits))
        }).await?;
        let mut rebuild = rebuild;
        if let (Some(notice), Some(cap)) = (&mut rebuild, options.rebuild_file_limit) {
            notice.omitted_files = notice.affected_files.len().saturating_sub(cap);
            notice.affected_files.truncate(cap);
        }
        let rebuild_tokens = if let Some(notice) = &rebuild {
            self.inner.backend.tokens(&serde_json::to_string(notice)?)?
        } else {
            0
        };
        let mut page = SearchPage {
            generation: generation.clone(),
            mode: options.mode.clone(),
            hits: vec![],
            next: None,
            omitted_hits: available.saturating_sub(offset),
            candidate_count: available,
            fresh_at_publication: fresh,
            rebuild: None,
            search_tokens: 0,
            rebuild_tokens,
            tokenizer: self.inner.backend.output_tokenizer(),
            judge: None,
        };
        let on_disk: Vec<(String, String)> = page_hits
            .iter()
            .map(|(hit, _)| (hit.path.clone(), hit.snapshot.clone()))
            .collect();
        let changed = tokio::task::spawn_blocking(move || changed_on_disk(&spec, on_disk)).await?;
        for (mut hit, text) in page_hits {
            hit.may_be_stale = affected.contains(&hit.path) || changed.contains(&hit.path);
            page.search_tokens += self.inner.backend.tokens(&text)?;
            hit.text = text;
            page.hits.push(hit);
        }
        page.omitted_hits = available.saturating_sub(offset + page.hits.len());
        page.next = if page.omitted_hits > 0 {
            Some(SearchCursor {
                generation,
                query_hash,
                offset: offset + page.hits.len(),
            })
        } else {
            None
        };
        page.rebuild = rebuild;
        Ok(page)
    }
    /// Exact snapshot text for selected IDs, optionally including neighboring chunks.
    /// Old complete generations remain readable until the index is explicitly deleted.
    pub async fn read_chunks(
        &self,
        index: &str,
        generation: &str,
        ids: &[String],
        neighbors: usize,
    ) -> Result<Vec<ChunkRead>> {
        let (index, generation, ids) = (index.to_owned(), generation.to_owned(), ids.to_vec());
        self.inner.db.call(move|c|{
            let valid= c.query_row("SELECT id FROM jobs WHERE id=?1 AND index_id=?2 AND phase='ready'",params![generation,index],|r|r.get::<_,String>(0)).optional()?;
            if valid.is_none(){return Err(Error::Stale("generation is unavailable".into()));}
            let mut selected=HashSet::new();
            for id in ids{
                let(path,start):(String,usize)=required(c.query_row("SELECT path,start FROM chunks WHERE job=?1 AND id=?2",params![generation,id],|r|Ok((r.get(0)?,r.get(1)?))).optional()?,"chunk in generation")?;
                selected.insert(id);
                for sql in ["SELECT id FROM chunks WHERE job=?1 AND path=?2 AND start<?3 ORDER BY start DESC LIMIT ?4", "SELECT id FROM chunks WHERE job=?1 AND path=?2 AND start>?3 ORDER BY start LIMIT ?4"]{
                    let mut statement=c.prepare(sql)?;
                    let rows:Vec<String>=statement.query_map(params![generation,path,start,i64::try_from(neighbors).unwrap_or(i64::MAX)],|r|r.get(0))?.collect::<std::result::Result<_,_>>()?;selected.extend(rows);
                }
            }
            let mut out=vec![];
            for id in selected{out.push(c.query_row("SELECT ch.id,ch.path,ch.hash,ch.start,ch.end,ch.first_line,ch.last_line,CAST(substr(CAST(s.content AS BLOB),ch.start+1,ch.end-ch.start) AS TEXT) FROM chunks ch JOIN snapshots s ON s.hash=ch.hash WHERE ch.job=?1 AND ch.id=?2",params![generation,id],|r|Ok(ChunkRead{id:r.get(0)?,path:r.get(1)?,snapshot:r.get(2)?,start_byte:r.get(3)?,end_byte:r.get(4)?,first_line:r.get(5)?,last_line:r.get(6)?,text:r.get(7)?}))?);}
            out.sort_by(|a,b|a.path.cmp(&b.path).then(a.start_byte.cmp(&b.start_byte)));Ok(out)
        }).await
    }
}
