//! Search an index2 state directory lexically (BM25), densely (embeddings), or both
//! (hybrid), with optional path and language filters and a context header per
//! result.
//!
//! **Hybrid** retrieval takes the best [`SearchRequest::candidates`] (default 50)
//! chunks from each side and fuses the two rankings ([`Fusion`]):
//!
//! * [`Fusion::Rrf`] (default): reciprocal rank fusion. A chunk's score is the sum,
//!   over the lists that contain it, of `weight / (k + rank)`, with ranks from 1,
//!   `k = 60` and both weights 1. Only ranks matter, so BM25 scores and cosine
//!   similarities need no common scale, and a chunk both sides rank well comes first.
//! * [`Fusion::Linear`]: each list's scores are min-max normalized to `[0, 1]` over its
//!   candidates (all 1 when they are equal), a missing chunk counts 0, and the score
//!   is `w · lexical + (1 − w) · dense`.
//!
//! Equal BM25 and fused scores are ordered by document id, which follows (path, start
//! byte), and equal similarities by vector id: results are deterministic.
use crate::{
    bench_v1::{Chunk, Error},
    dense::{DenseIndexInfo, Embedder, ModelDescriptor, embed_query},
    index2::Generation,
    vector::{Filter, VectorStore},
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::Path};

/// Which retrieval a search uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// BM25 over words and identifier parts. Needs no embedder and costs nothing.
    #[default]
    Lexical,
    /// Cosine similarity of the query's embedding to the chunks'.
    Dense,
    /// Both, fused.
    Hybrid,
}
impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Lexical => "lexical",
            Mode::Dense => "dense",
            Mode::Hybrid => "hybrid",
        }
    }
}
impl std::str::FromStr for Mode {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        match s {
            "lexical" => Ok(Mode::Lexical),
            "dense" => Ok(Mode::Dense),
            "hybrid" => Ok(Mode::Hybrid),
            _ => Err(format!("unknown search mode {s:?} (lexical, dense or hybrid)").into()),
        }
    }
}

/// How hybrid search combines the lexical and dense rankings (see the module docs).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "lowercase")]
pub enum Fusion {
    /// Reciprocal rank fusion: `Σ weight / (k + rank)`.
    Rrf {
        k: f64,
        lexical_weight: f64,
        dense_weight: f64,
    },
    /// `lexical_weight · lexical + (1 − lexical_weight) · dense`, over min-max
    /// normalized scores.
    Linear { lexical_weight: f64 },
}
impl Default for Fusion {
    fn default() -> Self {
        Fusion::Rrf {
            k: 60.0,
            lexical_weight: 1.0,
            dense_weight: 1.0,
        }
    }
}
impl Fusion {
    fn validate(&self) -> Result<(), Error> {
        let ok = |w: f64| w.is_finite() && w >= 0.0;
        match *self {
            Fusion::Rrf {
                k,
                lexical_weight,
                dense_weight,
            } if ok(k) && ok(lexical_weight) && ok(dense_weight) => Ok(()),
            Fusion::Linear { lexical_weight } if ok(lexical_weight) && lexical_weight <= 1.0 => {
                Ok(())
            }
            _ => Err(format!("invalid fusion settings {self:?}").into()),
        }
    }
}

/// One fused result of [`fuse`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fused {
    pub id: u64,
    pub score: f64,
    /// 1-based rank in each input list, when the id is in it.
    pub lexical_rank: Option<usize>,
    pub dense_rank: Option<usize>,
}

/// Fuses two rankings (each best first, `(id, score)`) into the best `limit`, by
/// fused score, then id ascending.
pub fn fuse(
    lexical: &[(u64, f64)],
    dense: &[(u64, f64)],
    fusion: &Fusion,
    limit: usize,
) -> Result<Vec<Fused>, Error> {
    fusion.validate()?;
    let mut by_id: HashMap<u64, Fused> = HashMap::new();
    fn slot(by_id: &mut HashMap<u64, Fused>, id: u64) -> &mut Fused {
        by_id.entry(id).or_insert(Fused {
            id,
            score: 0.0,
            lexical_rank: None,
            dense_rank: None,
        })
    }
    // Min-max normalization of a list's scores.
    let normalize = |list: &[(u64, f64)]| -> Vec<f64> {
        let max = list.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
        let min = list.iter().map(|x| x.1).fold(f64::INFINITY, f64::min);
        list.iter()
            .map(|x| {
                if max > min {
                    (x.1 - min) / (max - min)
                } else {
                    1.0
                }
            })
            .collect()
    };
    match *fusion {
        Fusion::Rrf {
            k,
            lexical_weight,
            dense_weight,
        } => {
            for (rank, (id, _)) in lexical.iter().enumerate() {
                let e = slot(&mut by_id, *id);
                if e.lexical_rank.is_none() {
                    e.lexical_rank = Some(rank + 1);
                    e.score += lexical_weight / (k + (rank + 1) as f64);
                }
            }
            for (rank, (id, _)) in dense.iter().enumerate() {
                let e = slot(&mut by_id, *id);
                if e.dense_rank.is_none() {
                    e.dense_rank = Some(rank + 1);
                    e.score += dense_weight / (k + (rank + 1) as f64);
                }
            }
        }
        Fusion::Linear { lexical_weight } => {
            for (rank, ((id, _), norm)) in lexical.iter().zip(normalize(lexical)).enumerate() {
                let e = slot(&mut by_id, *id);
                if e.lexical_rank.is_none() {
                    e.lexical_rank = Some(rank + 1);
                    e.score += lexical_weight * norm;
                }
            }
            for (rank, ((id, _), norm)) in dense.iter().zip(normalize(dense)).enumerate() {
                let e = slot(&mut by_id, *id);
                if e.dense_rank.is_none() {
                    e.dense_rank = Some(rank + 1);
                    e.score += (1.0 - lexical_weight) * norm;
                }
            }
        }
    }
    let mut fused: Vec<Fused> = by_id.into_values().collect();
    fused.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
    fused.truncate(limit);
    Ok(fused)
}

/// A search. `SearchRequest::new(query)` is lexical, ten results, no filter.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SearchRequest {
    pub query: String,
    pub mode: Mode,
    /// Results returned (default 10).
    pub limit: usize,
    /// Hybrid: chunks taken from each side before fusing (default 50; at least
    /// `limit` are taken).
    pub candidates: usize,
    pub fusion: Fusion,
    pub filter: Filter,
    /// Dense and hybrid: the query's embedding, when the caller already has it (from
    /// [`embed_query`] with the index's model); no embedder is needed then, and the
    /// search reports 0 query tokens.
    pub query_vector: Option<Vec<f32>>,
}
impl SearchRequest {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            mode: Mode::Lexical,
            limit: 10,
            candidates: 50,
            fusion: Fusion::default(),
            filter: Filter::default(),
            query_vector: None,
        }
    }
    pub fn with_mode(mut self, mode: Mode) -> Self {
        self.mode = mode;
        self
    }
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
    pub fn with_candidates(mut self, candidates: usize) -> Self {
        self.candidates = candidates;
        self
    }
    pub fn with_fusion(mut self, fusion: Fusion) -> Self {
        self.fusion = fusion;
        self
    }
    pub fn with_filter(mut self, filter: Filter) -> Self {
        self.filter = filter;
        self
    }
    pub fn with_query_vector(mut self, vector: Vec<f32>) -> Self {
        self.query_vector = Some(vector);
        self
    }
}

/// One result: the chunk, its context header, and its score.
#[derive(Clone, Debug, Serialize)]
#[non_exhaustive]
pub struct SearchHit {
    #[serde(flatten)]
    pub chunk: Chunk,
    /// [`Chunk::header`]: `path:start-end` and the scope.
    pub header: String,
    /// BM25 (lexical), cosine similarity (dense) or the fused score (hybrid).
    pub score: f64,
    /// 1-based rank in the lexical and dense lists (hybrid; the one list otherwise).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexical_rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dense_rank: Option<usize>,
}

/// The results of a search and what it cost.
#[derive(Clone, Debug, Serialize)]
#[non_exhaustive]
pub struct SearchResults {
    pub hits: Vec<SearchHit>,
    pub mode: Mode,
    /// Tokens billed to embed the query (0 for lexical search or a supplied vector).
    /// The future arko tool charges these to the calling agent.
    pub query_tokens: u64,
    /// The index's embeddings (model and coverage), when it has them.
    pub dense: Option<DenseIndexInfo>,
}

/// Searches the current generation of an index2 state directory. Dense and hybrid
/// modes need embeddings in the index and an `embedder` with the index's model (or
/// [`SearchRequest::query_vector`]).
pub fn search(
    state_dir: &Path,
    request: &SearchRequest,
    embedder: Option<&dyn Embedder>,
) -> Result<SearchResults, Error> {
    // A build may publish a new generation and remove the one being read, or the
    // vectors of its chunks that are gone; then the search starts over on the new
    // generation. The query is embedded (and billed) once.
    let mut embedded: Option<(ModelDescriptor, Vec<f32>)> = None;
    let mut query_tokens = 0;
    let current = || std::fs::read_to_string(state_dir.join("CURRENT2")).ok();
    for attempt in 0.. {
        let before = current();
        let result = search_once(
            state_dir,
            request,
            embedder,
            &mut embedded,
            &mut query_tokens,
        );
        let retry = attempt < 2
            && (result.is_err() || request.mode != Mode::Lexical)
            && current() != before;
        if !retry {
            return result;
        }
    }
    unreachable!()
}

fn search_once(
    state_dir: &Path,
    request: &SearchRequest,
    embedder: Option<&dyn Embedder>,
    embedded: &mut Option<(ModelDescriptor, Vec<f32>)>,
    query_tokens: &mut u64,
) -> Result<SearchResults, Error> {
    let generation = Generation::open(state_dir)?;
    let info = generation.dense_info()?;
    let limit = request.limit;
    let ranked: Vec<(u32, f64, Option<usize>, Option<usize>)> = match request.mode {
        Mode::Lexical => generation
            .lexical(&request.query, limit, &request.filter)?
            .into_iter()
            .enumerate()
            .map(|(i, (d, s))| (d, s, Some(i + 1), None))
            .collect(),
        Mode::Dense | Mode::Hybrid => {
            request.fusion.validate()?;
            let info = info.as_ref().ok_or(
                "this index has no embeddings: build it with embeddings (index2 --embed) or search lexically",
            )?;
            let vector = match (&request.query_vector, &*embedded) {
                (Some(v), _) => v.clone(),
                (None, Some((model, v))) if *model == info.embedder => v.clone(),
                (None, _) => {
                    let embedder = embedder.ok_or("dense search needs an embedder")?;
                    let have = embedder.descriptor();
                    if have != info.embedder {
                        return Err(format!(
                            "the index was embedded with {}, not {}",
                            info.embedder.display(),
                            have.display()
                        )
                        .into());
                    }
                    let (v, tokens) = embed_query(embedder, &request.query)?;
                    *query_tokens += tokens;
                    *embedded = Some((have, v.clone()));
                    v
                }
            };
            let store = generation.open_store(info)?;
            let depth = if request.mode == Mode::Dense {
                limit
            } else {
                request.candidates.max(limit)
            };
            // Vectors of another generation in the store (see `open_store`) are
            // skipped: fetch enough to keep `depth` of this one's.
            let others = store.len().saturating_sub(info.embedded as usize);
            let hits = store.search(&vector, depth.saturating_add(others), &request.filter)?;
            let ids: Vec<_> = hits.iter().map(|h| h.id).collect();
            let dense: Vec<(u64, f64)> = hits
                .iter()
                .zip(generation.docs_of(&ids)?)
                .filter_map(|(h, doc)| Some((u64::from(doc?), f64::from(h.score))))
                .take(depth)
                .collect();
            if request.mode == Mode::Dense {
                dense
                    .into_iter()
                    .enumerate()
                    .map(|(i, (d, s))| (d as u32, s, None, Some(i + 1)))
                    .collect()
            } else {
                let lexical: Vec<(u64, f64)> = generation
                    .lexical(&request.query, depth, &request.filter)?
                    .into_iter()
                    .map(|(d, s)| (u64::from(d), s))
                    .collect();
                fuse(&lexical, &dense, &request.fusion, limit)?
                    .into_iter()
                    .map(|f| (f.id as u32, f.score, f.lexical_rank, f.dense_rank))
                    .collect()
            }
        }
    };
    let docs: Vec<u32> = ranked.iter().map(|r| r.0).collect();
    let hits = generation
        .chunks(&docs)?
        .into_iter()
        .zip(ranked)
        .map(|(chunk, (_, score, lexical_rank, dense_rank))| SearchHit {
            header: chunk.header(),
            chunk,
            score,
            lexical_rank,
            dense_rank,
        })
        .collect();
    Ok(SearchResults {
        hits,
        mode: request.mode,
        query_tokens: *query_tokens,
        dense: info,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_rewards_agreement_and_breaks_ties_by_id() {
        let lexical = [(10, 9.0), (20, 5.0), (30, 1.0)];
        let dense = [(30, 0.9), (40, 0.8), (10, 0.1)];
        let fused = fuse(&lexical, &dense, &Fusion::default(), 10).unwrap();
        let ids: Vec<u64> = fused.iter().map(|f| f.id).collect();
        // 10: 1/61 + 1/63; 30: 1/63 + 1/61 (equal: id order); 20: 1/62; 40: 1/62.
        assert_eq!(ids, [10, 30, 20, 40]);
        assert_eq!(fused[0].score, fused[1].score);
        assert_eq!(
            (fused[0].lexical_rank, fused[0].dense_rank),
            (Some(1), Some(3))
        );
        assert_eq!(
            (fused[3].lexical_rank, fused[3].dense_rank),
            (None, Some(2))
        );
        assert!((fused[0].score - (1.0 / 61.0 + 1.0 / 63.0)).abs() < 1e-12);
        // Weights: dense only.
        let dense_only = Fusion::Rrf {
            k: 60.0,
            lexical_weight: 0.0,
            dense_weight: 1.0,
        };
        let ids: Vec<u64> = fuse(&lexical, &dense, &dense_only, 3)
            .unwrap()
            .iter()
            .map(|f| f.id)
            .collect();
        assert_eq!(ids, [30, 40, 10]);
        // Limit, empty lists.
        assert_eq!(
            fuse(&lexical, &dense, &Fusion::default(), 1).unwrap().len(),
            1
        );
        assert!(fuse(&[], &[], &Fusion::default(), 5).unwrap().is_empty());
    }

    #[test]
    fn linear_fusion_normalizes_each_list() {
        let lexical = [(1, 30.0), (2, 20.0), (3, 10.0)];
        let dense = [(3, 0.9), (2, 0.5), (4, 0.1)];
        let fusion = Fusion::Linear {
            lexical_weight: 0.5,
        };
        let fused = fuse(&lexical, &dense, &fusion, 10).unwrap();
        let got: Vec<(u64, f64)> = fused.iter().map(|f| (f.id, f.score)).collect();
        // 1: .5·1; 2: .5·.5 + .5·.5; 3: .5·0 + .5·1; 4: .5·0 -> ties by id.
        assert_eq!(got, [(1, 0.5), (2, 0.5), (3, 0.5), (4, 0.0)]);
        let lexical_only = Fusion::Linear {
            lexical_weight: 1.0,
        };
        let ids: Vec<u64> = fuse(&lexical, &dense, &lexical_only, 3)
            .unwrap()
            .iter()
            .map(|f| f.id)
            .collect();
        assert_eq!(ids, [1, 2, 3]);
        // One candidate, or all equal: normalized to 1.
        let one = fuse(&[(7, 3.0)], &[], &fusion, 1).unwrap();
        assert_eq!(one[0].score, 0.5);
    }

    #[test]
    fn invalid_fusion_settings_are_rejected() {
        for f in [
            Fusion::Rrf {
                k: -1.0,
                lexical_weight: 1.0,
                dense_weight: 1.0,
            },
            Fusion::Rrf {
                k: 60.0,
                lexical_weight: f64::NAN,
                dense_weight: 1.0,
            },
            Fusion::Linear {
                lexical_weight: 1.5,
            },
        ] {
            assert!(fuse(&[], &[], &f, 1).is_err(), "{f:?}");
        }
    }
}
