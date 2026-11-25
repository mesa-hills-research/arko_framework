# Changelog

## 0.3.3 (2026-11-18)

### Borrowed embeddings

- A build or an embedding run can copy vectors from the embedding caches of other index state
  directories, such as the indexes of other worktrees of the same project, instead of embedding
  unchanged chunks again. `DenseBuild::borrow_caches` (`with_borrow_caches`) names the
  directories, and `EmbedStats::borrowed` counts the texts copied.
- A donor's cache is read without its lock and never written. The index's own directory and a
  cache of another model are skipped, and a vector that is missing, corrupt or truncated counts as
  a miss. Copied vectors go into the index's own cache, so the index stays whole when a donor is
  compacted or deleted.
- `index2::build`, also with `embed_later`, writes the borrowed vectors to the vector store before
  it publishes the generation, so dense search finds the unchanged chunks at once.
  `index2::embed` looks in the donors before it counts tokens, so borrowed texts don't count
  toward `max_tokens`.
- The `arko_index_cli` example's embedding lines say how many texts were borrowed.
