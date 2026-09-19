# Chunk repair findings — framework run, 2026-09-13

## Current behavior — accuracy takes priority

The historical budgets and size targets described below have been removed. Code chunking preserves complete functions, impls and other logical components regardless of size; documentation follows topic/example boundaries. Only actual Voyage constraints require embedding subdivision. Whole source files are shown to Luna, and repairs receive complete source context and may return multiple legal boundaries.

Search returns complete chunk text with no preview or response budget, no hidden candidate cap, and no default hit-count restriction. `top_k` is an optional caller-requested page size. Repairs and transient transport retries have no attempt limit. Per-attempt events now expose starts, failures, waits and completion immediately. Changing the chunking policy invalidates old boundary reuse. The earlier 1,200–2,400 and experimental 6,000–12,000 character targets are historical measurements, not current guidance.

The updated regression suite checks source preservation and retrieval beyond the removed limits. See the package README for the current API; the entries below retain the measured causes that led to these corrections.

## Fragmented groups — 2026-09-14

The subsequent 137-file run contained fragmentation beyond imports. Repair 107 split related delimiter constants in `arko-agents/src/parsing/mod.rs`; repair 136 separated module declarations in `arko-workflows/src/lib.rs` into individual lines; repair 106 detached a serde-default helper from `AgentToolConfig`. Repairs 137 and 105 separated introductory prose from its example in the index and agents READMEs. Some small type declarations originated in initial request 98, so improving repairs alone would not address the initial grouping policy. A small complete unit is not inherently a failure.

Both prompts now share responsibility-based grouping guidance with concrete import and prose/example cases. Repairs explicitly distinguish possible boundaries from a checklist and demonstrate selecting the end of an import block. Full-file repair context remains in use. There are no new size thresholds or broad parser heuristics for semantic grouping.

One deterministic guard omission also surfaced: repair 121 selected L1 in `arko-ai/src/client.rs`, creating a chunk containing only `use crate::{`. The Rust visitor did not protect `ItemUse`. It now protects the existing AST span of each complete import statement, including multiline trees and attributes. Grouping several statements together remains allowed and model-guided. Tests verify internal import cuts are rejected while a single combined import group preserves every source byte.

These updates target observed causes. Live grouping checks are needed in addition to boundary validation: the earlier 39 repairs passed the then-implemented guards on their first attempts, but that did not establish semantic grouping quality or coverage of every Rust syntax construct.

Transient provider errors retry automatically without an attempt limit. Each failure increases the previous wait by about one second plus jitter, or uses a longer provider-requested delay. There is no configured wait ceiling; pending waits release the provider slot and remain pausable/cancellable. The retry regression test uses virtual time to verify four increasing waits, provider retry timing and success on the fifth attempt.

## Historical: why an allowance existed

The default two rounds were a trial attempt budget intended to prevent unproductive repeated model calls. It is not a time limit and does not make an invalid boundary acceptable. This run showed that incomplete repair context can consume the allowance without giving the model a solvable task. Increasing the allowance alone would repeat that mistake.

`Config.max_repair_rounds` now accepts `Some(n)` or `None` (unlimited). The default remains `Some(2)`; `Some(0)` permits deterministic fixes but no model repairs. There is no arbitrary maximum of ten. Existing serialized integer configurations remain compatible. Unlimited rounds retain source validation, context preflight, pause and cancellation. Resume uses the original configuration and preserves validated files/cuts; the allowance resets for blocked windows.

## Measured baseline

Corpus: 136 files, 1,529,583 LF-normalized UTF-8 bytes; 34 small-file bypasses. Luna High, six files/request, two active requests, eager scheduling. The first pass took 1,560.89 seconds and stopped with 133 files validated, after 18 initial calls and 56 targeted repair calls. No provider errors or transport retries occurred. This is whole-run wall time, not an estimate of individual request latency.

107 initial cut failures had overlapping causes: 69 suffix mismatches, 26 blank separators, 52 protected-unit splits and 12 non-internal boundaries. A cut can belong to multiple categories. The high repair count is the principal efficiency problem; final JSON length alone misses reasoning-token cost and additional round trips.

Evidence is retained in the local framework index database: job `52539b50-2bdc-47de-ace7-7e148d65d823`, its `requests` rows and file snapshots. The companion runtime `failure-audit.json` and `request-summary.json` contain compact derived counts. Row references below refer to those snapshots, since working files can change.

## Recurring cases and treatment

| Case | Evidence / root cause | Treatment |
| --- | --- | --- |
| Several disputed cuts, only the first visible | `terminal/runner.rs` cuts at rows 67, 117, 163 and 231 formed one region. Requests 29 and 71 showed neighborhoods centered only on row 67. Increasing radius from 12 to 60 still omitted later disputes. Both replies requested more context. | Implemented: cover every dispute and the context of remote exact-anchor candidates. Expand on subsequent requests, bounded by fixed neighbors and request preflight. |
| More context offered after all relevant context is shown | Radius previously stayed at 60 after the first expansion. Additional rounds could replay identical insufficient context. | Implemented: progressive expansion; remove `MORE_CONTEXT` from the tool enum when the entire interval is visible. Reject that response if it is unavailable. |
| Redundant final boundary mistaken for missing content | `terminal/io.rs` row 138 is `    })`; Luna submitted row 138 with tail `}`. The final function closes at row 139, which is EOF and therefore implicit. Requests 25/69 offered only `REMOVE`, but both replies requested context. `terminal/tests.rs` requests 32/72 repeated the same pattern. | Implemented: explain that removing a cut merges chunks while preserving every source byte. Exhaustively check the entire fixed-neighbor interval; if no legal nonblank internal row exists, remove disputed cuts deterministically and record the proof reason. A locally empty menu alone is insufficient. |
| Blank separator chosen with the preceding brace as anchor | `terminal/runner.rs` row 117 is blank; row 116 is `}`. `file_tree.rs` row 309 is blank; row 308 closes a test. | Current behavior: targeted validated candidates. Candidate optimization: canonicalize to the preceding nonblank row only when the anchor matches there, the intervening rows are blank, ordering is preserved and structure permits the boundary. This is not implemented yet. |
| Rust inner return-expression close mistaken for method close | The `io.rs` example confuses `})` with the next function-closing `}`. Short function/statement guards also rejected otherwise matching interior rows. | Existing initial prompt includes Rust brace/comma examples. Repairs keep parser checks. Candidate optimization: present compact legal-boundary hints for dense Rust functions, and benchmark failures per file against added input tokens. |
| Row/anchor disagreement | The mismatched Markdown anchors in this baseline all exist verbatim elsewhere in their snapshots: the reported row is wrong. For example, `arko-workflows/README.md` cut row 231 quotes the distiller paragraph, but that row contains hook documentation. Rust references also disagree with their rows. | Current behavior: source-backed candidate IDs eliminate recopying during repair. Candidate optimization: relocate an anchor only when its intended legal boundary is unambiguous within fixed neighbors; otherwise retain targeted candidate selection. Keep exact matching, with no byte cap. Prompt examples about capitalization would not address the measured Markdown failure. |
| EOF emitted as a cut | 12 baseline cuts were outside the strict internal window bounds, including window EOF. | Start and EOF are already implicit in the initial prompt and now explicit in repairs. Candidate optimization: remove an exact EOF reference deterministically after proving it is only a redundant terminator; preserve all earlier valid cuts. Broader canonicalization is not yet implemented. |

## Accuracy and efficiency rules

1. Source snapshots, exact byte spans, order, parser guards and reconstruction remain authoritative. A model proposes grouping, never replacement source.
2. A deterministic correction needs a proof. A unique nearby match or an empty local candidate list does not establish intent across the full interval.
3. Preserve accepted cuts and files. Supply only disputed regions and necessary neighboring context. Record the snapshot/revision on repairs so stale choices cannot apply.
4. Measure completion rate, repairs per file, input/output tokens including reasoning, and wall time at fixed batch/concurrency settings. Compare initial chunking and repair separately. Do not infer speed from reply character count or from a single call.
5. Keep Luna High for this run. Lower reasoning for candidate selection is a separate controlled benchmark; compare boundary validity and chunk usefulness as well as speed. Source validity alone cannot prove equivalent retrieval quality.

## Regression coverage

Tests cover grouped distant disputes, context for remote anchors, expansion beyond the old 60-row plateau, unavailable context requests, legacy/default/unlimited/zero allowances, deterministic completion with zero model-repair allowance, implicit EOF, exact source reconstruction, accepted-cut preservation, and a negative case where an unseen legal boundary prevents automatic removal. Existing tests still cover stale repair revisions, parser guards, checkpoint/resume and embedding validation.

## Larger-chunk follow-up

A six-file trial compared the current 1,200–2,400-character guidance with responsibility-based grouping and a loose 6,000–12,000-character target. The same 125,557 source bytes produced 62 baseline chunks versus 18 trial chunks (trial median 6,079 bytes). Both initial calls and repairs used Luna High; this run did not use Max.

Across six hand-selected, file-scoped source-evidence queries, the first result covered all selected evidence in 1/6 baseline cases and 5/6 trial cases. All four code cases fit the trial's first result. For memory generation/delivery, source tokens to coverage fell from 1,607 to 1,179. Larger groups were not uniformly better: the custom-tool documentation example went from a focused first result costing 229 source tokens to requiring two ranked results totaling 2,433 tokens. This is a small coverage test, not an end-to-end agent evaluation or full-corpus ranking benchmark; multiple chunks can already be fetched in one tool call.

Recommendation: prefer larger coherent groups of related code declarations; keep documentation sections and runnable examples focused instead of enforcing a large minimum. Preserve semantic boundaries, offer query-relevant previews with exact offsets, and fetch the complete group on demand. The trial prompt has not replaced the production default. Changing that default must also invalidate old boundary reuse through a persisted chunk-policy identity.

Two additional failure points were observed:

- Several full-baseline chunks contain only a closing brace (one or two bytes). They passed byte coverage and parser checks but provide no independent retrieval value. Keep enclosing closing delimiters with the preceding group. A deterministic merge must preserve source and embedding limits; distinguish these fragments from legitimate tiny whole files. The trial prompt avoided this fragment pattern in its six files; a deterministic tail merge is still a follow-up.
- The trial took 1,010.88 seconds through publication, but its first request suffered a streaming transport failure: `unexpected EOF during chunk size line`. The scheduler retried the six-file request because that attempt produced no accepted result. The successful initial response used 34,663 input / 2,074 output tokens; one targeted repair used 2,637 / 3,373. Failed-attempt usage was unavailable, so these figures are not complete trial token consumption. A separate tiny Luna High probe completed in 1.61 seconds. The long wall time cannot establish that larger chunks or the new prompt are slower.

Record attempt start, failure, backoff and completion events as they happen. Currently retry details become visible only with the eventual response/error, leaving an operator unable to distinguish useful model work from a stalled connection or retry wait. Preserve accepted batches across transport retries, and benchmark source-volume limits alongside the six-file cap to bound repeated work on a failed large request. Do not substitute a total wall-clock deadline for those diagnostics.
