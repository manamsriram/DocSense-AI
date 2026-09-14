# Backlog

RAG quality fixes identified but not yet implemented, each meant to be picked up
in its own session as one atomic change (per CLAUDE.md). Current best measured
baseline: commit `d32b07b` (grade_chunks truncation 300->1200), weighted_rag_score
~0.59-0.65 live, 0 failures, 35-case benchmark.

## Open items

### 1. q2_total_assets_2023 still fails (retrieval_recall 0.0)
Reranker scores a table-of-contents / overview mention of "Table 2-1" above the
table's actual data+totals chunk (observed at rerank rank 14/40). A blanket
top_n raise (8->15) for the whole "complex" query bucket was tried and reverted
— fixed this one case but regressed overall score (0.594/0.652 -> 0.525/0.426
across two runs, latency also rose ~25-31s -> ~38s) by adding noise to every
complex query's synthesis context.
Needs a narrower fix, e.g.:
- Boost table-marked chunks specifically in reranked ordering (not a blanket
  top_n change for all complex queries).
- Or detect "how many / total" style questions and bias retrieval toward
  chunks containing summary/totals rows.
Reverted commits: 343fcd5 (raise), e4b5f91 (revert).

### 2. Embedding / reranker model swap
Original question that opened this line of work. Deferred — current small
models (all-MiniLM-L6-v2 embed, ms-marco-MiniLM-L-6-v2 rerank) run on
separate 512MB Render dynos; a bigger model needs a RAM bump too. Only
worth revisiting once retrieval-side bugs (chunking, grading, cutoffs) are
exhausted, so any measured gain isn't just noise from an unrelated fix.

### 3. Benchmark data quality issue (flagged, not fixed)
`q11_center_highest_built_assets_2023`'s reference answer says "Wallops
Flight Facility with 625 built assets," but the actual 2023 table shows
Kennedy Space Center at 919 built assets (higher than WFF's 625) — looks
like an error in the benchmark's reference answer itself. Per CLAUDE.md
("do not silently change the benchmark"), needs explicit sign-off before
touching evals/benchmark.jsonl.

### 4. Remove temporary /debug/* endpoints
`/debug/retrieval` and `/debug/page_chunks` in app.py (marked with
`# ponytail: temporary diagnostic endpoint`) were added to inspect
pre-rerank candidates and stored chunks directly. Auth-gated, but still
live in prod. Remove once no longer needed for diagnosis.

### 5. Figure-caption dedup — lower priority, already partially done
Table caption+region exclusion from the prose splitter is done. Figures got
the same caption-exclusion treatment. Not fully re-verified across all 3
eval docs post-combined-fix (only spot-checked). Lower severity than table
fix since figures come up less often in the benchmark.

## Notes on process
- Always re-run `python evals/run_eval.py` against the live endpoint after
  any change — local eval doesn't reflect prod's ingestion path or real
  latency.
- Live eval runs show real run-to-run variance even with zero code change
  (seen 0.652 vs 0.594 back to back) — don't trust a single run when a
  change is borderline; re-run once before deciding revert vs keep.
- Reindexing eval docs on prod is required after any ingestion/chunking
  change (not needed for retrieval/grading/generation-only changes).
