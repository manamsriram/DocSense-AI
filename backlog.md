# Backlog

RAG quality fixes identified but not yet implemented, each meant to be picked up
in its own session as one atomic change (per CLAUDE.md). Current best measured
baseline: commit `d32b07b` (grade_chunks truncation 300->1200), weighted_rag_score
~0.59-0.65 live, 0 failures, 35-case benchmark.

## Open items

### 1. q2_total_assets_2023 still fails (retrieval_recall 0.0) — FIXED, VERIFIED LIVE
Reranker scores a table-of-contents / overview mention of "Table 2-1" above the
table's actual data+totals chunk (observed at rerank rank 14/40). A blanket
top_n raise (8->15) for the whole "complex" query bucket was tried and reverted
— fixed this one case but regressed overall score (0.594/0.652 -> 0.525/0.426
across two runs, latency also rose ~25-31s -> ~38s) by adding noise to every
complex query's synthesis context.
Reverted commits: 343fcd5 (raise), e4b5f91 (revert).

Fixed narrowly instead: added `_AGGREGATION_HINT_RE` ("how many"/"total"/"sum"/
"overall"/"count") in `app.py`. When a query matches and the reranked top_n
slice has no table-marked chunk (`[Table]` first line, see `_is_table_chunk`),
`_promote_table_chunk_for_aggregation()` swaps in the best-scoring table chunk
found lower in the same ranked pool — no top_n change, no effect on
non-aggregation queries. Wired into both `find_relevant_chunks` and
`find_relevant_chunks_with_graph`.
Verified: q2 retrieval_recall 0.0 -> 0.5, correctness 1.0, answer 5,409
correct. 35-case eval (local, real Qdrant/model services, run under partial
Groq/Gemini quota exhaustion so noisier than usual): weighted_rag_score 0.569,
0 failures — within/above the recent 0.426-0.594 noisy range, no metric
regressed. 89/89 unit tests pass.

Re-verified live against prod (2026-09-22, new Render URL
docsense-ai-7k4b.onrender.com after the old ai-pdf-reader-ezm2 instance was
suspended — README/vercel.json updated): weighted_rag_score 0.604,
correctness 0.635, retrieval_recall 0.586, citation_quality 0.829, latency
29.7s avg, 3/35 failures (q7/q8/q9, all generic "API error, please try
again" — transient, not code-related). q2_total_assets_2023 held:
correctness 1.0, weighted_rag_score 0.81. No metric regressed vs. the
0.426-0.652 noisy baseline range. Item closed.

### 2. Embedding / reranker model swap
Original question that opened this line of work. Embed side done (swapped
to BAAI/bge-small-en-v1.5, see 2026-09-22 entries above). Reranker side
tried and reverted (2026-09-24):

Attempted `Xenova/ms-marco-MiniLM-L-6-v2` -> `Xenova/ms-marco-MiniLM-L-12-v2`
in `model_service/service.py:32` and `Dockerfile:18` (commit 367c566),
deployed live. Render killed the rerank instance mid-eval-run with
"HTTP health check failed (timed out after 5 seconds)" — an OOM kill on
the 512MB dyno, not an actual health-check bug.

Root cause: `RERANK_BATCH_SIZE = 3` and the every-other-batch
`gc.collect()` cadence (`service.py:77-104`) were tuned for L-6-v2's ONNX
arena footprint on this dyno tier (see the MODEL_ROLE split comment at
`service.py:14-15`). onnxruntime's default CPU allocator uses an arena
that grows across inference calls and does **not** release memory back to
the OS between calls (`Dockerfile:41` already flagged this for L-6-v2).
`gc.collect()` only reclaims Python-level cyclic garbage — it cannot free
that native arena, so it was never actually bounding arena growth, just
Python object churn. L-6-v2 stayed under budget because its per-batch
arena ceiling was small enough that the ratchet never crossed 512MB in a
session's lifetime; L-12-v2's larger hidden-layer activations raised that
ceiling enough to cross it partway through the live eval's 35 sequential
questions (denser request volume than normal prod traffic).
Reverted: `service.py`/`Dockerfile` back to L-6-v2 (commit after 367c566).

Follow-up if L-12-v2 (or the bigger `mxbai-rerank-xsmall-v1`, not in
fastembed's supported list and needing a different loading path — separate
scoping) is revisited: shrinking `RERANK_BATCH_SIZE` further only slows
the ratchet, doesn't stop it, since the arena still never releases. Real
fix needs `enable_cpu_mem_arena=False` on the onnxruntime session (untested
tradeoff: avoids the ratchet, costs per-call allocation overhead instead),
or a bigger dyno.

**2nd attempt, `enable_cpu_mem_arena=False` (2026-09-24), tried and
reverted:** `enable_cpu_mem_arena=False` is natively supported by
fastembed 0.8.0 (`TextCrossEncoder`'s `**kwargs` -> `EXPOSED_SESSION_OPTIONS`,
see `fastembed/common/onnx_model.py`) — passed it directly, no OOM this
time, deploy stayed healthy. But the live 35-case eval, which normally
finishes in ~17min (~30s/question), was still running after 33+ minutes
and never completed — the main app's own `/health` check measured 66.6s
round-trip mid-run, meaning the eval was still far from done at 30min+, well
past any reasonable regression tolerance. Per-call allocation overhead from
disabling the arena appears to be severe enough on this dyno tier to make
L-12-v2 impractical here even without OOM.
Commits: `1798bd7` (attempt), `1ef3742` (revert).

Conclusion: L-12-v2 is not viable on the current 512MB rerank dyno under
either the default arena (OOMs) or `enable_cpu_mem_arena=False` (too
slow). Staying on L-6-v2. Reopening this only makes sense with a bigger
dyno (untested) — item 2 closed as "not viable at current tier" otherwise.

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

### 6. Self-hosted vector store per org (future, not started)
Qdrant Cloud is itself a third party that receives raw chunk text on every
upsert (payload `text` field), independent of any LLM-provider exposure.
Fully removing that would mean self-hosting the vector store (e.g. per-org
Qdrant instance or another self-hosted engine) instead of Qdrant Cloud —
a separate infra project (provisioning, scaling, backups per tenant), not
something to bundle into the pseudonymization work (item below). Raised
2026-09-22 while scoping pseudonymization: BM25 and dense embeddings need
raw text to retrieve well, so pseudonymizing what's stored in Qdrant isn't
a substitute for this — genuinely needs its own infra change.

### 7. CRAG_WALL_CLOCK_BUDGET_S (12s) is tighter than observed retrieval latency — root cause noted, not fixed
PR #3 (pseudonymization, merged 2026-09-24) eval runs came in at
weighted_rag_score 0.551-0.593 vs. best historical 0.652, which looked like
a regression from the diff at first. Investigated instead of blocking the
merge:

- `pseudonymize_text`/`deanonymize_text` are cache-backed (single-digit ms
  once warm) — not the cause. Retrieval itself
  (`find_relevant_chunks_with_graph`) and `CRAG_WALL_CLOCK_BUDGET_S` are
  byte-identical to master in this diff.
- Real cause #1 (fixed in this PR): `gemini-3.5-flash`'s free tier is
  20 requests/day — quota exhaustion (`429 RESOURCE_EXHAUSTED`) mid-eval
  silently dropped answers before the fix. Added `GEMINI_MODEL_CHAIN`
  fallback (flash -> flash-lite -> 3-flash-preview -> 2.5-flash ->
  2.5-flash-lite -> 2.0-flash) in `_call_gemini_with_fallback()`. Confirmed
  firing correctly on a live 429 mid-run; score improved 0.551 -> 0.593
  after this alone.
- Real cause #2 (NOT fixed, still open): `retrieval_iter0` was observed
  taking 12.9s-22.8s per call against cloud Qdrant from a local eval
  environment, but `CRAG_WALL_CLOCK_BUDGET_S` is hardcoded to 12s — so the
  CRAG loop aborts before grading even runs on any slow-retrieval query,
  producing "couldn't find relevant information" (0 retrieval_recall) on
  otherwise-answerable questions. This is pre-existing, unrelated to the
  pseudonymization diff, and is likely why the noisy 0.426-0.652 range in
  item 1/Notes above exists at all — whatever run scored 0.652 probably
  just had a faster Qdrant round-trip that run, not different code.
  Needs its own fix: either raise the budget, make it adaptive to observed
  retrieval latency, or investigate why retrieval itself is sometimes
  12-22s (network path, embedding-call latency, Qdrant load) as a separate
  session.

**Attempt (2026-09-24), tried and reverted:** raised
`CRAG_WALL_CLOCK_BUDGET_S` default 12->25 in `app.py`. Unit tests passed
(143/143), but a live local eval against real Qdrant/model services
crashed the prod rerank dyno mid-run (`503 Service Unavailable` then
`429 Too Many Requests` on `/rerank`, then a `HTTP health check failed
(timed out after 5 seconds)` restart) — the same OOM/arena-ratchet
signature as item 2, on the currently-deployed `L-6-v2` model, which had
been stable before this change.

Root cause: at 12s, most slow-retrieval questions got cut off after their
*first* CRAG iteration, since retrieval alone often already takes
12-33s — the tight budget was accidentally acting as a rate limiter on
how many `/rerank` calls a single slow question could generate. Raising
it to 25s let more iterations/sub-queries survive the check
(`retrieval_iter1` went from ~0 to 4 occurrences across 39 in one eval
run), increasing `/rerank` call volume in the same wall-clock window.
Since onnxruntime's arena never releases memory between calls
(`service.py:14-15`, same note as item 2) and the rerank dyno serializes
all calls through one lock (`service.py:82`, single 512MB instance), more
calls stacking up is enough to trip the same ratchet regardless of model
size — the budget fix and item 2's model-size fix hit the same downstream
constraint from two different directions.

Reverted `app.py` back to the 12s default (net no-op diff).

Conclusion: this item can't be fixed by raising the wall-clock budget
alone without first addressing rerank capacity (item 2's dyno is the
shared bottleneck) — doing so trades "some slow questions get cut short"
for "the rerank service falls over for everyone mid-eval." Needs either:
a bigger/less marginal rerank dyno, a per-iteration (not just overall)
rerank call budget/concurrency cap, or the adaptive-budget approach
(scale the timeout to *observed* retrieval latency per call rather than a
blanket raise) so slow questions don't multiply rerank load unboundedly.
Reopen only alongside a rerank-capacity fix.

**2026-09-24, two follow-ups landed:**

1. **Embed timing instrumentation.** Added `_timed()` wrapping around all
   four `get_embedding_model().embed(...)` call sites in `app.py`:
   `embed_ingest` (bulk ingestion), `embed_query` (hot path, once per CRAG
   iteration/sub-query in `hybrid_search`), `embed_cache_lookup` and
   `embed_cache_store` (semantic cache). Previously embed latency was
   folded into the outer `retrieval_iterN` total with no way to isolate
   it. Needed before any embed speed-up work — can't fix what isn't
   measured. Not yet run against live traffic to see the actual split.

2. **Rerank capacity fix, attempt 1: app-side concurrency cap.** Added
   `RERANK_MAX_CONCURRENT` (default 2, env-configurable) via a
   `threading.Semaphore` around `_RemoteReranker.rerank()` (app.py), logging
   `rerank_semaphore_wait` when a call queues >50ms. Caps how many
   concurrent `/rerank` HTTP calls this app instance sends, independent of
   `CRAG_WALL_CLOCK_BUDGET_S` — protects the single-worker rerank dyno from
   a multi-user concurrent-request pileup.
   **Caveat, important:** this does *not* fix the specific failure mode
   from the attempt above. That crash came from single-sequential eval
   traffic (one question at a time, no concurrency) accumulating enough
   total `/rerank` calls to hit the dyno's `--max-requests 40` recycle
   threshold sooner — a call-*volume* problem, not a call-*concurrency*
   problem. A semaphore of 2 does nothing to reduce total request count
   over time. This change is real protection for concurrent multi-user
   prod traffic, but reopening the CRAG budget increase still needs one
   of: a bigger dyno, a per-request/session rerank-call budget, or the
   adaptive-timeout approach — unchanged from the conclusion above.
   143/143 unit tests pass with both changes; not yet re-verified with a
   live eval run (rerank dyno was mid-recovery from the earlier crash at
   time of writing).

   **Live eval re-verification (same session, ~20min later):** 35 cases,
   0 failures, weighted_rag_score 0.4685, citation_quality 0.686,
   groundedness 0.452, latency_sec 27.9 (all in-line with the historical
   noisy 0.426-0.652 range — no gate tripped). Rerank dyno stayed fully
   healthy the entire run: zero 429/503/health-timeout errors, zero
   `rerank_semaphore_wait` events (expected — eval traffic is sequential,
   so the concurrency cap had nothing to queue; this run doesn't exercise
   it, only confirms it doesn't break anything). Note this run's absolute
   score is confounded by simultaneous quota exhaustion on all three LLM
   providers (Groq daily TPD maxed, Gemini free-tier maxed, OpenRouter
   `402 Payment Required` out of credits) — every answer fell through to
   the last-resort fallback model, depressing correctness/completeness
   independent of these changes. Retrieval-side metrics (citation_quality,
   groundedness, latency, failures) are what's actually informative here
   and all look normal.

   `embed_query` timing (new instrumentation, 32 samples): consistently
   300-700ms, no outliers. Confirms embed is *not* the retrieval
   bottleneck — `retrieval_iter0` totals of 15-33s are dominated by
   Qdrant/rerank/graph-expansion, not embedding. No embed speed-up work
   is warranted right now; closing that half of this investigation.

   **Kept both changes** (embed instrumentation + rerank semaphore) —
   no regression gate tripped, 143/143 tests pass.

## Notes on process
- Always re-run `python evals/run_eval.py` against the live endpoint after
  any change — local eval doesn't reflect prod's ingestion path or real
  latency.
- Live eval runs show real run-to-run variance even with zero code change
  (seen 0.652 vs 0.594 back to back) — don't trust a single run when a
  change is borderline; re-run once before deciding revert vs keep.
- Reindexing eval docs on prod is required after any ingestion/chunking
  change (not needed for retrieval/grading/generation-only changes).
