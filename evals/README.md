# DocSense-AI Eval Harness

## Goal
Evaluate the whole RAG system — retrieval, reranking, graph expansion, and
generation together — against a fixed corpus of uploaded PDFs, by driving
the real `/ask` endpoint over HTTP (no mocking).

## Corpus
Upload the evaluation PDFs into DocSense-AI first and wait until indexing
completes (locally this is synchronous; in production it runs via the
GitHub Actions ingestion offload — poll `/documents` until chunk counts
are non-zero before running the eval).

Recommended starter PDFs:
- NASA 2018–2020 Section 3 Triennial Report
- NASA 2021–2023 Section 3 Triennial Report
- Iron Mountain 2023 Annual Financial Report

## Benchmark
`benchmark.jsonl` (one JSON object per line) contains, per case:
- `id`, `question`
- `reference_answer` — used for the correctness proxy
- `must_include` — list of facts/strings the answer should contain (completeness)
- `metadata` — free-form, not scored

## Run
Set environment variables:

- `DOCSENSE_EVAL_BASE_URL` — e.g. `http://127.0.0.1:5000` or the Render prod URL
- `DOCSENSE_EVAL_BEARER_TOKEN` — Supabase JWT for a user with access to the indexed org
- `DOCSENSE_EVAL_SESSION_ID` — base session id; each question is sent under
  `{SESSION_ID}-{question_id}` so per-question sessions don't leak conversation
  history into each other's retrieval

```bash
export DOCSENSE_EVAL_BASE_URL=http://127.0.0.1:5000
export DOCSENSE_EVAL_BEARER_TOKEN=your_token_here
export DOCSENSE_EVAL_SESSION_ID=eval-session
python evals/run_eval.py
```

Each question is POSTed form-encoded to `/ask` (not JSON — that's what the
route expects) and scored against the live response.

## How scoring works
Metrics are computed from the real response (`answer`, `sources`), not an
LLM judge — cheap and deterministic, at the cost of being a token-overlap
proxy rather than a semantic grader:

| Metric | How it's computed |
|---|---|
| `correctness` | token-set overlap between the answer and `reference_answer` |
| `groundedness` | token-set overlap between the answer and the retrieved source chunks (proxy for "did the model actually use what it retrieved") |
| `citation_quality` | 1.0 if `sources` is non-empty, else 0.0 |
| `completeness` | fraction of `must_include` strings present in the answer |
| `latency_sec` | wall-clock time for the `/ask` call |
| `weighted_rag_score` | `0.40*correctness + 0.25*groundedness + 0.15*citation_quality + 0.20*completeness` |

A case that raises (HTTP error, or `/ask` returning `{"error": ...}`) counts
as a **failure**: all four quality metrics are recorded as `0.0` and
`latency_sec` as `null` for that case, and the run's `failures` counter
increments. Failures pull the aggregate score down but are tracked
separately in the summary so a crash-driven zero doesn't read the same as
a low-quality answer.

## Outputs
- `evals/latest_results.json`: `{ "summary": {...}, "results": [...] }` — full
  per-case answers, sources, and metrics from the most recent run
- `evals/history.json`: append-only list of `summary` objects, one per run —
  use this to track score trends over time

## Regression gates
Per the project's root `CLAUDE.md`, reject a change if, relative to the
current best run in `history.json`:
- `failures` increases materially
- `citation_quality` decreases
- `groundedness` decreases significantly
- `latency_sec` regresses unacceptably
- the app's own pytest suite fails

## Benchmark rules
- Keep the corpus fixed during comparisons.
- Do not delete hard questions to improve scores.
- Add new benchmark cases in separate commits.
- Rebuild reference answers only when the corpus changes intentionally.