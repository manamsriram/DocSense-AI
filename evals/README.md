# Eval Harness

## Purpose
Repeatable RAG evaluation for DocSense-AI. Calls the live `/ask` endpoint for
a fixed set of benchmark questions and scores each answer with deterministic,
LLM-free proxy metrics — fast and stable enough to run after every change.

Metric definitions and the primary `weighted_rag_score` formula are defined
in `/CLAUDE.md`. This script implements that formula exactly.

For a heavier, LLM-judged eval (RAGAS: faithfulness, answer_relevancy,
context_precision), use `eval_ragas.py` at the repo root instead — slower
and costs LLM calls, so it's not the per-iteration loop.

## Setup
1. Start the app locally: `python app.py` (or `flask run`) — needs the same
   `.env` as the main app (`SUPABASE_*`, `QDRANT_*`, `GROQ_API_KEY`, etc).
2. Upload at least one document for the account you'll eval with, so
   retrieval has something to find. The default `benchmark.jsonl` questions
   assume a document about a company security policy — replace them with
   questions about your own test document, or upload one matching this
   scenario.
3. Get a bearer token: log in via the browser, open devtools → Network, find
   an `/ask` request, copy the `Authorization: Bearer <token>` value (a
   Supabase JWT).
4. Export env vars:
   ```
   export DOCSENSE_EVAL_BASE_URL=http://127.0.0.1:5000   # default shown
   export DOCSENSE_EVAL_BEARER_TOKEN=<jwt>                # required
   ```
   Leave `DOCSENSE_EVAL_SESSION_ID` unset — the harness scores each question
   standalone. Setting it makes `/ask` treat questions as one multi-turn
   conversation, which skews retrieval and scores.

## Run
```
python evals/run_eval.py
```
No extra dependencies beyond what's already in `requirements.txt`
(`requests`, `python-dotenv`).

## Outputs
- `evals/latest_results.json` — full detail for the most recent run: every
  question, answer, sources, and per-item metrics.
- `evals/history.json` — one summary object appended per run (timestamped by
  order), used to compare runs and check regression gates. Never edited by
  hand.

Each run also prints a summary and a `PASS`/`WARN` line per regression gate
from `CLAUDE.md`, compared against the previous `history.json` entry. Gates
are informational in v1 — the script doesn't fail the run, a human (or Claude
session) reads the output and decides whether to keep or revert the change.
`failing test count` is not automated here; run `pytest -q` separately.

## Benchmark maintenance rules
`evals/benchmark.jsonl` is one JSON object per line:
```
{"id": "case_004", "question": "...", "reference_answer": "...", "must_include": ["phrase1", "phrase2"], "metadata": {"difficulty": "easy", "type": "factual"}}
```
- `id` — stable, unique, never reused.
- `question` — the query sent to `/ask`.
- `reference_answer` — free text; scored via word-overlap against the answer.
- `must_include` — phrases that must literally appear in the answer (used for
  `completeness`). Can be empty.
- `metadata` — freeform tags (`difficulty`, `type`, etc), not scored.

Rules:
- Add new cases with their own commit; don't bundle with a code change.
- Never remove or water down a case just because it scores poorly — that's
  hiding a regression, not fixing one.
- Keep the benchmark stable while comparing runs. Changing it invalidates
  comparison against prior `history.json` entries — note this in the run.
