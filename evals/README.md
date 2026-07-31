# DocSense-AI Eval Harness

## Goal
Evaluate the RAG system as a whole against a fixed corpus of uploaded PDFs.

## Corpus
Upload the evaluation PDFs into DocSense-AI first and wait until indexing completes.

Recommended starter PDFs:
- NASA 2018–2020 Section 3 Triennial Report
- NASA 2021–2023 Section 3 Triennial Report
- Iron Mountain 2023 Annual Financial Report

## Benchmark
`benchmark.jsonl` contains questions, reference answers, and required facts derived from the uploaded corpus.

## Run
Set environment variables:

- `DOCSENSE_EVAL_BASE_URL`
- `DOCSENSE_EVAL_BEARER_TOKEN`
- `DOCSENSE_EVAL_SESSION_ID`

Example:

```bash
export DOCSENSE_EVAL_BASE_URL=http://127.0.0.1:5000
export DOCSENSE_EVAL_BEARER_TOKEN=your_token_here
export DOCSENSE_EVAL_SESSION_ID=eval-session
python evals/run_eval.py
```

## Outputs
- `evals/latest_results.json`: full per-case results
- `evals/history.json`: summary history across runs

## Metrics
- correctness
- groundedness
- citation_quality
- completeness
- latency_sec
- weighted_rag_score

## Benchmark rules
- Keep the corpus fixed during comparisons.
- Do not delete hard questions to improve scores.
- Add new benchmark cases in separate commits.
- Rebuild reference answers only when the corpus changes intentionally.