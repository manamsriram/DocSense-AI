# DocSense-AI Claude Harness

## Project
DocSense-AI is a Flask-based document intelligence platform that answers questions over uploaded PDFs using hybrid retrieval, graph expansion, reranking, CRAG-style correction, and citation-grounded answer generation.

## Goal
Improve end-to-end RAG quality using a repeatable evaluation harness.

## Success metric
Primary metric: weighted_rag_score

weighted_rag_score =
0.40 * answer_correctness +
0.25 * groundedness +
0.20 * retrieval_relevance +
0.10 * citation_quality +
0.05 * completeness

Regression gates:
- hallucination_rate must not worsen by more than 0.03
- latency_p95 must not worsen by more than 20%
- failing test count must not increase
- citation_presence_rate must not decrease

## Commands
Install app deps:
`pip install -r requirements.txt`

Install eval deps:
`pip install -r requirements-eval.txt`

Run tests:
`pytest -q`

Run eval:
`python evals/run_eval.py`

## Architecture notes
- Main app entrypoint: `app.py`
- Online query endpoint: `POST /ask`
- Existing eval assets: `eval_dataset.json`, `eval_ragas.py`
- RAG system includes dense retrieval, BM25, RRF fusion, graph expansion, reranking, and CRAG-style correction

## Rules
- Make one atomic change per iteration.
- Always run `python evals/run_eval.py` after each change.
- Keep a change only if the primary metric improves and regression gates pass.
- Revert losing changes cleanly.
- Do not silently modify the benchmark.
- Do not remove difficult eval cases to improve scores.
- Prefer minimal diffs.
- Do not make destructive changes without confirmation.

## Output format for each iteration
1. Current best score
2. Hypothesis
3. Files changed
4. Eval result
5. Keep or revert
6. Next experiment