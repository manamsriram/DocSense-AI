# DocSense-AI Claude Harness

## Project
DocSense-AI is a Flask-based RAG application for question answering over uploaded PDFs with retrieval, reranking, graph expansion, and citation-grounded answers.

## Goal
Improve whole-system RAG quality on a fixed evaluation corpus.

## Verification command
`python evals/run_eval.py`

## Primary metric
`weighted_rag_score`

## Secondary metrics
- correctness
- groundedness
- citation_quality
- completeness
- latency_sec
- failures

## Regression gates
Reject a change if:
- the eval script fails
- failures increase materially
- citation quality decreases
- groundedness decreases significantly
- latency regresses unacceptably
- app tests fail

## Rules
- Make one atomic change per iteration.
- Run the verification command after every change.
- Keep only measured improvements.
- Revert losing changes cleanly.
- Do not silently change the benchmark.
- Do not remove hard examples.
- Prefer minimal diffs.

## Standard iteration output
1. Current best score
2. Hypothesis
3. Files changed
4. Eval result
5. Keep or revert
6. Next experiment