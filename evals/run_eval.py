import json
import os
import time
from pathlib import Path
from statistics import mean
import requests

EVAL_DIR = Path("evals")
BENCHMARK_PATH = EVAL_DIR / "benchmark.jsonl"
LATEST_RESULTS_PATH = EVAL_DIR / "latest_results.json"
HISTORY_PATH = EVAL_DIR / "history.json"

BASE_URL = os.getenv("DOCSENSE_EVAL_BASE_URL", "http://127.0.0.1:5000")
ASK_URL = f"{BASE_URL}/ask"
AUTH_TOKEN = os.getenv("DOCSENSE_EVAL_BEARER_TOKEN", "")
SESSION_ID = os.getenv("DOCSENSE_EVAL_SESSION_ID", "eval-session")

def load_benchmark(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def ask_question(question, qid):
    headers = {}
    if AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
    # /ask reads request.form (not JSON body) — must be form-encoded.
    # Each question gets its own session_id so /ask's multi-turn history lookup
    # doesn't leak prior questions' context into this question's retrieval.
    payload = {"question": question, "session_id": f"{SESSION_ID}-{qid}"}

    start = time.time()
    resp = requests.post(ASK_URL, data=payload, headers=headers, timeout=120)
    latency = time.time() - start
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")
    return {
        "answer": data.get("response", ""),
        "sources": data.get("sources", []),
        "latency_sec": latency
    }

def normalize(text):
    return " ".join(str(text).lower().strip().split())

def token_set(text):
    return set(normalize(text).split())

def overlap_score(a, b):
    a_tokens = token_set(a)
    b_tokens = token_set(b)
    if not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(b_tokens)

def contains_required_facts(answer, must_include):
    answer_n = normalize(answer)
    if not must_include:
        return 1.0
    hits = sum(1 for item in must_include if normalize(item) in answer_n)
    return hits / len(must_include)

def citation_presence_score(sources):
    return 1.0 if isinstance(sources, list) and len(sources) > 0 else 0.0

def source_text_blob(sources):
    parts = []
    for s in sources or []:
        parts.append(str(s.get("snippet", "")))
        parts.append(str(s.get("text", "")))
        parts.append(str(s.get("source", "")))
        parts.append(str(s.get("file_name", "")))
    return " ".join(parts)

def groundedness_proxy(answer, sources):
    src_text = source_text_blob(sources)
    if not src_text.strip():
        return 0.0
    return overlap_score(answer, src_text)

def correctness_proxy(answer, reference_answer):
    return overlap_score(answer, reference_answer)

def weighted_score(correctness, groundedness, citation_quality, completeness):
    return (
        0.40 * correctness +
        0.25 * groundedness +
        0.15 * citation_quality +
        0.20 * completeness
    )

def safe_mean(values):
    vals = [v for v in values if v is not None]
    return mean(vals) if vals else 0.0

def run():
    benchmark = load_benchmark(BENCHMARK_PATH)
    per_case = []
    agg = {
        "correctness": [],
        "groundedness": [],
        "citation_quality": [],
        "completeness": [],
        "latency_sec": [],
        "weighted_rag_score": [],
        "failures": 0
    }

    for row in benchmark:
        qid = row["id"]
        question = row["question"]
        reference_answer = row.get("reference_answer", "")
        must_include = row.get("must_include", [])
        metadata = row.get("metadata", {})

        try:
            result = ask_question(question, qid)
            answer = result["answer"]
            sources = result["sources"]
            latency_sec = result["latency_sec"]

            correctness = correctness_proxy(answer, reference_answer)
            groundedness = groundedness_proxy(answer, sources)
            citation_quality = citation_presence_score(sources)
            completeness = contains_required_facts(answer, must_include)
            total = weighted_score(correctness, groundedness, citation_quality, completeness)

            per_case.append({
                "id": qid,
                "question": question,
                "reference_answer": reference_answer,
                "answer": answer,
                "sources": sources,
                "metadata": metadata,
                "metrics": {
                    "correctness": correctness,
                    "groundedness": groundedness,
                    "citation_quality": citation_quality,
                    "completeness": completeness,
                    "latency_sec": latency_sec,
                    "weighted_rag_score": total
                }
            })

            agg["correctness"].append(correctness)
            agg["groundedness"].append(groundedness)
            agg["citation_quality"].append(citation_quality)
            agg["completeness"].append(completeness)
            agg["latency_sec"].append(latency_sec)
            agg["weighted_rag_score"].append(total)

        except Exception as e:
            agg["failures"] += 1
            per_case.append({
                "id": qid,
                "question": question,
                "reference_answer": reference_answer,
                "answer": "",
                "sources": [],
                "metadata": metadata,
                "error": str(e),
                "metrics": {
                    "correctness": 0.0,
                    "groundedness": 0.0,
                    "citation_quality": 0.0,
                    "completeness": 0.0,
                    "latency_sec": None,
                    "weighted_rag_score": 0.0
                }
            })
            agg["correctness"].append(0.0)
            agg["groundedness"].append(0.0)
            agg["citation_quality"].append(0.0)
            agg["completeness"].append(0.0)
            agg["weighted_rag_score"].append(0.0)

    summary = {
        "num_cases": len(benchmark),
        "failures": agg["failures"],
        "correctness": safe_mean(agg["correctness"]),
        "groundedness": safe_mean(agg["groundedness"]),
        "citation_quality": safe_mean(agg["citation_quality"]),
        "completeness": safe_mean(agg["completeness"]),
        "latency_sec": safe_mean(agg["latency_sec"]),
        "weighted_rag_score": safe_mean(agg["weighted_rag_score"])
    }

    output = {
        "summary": summary,
        "results": per_case
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(LATEST_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    history = []
    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(summary)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    run()