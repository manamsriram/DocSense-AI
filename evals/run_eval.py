import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
import requests
from dotenv import load_dotenv

load_dotenv()

EVAL_DIR = Path(__file__).parent
BENCHMARK_PATH = EVAL_DIR / "benchmark.jsonl"
LATEST_RESULTS_PATH = EVAL_DIR / "latest_results.json"
HISTORY_PATH = EVAL_DIR / "history.json"

BASE_URL = os.getenv("DOCSENSE_EVAL_BASE_URL", "http://127.0.0.1:5000")
ASK_URL = f"{BASE_URL}/ask"
AUTH_TOKEN = os.getenv("DOCSENSE_EVAL_BEARER_TOKEN", "")
# No session_id by default: each benchmark question is scored standalone.
# Setting one makes /ask anchor retrieval to prior-turn history, which would
# contaminate scores across unrelated questions.
SESSION_ID = os.getenv("DOCSENSE_EVAL_SESSION_ID", "")

def load_benchmark(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def ask_question(question):
    headers = {}
    if AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"
    # /ask reads request.form (not JSON) — must be form-encoded.
    payload = {"question": question}
    if SESSION_ID:
        payload["session_id"] = SESSION_ID

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
    return " ".join(text.lower().strip().split())

def contains_required_facts(answer, must_include):
    answer_n = normalize(answer)
    if not must_include:
        return 1.0
    hits = sum(1 for item in must_include if normalize(item) in answer_n)
    return hits / len(must_include)

def citation_presence_score(sources):
    return 1.0 if sources and len(sources) > 0 else 0.0

def simple_correctness(answer, reference_answer):
    ref_tokens = set(normalize(reference_answer).split())
    ans_tokens = set(normalize(answer).split())
    if not ref_tokens:
        return 0.0
    overlap = len(ref_tokens & ans_tokens) / len(ref_tokens)
    return overlap

def completeness_score(answer, must_include):
    return contains_required_facts(answer, must_include)

def source_text(s):
    return str(s.get("text", ""))

def groundedness_proxy(answer, sources):
    if not sources:
        return 0.0
    joined = normalize(" ".join(source_text(s) for s in sources))
    if not joined:
        return 0.5
    answer_tokens = set(normalize(answer).split())
    source_tokens = set(joined.split())
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & source_tokens) / max(len(answer_tokens), 1)

def retrieval_relevance_proxy(question, sources):
    """How much of the question's vocabulary shows up in retrieved chunks —
    proxy for 'did retrieval fetch chunks about this question'."""
    if not sources:
        return 0.0
    q_tokens = set(normalize(question).split())
    if not q_tokens:
        return 0.0
    joined = normalize(" ".join(source_text(s) for s in sources))
    source_tokens = set(joined.split())
    return len(q_tokens & source_tokens) / len(q_tokens)

# Matches CLAUDE.md's weighted_rag_score definition.
def weighted_score(correctness, groundedness, retrieval_relevance, citation_quality, completeness):
    return (
        0.40 * correctness +
        0.25 * groundedness +
        0.20 * retrieval_relevance +
        0.10 * citation_quality +
        0.05 * completeness
    )

def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, int(len(s) * p))
    return s[idx]

def check_gates(current, previous):
    """Informational only — v1 prints PASS/WARN, does not fail the run."""
    if previous is None:
        return ["no previous history entry — nothing to gate against"]
    lines = []

    hall_delta = current["hallucination_rate"] - previous["hallucination_rate"]
    status = "WARN" if hall_delta > 0.03 else "PASS"
    lines.append(f"[{status}] hallucination_rate {previous['hallucination_rate']:.3f} -> {current['hallucination_rate']:.3f} (delta {hall_delta:+.3f}, gate: worsen <= 0.03)")

    prev_p95 = previous.get("latency_p95_sec")
    if prev_p95:
        lat_pct = (current["latency_p95_sec"] - prev_p95) / prev_p95
        status = "WARN" if lat_pct > 0.20 else "PASS"
        lines.append(f"[{status}] latency_p95_sec {prev_p95:.2f} -> {current['latency_p95_sec']:.2f} (delta {lat_pct:+.1%}, gate: worsen <= 20%)")

    cite_delta = current["citation_presence_rate"] - previous["citation_presence_rate"]
    status = "WARN" if cite_delta < 0 else "PASS"
    lines.append(f"[{status}] citation_presence_rate {previous['citation_presence_rate']:.3f} -> {current['citation_presence_rate']:.3f} (delta {cite_delta:+.3f}, gate: must not decrease)")

    lines.append("[SKIP] failing test count — run `pytest -q` separately, not part of this eval run")
    return lines

def run():
    if not AUTH_TOKEN:
        print("DOCSENSE_EVAL_BEARER_TOKEN not set. Get a Supabase JWT (browser devtools ->\n"
              "Authorization header on any /ask request while logged in) and run:\n"
              "  export DOCSENSE_EVAL_BEARER_TOKEN=<jwt>\n"
              "See evals/README.md.")
        sys.exit(1)

    benchmark = load_benchmark(BENCHMARK_PATH)
    if not benchmark:
        print(f"Benchmark {BENCHMARK_PATH} is empty.")
        sys.exit(1)

    results = []
    aggregate = {
        "correctness": [], "groundedness": [], "retrieval_relevance": [],
        "citation_quality": [], "completeness": [], "latency_sec": [],
        "weighted_rag_score": [],
    }

    for row in benchmark:
        q = row["question"]
        reference_answer = row.get("reference_answer", "")
        must_include = row.get("must_include", [])

        try:
            out = ask_question(q)
        except requests.exceptions.ConnectionError:
            print(f"\nCould not reach {BASE_URL} — is the Flask app running?\n  python app.py")
            sys.exit(1)
        except Exception as e:
            results.append({
                "id": row["id"], "question": q, "error": str(e),
                "metrics": {
                    "correctness": 0.0, "groundedness": 0.0, "retrieval_relevance": 0.0,
                    "citation_quality": 0.0, "completeness": 0.0, "latency_sec": None,
                    "weighted_rag_score": 0.0,
                },
                "metadata": row.get("metadata", {}),
            })
            for k in ("correctness", "groundedness", "retrieval_relevance", "citation_quality", "completeness", "weighted_rag_score"):
                aggregate[k].append(0.0)
            continue

        answer = out["answer"]
        sources = out["sources"]
        latency_sec = out["latency_sec"]

        correctness = simple_correctness(answer, reference_answer)
        groundedness = groundedness_proxy(answer, sources)
        retrieval_relevance = retrieval_relevance_proxy(q, sources)
        citation_quality = citation_presence_score(sources)
        completeness = completeness_score(answer, must_include)
        total = weighted_score(correctness, groundedness, retrieval_relevance, citation_quality, completeness)

        results.append({
            "id": row["id"],
            "question": q,
            "answer": answer,
            "reference_answer": reference_answer,
            "sources": sources,
            "metrics": {
                "correctness": correctness,
                "groundedness": groundedness,
                "retrieval_relevance": retrieval_relevance,
                "citation_quality": citation_quality,
                "completeness": completeness,
                "latency_sec": latency_sec,
                "weighted_rag_score": total,
            },
            "metadata": row.get("metadata", {}),
        })

        aggregate["correctness"].append(correctness)
        aggregate["groundedness"].append(groundedness)
        aggregate["retrieval_relevance"].append(retrieval_relevance)
        aggregate["citation_quality"].append(citation_quality)
        aggregate["completeness"].append(completeness)
        aggregate["latency_sec"].append(latency_sec)
        aggregate["weighted_rag_score"].append(total)

    groundedness_mean = mean(aggregate["groundedness"]) if aggregate["groundedness"] else 0.0
    latencies = [x for x in aggregate["latency_sec"] if x is not None]

    summary = {
        "num_cases": len(results),
        "correctness": mean(aggregate["correctness"]) if aggregate["correctness"] else 0.0,
        "groundedness": groundedness_mean,
        "retrieval_relevance": mean(aggregate["retrieval_relevance"]) if aggregate["retrieval_relevance"] else 0.0,
        "citation_quality": mean(aggregate["citation_quality"]) if aggregate["citation_quality"] else 0.0,
        "completeness": mean(aggregate["completeness"]) if aggregate["completeness"] else 0.0,
        "weighted_rag_score": mean(aggregate["weighted_rag_score"]) if aggregate["weighted_rag_score"] else 0.0,
        "hallucination_rate": round(1 - groundedness_mean, 4),
        "citation_presence_rate": mean(aggregate["citation_quality"]) if aggregate["citation_quality"] else 0.0,
        "latency_p50_sec": percentile(latencies, 0.50),
        "latency_p95_sec": percentile(latencies, 0.95),
    }

    full = {"summary": summary, "results": results}

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(LATEST_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2)

    history = []
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
    previous = history[-1] if history else None
    history.append(summary)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("=== Summary ===")
    print(json.dumps(summary, indent=2))
    print("\n=== Regression gates vs previous run ===")
    for line in check_gates(summary, previous):
        print(line)
    print(f"\nFull results: {LATEST_RESULTS_PATH}")
    print(f"History: {HISTORY_PATH}")

if __name__ == "__main__":
    run()