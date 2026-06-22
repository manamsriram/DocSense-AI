"""
RAGAS evaluation script for DocsenseAI.

Calls the live /ask endpoint, collects (question, answer, contexts) triples,
and runs RAGAS faithfulness + answer_relevancy (+ context_precision when
ground_truth is provided in the dataset).

Usage:
    python eval_ragas.py --token <supabase-jwt> [--host http://localhost:5000] [--dataset eval_dataset.json]

Install eval deps first (separate from app requirements):
    pip install ragas langchain-google-genai datasets

Dataset format (eval_dataset.json):
    [{"question": "...", "ground_truth": "..."}]
    ground_truth is optional — omit or leave empty to skip context_precision.
"""
import argparse
import json
import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()


def call_ask(host, token, question):
    resp = requests.post(
        f"{host}/ask",
        data={"question": question},
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")
    answer = data.get("response", "")
    contexts = [s.get("text", "") for s in data.get("sources", []) if s.get("text")]
    return answer, contexts


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS eval against DocsenseAI")
    parser.add_argument("--host", default="http://localhost:5000")
    parser.add_argument("--token", required=True, help="Supabase JWT from browser devtools (Authorization header)")
    parser.add_argument("--dataset", default="eval_dataset.json")
    parser.add_argument("--output", default="eval_results.csv")
    args = parser.parse_args()

    with open(args.dataset) as f:
        samples = json.load(f)

    if not samples:
        print("Dataset is empty.")
        sys.exit(1)

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, s in enumerate(samples, 1):
        q = s["question"]
        print(f"[{i}/{len(samples)}] {q[:70]}...")
        ans, ctx = call_ask(args.host, args.token, q)
        questions.append(q)
        answers.append(ans)
        contexts.append(ctx)
        ground_truths.append(s.get("ground_truth", ""))

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from datasets import Dataset
    except ImportError:
        print("\nMissing eval deps. Run:\n  pip install ragas langchain-google-genai datasets")
        sys.exit(1)

    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("GEMINI_API_KEY not set — needed for RAGAS LLM judge.")
        sys.exit(1)

    llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"), google_api_key=gemini_key)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_key)

    has_ground_truth = any(gt.strip() for gt in ground_truths)
    metrics = [faithfulness, answer_relevancy]
    if has_ground_truth:
        metrics.append(context_precision)
        print("Ground truth detected — including context_precision.")

    data = {"question": questions, "answer": answers, "contexts": contexts}
    if has_ground_truth:
        data["ground_truth"] = ground_truths

    result = evaluate(Dataset.from_dict(data), metrics=metrics, llm=llm, embeddings=embeddings)

    print("\n=== RAGAS Results ===")
    print(result)

    df = result.to_pandas()
    df.to_csv(args.output, index=False)
    print(f"\nSaved per-question breakdown to {args.output}")


if __name__ == "__main__":
    main()
