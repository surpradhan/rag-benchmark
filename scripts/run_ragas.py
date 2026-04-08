"""
Post-hoc RAGAS evaluation script.

For each pattern, loads stored answers from the most recent raw JSON run,
re-runs retrieval only (no LLM generation) to get context texts, then
computes Faithfulness and Answer Relevance via RAGAS using Ollama locally.

Updates the raw JSON in-place with RAGAS scores and re-aggregates results.

Usage:
    python scripts/run_ragas.py [--patterns basic_rag hybrid_rag ...] [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import yaml

# ── RAGAS + LangChain setup ───────────────────────────────────────────────
def build_ragas_metrics():
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, faithfulness

    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = cfg["llm"].get("model", "llama3.1:8b-instruct-q8_0")
    base_url = cfg["llm"].get("base_url", "http://localhost:11434")
    embed_model = cfg["embedding"].get("model", "all-MiniLM-L6-v2")

    # 180s timeout per LLM call — must match RunConfig timeout to avoid
    # RAGAS retrying after the Ollama client has already given up.
    llm = ChatOllama(model=model, base_url=base_url, temperature=0, timeout=180)
    emb = HuggingFaceEmbeddings(model_name=embed_model)

    faithfulness.llm = LangchainLLMWrapper(llm)
    answer_relevancy.llm = LangchainLLMWrapper(llm)
    answer_relevancy.embeddings = LangchainEmbeddingsWrapper(emb)

    return faithfulness, answer_relevancy


# ── Load pattern + retrieve contexts ─────────────────────────────────────
def load_pattern(pattern_name: str, config: dict):
    """Instantiate a RAG pattern by name."""
    import importlib
    mod = importlib.import_module(f"rag_patterns.{pattern_name}")
    # Find the class (subclass of BaseRAG)
    from rag_patterns.base_retriever import BaseRAG
    for name in dir(mod):
        cls = getattr(mod, name)
        try:
            if isinstance(cls, type) and issubclass(cls, BaseRAG) and cls is not BaseRAG:
                return cls(config)
        except Exception:
            continue
    raise ValueError(f"No BaseRAG subclass found in rag_patterns.{pattern_name}")


def retrieve_contexts(pattern, question: str, top_k: int) -> list[str]:
    try:
        docs = pattern.retrieve(question, top_k)
        return [d.get("text", "") for d in docs if d.get("text")]
    except Exception as e:
        print(f"    [warn] retrieve failed: {e}")
        return []


# ── Find latest run JSON for a pattern ───────────────────────────────────
def find_latest_run(pattern_name: str, raw_dir: Path) -> Path | None:
    files = sorted(glob.glob(str(raw_dir / f"*_{pattern_name}_run*.json")))
    return Path(files[-1]) if files else None


# ── RAGAS batch eval ──────────────────────────────────────────────────────
def run_ragas_batch(questions, answers, contexts, ground_truths, faithfulness_m, relevancy_m):
    from datasets import Dataset
    from ragas import evaluate, RunConfig

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    # max_workers=1 forces sequential LLM calls — prevents Ollama from being
    # overwhelmed by 16 parallel requests and avoids cascading TimeoutErrors.
    run_cfg = RunConfig(max_workers=1, timeout=180, max_retries=2)
    result = evaluate(dataset, metrics=[faithfulness_m, relevancy_m],
                      run_config=run_cfg, raise_exceptions=False)
    faith = result["faithfulness"]
    rel   = result["answer_relevancy"]
    # Convert any NaN to None for clean JSON storage
    import math
    faith = [None if (v is None or (isinstance(v, float) and math.isnan(v))) else v for v in faith]
    rel   = [None if (v is None or (isinstance(v, float) and math.isnan(v))) else v for v in rel]
    return faith, rel


# ── Main ──────────────────────────────────────────────────────────────────
PATTERNS = [
    "zero_retrieval", "basic_rag", "oracle_rag", "hybrid_rag",
    "reranking_rag", "multiquery_rag", "hyde_rag", "parent_child_rag",
    "self_query_rag", "corrective_rag", "agentic_rag", "graph_rag",
    "tree_rag",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patterns", nargs="+", default=PATTERNS)
    parser.add_argument("--limit", type=int, default=None, help="Limit questions per pattern (for testing)")
    parser.add_argument("--sample", type=int, default=50, help="Random sample size per pattern (default 50; use 0 for all)")
    parser.add_argument("--raw-dir", default="results/raw")
    parser.add_argument("--force", action="store_true", help="Re-run RAGAS even if already computed")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    top_k = config["retrieval"].get("top_k", 5)
    if isinstance(top_k, list):
        top_k = top_k[0]

    print("Loading RAGAS metrics (Ollama + local embeddings)...")
    faithfulness_m, relevancy_m = build_ragas_metrics()
    print("Ready.\n")

    for pattern_name in args.patterns:
        print(f"{'='*60}")
        print(f"Pattern: {pattern_name}")

        json_path = find_latest_run(pattern_name, raw_dir)
        if json_path is None:
            print(f"  [skip] No raw JSON found for {pattern_name}")
            continue

        with open(json_path) as f:
            data = json.load(f)

        per_query = data.get("per_query_metrics", [])
        if args.limit:
            per_query = per_query[:args.limit]

        # Random sample for speed — seed=42 matches the project reproducibility
        # contract so the same 50 questions are always selected per pattern.
        if args.sample and args.sample > 0 and len(per_query) > args.sample:
            import random
            random.seed(42)  # must be set immediately before sample(), not earlier
            per_query = random.sample(per_query, args.sample)
            print(f"  Sampled {args.sample} from {len(data['per_query_metrics'])} queries in {json_path.name}")
        else:
            print(f"  Loaded {len(per_query)} queries from {json_path.name}")

        # Check if RAGAS already computed (NaN-safe check)
        import math
        existing_faith = data.get("aggregate_metrics", {}).get("faithfulness")
        already_done = (
            existing_faith is not None
            and not (isinstance(existing_faith, float) and math.isnan(existing_faith))
        )
        if already_done and not args.force:
            print(f"  [skip] RAGAS already computed (faithfulness={existing_faith:.4f}). Use --force to rerun.")
            continue

        # Instantiate pattern for retrieval
        print(f"  Instantiating pattern for retrieval...")
        try:
            pattern = load_pattern(pattern_name, config)
        except Exception as e:
            print(f"  [error] Could not load pattern: {e}")
            continue

        # Retrieve contexts for each query
        print(f"  Retrieving contexts ({len(per_query)} questions)...")
        questions, answers, contexts, ground_truths = [], [], [], []
        for i, q in enumerate(per_query):
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(per_query)}...")
            ctx = retrieve_contexts(pattern, q["question"], top_k)
            if not ctx:
                ctx = [""]  # RAGAS needs at least one context
            questions.append(q["question"])
            answers.append(q.get("answer", ""))
            contexts.append(ctx)
            ground_truths.append(q.get("gold_answer", q.get("answer", "")))

        # Load ground truth answers from questions JSON
        try:
            with open("data/processed/dev_questions.json") as f:
                dev_qs = {q["question_id"]: q.get("answer", "") for q in json.load(f)}
            ground_truths = [dev_qs.get(q.get("question_id", ""), "") for q in per_query]
        except Exception:
            pass  # fall back to stored answers as ground truth

        # Run RAGAS
        print(f"  Running RAGAS on {len(questions)} questions (this takes a while)...")
        t0 = time.time()
        try:
            faith_scores, rel_scores = run_ragas_batch(
                questions, answers, contexts, ground_truths,
                faithfulness_m, relevancy_m
            )
        except Exception as e:
            print(f"  [error] RAGAS failed: {e}")
            continue

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

        # Store per-query RAGAS scores back into JSON
        for i, q in enumerate(per_query):
            q["faithfulness"] = faith_scores[i] if i < len(faith_scores) else None
            q["answer_relevance"] = rel_scores[i] if i < len(rel_scores) else None

        valid_faith = [s for s in faith_scores if s is not None]
        valid_rel   = [s for s in rel_scores   if s is not None]
        mean_faith = sum(valid_faith) / len(valid_faith) if valid_faith else float("nan")
        mean_rel   = sum(valid_rel)   / len(valid_rel)   if valid_rel   else float("nan")

        data["aggregate_metrics"]["faithfulness"]     = mean_faith
        data["aggregate_metrics"]["answer_relevance"] = mean_rel
        data["aggregate_metrics"]["hallucination_rate"] = max(0.0, 1.0 - mean_faith)

        print(f"  Faithfulness:     {mean_faith:.4f}")
        print(f"  Answer Relevance: {mean_rel:.4f}")
        print(f"  Hallucination:    {max(0.0, 1-mean_faith):.4f}")

        # Write back
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved → {json_path.name}")

    print("\nAll done. Run aggregate_results.py to update comparison.csv.")


if __name__ == "__main__":
    main()
