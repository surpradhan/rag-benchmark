"""
Main evaluation runner.

Usage:
    python evaluation/run_eval.py --pattern zero_retrieval --run-id 1
    python evaluation/run_eval.py --pattern basic_rag --run-id 1 --dev   # dev set (500 q)
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.logger import RunLogger
from evaluation.metrics import (
    aggregate_latency,
    aggregate_retrieval,
    aggregate_tokens,
    compute_ragas_metrics,
    precision_at_k,
    recall_at_k,
)

PATTERN_MODULES = {
    "zero_retrieval": "rag_patterns.zero_retrieval",
    "basic_rag": "rag_patterns.basic_rag",
    "hybrid_rag": "rag_patterns.hybrid_rag",
    "reranking_rag": "rag_patterns.reranking_rag",
    "multiquery_rag": "rag_patterns.multiquery_rag",
    "hyde_rag": "rag_patterns.hyde_rag",
    "parent_child_rag": "rag_patterns.parent_child_rag",
    "self_query_rag": "rag_patterns.self_query_rag",
    "corrective_rag": "rag_patterns.corrective_rag",
    "agentic_rag": "rag_patterns.agentic_rag",
    "graph_rag": "rag_patterns.graph_rag",
    "tree_rag":  "rag_patterns.tree_rag",
    "oracle_rag": "rag_patterns.oracle_rag",
}


def load_pattern(pattern_name: str, config: dict):
    module_path = PATTERN_MODULES[pattern_name]
    module = importlib.import_module(module_path)
    # Convention: each module exposes a class with the same CamelCase name
    class_name = "".join(w.capitalize() for w in pattern_name.split("_"))
    cls = getattr(module, class_name)
    return cls(config)


def _checkpoint_path(results_dir: str, pattern_name: str, run_id: int) -> Path:
    return Path(results_dir) / f".ckpt_{pattern_name}_run{run_id}.json"


def _load_checkpoint(ckpt_path: Path) -> dict | None:
    """Load checkpoint if it exists. Returns None if not found or corrupt."""
    if not ckpt_path.exists():
        return None
    try:
        with ckpt_path.open() as f:
            return json.load(f)
    except Exception:
        return None


def _save_checkpoint(ckpt_path: Path, data: dict) -> None:
    """Atomically write checkpoint via a temp file."""
    tmp = ckpt_path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(data, f)
    tmp.replace(ckpt_path)


def run_eval(
    pattern_name: str,
    config: dict,
    questions: list[dict],
    run_id: int,
    k_values: list[int],
    use_ragas: bool = True,
    checkpoint_every: int = 50,
) -> Path:
    results_dir = config["evaluation"]["results_dir"]
    ckpt_path = _checkpoint_path(results_dir, pattern_name, run_id)

    logger = RunLogger(
        config=config,
        pattern_name=pattern_name,
        run_id=run_id,
        results_dir=results_dir,
    )

    pattern = load_pattern(pattern_name, config)
    top_k = max(k_values)

    ragas_questions, ragas_answers, ragas_contexts, ragas_golds = [], [], [], []
    latencies, token_counts = [], []
    per_query_retrieval = []

    # ── Resume from checkpoint if available ──────────────────────────────────
    ckpt = _load_checkpoint(ckpt_path)
    start_idx = 0
    if ckpt:
        print(f"  [resume] Checkpoint found — resuming from question {ckpt['next_idx']}/{len(questions)}")
        start_idx = ckpt["next_idx"]
        for rec in ckpt.get("per_query", []):
            logger.log_query(rec["question_id"], rec["question"],
                             {k: v for k, v in rec.items() if k not in ("question_id", "question")})
            latencies.append(rec["latency_ms"])
            token_counts.append(rec["token_count"])
            per_query_retrieval.append({
                "retrieved_titles": rec["retrieved_titles"],
                "gold_titles": rec["gold_titles"],
            })
            if use_ragas and rec.get("answer"):
                ragas_questions.append(rec["question"])
                ragas_answers.append(rec["answer"])
                ragas_contexts.append(rec.get("_contexts", [""]))
                ragas_golds.append(rec.get("_gold_answer", ""))

    print(f"\n[{pattern_name}] run {run_id} — {len(questions)} questions (starting at {start_idx})")
    for i, q in enumerate(questions[start_idx:], start=start_idx):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(questions)}...")
        try:
            gold_titles = q.get("supporting_titles", [])
            result = pattern.run(q["question"], top_k=top_k, gold_titles=gold_titles)

            retrieved_titles = [d["metadata"].get("title", "") for d in result.retrieved_docs]

            per_query_retrieval.append(
                {"retrieved_titles": retrieved_titles, "gold_titles": gold_titles}
            )
            latencies.append(result.latency_ms)
            token_counts.append(result.token_count)

            query_metrics = {
                "answer": result.answer,
                "latency_ms": result.latency_ms,
                "token_count": result.token_count,
                "retrieved_titles": retrieved_titles,
                "gold_titles": gold_titles,
                "metadata": result.metadata,
            }
            for k in k_values:
                query_metrics[f"recall@{k}"] = recall_at_k(retrieved_titles, gold_titles, k)
                query_metrics[f"precision@{k}"] = precision_at_k(retrieved_titles, gold_titles, k)

            logger.log_query(q["question_id"], q["question"], query_metrics)

            if use_ragas:
                ragas_questions.append(q["question"])
                ragas_answers.append(result.answer)
                ragas_contexts.append([d["text"] for d in result.retrieved_docs])
                ragas_golds.append(q.get("answer", ""))

            # ── Checkpoint every N questions ──────────────────────────────
            if checkpoint_every and (i + 1) % checkpoint_every == 0:
                ckpt_records = []
                for rec in logger.per_query:
                    r = dict(rec)
                    # stash RAGAS inputs so they survive resume
                    idx_in_ragas = len(ckpt_records)
                    if idx_in_ragas < len(ragas_questions):
                        r["_contexts"]    = ragas_contexts[idx_in_ragas]
                        r["_gold_answer"] = ragas_golds[idx_in_ragas]
                    ckpt_records.append(r)
                _save_checkpoint(ckpt_path, {"next_idx": i + 1, "per_query": ckpt_records})

        except Exception as e:
            logger.log_error(q["question_id"], traceback.format_exc())

    # Aggregate
    agg = {}
    agg.update(aggregate_retrieval(per_query_retrieval, k_values))
    agg.update(aggregate_latency(latencies))
    agg.update(aggregate_tokens(token_counts))

    if use_ragas and ragas_questions:
        print("  Running RAGAS...")
        agg.update(compute_ragas_metrics(ragas_questions, ragas_answers, ragas_contexts, ragas_golds))

    out_path = logger.save(agg)

    # Remove checkpoint now that the full result is saved
    if ckpt_path.exists():
        ckpt_path.unlink()

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", required=True, choices=list(PATTERN_MODULES.keys()))
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dev", action="store_true", help="Use dev set instead of test set")
    parser.add_argument("--no-ragas", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions (for quick testing)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    questions_path = (
        "data/processed/dev_questions.json" if args.dev else "data/processed/test_questions.json"
    )
    with open(questions_path) as f:
        questions = json.load(f)

    if args.limit:
        questions = questions[: args.limit]

    k_values = config["retrieval"].get("top_k_values", [5, 10])
    run_eval(
        pattern_name=args.pattern,
        config=config,
        questions=questions,
        run_id=args.run_id,
        k_values=k_values,
        use_ragas=not args.no_ragas,
    )


if __name__ == "__main__":
    main()
