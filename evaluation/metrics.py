"""
Evaluation metrics for the RAG benchmark.

Retrieval metrics:
  - Recall@K   : fraction of gold supporting titles found in top-K results
  - Precision@K: fraction of top-K results that are gold supporting titles

Generation metrics (via RAGAS):
  - Faithfulness     : claims in answer grounded in retrieved context
  - Answer Relevance : answer addresses the question
  - Hallucination Rate: 1 - faithfulness

System metrics:
  - Latency (ms)
  - Token cost
"""
from __future__ import annotations

import statistics
from typing import Any


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_titles: list[str], gold_titles: list[str], k: int) -> float:
    """Fraction of gold titles found in the top-k retrieved titles."""
    if not gold_titles:
        return 0.0
    top_k = set(t.lower() for t in retrieved_titles[:k])
    gold = set(t.lower() for t in gold_titles)
    return len(top_k & gold) / len(gold)


def precision_at_k(retrieved_titles: list[str], gold_titles: list[str], k: int) -> float:
    """Fraction of top-k retrieved titles that are gold titles."""
    if k == 0:
        return 0.0
    top_k = [t.lower() for t in retrieved_titles[:k]]
    gold = set(t.lower() for t in gold_titles)
    hits = sum(1 for t in top_k if t in gold)
    return hits / k


# ---------------------------------------------------------------------------
# RAGAS generation metrics (lightweight wrapper)
# ---------------------------------------------------------------------------

def compute_ragas_metrics(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict[str, float]:
    """
    Run RAGAS faithfulness and answer_relevancy on a batch.
    Returns dict with mean scores.
    Falls back to NaN on errors so the benchmark continues.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        faith = float(result["faithfulness"])
        relevancy = float(result["answer_relevancy"])
        return {
            "faithfulness": faith,
            "answer_relevance": relevancy,
            "hallucination_rate": max(0.0, 1.0 - faith),
        }
    except Exception as e:
        print(f"[RAGAS error] {e}")
        return {"faithfulness": float("nan"), "answer_relevance": float("nan"), "hallucination_rate": float("nan")}


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_retrieval(
    per_query: list[dict],
    k_values: list[int] = (5, 10),
) -> dict[str, float]:
    """Compute mean Recall@K and Precision@K across queries."""
    agg: dict[str, list[float]] = {}
    for k in k_values:
        agg[f"recall@{k}"] = []
        agg[f"precision@{k}"] = []

    for q in per_query:
        retrieved = q.get("retrieved_titles", [])
        gold = q.get("gold_titles", [])
        for k in k_values:
            agg[f"recall@{k}"].append(recall_at_k(retrieved, gold, k))
            agg[f"precision@{k}"].append(precision_at_k(retrieved, gold, k))

    return {key: statistics.mean(vals) if vals else 0.0 for key, vals in agg.items()}


def aggregate_latency(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"p50_ms": 0.0, "p90_ms": 0.0, "p99_ms": 0.0}
    sorted_l = sorted(latencies_ms)
    n = len(sorted_l)
    return {
        "p50_ms": sorted_l[int(n * 0.50)],
        "p90_ms": sorted_l[int(n * 0.90)],
        "p99_ms": sorted_l[min(int(n * 0.99), n - 1)],
        "mean_ms": statistics.mean(latencies_ms),
        "std_ms": statistics.stdev(latencies_ms) if n > 1 else 0.0,
    }


def aggregate_tokens(token_counts: list[int]) -> dict[str, float]:
    if not token_counts:
        return {"mean_tokens": 0.0, "total_tokens": 0}
    return {
        "mean_tokens": statistics.mean(token_counts),
        "total_tokens": sum(token_counts),
    }
