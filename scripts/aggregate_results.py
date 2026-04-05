"""
Aggregate per-run JSON results into comparison CSVs (mean ± std per pattern).

Usage:
    python scripts/aggregate_results.py --input results/raw/ --output results/aggregated/
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd


METRIC_KEYS = [
    "recall@5", "recall@10",
    "precision@5", "precision@10",
    "faithfulness", "answer_relevance", "hallucination_rate",
    "p50_ms", "p90_ms", "mean_tokens",
]


def load_runs(raw_dir: Path) -> dict[str, list[dict]]:
    """Load all result JSONs, grouped by pattern name."""
    by_pattern: dict[str, list[dict]] = {}
    for path in sorted(raw_dir.glob("*.json")):
        with path.open() as f:
            data = json.load(f)
        name = data["pattern_name"]
        by_pattern.setdefault(name, []).append(data)
    return by_pattern


def aggregate(by_pattern: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for pattern, runs in by_pattern.items():
        row = {"pattern": pattern, "n_runs": len(runs)}
        for key in METRIC_KEYS:
            values = [r["aggregate_metrics"].get(key) for r in runs if r["aggregate_metrics"].get(key) is not None]
            values = [v for v in values if v == v]  # drop NaN
            if values:
                row[f"{key}_mean"] = statistics.mean(values)
                row[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            else:
                row[f"{key}_mean"] = float("nan")
                row[f"{key}_std"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pattern")


def main(input_dir: str, output_dir: str) -> None:
    raw_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_pattern = load_runs(raw_dir)
    if not by_pattern:
        print(f"No result files found in {raw_dir}")
        return

    df = aggregate(by_pattern)
    out_path = out_dir / "comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"Aggregated results → {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/raw/")
    parser.add_argument("--output", default="results/aggregated/")
    args = parser.parse_args()
    main(args.input, args.output)
