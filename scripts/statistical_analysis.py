"""
Statistical analysis: per-query significance tests vs. Basic RAG baseline.

For each pattern × metric pair, computes:
  - Mean ± std (Basic RAG and challenger)
  - Absolute delta and % delta vs. Basic RAG
  - Wilcoxon signed-rank test (paired, non-parametric) — p-value
  - Cohen's d effect size
  - Significance label: *** p<0.001 / ** p<0.01 / * p<0.05 / ns

Output:
  results/aggregated/statistical_analysis.csv
  results/aggregated/statistical_analysis.md   (publication-ready table)

Usage:
    python scripts/statistical_analysis.py
"""
from __future__ import annotations

import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
from scipy import stats

RAW_DIR = Path("results/raw")
OUT_DIR = Path("results/aggregated")
BASELINE = "basic_rag"
METRICS = ["recall@5", "precision@5", "recall@10", "precision@10"]
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Load per-query scores from raw JSON files
# ---------------------------------------------------------------------------

def load_per_query(raw_dir: Path) -> dict[str, dict[str, list[float]]]:
    """
    Returns {pattern_name: {metric: [score_per_query, ...]}}
    Aggregates across all runs for a pattern (consistent with CLAUDE.md).
    """
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(str(raw_dir / "*.json"))):
        with open(path) as f:
            d = json.load(f)
        pattern = d["pattern_name"]
        for q in d.get("per_query_metrics", []):
            for metric in METRICS:
                val = q.get(metric)
                if val is not None:
                    data[pattern][metric].append(float(val))
    return data


# ---------------------------------------------------------------------------
# Cohen's d (pooled std)
# ---------------------------------------------------------------------------

def cohens_d(a: list[float], b: list[float]) -> float:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    mean_diff = sum(a) / n1 - sum(b) / n2
    var1 = sum((x - sum(a) / n1) ** 2 for x in a) / (n1 - 1)
    var2 = sum((x - sum(b) / n2) ** 2 for x in b) / (n2 - 1)
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return mean_diff / pooled_std if pooled_std > 0 else float("nan")


def sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def effect_label(d: float) -> str:
    if math.isnan(d):
        return "n/a"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_query = load_per_query(RAW_DIR)

    if BASELINE not in per_query:
        print(f"[error] Baseline '{BASELINE}' not found in raw results.")
        return

    rows = []
    for pattern, metrics in sorted(per_query.items()):
        if pattern == BASELINE:
            continue
        for metric in METRICS:
            baseline_scores = per_query[BASELINE].get(metric, [])
            challenger_scores = metrics.get(metric, [])

            if not baseline_scores or not challenger_scores:
                continue

            # Align lengths by taking the minimum (in case of unequal run counts)
            n = min(len(baseline_scores), len(challenger_scores))
            b = baseline_scores[:n]
            c = challenger_scores[:n]

            mean_b = sum(b) / len(b)
            mean_c = sum(c) / len(c)
            std_b = math.sqrt(sum((x - mean_b) ** 2 for x in b) / max(len(b) - 1, 1))
            std_c = math.sqrt(sum((x - mean_c) ** 2 for x in c) / max(len(c) - 1, 1))

            delta_abs = mean_c - mean_b
            delta_pct = (delta_abs / mean_b * 100) if mean_b != 0 else float("nan")

            # Wilcoxon signed-rank test (requires paired data, n >= 10)
            try:
                _, p_value = stats.wilcoxon(c, b, alternative="two-sided")
            except ValueError:
                p_value = float("nan")

            d = cohens_d(c, b)

            rows.append({
                "pattern": pattern,
                "metric": metric,
                "baseline_mean": round(mean_b, 4),
                "baseline_std": round(std_b, 4),
                "challenger_mean": round(mean_c, 4),
                "challenger_std": round(std_c, 4),
                "delta_abs": round(delta_abs, 4),
                "delta_pct": round(delta_pct, 2),
                "p_value": round(p_value, 4) if not math.isnan(p_value) else None,
                "significant": p_value < ALPHA if not math.isnan(p_value) else None,
                "sig_label": sig_label(p_value) if not math.isnan(p_value) else "n/a",
                "cohens_d": round(d, 3) if not math.isnan(d) else None,
                "effect_size": effect_label(d),
            })

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "statistical_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    # --- Markdown summary table ---
    md_lines = [
        "# Statistical Analysis vs. Basic RAG Baseline",
        "",
        "**Test:** Wilcoxon signed-rank (paired, two-sided) | **α = 0.05**",
        "**Effect size:** Cohen's d (pooled std) — negligible <0.2, small 0.2–0.5, medium 0.5–0.8, large >0.8",
        "",
    ]

    for metric in METRICS:
        md_lines += [f"## {metric.upper()}", ""]
        md_lines += ["| Pattern | Baseline | Challenger | Δ (abs) | Δ (%) | p-value | sig | Cohen's d | Effect |",
                     "|---|---|---|---|---|---|---|---|---|"]
        subset = df[df["metric"] == metric].sort_values("delta_abs", ascending=False)
        for _, row in subset.iterrows():
            p_str = f"{row['p_value']:.4f}" if row["p_value"] is not None else "n/a"
            d_str = f"{row['cohens_d']:.3f}" if row["cohens_d"] is not None else "n/a"
            md_lines.append(
                f"| {row['pattern']} "
                f"| {row['baseline_mean']:.3f} ± {row['baseline_std']:.3f} "
                f"| {row['challenger_mean']:.3f} ± {row['challenger_std']:.3f} "
                f"| {row['delta_abs']:+.3f} "
                f"| {row['delta_pct']:+.1f}% "
                f"| {p_str} "
                f"| {row['sig_label']} "
                f"| {d_str} "
                f"| {row['effect_size']} |"
            )
        md_lines.append("")

    md_path = OUT_DIR / "statistical_analysis.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Saved → {md_path}")

    # --- Console summary ---
    print("\n=== Patterns significantly better than Basic RAG (p < 0.05) ===")
    sig_better = df[(df["significant"] == True) & (df["delta_abs"] > 0)]
    if sig_better.empty:
        print("  None")
    else:
        for _, row in sig_better.iterrows():
            print(f"  {row['pattern']:20s} {row['metric']:15s}  "
                  f"Δ={row['delta_abs']:+.3f} ({row['delta_pct']:+.1f}%)  "
                  f"p={row['p_value']:.4f} {row['sig_label']}  d={row['cohens_d']}")

    print("\n=== Patterns significantly worse than Basic RAG (p < 0.05) ===")
    sig_worse = df[(df["significant"] == True) & (df["delta_abs"] < 0)]
    if sig_worse.empty:
        print("  None")
    else:
        for _, row in sig_worse.iterrows():
            print(f"  {row['pattern']:20s} {row['metric']:15s}  "
                  f"Δ={row['delta_abs']:+.3f} ({row['delta_pct']:+.1f}%)  "
                  f"p={row['p_value']:.4f} {row['sig_label']}  d={row['cohens_d']}")


if __name__ == "__main__":
    main()
