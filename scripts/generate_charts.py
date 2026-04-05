"""
Generate all 8 required benchmark charts from aggregated results.
Output: PNG (300 DPI) + SVG per chart.

Usage:
    python scripts/generate_charts.py --input results/aggregated/ --output results/charts/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
PALETTE = sns.color_palette("husl", 12)
FIGSIZE_STD = (10, 6)
FIGSIZE_RADAR = (8, 8)
DPI = 300


def save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    print(f"  Saved {name}.png / .svg")
    plt.close(fig)


def chart1_overall_dashboard(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["recall@5_mean", "precision@5_mean", "faithfulness_mean", "answer_relevance_mean"]
    labels = ["Recall@5", "Precision@5", "Faithfulness", "Answer Relevance"]
    melted = df[["pattern"] + metrics].melt(id_vars="pattern", var_name="metric", value_name="score")
    melted["metric"] = melted["metric"].map(dict(zip(metrics, labels)))
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    sns.barplot(data=melted, x="pattern", y="score", hue="metric", palette=PALETTE[:4], ax=ax)
    ax.set_title("Overall Metric Dashboard", fontsize=14)
    ax.set_xlabel("Pattern", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="Metric")
    fig.tight_layout()
    save_fig(fig, out_dir, "1_overall_dashboard")


def chart2_accuracy_vs_cost(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    ax.scatter(df["mean_tokens_mean"], df["answer_relevance_mean"], s=80, c=PALETTE[:len(df)])
    for _, row in df.iterrows():
        ax.annotate(row["pattern"], (row["mean_tokens_mean"], row["answer_relevance_mean"]),
                    fontsize=9, ha="left", va="bottom")
    ax.set_title("Accuracy vs. Token Cost", fontsize=14)
    ax.set_xlabel("Mean Tokens / Query", fontsize=12)
    ax.set_ylabel("Answer Relevance", fontsize=12)
    save_fig(fig, out_dir, "2_accuracy_vs_cost")


def chart3_accuracy_vs_latency(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    ax.scatter(df["p50_ms_mean"], df["faithfulness_mean"], s=80, c=PALETTE[:len(df)])
    for _, row in df.iterrows():
        ax.annotate(row["pattern"], (row["p50_ms_mean"], row["faithfulness_mean"]),
                    fontsize=9, ha="left", va="bottom")
    ax.set_title("Accuracy vs. Latency", fontsize=14)
    ax.set_xlabel("Latency p50 (ms)", fontsize=12)
    ax.set_ylabel("Faithfulness", fontsize=12)
    save_fig(fig, out_dir, "3_accuracy_vs_latency")


def chart5_latency_distribution(raw_dir: Path, out_dir: Path) -> None:
    """Box plot of per-query latencies from raw result files."""
    import json, glob
    data = {}
    for path in glob.glob(str(raw_dir / "*.json")):
        with open(path) as f:
            d = json.load(f)
        name = d["pattern_name"]
        lats = [q.get("latency_ms", 0) for q in d.get("per_query_metrics", [])]
        data.setdefault(name, []).extend(lats)
    if not data:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    ax.boxplot(data.values(), labels=data.keys(), patch_artist=True)
    ax.set_title("Latency Distribution per Pattern", fontsize=14)
    ax.set_xlabel("Pattern", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    save_fig(fig, out_dir, "5_latency_distribution")


def chart4_recall_at_k(df: pd.DataFrame, out_dir: Path) -> None:
    """Line chart: Recall@K (K=5,10) per pattern."""
    # Sort by recall@5 descending for a clean legend order
    df_sorted = df.sort_values("recall@5_mean", ascending=False).reset_index(drop=True)
    ks = [5, 10]
    cols = ["recall@5_mean", "recall@10_mean"]

    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    colors = sns.color_palette("husl", len(df_sorted))

    for i, (_, row) in enumerate(df_sorted.iterrows()):
        values = [row[c] for c in cols]
        ax.plot(ks, values, marker="o", label=row["pattern"], color=colors[i], linewidth=1.8)

    ax.set_title("Recall@K Curves by Pattern", fontsize=14)
    ax.set_xlabel("K (number of retrieved docs)", fontsize=12)
    ax.set_ylabel("Recall@K", fontsize=12)
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Pattern", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir, "4_recall_at_k")


def chart6_radar(df: pd.DataFrame, out_dir: Path) -> None:
    """Radar/spider chart for top-5 patterns using available metrics."""
    import numpy as np

    # Select top-5 patterns by recall@5 (excluding oracle and zero as they are trivial bounds)
    exclude = {"oracle_rag", "zero_retrieval"}
    top5 = (
        df[~df["pattern"].isin(exclude)]
        .sort_values("recall@5_mean", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )

    # Metrics to plot (all on 0-1 scale after normalisation)
    metric_labels = ["Recall@5", "Recall@10", "Precision@5", "Precision@10",
                     "Speed\n(inv latency)", "Token\nEfficiency"]

    def norm(series: pd.Series) -> pd.Series:
        mn, mx = series.min(), series.max()
        return (series - mn) / (mx - mn) if mx > mn else series * 0 + 0.5

    # Invert latency and tokens so higher = better on radar
    top5["speed_score"] = norm(1 / top5["p50_ms_mean"])
    top5["token_eff_score"] = norm(1 / top5["mean_tokens_mean"])

    raw_cols = ["recall@5_mean", "recall@10_mean", "precision@5_mean",
                "precision@10_mean", "speed_score", "token_eff_score"]

    # Normalise retrieval metrics too for fair visual scaling
    for col in raw_cols[:4]:
        top5[col + "_norm"] = norm(top5[col])

    value_cols = [c + "_norm" for c in raw_cols[:4]] + ["speed_score", "token_eff_score"]

    N = len(metric_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=FIGSIZE_RADAR, subplot_kw={"polar": True})
    colors = sns.color_palette("husl", len(top5))

    for i, (_, row) in enumerate(top5.iterrows()):
        values = [row[c] for c in value_cols]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, color=colors[i], label=row["pattern"])
        ax.fill(angles, values, alpha=0.08, color=colors[i])

    ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Top-5 Patterns — Multi-Dimensional Comparison\n(normalised per metric)",
                 fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    save_fig(fig, out_dir, "6_radar")


def chart7_improvement_over_baseline(df: pd.DataFrame, out_dir: Path) -> None:
    if "basic_rag" not in df["pattern"].values:
        return
    baseline = df[df["pattern"] == "basic_rag"].iloc[0]
    others = df[df["pattern"] != "basic_rag"].copy()
    for col in ["recall@5_mean", "faithfulness_mean", "answer_relevance_mean"]:
        others[f"delta_{col}"] = (others[col] - float(baseline[col])) * 100
    melted = others[["pattern", "delta_recall@5_mean", "delta_faithfulness_mean", "delta_answer_relevance_mean"]].melt(
        id_vars="pattern", var_name="metric", value_name="delta_pct"
    )
    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    sns.barplot(data=melted, x="delta_pct", y="pattern", hue="metric", orient="h", ax=ax)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("% Improvement over Basic RAG Baseline", fontsize=14)
    ax.set_xlabel("Delta (%)", fontsize=12)
    ax.set_ylabel("Pattern", fontsize=12)
    save_fig(fig, out_dir, "7_improvement_over_baseline")


def chart8_error_heatmap(raw_dir: Path, questions_path: Path, out_dir: Path) -> None:
    """Heatmap of Recall@5 by question type (bridge/comparison) × pattern."""
    import json, glob
    import numpy as np

    # Build question_id → type mapping
    if not questions_path.exists():
        print("  [skip] chart8: questions file not found")
        return
    with open(questions_path) as f:
        questions = json.load(f)
    qid_to_type = {q["question_id"]: q.get("type", "unknown") for q in questions}

    # Collect per-pattern recall@5 by question type
    pattern_type_scores: dict[str, dict[str, list[float]]] = {}
    for path in sorted(glob.glob(str(raw_dir / "*.json"))):
        with open(path) as f:
            d = json.load(f)
        name = d["pattern_name"]
        if name not in pattern_type_scores:
            pattern_type_scores[name] = {}
        for q in d.get("per_query_metrics", []):
            qtype = qid_to_type.get(q.get("question_id"), "unknown")
            pattern_type_scores[name].setdefault(qtype, []).append(q.get("recall@5", 0.0))

    if not pattern_type_scores:
        return

    # Build matrix: rows = patterns sorted by overall recall@5, cols = question types
    qtypes = sorted({t for scores in pattern_type_scores.values() for t in scores})
    patterns = sorted(pattern_type_scores.keys(),
                      key=lambda p: sum(sum(v) / len(v) for v in pattern_type_scores[p].values()
                                        if v) / max(len(pattern_type_scores[p]), 1),
                      reverse=True)

    matrix = np.array([
        [sum(pattern_type_scores[p].get(t, [0])) / max(len(pattern_type_scores[p].get(t, [0])), 1)
         for t in qtypes]
        for p in patterns
    ])

    fig, ax = plt.subplots(figsize=(max(6, len(qtypes) * 2), max(6, len(patterns) * 0.55)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Recall@5")

    ax.set_xticks(range(len(qtypes)))
    ax.set_xticklabels(qtypes, fontsize=11)
    ax.set_yticks(range(len(patterns)))
    ax.set_yticklabels(patterns, fontsize=9)

    for i in range(len(patterns)):
        for j in range(len(qtypes)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black")

    ax.set_title("Recall@5 by Question Type × Pattern", fontsize=14)
    ax.set_xlabel("Question Type", fontsize=12)
    ax.set_ylabel("Pattern", fontsize=12)
    fig.tight_layout()
    save_fig(fig, out_dir, "8_error_heatmap")


def main(input_dir: str, output_dir: str) -> None:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    csv_path = in_dir / "comparison.csv"

    if not csv_path.exists():
        print(f"No comparison.csv in {in_dir}. Run aggregate_results.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Generating charts for {len(df)} patterns...")

    chart1_overall_dashboard(df, out_dir)
    chart2_accuracy_vs_cost(df, out_dir)
    chart3_accuracy_vs_latency(df, out_dir)
    chart4_recall_at_k(df, out_dir)
    chart5_latency_distribution(in_dir.parent / "raw", out_dir)
    chart6_radar(df, out_dir)
    chart7_improvement_over_baseline(df, out_dir)
    chart8_error_heatmap(in_dir.parent / "raw",
                         Path("data/processed/dev_questions.json"), out_dir)

    print(f"\nAll 8 charts saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/aggregated/")
    parser.add_argument("--output", default="results/charts/")
    args = parser.parse_args()
    main(args.input, args.output)
