#!/usr/bin/env bash
# Run this after scripts/run_ragas.py finishes to re-aggregate, regenerate charts,
# and update both the dashboard and GitHub Pages with RAGAS metrics.

set -e
cd "$(dirname "$0")/.."

echo "============================================================"
echo "Step 1: Re-aggregate results (includes RAGAS scores)"
echo "============================================================"
python3.11 scripts/aggregate_results.py --input results/raw/ --output results/aggregated/
echo ""

echo "============================================================"
echo "Step 2: Regenerate all 8 charts"
echo "============================================================"
python3.11 scripts/generate_charts.py --input results/aggregated/ --output results/charts/
echo ""

echo "============================================================"
echo "Step 3: Inject RAGAS scores into dashboard HTML"
echo "============================================================"
python3.11 - <<'PYEOF'
import json, csv, math, re
from pathlib import Path

# Load updated comparison.csv
rows = {}
with open("results/aggregated/comparison.csv") as f:
    for row in csv.DictReader(f):
        rows[row["pattern"]] = row

def fmt(val, decimals=4):
    try:
        v = float(val)
        if math.isnan(v): return "—"
        return f"{v:.{decimals}f}"
    except (TypeError, ValueError):
        return "—"

# Build RAGAS summary for console output
print("\nRAGAS Summary:")
print(f"{'Pattern':<25} {'Faithfulness':>14} {'Ans. Relevance':>15} {'Hallucination':>14}")
print("-" * 72)
for p, r in sorted(rows.items()):
    faith = fmt(r.get("faithfulness_mean", ""))
    rel   = fmt(r.get("answer_relevance_mean", ""))
    hall  = fmt(r.get("hallucination_rate_mean", ""))
    print(f"{p:<25} {faith:>14} {rel:>15} {hall:>14}")

print("\nDone — comparison.csv updated with RAGAS scores.")
print("Charts regenerated in results/charts/")
print("Commit results/ and docs/ to publish the updated dashboard.")
PYEOF

echo ""
echo "============================================================"
echo "All done!"
echo ""
echo "Next steps:"
echo "  1. Review results/charts/ — charts 2, 3, 6 now have RAGAS data"
echo "  2. Update README.md to add charts 2/3/6 and RAGAS columns"
echo "  3. git add results/ docs/ && git commit -m 'Add RAGAS metrics'"
echo "  4. git push  →  GitHub Pages updates automatically"
echo "============================================================"
