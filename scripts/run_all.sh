#!/usr/bin/env bash
# run_all.sh — full benchmark runner (one command)
# Usage: ./scripts/run_all.sh [--dev] [--no-ragas] [--limit N]
set -euo pipefail

PATTERNS=(
  zero_retrieval
  basic_rag
  oracle_rag
  hybrid_rag
  reranking_rag
  multiquery_rag
  hyde_rag
  parent_child_rag
  self_query_rag
  corrective_rag
  agentic_rag
  graph_rag
  tree_rag
)
N_RUNS=3
DEV_FLAG=""
RAGAS_FLAG=""
LIMIT_FLAG=""

for arg in "$@"; do
  case $arg in
    --dev)       DEV_FLAG="--dev" ;;
    --no-ragas)  RAGAS_FLAG="--no-ragas" ;;
    --limit=*)   LIMIT_FLAG="--limit=${arg#*=}" ;;
  esac
done

echo "================================================"
echo "  RAG Benchmark — full run"
echo "  Patterns : ${#PATTERNS[@]}"
echo "  Runs/each: $N_RUNS"
echo "================================================"

# Step 1: dataset + indexes
python scripts/prepare_dataset.py
python scripts/build_indexes.py

# Step 2: evaluate each pattern x N_RUNS
for pattern in "${PATTERNS[@]}"; do
  for run in $(seq 1 $N_RUNS); do
    echo ""
    echo ">>> $pattern  run $run / $N_RUNS"
    python evaluation/run_eval.py \
      --pattern "$pattern" \
      --run-id "$run" \
      --config config/config.yaml \
      $DEV_FLAG $RAGAS_FLAG $LIMIT_FLAG || {
        echo "[warn] $pattern run $run failed — continuing"
      }
  done
done

# Step 3: aggregate
python scripts/aggregate_results.py --input results/raw/ --output results/aggregated/

# Step 4: charts
python scripts/generate_charts.py --input results/aggregated/ --output results/charts/

echo ""
echo "================================================"
echo "  Benchmark complete. Results in results/"
echo "================================================"
