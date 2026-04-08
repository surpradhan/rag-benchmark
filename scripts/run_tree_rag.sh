#!/usr/bin/env bash
# Run all 3 tree_rag eval runs + RAGAS + aggregate + charts.
# Designed to run as a detached background process (nohup).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=/opt/anaconda3/bin/python3.11
LOG=/tmp/tree_rag_pipeline.log

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== tree_rag run 1 ==="
PYTHONUNBUFFERED=1 $PYTHON evaluation/run_eval.py --pattern tree_rag --run-id 1 --no-ragas >> "$LOG" 2>&1

log "=== tree_rag run 2 ==="
PYTHONUNBUFFERED=1 $PYTHON evaluation/run_eval.py --pattern tree_rag --run-id 2 --no-ragas >> "$LOG" 2>&1

log "=== tree_rag run 3 ==="
PYTHONUNBUFFERED=1 $PYTHON evaluation/run_eval.py --pattern tree_rag --run-id 3 --no-ragas >> "$LOG" 2>&1

log "=== RAGAS ==="
PYTHONUNBUFFERED=1 $PYTHON scripts/run_ragas.py --patterns tree_rag >> "$LOG" 2>&1

log "=== aggregate ==="
PYTHONUNBUFFERED=1 $PYTHON scripts/aggregate_results.py --input results/raw/ --output results/aggregated/ >> "$LOG" 2>&1

log "=== statistical analysis ==="
PYTHONUNBUFFERED=1 $PYTHON scripts/statistical_analysis.py >> "$LOG" 2>&1

log "=== charts ==="
PYTHONUNBUFFERED=1 $PYTHON scripts/generate_charts.py --input results/aggregated/ --output results/charts/ >> "$LOG" 2>&1

log "=== DONE ==="
