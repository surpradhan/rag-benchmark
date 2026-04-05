"""
Structured run logger — writes timestamped JSON to results/raw/.
Each file: pattern_name, config_hash, timestamp, per_query_metrics, aggregate_metrics, errors.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def config_hash(config: dict) -> str:
    """Stable SHA-256 hash of the config dict."""
    serialised = json.dumps(config, sort_keys=True)
    return hashlib.sha256(serialised.encode()).hexdigest()[:12]


class RunLogger:
    def __init__(self, config: dict, pattern_name: str, run_id: int, results_dir: str = "results/raw"):
        self.config = config
        self.pattern_name = pattern_name
        self.run_id = run_id
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.start_time = datetime.utcnow().isoformat()
        self.per_query: list[dict] = []
        self.errors: list[dict] = []

    def log_query(self, question_id: str, question: str, result: dict) -> None:
        self.per_query.append({"question_id": question_id, "question": question, **result})

    def log_error(self, question_id: str, error: str) -> None:
        self.errors.append({"question_id": question_id, "error": error})

    def save(self, aggregate_metrics: dict[str, Any]) -> Path:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        filename = f"{date_str}_{self.pattern_name}_run{self.run_id}.json"
        out_path = self.results_dir / filename

        payload = {
            "pattern_name": self.pattern_name,
            "run_id": self.run_id,
            "config_hash": config_hash(self.config),
            "timestamp": self.start_time,
            "n_queries": len(self.per_query),
            "n_errors": len(self.errors),
            "aggregate_metrics": aggregate_metrics,
            "per_query_metrics": self.per_query,
            "errors": self.errors,
        }
        with out_path.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved → {out_path}")
        return out_path
