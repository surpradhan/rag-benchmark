"""
Build FAISS and BM25 indexes from the processed chunks.
Run after prepare_dataset.py.

Usage:
    python scripts/build_indexes.py
    python scripts/build_indexes.py --force   # rebuild even if indexes exist
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_patterns.chunking import load_and_chunk
from rag_patterns.indexing import build_bm25_index, build_faiss_index

CORPUS_PATH = Path("data/processed/corpus.json")
CHUNKS_PATH = Path("data/processed/chunks.json")


def main(config_path: str = "config/config.yaml", force: bool = False) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    emb_cfg = config["embedding"]
    vs_cfg = config["vector_store"]
    bm25_cfg = config["bm25"]

    # 1. Chunk
    chunks = load_and_chunk(CORPUS_PATH, CHUNKS_PATH, config, force=force)
    print(f"Total chunks: {len(chunks):,}")

    # 2. FAISS
    build_faiss_index(
        chunks,
        persist_dir=vs_cfg["persist_dir"],
        model_name=emb_cfg["model"],
        dimension=emb_cfg["dimension"],
        force=force,
    )

    # 3. BM25
    build_bm25_index(chunks, persist_path=bm25_cfg["persist_path"], force=force)

    print("\nAll indexes built successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--force", action="store_true", help="Rebuild even if indexes exist")
    args = parser.parse_args()
    main(args.config, args.force)
