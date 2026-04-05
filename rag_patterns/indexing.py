"""
Indexing pipeline: chunks → FAISS IndexFlatIP + BM25 index.
Both indexes are serialised to disk and reloaded on subsequent runs.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_texts(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    normalize: bool = True,
    batch_size: int = 256,
) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# FAISS
# ---------------------------------------------------------------------------

FAISS_INDEX_FILE = "index.faiss"
FAISS_META_FILE = "metadata.pkl"


def build_faiss_index(
    chunks: list[dict],
    persist_dir: str | Path,
    model_name: str = "all-MiniLM-L6-v2",
    dimension: int = 384,
    force: bool = False,
) -> tuple[faiss.Index, list[dict]]:
    persist_dir = Path(persist_dir)
    idx_path = persist_dir / FAISS_INDEX_FILE
    meta_path = persist_dir / FAISS_META_FILE

    if idx_path.exists() and meta_path.exists() and not force:
        print(f"[skip] FAISS index already exists at {persist_dir} — loading")
        index = faiss.read_index(str(idx_path))
        with meta_path.open("rb") as f:
            metadata = pickle.load(f)
        return index, metadata

    persist_dir.mkdir(parents=True, exist_ok=True)
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts):,} chunks with {model_name}...")
    embeddings = embed_texts(texts, model_name=model_name)

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(idx_path))
    # Preserve all fields from each chunk so extra keys (e.g. parent_id) are retained
    metadata = [dict(c) for c in chunks]
    with meta_path.open("wb") as f:
        pickle.dump(metadata, f)

    print(f"FAISS index: {index.ntotal:,} vectors → {persist_dir}")
    return index, metadata


def load_faiss_index(persist_dir: str | Path) -> tuple[faiss.Index, list[dict]]:
    persist_dir = Path(persist_dir)
    index = faiss.read_index(str(persist_dir / FAISS_INDEX_FILE))
    with (persist_dir / FAISS_META_FILE).open("rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def faiss_search(
    index: faiss.Index,
    metadata: list[dict],
    query_embedding: np.ndarray,
    top_k: int,
) -> list[dict]:
    """Return top_k results as [{text, metadata, score}]."""
    scores, indices = index.search(query_embedding.reshape(1, -1), top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        meta = metadata[idx]
        results.append({"text": meta["text"], "metadata": meta, "score": float(score)})
    return results


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

BM25_FILE = "bm25_index.pkl"


def build_bm25_index(
    chunks: list[dict],
    persist_path: str | Path,
    force: bool = False,
) -> tuple[BM25Okapi, list[dict]]:
    persist_path = Path(persist_path)
    if persist_path.exists() and not force:
        print(f"[skip] BM25 index already exists at {persist_path} — loading")
        with persist_path.open("rb") as f:
            data = pickle.load(f)
        return data["bm25"], data["metadata"]

    persist_path.parent.mkdir(parents=True, exist_ok=True)
    tokenized = [c["text"].lower().split() for c in tqdm(chunks, desc="Tokenising for BM25")]
    bm25 = BM25Okapi(tokenized)
    metadata = [
        {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "title": c["title"], "text": c["text"]}
        for c in chunks
    ]
    with persist_path.open("wb") as f:
        pickle.dump({"bm25": bm25, "metadata": metadata}, f)
    print(f"BM25 index: {len(chunks):,} documents → {persist_path}")
    return bm25, metadata


def load_bm25_index(persist_path: str | Path) -> tuple[BM25Okapi, list[dict]]:
    with open(persist_path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["metadata"]


def bm25_search(
    bm25: BM25Okapi,
    metadata: list[dict],
    query: str,
    top_k: int,
) -> list[dict]:
    """Return top_k BM25 results as [{text, metadata, score}]."""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {"text": metadata[i]["text"], "metadata": metadata[i], "score": float(scores[i])}
        for i in top_indices
        if scores[i] > 0
    ]
