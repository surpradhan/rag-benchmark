"""
P2: Hybrid RAG — BM25 + FAISS vector search fused with Reciprocal Rank Fusion (RRF).
BM25 weight intentionally not tuned (fusion via rank, not score magnitude).
Config: retrieval.rrf_k=60, retrieval.bm25_top_k_multiplier=3
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .base_retriever import BaseRAG, RAGResult
from .indexing import (
    bm25_search,
    faiss_search,
    load_bm25_index,
    load_faiss_index,
)
from .llm_client import LLMClient


def _load_prompt(name: str) -> str:
    return (Path("config/prompts") / name).read_text().strip()


def _rrf_fuse(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    top_n: int = 5,
) -> list[dict]:
    """Reciprocal Rank Fusion. k=60 per the PRD spec."""
    scores: dict[str, float] = {}
    docs_by_chunk_id: dict[str, dict] = {}

    for rank, doc in enumerate(vector_results):
        cid = doc["metadata"]["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        docs_by_chunk_id[cid] = doc

    for rank, doc in enumerate(bm25_results):
        cid = doc["metadata"]["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        docs_by_chunk_id[cid] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [
        {**docs_by_chunk_id[cid], "score": rrf_score}
        for cid, rrf_score in ranked
    ]


class HybridRag(BaseRAG):
    def __init__(self, config: dict):
        super().__init__(config)
        embed_cfg = config.get("embedding", {})
        vs_cfg = config.get("vector_store", {})
        bm25_cfg = config.get("bm25", {})
        retrieval_cfg = config.get("retrieval", {})

        self.embed_model = SentenceTransformer(embed_cfg.get("model", "all-MiniLM-L6-v2"))
        self.normalize = embed_cfg.get("normalize", True)
        self.faiss_index, self.faiss_meta = load_faiss_index(vs_cfg.get("persist_dir", "./data/faiss_index"))
        self.bm25, self.bm25_meta = load_bm25_index(bm25_cfg.get("persist_path", "./data/bm25_index/bm25_index.pkl"))
        self.rrf_k = retrieval_cfg.get("rrf_k", 60)
        self.llm = LLMClient(config)
        self._prompt_template = _load_prompt("basic_qa.txt")

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self.embed_model.encode([query], normalize_embeddings=self.normalize, convert_to_numpy=True)
        return vec[0].astype(np.float32)

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        # Retrieve more candidates from each source before fusing
        candidate_k = top_k * 3
        vector_results = faiss_search(self.faiss_index, self.faiss_meta, self._embed_query(query), candidate_k)
        bm25_results = bm25_search(self.bm25, self.bm25_meta, query, candidate_k)
        return _rrf_fuse(vector_results, bm25_results, k=self.rrf_k, top_n=top_k)

    def generate(self, query: str, docs: list[dict]) -> tuple[str, int]:
        context = "\n\n".join(d["text"] for d in docs)
        prompt = self._prompt_template.format(context=context, question=query)
        return self.llm.complete(prompt)
