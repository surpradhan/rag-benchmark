"""
P3: Re-ranking RAG — FAISS retrieves top-20, cross-encoder re-ranks to top-5.
Cross-encoder: ms-marco-MiniLM-L-6-v2 (per PRD spec).
Latency is tracked separately for retrieval vs. re-ranking.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from .base_retriever import BaseRAG, RAGResult
from .indexing import faiss_search, load_faiss_index
from .llm_client import LLMClient


def _load_prompt(name: str) -> str:
    return (Path("config/prompts") / name).read_text().strip()


class RerankingRag(BaseRAG):
    def __init__(self, config: dict):
        super().__init__(config)
        embed_cfg = config.get("embedding", {})
        vs_cfg = config.get("vector_store", {})
        retrieval_cfg = config.get("retrieval", {})

        self.embed_model = SentenceTransformer(embed_cfg.get("model", "all-MiniLM-L6-v2"))
        self.normalize = embed_cfg.get("normalize", True)
        self.faiss_index, self.faiss_meta = load_faiss_index(vs_cfg.get("persist_dir", "./data/faiss_index"))

        cross_encoder_model = retrieval_cfg.get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        self.initial_k = retrieval_cfg.get("rerank_initial_k", 20)

        self.llm = LLMClient(config)
        self._prompt_template = _load_prompt("basic_qa.txt")
        self._rerank_latency_ms: float = 0.0

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self.embed_model.encode([query], normalize_embeddings=self.normalize, convert_to_numpy=True)
        return vec[0].astype(np.float32)

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        # Step 1: broad FAISS retrieval
        candidates = faiss_search(self.faiss_index, self.faiss_meta, self._embed_query(query), self.initial_k)

        # Step 2: cross-encoder re-ranking
        t0 = time.perf_counter()
        pairs = [(query, d["text"]) for d in candidates]
        ce_scores = self.cross_encoder.predict(pairs)
        self._rerank_latency_ms = (time.perf_counter() - t0) * 1000

        ranked = sorted(zip(ce_scores, candidates), key=lambda x: x[0], reverse=True)
        return [
            {**doc, "score": float(score), "metadata": {**doc["metadata"], "ce_score": float(score)}}
            for score, doc in ranked[:top_k]
        ]

    def generate(self, query: str, docs: list[dict]) -> tuple[str, int]:
        context = "\n\n".join(d["text"] for d in docs)
        prompt = self._prompt_template.format(context=context, question=query)
        return self.llm.complete(prompt)

    def run(self, query: str, top_k: int | None = None, **kwargs) -> RAGResult:
        if top_k is None:
            top_k = self.config.get("retrieval", {}).get("top_k", 5)
        t0 = time.perf_counter()
        docs = self.retrieve(query, top_k)
        answer, tokens = self.generate(query, docs)
        latency_ms = (time.perf_counter() - t0) * 1000
        return RAGResult(
            answer=answer,
            retrieved_docs=docs,
            latency_ms=latency_ms,
            token_count=tokens,
            metadata={
                "pattern": "RerankingRag",
                "rerank_latency_ms": self._rerank_latency_ms,
                "initial_k": self.initial_k,
            },
        )
