"""
P4: Multi-Query RAG — LLM expands a query into N variants (config: retrieval.multiquery_n_variants),
retrieves for each, deduplicates by chunk_id, then generates from the merged context.

Token counting covers ALL LLM calls: N expansion calls + 1 generation call.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .base_retriever import BaseRAG, RAGResult
from .indexing import faiss_search, load_faiss_index
from .llm_client import LLMClient


def _load_prompt(name: str) -> str:
    return (Path("config/prompts") / name).read_text().strip()


class MultiqueryRag(BaseRAG):
    def __init__(self, config: dict):
        super().__init__(config)
        embed_cfg = config.get("embedding", {})
        vs_cfg = config.get("vector_store", {})
        retrieval_cfg = config.get("retrieval", {})

        self.embed_model = SentenceTransformer(embed_cfg.get("model", "all-MiniLM-L6-v2"))
        self.normalize = embed_cfg.get("normalize", True)
        self.faiss_index, self.faiss_meta = load_faiss_index(vs_cfg.get("persist_dir", "./data/faiss_index"))
        self.n_variants = retrieval_cfg.get("multiquery_n_variants", 3)
        self.llm = LLMClient(config)
        self._expand_template = _load_prompt("multiquery_expand.txt")
        self._qa_template = _load_prompt("basic_qa.txt")

    def _embed(self, text: str) -> np.ndarray:
        vec = self.embed_model.encode([text], normalize_embeddings=self.normalize, convert_to_numpy=True)
        return vec[0].astype(np.float32)

    def _expand_query(self, query: str) -> tuple[list[str], int]:
        """Return (query_variants_including_original, expansion_token_count)."""
        prompt = self._expand_template.format(question=query)
        response, tokens = self.llm.complete(prompt)
        variants = [line.strip() for line in response.strip().splitlines() if line.strip()]
        # Always include original; cap at n_variants additional queries
        queries = [query] + variants[: self.n_variants]
        return list(dict.fromkeys(queries)), tokens  # preserve order, deduplicate

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """Retrieve-only path (expansion tokens not tracked here; use run() for full accounting)."""
        queries, _ = self._expand_query(query)
        return self._retrieve_for_queries(queries, top_k)

    def _retrieve_for_queries(self, queries: list[str], top_k: int) -> list[dict]:
        seen: dict[int, dict] = {}  # chunk_id → best-scoring doc
        for q in queries:
            for doc in faiss_search(self.faiss_index, self.faiss_meta, self._embed(q), top_k):
                cid = doc["metadata"]["chunk_id"]
                if cid not in seen or doc["score"] > seen[cid]["score"]:
                    seen[cid] = doc
        ranked = sorted(seen.values(), key=lambda d: d["score"], reverse=True)
        return ranked[:top_k]

    def generate(self, query: str, docs: list[dict]) -> tuple[str, int]:
        context = "\n\n".join(d["text"] for d in docs)
        prompt = self._qa_template.format(context=context, question=query)
        return self.llm.complete(prompt)

    def run(self, query: str, top_k: int | None = None, **kwargs) -> RAGResult:
        """Full pipeline: accumulates tokens from ALL LLM calls (expansion + generation)."""
        if top_k is None:
            top_k = self.config.get("retrieval", {}).get("top_k", 5)
        t0 = time.perf_counter()
        queries, expand_tokens = self._expand_query(query)
        docs = self._retrieve_for_queries(queries, top_k)
        answer, gen_tokens = self.generate(query, docs)
        latency_ms = (time.perf_counter() - t0) * 1000
        return RAGResult(
            answer=answer,
            retrieved_docs=docs,
            latency_ms=latency_ms,
            token_count=expand_tokens + gen_tokens,  # all LLM calls accounted for
            metadata={
                "pattern": "MultiqueryRag",
                "n_queries": len(queries),
                "expanded_queries": queries,
                "expand_tokens": expand_tokens,
                "gen_tokens": gen_tokens,
            },
        )
