"""
P5: HyDE (Hypothetical Document Embeddings) RAG.
LLM generates a hypothetical answer paragraph → embed that → FAISS search → generate real answer.

Token counting covers BOTH LLM calls: hypothesis generation + answer generation.
retrieve() stores hypo state so run() can pick it up without a second LLM call.
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


class HydeRag(BaseRAG):
    def __init__(self, config: dict):
        super().__init__(config)
        embed_cfg = config.get("embedding", {})
        vs_cfg = config.get("vector_store", {})

        self.embed_model = SentenceTransformer(embed_cfg.get("model", "all-MiniLM-L6-v2"))
        self.normalize = embed_cfg.get("normalize", True)
        self.faiss_index, self.faiss_meta = load_faiss_index(vs_cfg.get("persist_dir", "./data/faiss_index"))
        self.llm = LLMClient(config)
        self._hyde_template = _load_prompt("hyde_generate.txt")
        self._qa_template = _load_prompt("basic_qa.txt")
        # State stashed by retrieve() so run() can add hypo tokens without a second LLM call
        self._last_hypo_doc: str = ""
        self._last_hypo_tokens: int = 0

    def _embed(self, text: str) -> np.ndarray:
        vec = self.embed_model.encode([text], normalize_embeddings=self.normalize, convert_to_numpy=True)
        return vec[0].astype(np.float32)

    def _generate_hypothetical_doc(self, query: str) -> tuple[str, int]:
        prompt = self._hyde_template.format(question=query)
        return self.llm.complete(prompt)

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """Generate hypothetical doc, embed it, search FAISS. Stashes hypo tokens for run()."""
        hypo_doc, hypo_tokens = self._generate_hypothetical_doc(query)
        self._last_hypo_doc = hypo_doc
        self._last_hypo_tokens = hypo_tokens
        return faiss_search(self.faiss_index, self.faiss_meta, self._embed(hypo_doc), top_k)

    def generate(self, query: str, docs: list[dict]) -> tuple[str, int]:
        context = "\n\n".join(d["text"] for d in docs)
        prompt = self._qa_template.format(context=context, question=query)
        return self.llm.complete(prompt)

    def run(self, query: str, top_k: int | None = None, **kwargs) -> RAGResult:
        """Full pipeline via retrieve() → generate(). Counts tokens from both LLM calls."""
        if top_k is None:
            top_k = self.config.get("retrieval", {}).get("top_k", 5)
        t0 = time.perf_counter()
        docs = self.retrieve(query, top_k)              # sets _last_hypo_tokens/_last_hypo_doc
        answer, answer_tokens = self.generate(query, docs)
        latency_ms = (time.perf_counter() - t0) * 1000
        return RAGResult(
            answer=answer,
            retrieved_docs=docs,
            latency_ms=latency_ms,
            token_count=self._last_hypo_tokens + answer_tokens,
            metadata={
                "pattern": "HydeRag",
                "hypothetical_doc": self._last_hypo_doc,
                "hypo_tokens": self._last_hypo_tokens,
                "gen_tokens": answer_tokens,
            },
        )
