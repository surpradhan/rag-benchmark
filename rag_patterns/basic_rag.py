"""
B1: Basic RAG — the primary comparison baseline for all other patterns.
FAISS vector search → top-K chunks → LLM generation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .base_retriever import BaseRAG, RAGResult
from .indexing import faiss_search, load_faiss_index
from .llm_client import LLMClient


def _load_prompt(name: str) -> str:
    return (Path("config/prompts") / name).read_text().strip()


class BasicRag(BaseRAG):
    def __init__(self, config: dict):
        super().__init__(config)
        embed_cfg = config.get("embedding", {})
        vs_cfg = config.get("vector_store", {})

        self.embed_model = SentenceTransformer(embed_cfg.get("model", "all-MiniLM-L6-v2"))
        self.normalize = embed_cfg.get("normalize", True)
        self.faiss_index, self.faiss_meta = load_faiss_index(vs_cfg.get("persist_dir", "./data/faiss_index"))
        self.llm = LLMClient(config)
        self._prompt_template = _load_prompt("basic_qa.txt")

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self.embed_model.encode([query], normalize_embeddings=self.normalize, convert_to_numpy=True)
        return vec[0].astype(np.float32)

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        q_vec = self._embed_query(query)
        return faiss_search(self.faiss_index, self.faiss_meta, q_vec, top_k)

    def generate(self, query: str, docs: list[dict]) -> tuple[str, int]:
        context = "\n\n".join(d["text"] for d in docs)
        prompt = self._prompt_template.format(context=context, question=query)
        return self.llm.complete(prompt)
