"""
P7: Self-Query RAG.
LLM extracts a structured filter from the question (title, chunk_id).
Filtered candidates are retrieved from FAISS via post-filtering on metadata.
Falls back to Basic RAG if the LLM fails to return parseable JSON.

Config: retrieval.self_query_candidate_k (default 50) — broad FAISS fetch before filter.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .base_retriever import BaseRAG, RAGResult
from .indexing import faiss_search, load_faiss_index
from .llm_client import LLMClient


def _load_prompt(name: str) -> str:
    return (Path("config/prompts") / name).read_text().strip()


def _parse_filter(response: str) -> dict | None:
    """Extract the first valid JSON object from the LLM response.

    Uses raw_decode() from the first '{' so multiple JSON objects or trailing
    text don't cause the greedy-regex problem.
    """
    idx = response.find("{")
    if idx == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(response, idx)
        if not isinstance(obj, dict) or "query" not in obj:
            return None
        return obj
    except json.JSONDecodeError:
        return None


def _apply_filters(docs: list[dict], filters: dict) -> list[dict]:
    """Post-filter retrieved docs by metadata criteria."""
    if not filters:
        return docs
    result = []
    for doc in docs:
        meta = doc["metadata"]
        match = True
        for field, value in filters.items():
            if field == "source":
                # Match against title (case-insensitive substring)
                if value.lower() not in meta.get("title", "").lower():
                    match = False
                    break
            elif field == "chunk_id":
                if value is None or meta.get("chunk_id") != int(value):
                    match = False
                    break
            # word_count / other fields: skip gracefully (not in index metadata)
        if match:
            result.append(doc)
    return result


class SelfQueryRag(BaseRAG):
    def __init__(self, config: dict):
        super().__init__(config)
        embed_cfg = config.get("embedding", {})
        vs_cfg = config.get("vector_store", {})
        retrieval_cfg = config.get("retrieval", {})

        self.embed_model = SentenceTransformer(embed_cfg.get("model", "all-MiniLM-L6-v2"))
        self.normalize = embed_cfg.get("normalize", True)
        self.faiss_index, self.faiss_meta = load_faiss_index(vs_cfg.get("persist_dir", "./data/faiss_index"))
        self.candidate_k = retrieval_cfg.get("self_query_candidate_k", 50)
        self.llm = LLMClient(config)
        self._parse_template = _load_prompt("selfquery_parse.txt")
        self._qa_template = _load_prompt("basic_qa.txt")

        # State stashed by retrieve() for run() metadata reporting
        self._last_parsed_query: str = ""
        self._last_filters: dict = {}
        self._last_used_fallback: bool = False
        self._last_parse_tokens: int = 0

    def _embed(self, text: str) -> np.ndarray:
        vec = self.embed_model.encode([text], normalize_embeddings=self.normalize, convert_to_numpy=True)
        return vec[0].astype(np.float32)

    def _extract_filter(self, query: str) -> tuple[str, dict, int]:
        """Ask LLM to parse query into (search_query, filters). Returns (query, filters, tokens)."""
        prompt = self._parse_template.format(question=query)
        response, tokens = self.llm.complete(prompt)
        parsed = _parse_filter(response)
        if parsed is None:
            return query, {}, tokens
        search_query = parsed.get("query", query)
        filters = parsed.get("filters", {})
        if not isinstance(filters, dict):
            filters = {}
        return search_query, filters, tokens

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        search_query, filters, parse_tokens = self._extract_filter(query)
        self._last_parsed_query = search_query
        self._last_filters = filters
        self._last_parse_tokens = parse_tokens
        self._last_used_fallback = False

        # Fetch broad candidates then post-filter
        candidates = faiss_search(self.faiss_index, self.faiss_meta, self._embed(search_query), self.candidate_k)
        filtered = _apply_filters(candidates, filters)

        if len(filtered) == 0:
            # Fallback: filter matched nothing → use unfiltered candidates (PRD spec)
            self._last_used_fallback = True
            filtered = candidates
        # If filter matched some docs (even < top_k), return those — don't override with unfiltered

        return filtered[:top_k]

    def generate(self, query: str, docs: list[dict]) -> tuple[str, int]:
        context = "\n\n".join(d["text"] for d in docs)
        prompt = self._qa_template.format(context=context, question=query)
        return self.llm.complete(prompt)

    def run(self, query: str, top_k: int | None = None, **kwargs) -> RAGResult:
        if top_k is None:
            top_k = self.config.get("retrieval", {}).get("top_k", 5)
        t0 = time.perf_counter()
        docs = self.retrieve(query, top_k)
        answer, gen_tokens = self.generate(query, docs)
        latency_ms = (time.perf_counter() - t0) * 1000
        return RAGResult(
            answer=answer,
            retrieved_docs=docs,
            latency_ms=latency_ms,
            token_count=self._last_parse_tokens + gen_tokens,
            metadata={
                "pattern": "SelfQueryRag",
                "parsed_query": self._last_parsed_query,
                "filters": self._last_filters,
                "used_fallback": self._last_used_fallback,
                "parse_tokens": self._last_parse_tokens,
                "gen_tokens": gen_tokens,
            },
        )
