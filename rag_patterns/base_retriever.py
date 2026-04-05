"""
BaseRAG abstract interface and RAGResult dataclass.
Every RAG pattern must implement this interface — the evaluation pipeline depends on it.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RAGResult:
    answer: str
    retrieved_docs: list[dict]   # [{text, metadata, score}]
    latency_ms: float
    token_count: int             # total tokens used (input + output)
    metadata: dict = field(default_factory=dict)  # pattern-specific data


class BaseRAG(ABC):
    """Abstract base class for all RAG patterns."""

    def __init__(self, config: dict):
        self.config = config
        self._token_count = 0

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """Return top_k documents as [{text, metadata, score}]."""

    @abstractmethod
    def generate(self, query: str, docs: list[dict]) -> tuple[str, int]:
        """Return (answer_str, tokens_used)."""

    def run(self, query: str, top_k: int | None = None, **kwargs) -> RAGResult:
        """Execute retrieve → generate and return a RAGResult."""
        if top_k is None:
            top_k = self.config.get("retrieval", {}).get("top_k", 5)

        start = time.perf_counter()
        docs = self.retrieve(query, top_k)
        answer, tokens = self.generate(query, docs)
        latency_ms = (time.perf_counter() - start) * 1000

        return RAGResult(
            answer=answer,
            retrieved_docs=docs,
            latency_ms=latency_ms,
            token_count=tokens,
            metadata={"pattern": self.__class__.__name__},
        )

    def retrieve_and_generate(self, query: str) -> RAGResult:
        """Convenience alias — calls run() with default top_k."""
        return self.run(query)
