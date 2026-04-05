"""
B0: Zero-retrieval baseline.
LLM answers from parametric knowledge only — no documents retrieved.
Serves as the lower bound; shows whether retrieval helps at all.
"""
from __future__ import annotations

from .base_retriever import BaseRAG, RAGResult
from .llm_client import LLMClient


_ZERO_RETRIEVAL_PROMPT = (
    "Answer the following question to the best of your knowledge. "
    "Be concise.\n\nQuestion: {question}\n\nAnswer:"
)


class ZeroRetrieval(BaseRAG):
    def __init__(self, config: dict):
        super().__init__(config)
        self.llm = LLMClient(config)

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        return []  # no retrieval

    def generate(self, query: str, docs: list[dict]) -> tuple[str, int]:
        prompt = _ZERO_RETRIEVAL_PROMPT.format(question=query)
        return self.llm.complete(prompt)

    def run(self, query: str, top_k: int | None = None, **kwargs) -> RAGResult:
        import time
        start = time.perf_counter()
        answer, tokens = self.generate(query, [])
        latency_ms = (time.perf_counter() - start) * 1000
        return RAGResult(
            answer=answer,
            retrieved_docs=[],
            latency_ms=latency_ms,
            token_count=tokens,
            metadata={"pattern": "ZeroRetrieval"},
        )
