"""
B2: Oracle RAG — upper bound baseline.
Gold supporting documents are injected directly; no retrieval is performed.
Recall@K is 1.0 by construction. Shows the ceiling for the LLM + context combination.
"""
from __future__ import annotations

import json
from pathlib import Path

from .base_retriever import BaseRAG, RAGResult
from .llm_client import LLMClient


def _load_prompt(name: str) -> str:
    return (Path("config/prompts") / name).read_text().strip()


class OracleRag(BaseRAG):
    def __init__(self, config: dict):
        super().__init__(config)
        self.llm = LLMClient(config)
        self._prompt_template = _load_prompt("basic_qa.txt")

        # Build title → text lookup from corpus
        corpus_path = Path("data/processed/corpus.json")
        with corpus_path.open() as f:
            corpus = json.load(f)
        self._title_to_text: dict[str, str] = {doc["title"]: doc["text"] for doc in corpus}

    def retrieve(self, query: str, top_k: int, gold_titles: list[str] | None = None) -> list[dict]:
        """Return the gold supporting documents. gold_titles must be passed via run()."""
        if not gold_titles:
            return []
        docs = []
        for title in gold_titles:
            text = self._title_to_text.get(title, "")
            if text:
                docs.append({
                    "text": text,
                    "metadata": {"title": title, "source": "oracle"},
                    "score": 1.0,
                })
        return docs

    def generate(self, query: str, docs: list[dict]) -> tuple[str, int]:
        context = "\n\n".join(d["text"] for d in docs)
        prompt = self._prompt_template.format(context=context, question=query)
        return self.llm.complete(prompt)

    def run(self, query: str, top_k: int | None = None, gold_titles: list[str] | None = None) -> RAGResult:
        import time
        start = time.perf_counter()
        docs = self.retrieve(query, top_k or 0, gold_titles=gold_titles)
        answer, tokens = self.generate(query, docs)
        latency_ms = (time.perf_counter() - start) * 1000
        return RAGResult(
            answer=answer,
            retrieved_docs=docs,
            latency_ms=latency_ms,
            token_count=tokens,
            metadata={"pattern": "OracleRag", "gold_titles": gold_titles or []},
        )
