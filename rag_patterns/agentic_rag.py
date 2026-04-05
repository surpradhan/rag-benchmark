"""
P9: Agentic RAG (ReAct loop).
The LLM reasons step-by-step and calls tools iteratively:
  - vector_search(query) — FAISS semantic search
  - bm25_search(query)   — BM25 lexical search
  - finish(answer)       — submit final answer

Max iterations: config.retrieval.agentic_max_iterations (default 5).
All retrieved passages are accumulated across iterations for context.
Token accounting covers every LLM call in the loop.
"""
from __future__ import annotations

import re
import time
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


# ---------------------------------------------------------------------------
# Tool-call parsing
# ---------------------------------------------------------------------------
_TOOL_RE = re.compile(
    r"(vector_search|bm25_search|finish)\s*\(\s*['\"]?(.*?)['\"]?\s*\)",
    re.IGNORECASE | re.DOTALL,
)


def _clean_arg(arg: str) -> str:
    """Strip named-parameter prefix (e.g. 'answer=') and surrounding quotes."""
    arg = re.sub(r'^\w+\s*=\s*', '', arg.strip())  # strip 'answer=' style prefix
    return arg.strip().strip("'\"")


def _parse_tool_call(text: str) -> tuple[str, str] | None:
    """Return (tool_name, argument) from the last tool call found in text, or None."""
    matches = _TOOL_RE.findall(text)
    if not matches:
        return None
    tool, arg = matches[-1]
    return tool.lower(), _clean_arg(arg)


def _format_passages(docs: list[dict]) -> str:
    return "\n\n".join(
        f"[{i+1}] (title: {d['metadata'].get('title','?')}) {d['text']}"
        for i, d in enumerate(docs)
    )


class AgenticRag(BaseRAG):
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
        self.max_iterations = retrieval_cfg.get("agentic_max_iterations", 5)
        self.tool_top_k = retrieval_cfg.get("agentic_tool_top_k", 3)
        self.llm = LLMClient(config)
        self._react_template = _load_prompt("agent_react.txt")

    def _embed(self, text: str) -> np.ndarray:
        vec = self.embed_model.encode([text], normalize_embeddings=self.normalize, convert_to_numpy=True)
        return vec[0].astype(np.float32)

    def _vector_search(self, query: str) -> list[dict]:
        return faiss_search(self.faiss_index, self.faiss_meta, self._embed(query), self.tool_top_k)

    def _bm25_search(self, query: str) -> list[dict]:
        return bm25_search(self.bm25, self.bm25_meta, query, self.tool_top_k)

    # retrieve() and generate() satisfy the BaseRAG interface but the real logic is in run()
    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """Single-pass vector search (used when run() is not called directly)."""
        return faiss_search(self.faiss_index, self.faiss_meta, self._embed(query), top_k)

    def generate(self, query: str, docs: list[dict]) -> tuple[str, int]:
        """Not used in normal agentic flow — here for interface compliance."""
        qa_template = (Path("config/prompts") / "basic_qa.txt").read_text().strip()
        context = "\n\n".join(d["text"] for d in docs)
        prompt = qa_template.format(context=context, question=query)
        return self.llm.complete(prompt)

    def run(self, query: str, top_k: int | None = None, **kwargs) -> RAGResult:
        t0 = time.perf_counter()
        total_tokens = 0
        all_docs: list[dict] = []          # accumulated across all tool calls
        seen_chunk_ids: set[int] = set()   # dedup
        final_answer = ""
        n_iterations = 0
        tool_call_log: list[dict] = []

        # Build conversation incrementally — append only new content each iteration
        # so context grows O(n), not O(n²)
        conversation = self._react_template.format(question=query)

        for iteration in range(self.max_iterations):
            n_iterations += 1

            llm_response, tokens = self.llm.complete(conversation)
            total_tokens += tokens

            # Append LLM turn to conversation for next iteration
            conversation += f"\n{llm_response}"

            tool_call = _parse_tool_call(llm_response)

            if tool_call is None:
                if not all_docs:
                    # LLM gave a free-text response before searching — force a search
                    conversation += "\nSystem: You must call vector_search() or bm25_search() before answering."
                    continue
                # LLM gave a free-text answer after searching — accept it
                final_answer = llm_response.strip()
                break

            tool_name, tool_arg = tool_call
            tool_call_log.append({"tool": tool_name, "arg": tool_arg, "iteration": iteration + 1})

            if tool_name == "finish":
                if not all_docs:
                    # LLM tried to answer without searching — inject a hard constraint
                    # and continue the loop to force at least one search call
                    conversation += "\nSystem: You must call vector_search() or bm25_search() before finish()."
                    tool_call_log.append({"tool": "finish_rejected", "arg": tool_arg, "iteration": iteration + 1})
                    continue
                final_answer = tool_arg
                break

            # Execute tool
            if tool_name == "vector_search":
                results = self._vector_search(tool_arg)
            elif tool_name == "bm25_search":
                results = self._bm25_search(tool_arg)
            else:
                results = []

            # Accumulate unique docs
            new_docs = []
            for doc in results:
                cid = doc["metadata"].get("chunk_id", id(doc))
                if cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    all_docs.append(doc)
                    new_docs.append(doc)

            # Append observation to conversation (linear growth)
            observation = f"\nObservation: {_format_passages(new_docs) if new_docs else 'No new results found.'}"
            conversation += observation

        # If the loop exhausted without finish(), generate from accumulated docs
        if not final_answer and all_docs:
            fallback_answer, fallback_tokens = self.generate(query, all_docs[:top_k or 5])
            final_answer = fallback_answer
            total_tokens += fallback_tokens
        elif not final_answer:
            final_answer = "Unable to find an answer."

        latency_ms = (time.perf_counter() - t0) * 1000

        return RAGResult(
            answer=final_answer,
            retrieved_docs=all_docs,
            latency_ms=latency_ms,
            token_count=total_tokens,
            metadata={
                "pattern": "AgenticRag",
                "n_iterations": n_iterations,
                "tool_calls": tool_call_log,
                "total_docs_retrieved": len(all_docs),
            },
        )
