"""
P8: Corrective RAG (CRAG).
Step 1 — Retrieve top_k chunks with FAISS.
Step 2 — Rate each chunk as RELEVANT / PARTIAL / IRRELEVANT using an LLM judge.
Step 3 — If fewer than 50% are RELEVANT, reformulate the query via LLM and re-retrieve.
          Up to 2 re-retrieval attempts (config: retrieval.crag_max_retries).
Step 4 — Generate from the best chunks collected across all attempts.

Token accounting: initial retrieval judge calls + (optional) reformulation + generation.
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


_RELEVANCE_ORDER = {"RELEVANT": 2, "PARTIAL": 1, "IRRELEVANT": 0}


def _parse_rating(response: str) -> str:
    """Extract RELEVANT / PARTIAL / IRRELEVANT from LLM response."""
    upper = response.strip().upper()
    for label in ("RELEVANT", "PARTIAL", "IRRELEVANT"):
        if label in upper:
            return label
    return "IRRELEVANT"


class CorrectiveRag(BaseRAG):
    def __init__(self, config: dict):
        super().__init__(config)
        embed_cfg = config.get("embedding", {})
        vs_cfg = config.get("vector_store", {})
        retrieval_cfg = config.get("retrieval", {})

        self.embed_model = SentenceTransformer(embed_cfg.get("model", "all-MiniLM-L6-v2"))
        self.normalize = embed_cfg.get("normalize", True)
        self.faiss_index, self.faiss_meta = load_faiss_index(vs_cfg.get("persist_dir", "./data/faiss_index"))
        self.max_retries = retrieval_cfg.get("crag_max_retries", 2)
        self.relevance_threshold = retrieval_cfg.get("crag_relevance_threshold", 0.5)
        self.llm = LLMClient(config)
        self._eval_template = _load_prompt("corrective_eval.txt")
        self._qa_template = _load_prompt("basic_qa.txt")

        # State for run() metadata reporting
        self._last_ratings: list[str] = []
        self._last_n_retries: int = 0
        self._last_judge_tokens: int = 0
        self._last_reformulate_tokens: int = 0

    def _embed(self, text: str) -> np.ndarray:
        vec = self.embed_model.encode([text], normalize_embeddings=self.normalize, convert_to_numpy=True)
        return vec[0].astype(np.float32)

    def _rate_docs(self, question: str, docs: list[dict]) -> tuple[list[str], int]:
        """Rate each doc. Returns (ratings_list, total_judge_tokens)."""
        ratings, total_tokens = [], 0
        for doc in docs:
            prompt = self._eval_template.format(question=question, document=doc["text"])
            response, tokens = self.llm.complete(prompt)
            ratings.append(_parse_rating(response))
            total_tokens += tokens
        return ratings, total_tokens

    def _reformulate_query(self, original_query: str, attempt: int) -> tuple[str, int]:
        """Ask LLM to rephrase the query for better retrieval on retry."""
        prompt = (
            f"The following question did not return useful search results. "
            f"Rephrase it to improve document retrieval. Return only the rephrased question, no explanation.\n\n"
            f"Original question: {original_query}"
        )
        response, tokens = self.llm.complete(prompt)
        rephrased = response.strip().strip('"').strip("'")
        return rephrased if rephrased else original_query, tokens

    def _relevant_fraction(self, ratings: list[str]) -> float:
        if not ratings:
            return 0.0
        return sum(1 for r in ratings if r == "RELEVANT") / len(ratings)

    def _update_pool(
        self,
        pool: dict[int, tuple[dict, str]],
        docs: list[dict],
        ratings: list[str],
    ) -> None:
        """Merge docs+ratings into pool, keeping best rating (then score) per chunk."""
        for doc, rating in zip(docs, ratings):
            cid = doc["metadata"]["chunk_id"]
            existing = pool.get(cid)
            if existing is None:
                pool[cid] = (doc, rating)
            else:
                existing_rating_score = _RELEVANCE_ORDER.get(existing[1], 0)
                new_rating_score = _RELEVANCE_ORDER.get(rating, 0)
                if new_rating_score > existing_rating_score or (
                    new_rating_score == existing_rating_score and doc["score"] > existing[0]["score"]
                ):
                    pool[cid] = (doc, rating)

    def _corrective_retrieve(
        self, query: str, top_k: int
    ) -> tuple[list[dict], list[str], int, int, int]:
        """Run the corrective retrieval loop (without generation).

        Returns:
            (final_docs, final_ratings, judge_tokens, reformulate_tokens, n_retries)
        """
        total_judge_tokens = 0
        total_reformulate_tokens = 0
        n_retries = 0

        # pool: chunk_id → (doc, rating) — accumulates best docs across ALL attempts
        pool: dict[int, tuple[dict, str]] = {}

        # Initial retrieval
        docs = faiss_search(self.faiss_index, self.faiss_meta, self._embed(query), top_k)
        ratings, judge_tokens = self._rate_docs(query, docs)
        total_judge_tokens += judge_tokens
        self._update_pool(pool, docs, ratings)

        current_query = query
        for attempt in range(self.max_retries):
            pool_ratings = [r for _, r in pool.values()]
            if self._relevant_fraction(pool_ratings) >= self.relevance_threshold:
                break
            # Not enough relevant docs — reformulate from the ORIGINAL query each time
            # (reformulating from current_query causes intent drift across retries)
            n_retries += 1
            new_query, ref_tokens = self._reformulate_query(query, attempt + 1)
            total_reformulate_tokens += ref_tokens
            current_query = new_query
            new_docs = faiss_search(self.faiss_index, self.faiss_meta, self._embed(current_query), top_k)
            new_ratings, judge_tokens = self._rate_docs(query, new_docs)
            total_judge_tokens += judge_tokens
            self._update_pool(pool, new_docs, new_ratings)

        # If retries occurred, sort pool by (rating, score) to surface best docs
        # across multiple retrieval attempts. If no retries, preserve original
        # FAISS cosine order — judge re-ranking without reformulation degrades
        # recall on distractor-heavy datasets (promotes related distractors over
        # gold docs rated PARTIAL).
        if n_retries > 0:
            sorted_pool = sorted(
                pool.values(),
                key=lambda x: (_RELEVANCE_ORDER.get(x[1], 0), x[0]["score"]),
                reverse=True,
            )
        else:
            sorted_pool = list(pool.values())  # already in FAISS cosine order

        final_docs = [d for d, _ in sorted_pool[:top_k]]
        final_ratings = [r for _, r in sorted_pool[:top_k]]
        return final_docs, final_ratings, total_judge_tokens, total_reformulate_tokens, n_retries

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        final_docs, final_ratings, judge_tokens, reformulate_tokens, n_retries = (
            self._corrective_retrieve(query, top_k)
        )
        self._last_ratings = final_ratings
        self._last_n_retries = n_retries
        self._last_judge_tokens = judge_tokens
        self._last_reformulate_tokens = reformulate_tokens
        return final_docs

    def generate(self, query: str, docs: list[dict]) -> tuple[str, int]:
        context = "\n\n".join(d["text"] for d in docs)
        prompt = self._qa_template.format(context=context, question=query)
        return self.llm.complete(prompt)

    def run(self, query: str, top_k: int | None = None, **kwargs) -> RAGResult:
        if top_k is None:
            top_k = self.config.get("retrieval", {}).get("top_k", 5)

        t0 = time.perf_counter()
        final_docs, final_ratings, total_judge_tokens, total_reformulate_tokens, n_retries = (
            self._corrective_retrieve(query, top_k)
        )

        answer, gen_tokens = self.generate(query, final_docs)
        latency_ms = (time.perf_counter() - t0) * 1000

        self._last_ratings = final_ratings
        self._last_n_retries = n_retries
        self._last_judge_tokens = total_judge_tokens
        self._last_reformulate_tokens = total_reformulate_tokens

        return RAGResult(
            answer=answer,
            retrieved_docs=final_docs,
            latency_ms=latency_ms,
            token_count=total_judge_tokens + total_reformulate_tokens + gen_tokens,
            metadata={
                "pattern": "CorrectiveRag",
                "ratings": final_ratings,
                "n_retries": n_retries,
                "relevant_fraction": self._relevant_fraction(final_ratings),
                "judge_tokens": total_judge_tokens,
                "reformulate_tokens": total_reformulate_tokens,
                "gen_tokens": gen_tokens,
            },
        )
