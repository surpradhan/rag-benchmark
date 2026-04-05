"""
P6: Parent-Child RAG.
Child chunks (256 tok / 25 overlap) are indexed for retrieval precision.
Each child stores a parent_id pointing to larger parent chunks (1024 tok / 100 overlap).
The LLM receives full parent context, giving broader context without sacrificing recall.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from .base_retriever import BaseRAG, RAGResult
from .indexing import build_faiss_index, faiss_search, load_faiss_index
from .llm_client import LLMClient


def _load_prompt(name: str) -> str:
    return (Path("config/prompts") / name).read_text().strip()


def _token_len(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text))


def build_parent_child_chunks(
    corpus: list[dict],
    child_size: int = 256,
    child_overlap: int = 25,
    parent_size: int = 1024,
    parent_overlap: int = 100,
    separators: list[str] | None = None,
    tokenizer_name: str = "cl100k_base",
) -> tuple[list[dict], list[dict]]:
    """Build parent and child chunk lists with child→parent linkage."""
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]
    enc = tiktoken.get_encoding(tokenizer_name)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=parent_overlap,
        separators=separators,
        length_function=lambda t: _token_len(t, enc),
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
        separators=separators,
        length_function=lambda t: _token_len(t, enc),
    )

    parents: list[dict] = []
    children: list[dict] = []

    for doc in corpus:
        parent_splits = parent_splitter.split_text(doc["text"])
        for p_text in parent_splits:
            parent_id = len(parents)
            parents.append({
                "parent_id": parent_id,
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "text": p_text,
                "token_count": _token_len(p_text, enc),
            })
            child_splits = child_splitter.split_text(p_text)
            for c_text in child_splits:
                children.append({
                    "chunk_id": len(children),
                    "parent_id": parent_id,
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "text": c_text,
                    "token_count": _token_len(c_text, enc),
                })

    return parents, children


def build_parent_child_index(
    corpus_path: str | Path,
    persist_dir: str | Path,
    config: dict,
    force: bool = False,
) -> tuple:
    """Build (or load cached) parent-child FAISS index + parent lookup."""
    persist_dir = Path(persist_dir)
    child_index_dir = persist_dir / "child_index"
    parents_path = persist_dir / "parents.pkl"

    if child_index_dir.exists() and parents_path.exists() and not force:
        print("[skip] Parent-child index already exists — loading")
        faiss_index, child_meta = load_faiss_index(child_index_dir)
        with parents_path.open("rb") as f:
            parents = pickle.load(f)
        return faiss_index, child_meta, parents

    with open(corpus_path) as f:
        corpus = json.load(f)

    pc_cfg = config.get("parent_child", {})
    chunk_cfg = config.get("chunking", {})
    embed_cfg = config.get("embedding", {})

    parents, children = build_parent_child_chunks(
        corpus,
        child_size=pc_cfg.get("child_size", 256),
        child_overlap=pc_cfg.get("child_overlap", 25),
        parent_size=pc_cfg.get("parent_size", 1024),
        parent_overlap=pc_cfg.get("parent_overlap", 100),
        separators=chunk_cfg.get("separators"),
        tokenizer_name=chunk_cfg.get("tokenizer", "cl100k_base"),
    )
    print(f"Parent-child: {len(parents):,} parents, {len(children):,} children")

    faiss_index, child_meta = build_faiss_index(
        children,
        persist_dir=child_index_dir,
        model_name=embed_cfg.get("model", "all-MiniLM-L6-v2"),
        dimension=embed_cfg.get("dimension", 384),
        force=force,
    )

    persist_dir.mkdir(parents=True, exist_ok=True)
    with parents_path.open("wb") as f:
        pickle.dump(parents, f)

    print(f"Parent-child index built → {persist_dir}")
    return faiss_index, child_meta, parents


class ParentChildRag(BaseRAG):
    def __init__(self, config: dict):
        super().__init__(config)
        embed_cfg = config.get("embedding", {})
        pc_cfg = config.get("parent_child", {})
        dataset_cfg = config.get("dataset", {})

        # Both paths come from config — no hardcoding
        corpus_path = dataset_cfg.get("corpus_path", "data/processed/corpus.json")
        pc_persist = Path(pc_cfg.get("persist_dir", "./data/parent_child_index"))

        self.embed_model = SentenceTransformer(embed_cfg.get("model", "all-MiniLM-L6-v2"))
        self.normalize = embed_cfg.get("normalize", True)

        self.child_index, self.child_meta, parents_list = build_parent_child_index(
            corpus_path=corpus_path,
            persist_dir=pc_persist,
            config=config,
        )
        # parent_id → parent dict for O(1) lookup
        self._parents: dict[int, dict] = {p["parent_id"]: p for p in parents_list}

        self.llm = LLMClient(config)
        self._qa_template = _load_prompt("basic_qa.txt")

    def _embed(self, text: str) -> np.ndarray:
        vec = self.embed_model.encode([text], normalize_embeddings=self.normalize, convert_to_numpy=True)
        return vec[0].astype(np.float32)

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """Retrieve child chunks, expand to unique parents (keeping max child score), return top_k parents."""
        child_hits = faiss_search(self.child_index, self.child_meta, self._embed(query), top_k * 2)

        seen_parents: dict[int, dict] = {}
        for child in child_hits:
            pid = child["metadata"]["parent_id"]
            if pid not in seen_parents:
                parent = self._parents[pid]
                seen_parents[pid] = {
                    "text": parent["text"],
                    "metadata": {
                        "parent_id": pid,
                        "doc_id": parent["doc_id"],
                        "title": parent["title"],
                        "best_child_score": child["score"],
                    },
                    "score": child["score"],
                }
            else:
                # Keep the highest child score so parent ranking is correct
                if child["score"] > seen_parents[pid]["score"]:
                    seen_parents[pid]["score"] = child["score"]
                    seen_parents[pid]["metadata"]["best_child_score"] = child["score"]

        ranked = sorted(seen_parents.values(), key=lambda d: d["score"], reverse=True)
        return ranked[:top_k]

    def generate(self, query: str, docs: list[dict]) -> tuple[str, int]:
        context = "\n\n".join(d["text"] for d in docs)
        prompt = self._qa_template.format(context=context, question=query)
        return self.llm.complete(prompt)
