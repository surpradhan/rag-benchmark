"""
Chunking pipeline: raw corpus documents → token-aware chunks.
Uses LangChain's RecursiveCharacterTextSplitter with tiktoken token counting.
"""
from __future__ import annotations

import json
from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


def _token_len(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text))


def chunk_corpus(
    corpus: list[dict],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    separators: list[str] | None = None,
    tokenizer_name: str = "cl100k_base",
) -> list[dict]:
    """
    Split each corpus document into chunks.

    Returns list of chunks, each:
        {chunk_id, doc_id, title, text, token_count}
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    enc = tiktoken.get_encoding(tokenizer_name)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=lambda t: _token_len(t, enc),
    )

    chunks: list[dict] = []
    for doc in tqdm(corpus, desc="Chunking"):
        splits = splitter.split_text(doc["text"])
        for split in splits:
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "text": split,
                    "token_count": _token_len(split, enc),
                }
            )
    return chunks


def load_and_chunk(
    corpus_path: str | Path,
    chunks_path: str | Path,
    config: dict,
    force: bool = False,
) -> list[dict]:
    """Load corpus, chunk, save, and return chunks. Skips if already done."""
    chunks_path = Path(chunks_path)
    if chunks_path.exists() and not force:
        print(f"[skip] {chunks_path} already exists — loading cached chunks")
        with chunks_path.open() as f:
            return json.load(f)

    with open(corpus_path) as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus):,} documents")

    c = config.get("chunking", {})
    chunks = chunk_corpus(
        corpus,
        chunk_size=c.get("chunk_size", 512),
        chunk_overlap=c.get("chunk_overlap", 50),
        separators=c.get("separators"),
        tokenizer_name=c.get("tokenizer", "cl100k_base"),
    )

    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("w") as f:
        json.dump(chunks, f)
    print(f"Saved {len(chunks):,} chunks → {chunks_path}")
    return chunks
