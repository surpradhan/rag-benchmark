"""
verify_setup.py — pre-flight checklist for the RAG benchmark.
Run this before starting any evaluation. All checks must pass (exit 0).

Usage:
    python scripts/verify_setup.py
"""
from __future__ import annotations

import importlib
import json
import pickle
import sys
from pathlib import Path

import yaml

# Ensure project root is on the path so rag_patterns is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"


def check(label: str, ok: bool, detail: str = "") -> bool:
    icon = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {icon}  {label}{suffix}")
    return ok


def section(title: str) -> None:
    print(f"\n{'─'*50}\n  {title}\n{'─'*50}")


def main() -> int:
    failures = 0
    root = Path(__file__).parent.parent

    # ------------------------------------------------------------------
    section("Python environment")
    # ------------------------------------------------------------------
    import platform
    py_ver = platform.python_version()
    failures += not check("Python 3.11+", py_ver.startswith("3.11"), py_ver)

    required_packages = [
        "faiss", "sentence_transformers", "rank_bm25",
        "langchain_text_splitters", "ragas", "tiktoken",
        "numpy", "pandas", "seaborn", "matplotlib", "tqdm", "yaml",
    ]
    for pkg in required_packages:
        mod_name = pkg.replace("-", "_")
        ok = importlib.util.find_spec(mod_name) is not None
        failures += not check(f"Package: {pkg}", ok)

    # ------------------------------------------------------------------
    section("Config")
    # ------------------------------------------------------------------
    config_path = root / "config" / "config.yaml"
    ok = config_path.exists()
    failures += not check("config/config.yaml exists", ok)
    if ok:
        with config_path.open() as f:
            config = yaml.safe_load(f)
        required_keys = ["dataset", "chunking", "embedding", "vector_store", "llm", "retrieval", "evaluation"]
        for k in required_keys:
            failures += not check(f"  config key: {k}", k in config)

    # ------------------------------------------------------------------
    section("Prompt templates")
    # ------------------------------------------------------------------
    prompts = [
        "system.txt", "basic_qa.txt", "multiquery_expand.txt",
        "hyde_generate.txt", "selfquery_parse.txt",
        "corrective_eval.txt", "agent_react.txt",
    ]
    for p in prompts:
        path = root / "config" / "prompts" / p
        failures += not check(f"Prompt: {p}", path.exists())

    # ------------------------------------------------------------------
    section("Dataset")
    # ------------------------------------------------------------------
    corpus_path = root / "data" / "processed" / "corpus.json"
    test_path = root / "data" / "processed" / "test_questions.json"
    dev_path = root / "data" / "processed" / "dev_questions.json"
    chunks_path = root / "data" / "processed" / "chunks.json"

    ok = corpus_path.exists()
    failures += not check("corpus.json exists", ok)
    if ok:
        with corpus_path.open() as f:
            corpus = json.load(f)
        failures += not check(f"  corpus size > 1000", len(corpus) > 1000, f"{len(corpus):,} docs")

    ok = test_path.exists()
    failures += not check("test_questions.json exists", ok)
    if ok:
        with test_path.open() as f:
            test_q = json.load(f)
        expected = config["dataset"]["test_size"]
        failures += not check(f"  test set size == {expected}", len(test_q) == expected, str(len(test_q)))

    ok = dev_path.exists()
    failures += not check("dev_questions.json exists", ok)

    ok = chunks_path.exists()
    failures += not check("chunks.json exists", ok)
    if ok:
        with chunks_path.open() as f:
            chunks = json.load(f)
        failures += not check(f"  chunks size > 1000", len(chunks) > 1000, f"{len(chunks):,} chunks")

    # ------------------------------------------------------------------
    section("Indexes")
    # ------------------------------------------------------------------
    faiss_dir = root / config["vector_store"]["persist_dir"].lstrip("./")
    faiss_index = faiss_dir / "index.faiss"
    faiss_meta = faiss_dir / "metadata.pkl"
    failures += not check("FAISS index.faiss exists", faiss_index.exists())
    failures += not check("FAISS metadata.pkl exists", faiss_meta.exists())
    if faiss_index.exists():
        import faiss
        idx = faiss.read_index(str(faiss_index))
        failures += not check(f"  FAISS vectors > 1000", idx.ntotal > 1000, f"{idx.ntotal:,} vectors")

    bm25_path = root / config["bm25"]["persist_path"].lstrip("./")
    failures += not check("BM25 index exists", bm25_path.exists())

    # ------------------------------------------------------------------
    section("Interfaces")
    # ------------------------------------------------------------------
    try:
        from rag_patterns.base_retriever import BaseRAG, RAGResult
        failures += not check("BaseRAG importable", True)
        failures += not check("RAGResult importable", True)
    except ImportError as e:
        failures += 1
        check("BaseRAG/RAGResult importable", False, str(e))

    try:
        from rag_patterns.zero_retrieval import ZeroRetrieval
        failures += not check("ZeroRetrieval importable", True)
        zr = ZeroRetrieval.__new__(ZeroRetrieval)
        failures += not check("ZeroRetrieval is BaseRAG subclass", issubclass(ZeroRetrieval, BaseRAG))
    except ImportError as e:
        failures += 1
        check("ZeroRetrieval importable", False, str(e))

    # ------------------------------------------------------------------
    section("Results directory")
    # ------------------------------------------------------------------
    results_dir = root / "results" / "raw"
    results_dir.mkdir(parents=True, exist_ok=True)
    failures += not check("results/raw/ writable", True)

    # ------------------------------------------------------------------
    section("LLM connectivity (Ollama)")
    # ------------------------------------------------------------------
    try:
        import ollama as _ollama
        models = _ollama.list()
        model_names = [m.model for m in models.models] if hasattr(models, "models") else []
        target = config["llm"]["model"]
        found = any(target in m for m in model_names)
        if not found:
            print(f"  {WARN}  Ollama model '{target}' not found (available: {model_names})")
            print(f"       Run: ollama pull {target}")
        else:
            check(f"Ollama model '{target}' available", True)
    except Exception as e:
        print(f"  {WARN}  Ollama not reachable ({e})")
        print("       Start Ollama or set provider=together in config.yaml")

    # ------------------------------------------------------------------
    print(f"\n{'═'*50}")
    if failures == 0:
        print(f"  {PASS}  All checks passed. Ready to benchmark!")
    else:
        print(f"  {FAIL}  {failures} check(s) failed. Fix above issues before running.")
    print(f"{'═'*50}\n")
    return failures


if __name__ == "__main__":
    sys.exit(main())
