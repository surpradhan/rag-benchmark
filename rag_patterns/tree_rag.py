"""
P11: Tree RAG (Vectorless RAG) — PageIndex-inspired, adapted for multi-document QA.

Instead of vector similarity, retrieval uses a two-stage LLM-guided pipeline:
  Stage 1 — BM25 shortlist: fast lexical search narrows the corpus to N candidate articles
  Stage 2 — LLM article selection: LLM picks the M most relevant articles by title + summary
  Stage 3 — LLM node navigation: for each selected article, LLM picks relevant paragraph nodes

No embeddings at query time. The tree index is pre-built at ingestion time and stored as JSON.
All LLM calls are token-accounted separately (article_selection, node_navigation, generation).

Config keys: tree_rag.*

Design note — HotpotQA adaptation:
  The canonical PageIndex approach targets single long structured documents (SEC filings, manuals).
  HotpotQA requires multi-hop retrieval across a ~5,000-article Wikipedia corpus. This
  implementation wraps each article in its own shallow two-level tree (article root → paragraphs)
  and uses BM25 to pre-filter before LLM tree traversal, making the pattern tractable at corpus
  scale while preserving the core vectorless-reasoning principle.

Reference: VectifyAI/PageIndex (https://github.com/VectifyAI/PageIndex)
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .base_retriever import BaseRAG, RAGResult
from .indexing import bm25_search, load_bm25_index
from .llm_client import LLMClient


def _load_prompt(name: str) -> str:
    return (Path("config/prompts") / name).read_text().strip()


# ---------------------------------------------------------------------------
# Tree index building and persistence
# ---------------------------------------------------------------------------

def build_article_tree(article: dict) -> dict:
    """
    Convert a HotpotQA corpus article into a two-level JSON tree.

    HotpotQA article schema:
        {"title": str, "paragraphs": [[sent0, sent1, ...], ...]}

    Output tree schema:
        {
            "doc_id":  str,        # article title (unique key in HotpotQA)
            "title":   str,
            "summary": str,        # first sentence of first paragraph (extractive)
            "nodes": [
                {
                    "node_id": str,    # "<title>_<para_idx>"
                    "title":   str,    # "Paragraph N"
                    "text":    str,    # full paragraph text (all sentences joined)
                    "summary": str,    # first sentence, truncated to 150 chars
                },
                ...
            ]
        }

    Summaries are extractive (first sentence) to avoid LLM cost at index-build time.
    The LLM only runs at retrieval time when navigating the tree.
    """
    title = article.get("title", "Untitled")
    paragraphs = article.get("paragraphs", [])

    nodes = []
    for i, para in enumerate(paragraphs):
        sentences: list[str] = para if isinstance(para, list) else [str(para)]
        text = " ".join(sentences).strip()
        if not text:
            continue

        first_sent = sentences[0] if sentences else text
        summary = first_sent[:150].rstrip()
        if len(first_sent) > 150:
            summary += "..."

        nodes.append({
            "node_id": f"{title}_{i}",
            "title":   f"Paragraph {i + 1}",
            "text":    text,
            "summary": summary,
        })

    article_summary = nodes[0]["summary"] if nodes else title

    return {
        "doc_id":  title,
        "title":   title,
        "summary": article_summary,
        "nodes":   nodes,
    }


def build_tree_index(corpus_path: str, persist_dir: str, force: bool = False) -> dict[str, dict]:
    """
    Build and persist a tree index for every article in the corpus JSON.

    Handles two corpus formats:
      1. Raw HotpotQA:  [{"title": str, "paragraphs": [[sent, ...], ...]}, ...]
      2. Processed flat: [{"title": str, "text": str}, ...]   ← one entry per paragraph

    Format 2 is what prepare_dataset.py produces: each corpus entry is a single
    joined paragraph (one entry per context paragraph in the HotpotQA distractor set).
    Multiple entries share the same title. We group them by title to reconstruct the
    multi-paragraph article structure before building trees.

    Index is persisted as a single `tree_index.json` under persist_dir.
    Skip if already exists unless force=True.
    """
    out_path = Path(persist_dir) / "tree_index.json"
    if out_path.exists() and not force:
        print(f"Tree index already exists at {out_path}. Use --force to rebuild.")
        return load_tree_index(persist_dir)

    print(f"Building tree index from {corpus_path} ...")
    with open(corpus_path) as f:
        corpus: list[dict] = json.load(f)

    # Detect format by checking first entry for "paragraphs" vs "text"
    first = corpus[0] if corpus else {}
    if "paragraphs" in first:
        # Format 1: each entry is a full article with a paragraphs list-of-lists
        articles = corpus
    else:
        # Format 2: each entry is one paragraph — group by title to reconstruct articles.
        # Preserve insertion order (= paragraph order within each article).
        from collections import OrderedDict
        grouped: dict[str, list[str]] = OrderedDict()
        for entry in corpus:
            title = entry.get("title", "Untitled")
            text  = entry.get("text", "").strip()
            if text:
                grouped.setdefault(title, []).append(text)
        # Convert to the paragraphs list-of-lists format expected by build_article_tree
        articles = [
            {"title": title, "paragraphs": [[para] for para in paras]}
            for title, paras in grouped.items()
        ]
        print(f"  Grouped {len(corpus):,} paragraph entries → {len(articles):,} articles")

    trees: dict[str, dict] = {}
    for article in articles:
        tree = build_article_tree(article)
        trees[tree["doc_id"]] = tree

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(trees, f)

    print(f"Tree index built: {len(trees):,} articles → {out_path}")
    return trees


def load_tree_index(persist_dir: str) -> dict[str, dict]:
    """Load pre-built tree index from disk. Returns {doc_id: tree_dict}."""
    index_file = Path(persist_dir) / "tree_index.json"
    if index_file.exists():
        with open(index_file) as f:
            return json.load(f)

    # Fallback: load individual JSON files (legacy format)
    trees: dict[str, dict] = {}
    for fp in Path(persist_dir).glob("*.json"):
        with open(fp) as f:
            tree = json.load(f)
        trees[tree["doc_id"]] = tree
    return trees


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

def _parse_id_list(text: str, valid_ids: list[str]) -> list[str]:
    """
    Extract an ordered list of IDs from LLM output.
    Accepts JSON arrays, comma-separated values, or one-per-line.
    Falls back to returning all valid_ids if nothing parseable is found.
    """
    valid_set = set(valid_ids)

    # Try JSON array first — most reliable when the prompt asks for it
    m = re.search(r'\[([^\]]*)\]', text, re.DOTALL)
    if m:
        try:
            raw = json.loads(f"[{m.group(1)}]")
            matched = [str(v).strip().strip("'\" ") for v in raw]
            matched = [v for v in matched if v in valid_set]
            if matched:
                return matched
        except (json.JSONDecodeError, TypeError):
            pass

    # Try quoted tokens
    quoted = re.findall(r'["\']([^"\']+)["\']', text)
    matched = [q for q in quoted if q in valid_set]
    if matched:
        return matched

    # Last resort: any token that matches a valid ID
    tokens = re.split(r'[\s,;]+', text)
    matched = [t.strip() for t in tokens if t.strip() in valid_set]
    if matched:
        return matched

    return valid_ids  # fallback: all candidates


# ---------------------------------------------------------------------------
# TreeRag
# ---------------------------------------------------------------------------

class TreeRag(BaseRAG):
    """
    P11: Vectorless Tree RAG.

    retrieve() pipeline:
        1. BM25 shortlist → top bm25_candidate_k articles (no LLM)
        2. LLM article selection → top article_select_k articles by title + summary
        3. LLM node navigation → top max_nodes_per_article paragraphs per article
        4. Merge, rank by traversal order, return top_k docs

    run() additionally accounts for all LLM tokens (article selection +
    per-article navigation + generation) and logs a per-stage token breakdown
    in the RAGResult metadata for cost analysis.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        bm25_cfg  = config.get("bm25", {})
        tree_cfg  = config.get("tree_rag", {})

        self.bm25, self.bm25_meta = load_bm25_index(
            bm25_cfg.get("persist_path", "./data/bm25_index/bm25_index.pkl")
        )
        self.trees: dict[str, dict] = load_tree_index(
            tree_cfg.get("persist_dir", "./data/tree_index")
        )

        self.bm25_candidate_k: int      = tree_cfg.get("bm25_candidate_k", 15)
        self.article_select_k: int      = tree_cfg.get("article_select_k", 5)
        self.max_nodes_per_article: int = tree_cfg.get("max_nodes_per_article", 3)

        self.llm = LLMClient(config)
        self._article_select_tpl = _load_prompt("tree_select_articles.txt")
        self._node_navigate_tpl  = _load_prompt("tree_navigate_nodes.txt")
        self._qa_tpl             = _load_prompt("basic_qa.txt")

    # ------------------------------------------------------------------
    # Stage 1: BM25 article shortlist (no LLM)
    # ------------------------------------------------------------------

    def _bm25_candidate_doc_ids(self, query: str) -> list[str]:
        """Return doc_ids of top BM25 candidate articles, deduplicated."""
        results = bm25_search(
            self.bm25, self.bm25_meta, query, self.bm25_candidate_k
        )
        seen: set[str] = set()
        doc_ids: list[str] = []
        for r in results:
            # BM25 index stores chunks; map back to article via title metadata
            doc_id = r["metadata"].get("title") or r["metadata"].get("doc_id", "")
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                doc_ids.append(doc_id)
        return doc_ids

    # ------------------------------------------------------------------
    # Stage 2: LLM article selection
    # ------------------------------------------------------------------

    def _llm_select_articles(
        self, query: str, candidate_ids: list[str]
    ) -> tuple[list[str], int]:
        """
        Present article titles + summaries to the LLM; return selected doc_ids.
        Returns (selected_doc_ids, tokens_used).
        """
        lines = []
        for doc_id in candidate_ids:
            tree = self.trees.get(doc_id)
            summary = tree["summary"] if tree else ""
            lines.append(f'"{doc_id}": {summary}')

        prompt = self._article_select_tpl.format(
            question=query,
            articles="\n".join(f"- {l}" for l in lines),
            k=self.article_select_k,
        )
        response, tokens = self.llm.complete(prompt)
        selected = _parse_id_list(response, candidate_ids)
        return selected[: self.article_select_k], tokens

    # ------------------------------------------------------------------
    # Stage 3: LLM node navigation (one call per selected article)
    # ------------------------------------------------------------------

    def _llm_navigate_nodes(
        self, query: str, doc_id: str
    ) -> tuple[list[dict], int]:
        """
        Navigate the tree of one article; return relevant paragraph nodes.
        Returns (docs_list, tokens_used).
        """
        tree = self.trees.get(doc_id)
        if not tree or not tree.get("nodes"):
            return [], 0

        nodes = tree["nodes"]
        node_lines = "\n".join(
            f'  node_id: "{n["node_id"]}" | {n["title"]} | "{n["summary"]}"'
            for n in nodes
        )
        prompt = self._node_navigate_tpl.format(
            question=query,
            article_title=doc_id,
            article_summary=tree.get("summary", ""),
            nodes=node_lines,
            k=self.max_nodes_per_article,
        )
        response, tokens = self.llm.complete(prompt)

        nodes_by_id = {n["node_id"]: n for n in nodes}
        selected_ids = _parse_id_list(response, list(nodes_by_id.keys()))

        result_docs: list[dict] = []
        for rank, nid in enumerate(selected_ids[: self.max_nodes_per_article]):
            node = nodes_by_id.get(nid)
            if not node:
                continue
            result_docs.append({
                "text": node["text"],
                "metadata": {
                    "doc_id":           doc_id,
                    "node_id":          nid,
                    "title":            doc_id,
                    "paragraph_title":  node["title"],
                    "summary":          node["summary"],
                    "retrieval_stage":  "tree_navigation",
                },
                # Rank-based score: first selected node scores highest
                "score": 1.0 / (rank + 1),
            })
        return result_docs, tokens

    # ------------------------------------------------------------------
    # BaseRAG interface
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """
        Vectorless two-stage retrieval. Token counts are discarded here;
        use run() for full token accounting.
        """
        candidate_ids = self._bm25_candidate_doc_ids(query)
        if not candidate_ids:
            return []

        selected_ids, _ = self._llm_select_articles(query, candidate_ids)

        all_nodes: list[dict] = []
        for doc_id in selected_ids:
            nodes, _ = self._llm_navigate_nodes(query, doc_id)
            all_nodes.extend(nodes)

        all_nodes.sort(key=lambda x: x["score"], reverse=True)
        return all_nodes[:top_k]

    def generate(self, query: str, docs: list[dict]) -> tuple[str, int]:
        context = "\n\n".join(
            f"[{d['metadata'].get('title', '?')} — {d['metadata'].get('paragraph_title', '')}]\n{d['text']}"
            for d in docs
        )
        prompt = self._qa_tpl.format(context=context, question=query)
        return self.llm.complete(prompt)

    def run(self, query: str, top_k: int | None = None, **kwargs) -> RAGResult:
        """Full pipeline with per-stage token and latency accounting."""
        if top_k is None:
            top_k = self.config.get("retrieval", {}).get("top_k", 5)

        t0 = time.perf_counter()
        total_tokens = 0

        # ── Stage 1: BM25 shortlist (no LLM) ──
        candidate_ids = self._bm25_candidate_doc_ids(query)
        if not candidate_ids:
            return RAGResult(
                answer="No relevant documents found.",
                retrieved_docs=[],
                latency_ms=(time.perf_counter() - t0) * 1000,
                token_count=0,
                metadata={"pattern": "TreeRag", "stage1_candidates": 0},
            )

        # ── Stage 2: LLM article selection ──
        selected_ids, select_tokens = self._llm_select_articles(query, candidate_ids)
        total_tokens += select_tokens

        # ── Stage 3: LLM node navigation (one LLM call per article) ──
        all_nodes: list[dict] = []
        nav_tokens_by_article: dict[str, int] = {}
        for doc_id in selected_ids:
            nodes, nav_tok = self._llm_navigate_nodes(query, doc_id)
            nav_tokens_by_article[doc_id] = nav_tok
            total_tokens += nav_tok
            all_nodes.extend(nodes)

        all_nodes.sort(key=lambda x: x["score"], reverse=True)
        retrieved_docs = all_nodes[:top_k]

        # ── Stage 4: Generation ──
        answer, gen_tokens = self.generate(query, retrieved_docs)
        total_tokens += gen_tokens

        return RAGResult(
            answer=answer,
            retrieved_docs=retrieved_docs,
            latency_ms=(time.perf_counter() - t0) * 1000,
            token_count=total_tokens,
            metadata={
                "pattern":                  "TreeRag",
                "stage1_candidates":        len(candidate_ids),
                "stage2_selected_articles": selected_ids,
                "stage3_articles_navigated": len(selected_ids),
                "total_nodes_before_trim":  len(all_nodes),
                "token_breakdown": {
                    "article_selection": select_tokens,
                    "node_navigation":   sum(nav_tokens_by_article.values()),
                    "generation":        gen_tokens,
                },
            },
        )
