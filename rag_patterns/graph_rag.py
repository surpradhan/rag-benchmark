"""
P10: Graph RAG (custom implementation).

Microsoft `graphrag` requires cloud LLMs (OpenAI/Azure) — incompatible with our
Ollama-local setup. This implements the core algorithm faithfully:

  Build phase (one-time, cached):
  1. Extract named-entity nodes from corpus document titles
  2. Build co-reference edges: doc A → doc B when B's title appears in A's text
     (Wikipedia-style cross-reference graph). Uses an inverted word-index for speed.
  3. Louvain community detection (NetworkX) — produces topic clusters
  4. Embed each community's representative text → community FAISS index

  Query phase:
  1. Embed query → search community FAISS → top-K relevant communities
  2. Collect member titles from matched communities
  3. Search main FAISS with the query, filter to community members only
  4. Return top_k chunks ranked by query-chunk similarity

Config (all in config.yaml under graph_rag):
  persist_dir           ./data/graph_rag_index
  min_community_size    2     # discard singleton communities
  max_communities       1000  # cap indexed communities
  community_repr_docs   8     # docs per community in representative text
  community_repr_chars  200   # chars per doc in representative text
  top_communities       5     # communities retrieved per query
"""
from __future__ import annotations

import json
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path

import faiss
import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .base_retriever import BaseRAG, RAGResult
from .indexing import (
    FAISS_INDEX_FILE,
    build_faiss_index,
    faiss_search,
    load_faiss_index,
)
from .llm_client import LLMClient


def _load_prompt(name: str) -> str:
    return (Path("config/prompts") / name).read_text().strip()


# ---------------------------------------------------------------------------
# Entity / graph helpers
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "in", "of", "and", "or", "to", "is", "was", "are",
    "were", "for", "by", "with", "at", "from", "as", "its", "it", "this",
    "that", "he", "she", "his", "her", "they", "their", "who", "which",
    "been", "has", "have", "had", "be", "on", "not", "but", "also",
}


def _sig_words(text: str) -> list[str]:
    """Significant words: 3+ chars, not stopwords, lowercased."""
    return [
        w for w in re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        if w not in _STOPWORDS
    ]


def _build_coref_graph(corpus: list[dict], min_edge_weight: int = 1) -> nx.Graph:
    """
    Build an undirected title co-reference graph.
    Edge (A, B) added when title B appears as a substring in doc A's text.
    Uses an inverted word-index for O(n * avg_candidates) rather than O(n²).
    """
    G = nx.Graph()
    all_titles = [doc["title"] for doc in corpus]
    for t in all_titles:
        G.add_node(t)

    # Inverted index: significant word → set of titles containing that word
    word_to_titles: dict[str, set[str]] = defaultdict(set)
    title_sigwords: dict[str, list[str]] = {}
    for title in all_titles:
        sw = _sig_words(title)
        title_sigwords[title] = sw
        for w in sw:
            word_to_titles[w].add(title)

    edge_weight: dict[tuple[str, str], int] = defaultdict(int)

    for doc in tqdm(corpus, desc="Building co-reference graph", leave=False):
        text_lower = doc["title"].lower() + " " + doc["text"].lower()  # include own title
        text_word_set = set(text_lower.split())

        # Candidate titles that share ≥1 significant word with this text
        candidates: set[str] = set()
        for w in text_word_set:
            if w in word_to_titles:
                candidates.update(word_to_titles[w])
        candidates.discard(doc["title"])

        for cand in candidates:
            sig = title_sigwords.get(cand, [])
            if not sig:
                continue
            # Pre-filter: ALL significant words of the candidate title must
            # appear in the text word-set (fast set lookup)
            if not all(w in text_word_set for w in sig):
                continue
            # Final verification: full title substring match (case-insensitive)
            if cand.lower() in text_lower:
                key = (min(doc["title"], cand), max(doc["title"], cand))
                edge_weight[key] += 1

    for (a, b), w in edge_weight.items():
        if w >= min_edge_weight:
            G.add_edge(a, b, weight=w)

    return G


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------

def build_graph_index(
    corpus_path: str | Path,
    persist_dir: str | Path,
    config: dict,
    embed_model_name: str = "all-MiniLM-L6-v2",
    force: bool = False,
) -> tuple:
    """
    Build (or load cached) graph index.
    Returns (G, communities, title_to_community, comm_index, comm_meta).
    """
    persist_dir = Path(persist_dir)
    graph_path = persist_dir / "graph.pkl"
    communities_path = persist_dir / "communities.json"
    t2c_path = persist_dir / "title_to_community.pkl"
    comm_index_dir = persist_dir / "community_index"

    if (
        graph_path.exists()
        and communities_path.exists()
        and t2c_path.exists()
        and (comm_index_dir / FAISS_INDEX_FILE).exists()
        and not force
    ):
        print("[skip] Graph RAG index already exists — loading")
        with graph_path.open("rb") as f:
            G = pickle.load(f)
        with communities_path.open() as f:
            communities = json.load(f)
        with t2c_path.open("rb") as f:
            title_to_community = pickle.load(f)
        comm_index, comm_meta = load_faiss_index(comm_index_dir)
        return G, communities, title_to_community, comm_index, comm_meta

    persist_dir.mkdir(parents=True, exist_ok=True)

    with open(corpus_path) as f:
        corpus = json.load(f)

    gr_cfg = config.get("graph_rag", {})
    min_edge_weight = gr_cfg.get("min_edge_weight", 1)
    min_community_size = gr_cfg.get("min_community_size", 2)
    max_communities = gr_cfg.get("max_communities", 1000)
    repr_docs = gr_cfg.get("community_repr_docs", 8)
    repr_chars = gr_cfg.get("community_repr_chars", 200)

    # 1. Build co-reference graph
    print("Step 1/3: Building title co-reference graph...")
    G = _build_coref_graph(corpus, min_edge_weight=min_edge_weight)
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    with graph_path.open("wb") as f:
        pickle.dump(G, f)

    # 2. Louvain community detection
    print("Step 2/3: Detecting communities (Louvain)...")
    raw_communities = louvain_communities(G, seed=42)
    # Sort by size descending, cap at max_communities
    raw_communities = sorted(raw_communities, key=len, reverse=True)
    raw_communities = [
        c for c in raw_communities if len(c) >= min_community_size
    ][:max_communities]
    print(f"  {len(raw_communities):,} communities (min_size={min_community_size})")

    # Build title → doc lookup
    title_to_doc = {doc["title"]: doc for doc in corpus}

    # 3. Build community metadata + representative texts
    communities: list[dict] = []
    title_to_community: dict[str, int] = {}

    for cid, member_set in enumerate(raw_communities):
        # Sort by degree descending so most-connected titles lead the representative text
        member_titles = sorted(member_set, key=lambda t: G.degree(t), reverse=True)
        for t in member_titles:
            title_to_community[t] = cid

        # Representative text: titles + snippets of top repr_docs members by degree
        snippets = []
        for title in member_titles[:repr_docs]:
            doc = title_to_doc.get(title)
            if doc:
                snippets.append(f"{title}: {doc['text'][:repr_chars]}")

        communities.append({
            "community_id": cid,
            "member_titles": member_titles,
            "size": len(member_titles),
            "representative_text": " | ".join(snippets),
        })

    with communities_path.open("w") as f:
        json.dump(communities, f)
    with t2c_path.open("wb") as f:
        pickle.dump(title_to_community, f)

    # 4. Embed community representative texts → FAISS
    print("Step 3/3: Embedding community representatives...")
    comm_chunks = [
        {
            "chunk_id": c["community_id"],
            "community_id": c["community_id"],
            "size": c["size"],
            "member_titles": c["member_titles"],
            "text": c["representative_text"],
        }
        for c in communities
    ]
    comm_index, comm_meta = build_faiss_index(
        comm_chunks,
        persist_dir=comm_index_dir,
        model_name=embed_model_name,
        dimension=config.get("embedding", {}).get("dimension", 384),
        force=force,
    )

    print(f"Graph RAG index built → {persist_dir}")
    return G, communities, title_to_community, comm_index, comm_meta


# ---------------------------------------------------------------------------
# Pattern class
# ---------------------------------------------------------------------------

class GraphRag(BaseRAG):
    def __init__(self, config: dict):
        super().__init__(config)
        embed_cfg = config.get("embedding", {})
        vs_cfg = config.get("vector_store", {})
        gr_cfg = config.get("graph_rag", {})
        dataset_cfg = config.get("dataset", {})

        self.embed_model = SentenceTransformer(embed_cfg.get("model", "all-MiniLM-L6-v2"))
        self.normalize = embed_cfg.get("normalize", True)
        self.top_communities = gr_cfg.get("top_communities", 5)
        self.broad_k_multiplier = gr_cfg.get("broad_k_multiplier", 20)

        corpus_path = dataset_cfg.get("corpus_path", "data/processed/corpus.json")
        persist_dir = Path(gr_cfg.get("persist_dir", "./data/graph_rag_index"))

        # Load / build graph index
        self.G, self.communities, self.title_to_community, self.comm_index, self.comm_meta = (
            build_graph_index(
                corpus_path=corpus_path,
                persist_dir=persist_dir,
                config=config,
                embed_model_name=embed_cfg.get("model", "all-MiniLM-L6-v2"),
            )
        )

        # Pre-build set of member titles per community for fast lookup
        self._community_titles: dict[int, set[str]] = {
            c["community_id"]: set(c["member_titles"]) for c in self.communities
        }

        # Main chunk FAISS (for within-community relevance ranking)
        self.faiss_index, self.faiss_meta = load_faiss_index(
            vs_cfg.get("persist_dir", "./data/faiss_index")
        )

        self.llm = LLMClient(config)
        self._qa_template = _load_prompt("basic_qa.txt")

    def _embed(self, text: str) -> np.ndarray:
        vec = self.embed_model.encode(
            [text], normalize_embeddings=self.normalize, convert_to_numpy=True
        )
        return vec[0].astype(np.float32)

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        query_emb = self._embed(query)

        # Step 1: Find relevant communities via community FAISS
        comm_hits = faiss_search(self.comm_index, self.comm_meta, query_emb, self.top_communities)
        community_ids = [h["metadata"]["community_id"] for h in comm_hits]

        # Step 2: Collect member titles from matched communities
        relevant_titles: set[str] = set()
        for cid in community_ids:
            relevant_titles.update(self._community_titles.get(cid, set()))

        if not relevant_titles:
            # Fallback: plain FAISS search
            return faiss_search(self.faiss_index, self.faiss_meta, query_emb, top_k)

        # Step 3: Search main FAISS (broad), filter to community members, return top_k
        broad_k = min(top_k * self.broad_k_multiplier, self.faiss_index.ntotal)
        broad_results = faiss_search(self.faiss_index, self.faiss_meta, query_emb, broad_k)
        filtered = [r for r in broad_results if r["metadata"]["title"] in relevant_titles]

        if len(filtered) < top_k:
            # Pad with unfiltered results not already present
            seen = {r["metadata"]["chunk_id"] for r in filtered}
            for r in broad_results:
                if r["metadata"]["chunk_id"] not in seen:
                    filtered.append(r)
                    seen.add(r["metadata"]["chunk_id"])
                if len(filtered) >= top_k:
                    break

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
        answer, tokens = self.generate(query, docs)
        latency_ms = (time.perf_counter() - t0) * 1000

        retrieved_titles = list({d["metadata"]["title"] for d in docs})
        return RAGResult(
            answer=answer,
            retrieved_docs=docs,
            latency_ms=latency_ms,
            token_count=tokens,
            metadata={
                "pattern": "GraphRag",
                "top_communities": self.top_communities,
                "n_communities_total": len(self.communities),
                "graph_nodes": self.G.number_of_nodes(),
                "graph_edges": self.G.number_of_edges(),
                "retrieved_titles": retrieved_titles,
            },
        )
