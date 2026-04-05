# I Benchmarked 10 RAG Architectures So You Don't Have To

*A controlled comparison on 500 multi-hop questions. Most advanced patterns don't beat the baseline.*

---

Everyone building AI applications today has heard the pitch for "advanced RAG." Query your documents, sure — but why stop at basic vector search when you could have HyDE, Corrective RAG, or an Agentic loop? The papers are compelling. The GitHub repos are starred. The promise is better answers.

I decided to actually test this. I built a reproducible benchmark comparing 10 RAG architectures on the same dataset, embedding model, and LLM — changing only the retrieval strategy. The results were more surprising than I expected.

**Spoiler: most advanced patterns don't beat the baseline.**

---

## The Setup

Here's what I controlled:

- **Dataset:** 500 questions from HotpotQA distractor dev set — multi-hop questions that require reasoning across 2+ documents
- **Embedding model:** `all-MiniLM-L6-v2` (384-dim, L2-normalized)
- **Vector store:** FAISS IndexFlatIP (cosine via normalized inner product)
- **LLM:** `llama3.1:8b-instruct-q8_0` via Ollama locally — zero API costs, full reproducibility
- **Chunk size:** 512 tokens, 50-token overlap
- **3 runs per pattern** with seed=42 and temperature=0

The 10 patterns I tested, plus two baselines:

| Pattern | The idea |
|---|---|
| **Zero-retrieval** | LLM uses only its training knowledge (lower bound) |
| **Basic RAG** | Standard vector search (primary baseline) |
| **Oracle RAG** | Gold supporting facts injected directly (upper bound) |
| **Hybrid RAG** | BM25 + vector search fused via Reciprocal Rank Fusion |
| **Re-ranking RAG** | FAISS top-20 → cross-encoder → top-5 |
| **Multi-query RAG** | Generate 3 query variants, take the union |
| **HyDE** | Generate a hypothetical document, embed that instead of the query |
| **Parent-Child RAG** | Retrieve small chunks, return their larger parents |
| **Self-Query RAG** | LLM extracts metadata filters from the query |
| **Corrective RAG** | Re-retrieve if initial results aren't good enough |
| **Agentic RAG** | ReAct loop with search/lookup/answer tools |
| **Graph RAG** | Entity extraction → knowledge graph → community summaries |

---

## The Results

Here's the full table. Every number is the mean across 3 runs on 500 questions.

| Pattern | Recall@5 | Recall@10 | p50 Latency | Tokens |
|---|---|---|---|---|
| Oracle *(upper bound)* | **1.000** | **1.000** | 2,210 ms | 178 |
| Re-ranking RAG | **0.706** | 0.732 | 7,947 ms | 858 |
| Hybrid RAG | 0.651 | **0.745** | 6,829 ms | 864 |
| **Basic RAG** *(baseline)* | 0.609 | 0.684 | 6,774 ms | 833 |
| Multi-query RAG | 0.607 | 0.685 | 10,000 ms | 952 |
| HyDE | 0.602 | 0.679 | 14,046 ms | 1,039 |
| Parent-Child RAG | 0.605 | 0.679 | 6,707 ms | 913 |
| Corrective RAG | 0.604 | 0.681 | 15,596 ms | 2,415 |
| Self-Query RAG | 0.579 | 0.645 | 8,763 ms | 900 |
| Agentic RAG | 0.498 | 0.508 | 12,206 ms | 858 |
| Graph RAG | 0.348 | 0.559 | 5,964 ms | 786 |
| Zero-retrieval *(lower bound)* | 0.000 | 0.000 | 568 ms | 36 |

The delta vs. Basic RAG, with statistical significance (Wilcoxon signed-rank test + Cohen's d):

| Pattern | Δ Recall@5 | Significant? | Practical effect? |
|---|---|---|---|
| Re-ranking | **+15.9%** | Yes (p<0.001) | Yes (d=0.32, small) |
| Hybrid | **+6.9%** | Yes (p<0.001) | Barely (d=0.14) |
| Multi-query | −0.3% | No (p=0.688) | No |
| HyDE | −1.1% | No (p=0.431) | No |
| Parent-Child | −0.7% | Yes (p=0.006) | No (d=−0.013) |
| Corrective | −0.8% | Yes (p=0.001) | No (d=−0.016) |
| Self-Query | −4.9% | Yes (p<0.001) | No (d=−0.093) |
| Agentic | **−18.2%** | Yes (p<0.001) | Yes (d=−0.353, negative) |
| Graph RAG | **−42.9%** | Yes (p<0.001) | Yes (d=−0.777, negative) |

---

## What I Actually Learned

### 1. Re-ranking is worth it

Cross-encoder re-ranking is the only pattern that shows a real, statistically significant improvement over Basic RAG. The intuition makes sense: cosine similarity between independently-encoded query and document vectors is a rough proxy for relevance. A cross-encoder that jointly reads query + document makes much better judgments.

The cost: 17% more latency (7,947 ms vs. 6,774 ms). For most production applications, that's a reasonable trade.

### 2. Hybrid RAG should be your default, not Basic RAG

Adding BM25 to vector search takes maybe half a day of engineering. The result is Recall@10 of 0.745 vs. 0.684 for Basic RAG — the best @10 of any non-oracle pattern. The latency is essentially identical (6,829 ms vs. 6,774 ms).

If you're starting a new RAG project, begin with Hybrid RAG, not Basic RAG. The ROI is high.

### 3. HyDE nearly broke — and the reason matters

My initial HyDE implementation had Recall@5 = 0.142. That's catastrophically bad. The original prompt asked the LLM to "be factual and specific as an encyclopedia entry." The LLM interpreted this as a request for factual claims it couldn't verify, and responded "I don't have enough information" 81% of the time.

I rewrote the prompt:
> *"Write a short hypothetical passage (3-5 sentences) that would appear in a document containing the answer to the question below. This passage will be used for semantic search — it does not need to be factually verified. Write a plausible, specific passage even if you are uncertain. Do not say 'I don't know' or refuse — always write a passage."*

After that, HyDE reached 0.602 — basically matching Basic RAG.

The lesson: HyDE is extremely prompt-sensitive. The technique works when the LLM generates document-like text. It falls apart the moment the LLM decides to refuse. If you deploy HyDE, budget time for prompt iteration and monitor refusal rates in production.

### 4. Multi-query RAG and Corrective RAG: complex pipelines, no gain

Multi-query RAG generates 3 query variants and merges results. Corrective RAG evaluates retrieved chunks and re-retrieves if they're not good enough. Both sound reasonable. Both add significant overhead (1.5–2.5× latency and/or tokens). Neither improves Recall@5 in a practically meaningful way.

For Multi-query: the problem on HotpotQA isn't query phrasing. It's that multi-hop questions require two separate documents that share no keywords. Generating variants of the same query doesn't fix that. The three queries converge on similar retrieval results.

For Corrective RAG: on a closed corpus (where the answer always exists), the initial retrieval usually finds the right documents. The correctiveness check mostly says "yes, these are fine" and adds token cost. CRAG's benefit would appear on open-domain questions where initial retrieval genuinely fails.

### 5. Agentic RAG underperforms at 8B scale

Agentic RAG with a ReAct loop (max 5 iterations) achieves Recall@5 = 0.498 — significantly worse than the 0.609 baseline. The gap is both statistically significant and practically meaningful (d=−0.353).

The iterative loop is supposed to help with multi-hop reasoning. In practice, a bad first retrieval leads to a bad second query, which leads to a bad third query. Errors compound. The 8B model also struggles to reliably select the right tool (search vs. lookup vs. answer) across iterations.

I suspect this result reverses with a 70B model. The ReAct pattern requires reliable instruction-following to work well, and 8B is at the edge of that capability.

### 6. Graph RAG is not for factoid retrieval

Graph RAG (Microsoft's implementation) achieves Recall@5 = 0.348 — the worst of all patterns. It's not close. Community summaries are designed for dataset-level questions ("what are the main themes in this corpus?"), not specific factoid retrieval.

The interesting data point: Graph RAG has the largest Recall@5→Recall@10 gap (+0.211), meaning it retrieves broadly once you look at more results. The summaries cover many related concepts, just not the precise supporting facts needed for multi-hop answer verification.

If your application needs high-level synthesis across a large document corpus, Graph RAG may be worth the significant build cost. For question answering, it's the wrong tool.

---

## The Practical Playbook

Based on this benchmark:

**Start here:**
```
Hybrid RAG (BM25 + Vector + RRF)
→ near-zero additional latency
→ best Recall@10
→ should be the default, not Basic RAG
```

**Add this if precision matters:**
```
+ Cross-encoder re-ranking (top-20 → top-5)
→ best Recall@5 (+15.9%)
→ +17% latency overhead
→ worth it for quality-sensitive applications
```

**Avoid these unless you have a specific reason:**
```
- Multi-query RAG: adds 50% latency, no gain on multi-hop
- HyDE: same performance as Basic RAG with 2× latency; prompt-fragile
- Corrective RAG: 2.9× token cost, marginal or no retrieval benefit
- Agentic RAG at 8B: actively hurts retrieval; needs 70B+ to work
- Graph RAG for factoid QA: worst Recall@5; wrong use case
```

---

## What I'd Test Next

A few obvious limitations in this benchmark that future work should address:

1. **Larger LLMs**: Agentic and Corrective patterns are designed for capable models. Testing with Llama 70B or GPT-4o-mini would show whether the null results hold or reverse.

2. **Open-domain dataset**: HotpotQA guarantees the answer is in the corpus. On NaturalQuestions or TriviaQA (where answers may not exist in your index), Corrective RAG's re-retrieval step would actually matter.

3. **Chunking ablation**: I fixed chunk size at 512 tokens. Parent-Child RAG uses 256/1024 token split. It's possible a different base chunk size would change results for all patterns.

4. **Better embeddings**: `all-MiniLM-L6-v2` is fast and free but not state-of-the-art. `text-embedding-3-small` or `bge-large-en-v1.5` might change the relative ordering.

---

## The Code

Everything is reproducible: same config → same output. All runs use temperature=0, seed=42.

→ [GitHub: surpradhan/rag-benchmark](https://github.com/surpradhan/rag-benchmark)

```bash
# Install + run
pip install -r requirements.txt
ollama pull llama3.1:8b-instruct-q8_0
python scripts/prepare_dataset.py && python scripts/build_indexes.py
bash scripts/run_all.sh
```

The `results/` directory has timestamped JSON for every run, `comparison.csv` with mean ± std across runs, and all 8 charts as PNG + SVG.

---

*Built as a portfolio project to understand RAG architectures in depth. Feedback, corrections, and additions welcome — open an issue or PR.*
