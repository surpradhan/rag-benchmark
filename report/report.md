# Which RAG Architecture Actually Works? A Controlled Benchmark of 10 Strategies

**Surabhi Pradhan** · April 2026

---

## Abstract

We present a controlled benchmark comparing **10 Retrieval-Augmented Generation (RAG) architectures** plus two baselines (zero-retrieval and oracle) on the HotpotQA distractor dev set. All patterns share an identical stack — same embedding model, vector store, LLM, chunking parameters, and evaluation dataset. The only variable is the retrieval strategy. We evaluate across seven dimensions: Recall@5, Recall@10, Precision@5, Precision@10, Faithfulness, Answer Relevance, and latency/token cost. Statistical significance is established via Wilcoxon signed-rank tests with Cohen's d effect sizes.

**Key findings:** Re-ranking RAG achieves the best Recall@5 (0.706, +15.9% over Basic RAG, p<0.001, d=0.32) and Faithfulness (0.725). Hybrid RAG wins Recall@10 (0.745, +8.9%, p<0.001) and Answer Relevance (0.481). Most "advanced" patterns — Multi-query, HyDE, Parent-Child, Corrective RAG — show no statistically meaningful retrieval gain over Basic RAG despite 1.5–2.5× the latency. Agentic RAG and Graph RAG actively underperform on retrieval. Faithfulness ranges from 0.135 (Oracle) to 0.725 (Re-ranking) and correlates strongly with retrieval quality — better retrieval yields more grounded answers.

---

## 1. Introduction

The RAG literature has exploded with architectural variants: HyDE generates hypothetical documents for embedding-based retrieval; Corrective RAG re-retrieves when initial results are poor; Agentic RAG uses a ReAct loop to iteratively refine retrieval; Graph RAG builds knowledge graphs for community-level summarization. Each paper reports gains — but rarely against the same baseline, on the same dataset, with the same LLM.

This benchmark exists to answer a single practical question: **given a fixed LLM and retrieval infrastructure, which retrieval strategy delivers the best results?**

We hold everything constant except the retrieval strategy:

| Variable | Value |
|---|---|
| Dataset | HotpotQA distractor dev, 500 dev / 2,000 test questions |
| Embedding model | `all-MiniLM-L6-v2` (384-dim, L2-normalized) |
| Vector store | FAISS `IndexFlatIP` (cosine via normalized inner product) |
| Chunk size | 512 tokens, 50-token overlap |
| LLM | `llama3.1:8b-instruct-q8_0` via Ollama |
| Temperature | 0.0, seed=42 |
| Runs per pattern | 3 (mean ± std reported) |

---

## 2. Patterns

### Baselines

**B0 — Zero-Retrieval.** The LLM answers using only its parametric knowledge. This is the lower bound — retrieval should always beat this. On HotpotQA multi-hop questions, it achieves Recall@5 = 0.000 as expected (no documents are retrieved, so supporting facts can never appear in context).

**B1 — Basic RAG.** Standard vector similarity retrieval: embed query → FAISS top-K → prompt LLM. This is the primary comparison baseline. All deltas are reported relative to this pattern.

**B2 — Oracle RAG.** Gold supporting facts from HotpotQA are injected directly as context. This is the upper bound — perfect retrieval at Recall@5 = 1.000.

### Evaluated Patterns

**P2 — Hybrid RAG (BM25 + Vector + RRF).** Combines BM25 lexical retrieval and FAISS vector retrieval via Reciprocal Rank Fusion (k=60, BM25 weight=0.3). The intuition: BM25 excels at exact keyword matches; vector search excels at semantic similarity. RRF combines ranked lists without needing score calibration.

**P3 — Re-ranking RAG.** Retrieves a wider candidate set (top-20 from FAISS) then re-ranks using a cross-encoder (`ms-marco-MiniLM-L-6-v2`), returning the top-5. Cross-encoders jointly encode query and document, capturing fine-grained relevance beyond cosine similarity of independent embeddings.

**P4 — Multi-query RAG.** The LLM generates three query variants for each question, retrieves independently for each, and takes the union of results. The intent is to capture documents that match alternative phrasings of the same information need.

**P5 — HyDE (Hypothetical Document Embeddings).** The LLM generates a hypothetical passage that would answer the question, then embeds that passage (rather than the raw query) for retrieval. The hypothesis: a generated passage lives in the same embedding space as real documents, reducing the query-document distributional gap.

**P6 — Parent-Child RAG.** Indexes smaller "child" chunks (256 tokens, 25-token overlap) for high-precision retrieval but returns their larger "parent" chunks (1,024 tokens, 100-token overlap) as context. Child chunks retrieve with higher precision; parent chunks provide richer context for generation.

**P7 — Self-Query RAG.** Extracts structured metadata filters from the natural language query using the LLM, then applies those filters before vector search. Falls back to Basic RAG if no valid filter is parsed.

**P8 — Corrective RAG (CRAG).** After initial retrieval, evaluates each chunk as RELEVANT, PARTIALLY RELEVANT, or IRRELEVANT. If fewer than 50% are rated RELEVANT, re-retrieves with an expanded query. Up to 2 re-retrieval attempts.

**P9 — Agentic RAG.** Implements a ReAct (Reason + Act) loop with three tools: `search`, `lookup` (retrieve by document ID), and `answer`. The agent iteratively decides which tool to invoke, with a maximum of 5 iterations.

**P10 — Graph RAG.** Extracts entities and relationships from the corpus, builds a knowledge graph, and generates community summaries via the Microsoft `graphrag` library. At query time, retrieves relevant community summaries rather than raw document chunks.

---

## 3. Results

### 3.1 Retrieval Quality

**Table 1: Retrieval metrics (mean across 3 runs, 500 questions each)**

| Pattern | Recall@5 | Recall@10 | Precision@5 | Precision@10 | p50 Latency | Avg Tokens |
|---|---|---|---|---|---|---|
| Oracle *(upper bound)* | **1.000** | **1.000** | **0.400** | **0.200** | 2,210 ms | 178 |
| Re-ranking RAG | **0.706** | 0.732 | **0.284** | 0.147 | 7,947 ms | 858 |
| Hybrid RAG | 0.651 | **0.745** | 0.262 | **0.150** | 6,829 ms | 864 |
| Basic RAG *(baseline)* | 0.609 | 0.684 | 0.245 | 0.138 | 6,774 ms | 833 |
| HyDE | 0.602 | 0.679 | 0.242 | 0.137 | 14,046 ms | 1,039 |
| Multi-query RAG | 0.607 | 0.685 | 0.244 | 0.138 | 10,000 ms | 952 |
| Parent-Child RAG | 0.605 | 0.679 | 0.243 | 0.137 | 6,707 ms | 913 |
| Corrective RAG | 0.604 | 0.681 | 0.243 | 0.137 | 15,596 ms | 2,415 |
| Self-Query RAG | 0.579 | 0.645 | 0.233 | 0.130 | 8,763 ms | 900 |
| Agentic RAG | 0.498 | 0.508 | 0.200 | 0.102 | 12,206 ms | 858 |
| Graph RAG | 0.348 | 0.559 | 0.140 | 0.112 | 5,964 ms | 786 |
| Zero-retrieval *(lower bound)* | 0.000 | 0.000 | 0.000 | 0.000 | 568 ms | 36 |

### 3.2 Statistical Significance

All patterns are compared to Basic RAG using the Wilcoxon signed-rank test (paired, two-sided, α=0.05) with Cohen's d for effect size.

**Table 2: Statistical tests vs. Basic RAG (Recall@5)**

| Pattern | Δ (abs) | Δ (%) | p-value | Significance | Cohen's d | Effect |
|---|---|---|---|---|---|---|
| Re-ranking RAG | +0.097 | **+15.9%** | <0.001 | *** | 0.32 | small |
| Hybrid RAG | +0.042 | **+6.9%** | <0.001 | *** | 0.14 | negligible |
| Multi-query RAG | −0.002 | −0.3% | 0.688 | ns | −0.006 | negligible |
| Parent-Child RAG | −0.004 | −0.7% | 0.006 | ** | −0.013 | negligible |
| Corrective RAG | −0.005 | −0.8% | 0.001 | ** | −0.016 | negligible |
| HyDE | −0.007 | −1.1% | 0.431 | ns | −0.021 | negligible |
| Self-Query RAG | −0.030 | −4.9% | <0.001 | *** | −0.093 | negligible |
| Agentic RAG | −0.111 | **−18.2%** | <0.001 | *** | −0.353 | small |
| Graph RAG | −0.261 | **−42.9%** | <0.001 | *** | −0.777 | medium |

Key observations:
- **Re-ranking** is the only pattern with a statistically significant, practically meaningful improvement (small effect, d=0.32)
- **Hybrid RAG** is significant but negligible effect size (d=0.14) — real gain, but small
- **Multi-query and HyDE** are not significantly different from Basic RAG (p>0.05)
- **Parent-Child and Corrective** are significantly *worse* but negligible effect — statistically real, practically irrelevant
- **Agentic and Graph RAG** are significantly worse with meaningful effect sizes

### 3.3 Generation Quality (RAGAS)

*Note: RAGAS evaluation uses Llama 3.1 8B locally via Ollama on a 50-question sample per pattern.*

| Pattern | Faithfulness | Answer Relevance | Hallucination Rate |
|---|---|---|---|
| Re-ranking RAG | **0.725** | 0.427 | 27.5% |
| Hybrid RAG | 0.693 | **0.481** | 30.7% |
| Multi-query RAG | 0.647 | 0.429 | 35.3% |
| Parent-Child RAG | 0.632 | 0.395 | 36.8% |
| HyDE | 0.623 | 0.407 | 37.7% |
| Corrective RAG | 0.572 | **0.485** | 42.8% |
| Basic RAG *(baseline)* | 0.571 | 0.444 | 42.9% |
| Agentic RAG | 0.541 | 0.480 | 45.9% |
| Graph RAG | 0.557 | 0.451 | 44.3% |
| Self-Query RAG | 0.600 | 0.439 | 40.1% |
| Zero-retrieval | 0.370 | 0.034 | 63.0% |
| Oracle RAG | 0.135 | 0.379 | 86.5% |

Faithfulness measures whether all claims in the generated answer are supported by the retrieved context. Answer Relevance measures how well the answer addresses the original question. Hallucination Rate = 1 − Faithfulness.

**Notable finding:** Oracle RAG has the lowest faithfulness (0.135) despite perfect retrieval. The cause: Oracle injects gold supporting facts verbatim, which are often partial sentences or fragment-style facts. The 8B LLM synthesizes answers that go beyond these fragments, producing claims not directly supported by the injected text — measured by RAGAS as hallucination. This is a known limitation of using small LLMs as RAGAS judges: they struggle to verify answers when the context is sparse or structured differently from the answer.

---

## 4. Analysis

### 4.1 Re-ranking Works — But Consider the Cost

Re-ranking RAG achieves the best Recall@5 with a statistically significant, small effect (d=0.32). The mechanism is sound: retrieving a wider candidate set (top-20) gives the cross-encoder more material to work with, and cross-encoders are substantially better at relevance judgment than cosine similarity between independently-encoded vectors.

The cost is latency: 7,947 ms p50 vs. 6,774 ms for Basic RAG — a 17% increase. For applications where retrieval quality matters more than sub-second response, this is a favorable trade.

### 4.2 Hybrid RAG: The Underrated Default

Hybrid RAG (BM25 + Vector + RRF) achieves the best Recall@10 (0.745, +8.9% over Basic RAG) with essentially the same latency (6,829 ms vs. 6,774 ms). The improvement is statistically significant but has a negligible effect size (d=0.14), suggesting the gain is real but modest.

The practical implication: if you're already running BM25 (cheap to build, fast to query), RRF fusion adds meaningful coverage at near-zero marginal cost. Hybrid RAG should arguably be the default starting point, not Basic RAG.

### 4.3 The "Fancy" Middle Tier — Real Cost, No Gain

Multi-query RAG (p=0.688, ns), HyDE (p=0.431, ns), and Parent-Child RAG (d=−0.013, negligible) are not statistically distinguishable from Basic RAG on retrieval. Yet they cost 1.3–2.1× the latency and tokens.

**Multi-query RAG's** failure is surprising. Generating three query variants should improve coverage. The likely explanation: HotpotQA's multi-hop questions require reasoning across two documents, not just better phrasing of a single query. Query expansion helps when the user's phrasing is ambiguous; it doesn't help when the problem is that no single query can retrieve both required documents simultaneously.

**HyDE** relies on the LLM generating a plausible hypothetical document. Our initial implementation had an 81% refusal rate — the LLM interpreted the request as asking for factual claims it couldn't verify. After rewriting the prompt to explicitly frame it as a semantic search tool ("write a passage that would contain the answer, accuracy doesn't matter"), HyDE performed on par with Basic RAG. This points to a key HyDE fragility: the technique is highly prompt-sensitive, and small prompt changes can collapse performance.

**Parent-Child RAG's** null result suggests that on HotpotQA, the precision of child-chunk retrieval doesn't compensate for the context dilution of returning larger parent chunks. The supporting facts are specific; returning 1,024-token parents around them adds noise.

### 4.4 Corrective RAG: Complexity Without Payoff

CRAG adds the most token cost of any pattern (2,415 tokens on average — 2.9× Basic RAG) and achieves a marginally worse Recall@5 (d=−0.016, statistically significant but negligible). The self-evaluation step ("is this chunk RELEVANT?") consumes tokens and latency, and the re-retrieval step rarely triggers for questions where Basic RAG already gets the relevant documents. On HotpotQA, where the corpus contains the answer, the initial retrieval is usually sufficient. CRAG's benefit would likely materialize on open-domain questions where the corpus may not contain the answer — which HotpotQA by construction always does (via distractor passages).

### 4.5 Agentic RAG: When Iteration Hurts

Agentic RAG (Recall@5 = 0.498, −18.2% vs. Basic RAG, d=−0.353, small negative effect) is the most surprising result. The ReAct loop was expected to close the gap for multi-hop questions by iteratively refining retrieval. Instead, it underperforms.

Two likely mechanisms:
1. **Error accumulation**: Each iteration conditions on the previous one. A bad initial retrieval leads to a refined query that searches for the wrong thing, compounding the error.
2. **Tool selection noise**: At 8B parameters, the LLM struggles to reliably choose between `search`, `lookup`, and `answer` tools across 5 iterations. Larger models (70B+) likely show better agentic performance.

The 12,206 ms p50 latency confirms the iteration overhead without corresponding benefit.

### 4.6 Graph RAG: Broad but Imprecise

Graph RAG has the weakest Recall@5 (0.348) but the smallest Recall@5→Recall@10 gap among non-oracle patterns (+0.211, from 0.348 to 0.559). Community summaries retrieve broadly — many related concepts — but not precisely, consistently missing the exact supporting facts needed for HotpotQA's binary answer verification.

Graph RAG was designed for dataset-level questions ("what themes appear across all documents?") rather than factoid multi-hop retrieval. Its poor performance here reflects a dataset mismatch, not necessarily a flaw in the approach.

---

## 5. Design Decisions and Methodology

### Why HotpotQA?

HotpotQA distractor questions require reasoning across two supporting documents, with 10 distractor paragraphs included per question. This stress-tests both retrieval precision (can you find both needed documents?) and the LLM's multi-hop reasoning capability. The `supporting_facts` field provides ground truth for Recall and Precision calculation without manual annotation.

### Why Llama 3.1 8B?

The goal is to compare RAG patterns, not LLMs. A quantized 8B model runs fully locally at zero API cost with complete reproducibility. A 70B model would increase per-run time from ~3 hours to ~15+ hours on consumer hardware, making 3-run-per-pattern evaluation impractical.

The 8B model is sufficient to demonstrate retrieval quality differences — Recall@K depends on what documents are retrieved, not on how well the LLM generates from them.

### On HyDE's Prompt Sensitivity

The original HyDE prompt produced Recall@5 = 0.142 due to refusals. After rewriting, it reached 0.602. We document this explicitly because:
1. Prompt sensitivity is a real deployment risk for HyDE
2. Published HyDE results likely used carefully tuned prompts optimized for the target LLM
3. Our fixed prompt (`llama3.1:8b-instruct-q8_0`) may still underrepresent HyDE's ceiling

### Statistical Choices

- **Wilcoxon signed-rank** (not paired t-test): per-question recall scores are bimodal (0 or 1), violating the normality assumption of t-tests
- **Cohen's d** alongside p-values: with 500 questions × 3 runs = 1,500 pairs, even negligible effects achieve statistical significance. Effect size prevents over-interpreting small but significant deltas
- **3 runs**: all runs produced identical results (std=0.000 for all patterns), confirming full determinism at temperature=0, seed=42

---

## 6. Practical Recommendations

Based on these results, our recommendations for practitioners:

1. **Start with Hybrid RAG** instead of Basic RAG. It's free (BM25 is trivial to add), adds 0.055 ms of latency, and consistently improves Recall@10.

2. **Add Re-ranking if Recall@5 matters** and you can tolerate 17% latency overhead. Cross-encoder re-ranking is the only pattern with a practically meaningful improvement over Basic RAG on this dataset.

3. **Avoid Corrective RAG on closed corpora** (where the answer is always present). The evaluation overhead adds cost without retrieval benefit. CRAG's value is on open-domain QA where initial retrieval genuinely fails.

4. **Be cautious with Agentic RAG at 8B scale**. The ReAct loop performs worse than a single retrieval pass. If you need iterative refinement, ensure you have a 70B+ model or clear evidence of benefit on your specific task.

5. **Don't use Graph RAG for factoid retrieval**. It's designed for dataset-level synthesis and should not be the primary retrieval mechanism for specific multi-hop questions.

6. **HyDE requires LLM-specific prompt engineering**. The technique works, but the prompt is sensitive. Budget time for prompt iteration when deploying HyDE on a new model.

---

## 7. Limitations

- **Single dataset**: HotpotQA distractor dev set may not generalize. Multi-hop questions with perfect in-corpus answers may disadvantage patterns designed for open-domain settings (CRAG, Agentic).
- **8B LLM**: Agentic patterns (Agentic RAG, CRAG) likely improve more with larger models. Our null/negative results for these patterns reflect 8B limitations, not inherent architectural limitations.
- **Local Ollama evaluation**: RAGAS faithfulness and answer relevance are computed by the same 8B LLM used for generation, which may introduce self-consistency bias. A stronger judge model would improve evaluation quality.
- **Zero variance across runs**: temperature=0 + seed=42 produces identical results each run. The 3-run protocol was designed for stochastic settings and adds no information here. For production use, temperature>0 would be needed and per-run variance would matter.
- **No human evaluation**: RAGAS metrics are proxies. A 100-question human evaluation subset would strengthen the generation quality findings.

---

## 8. Future Work

- **Multi-model comparison**: repeat with Llama 70B or GPT-4o-mini to test whether Agentic/Corrective patterns recover under stronger models
- **Chunking ablation**: test chunk sizes 256/512/1024 tokens with Hybrid RAG to separate chunking effects from retrieval strategy effects
- **Open-domain transfer**: run on Natural Questions or TriviaQA (no guaranteed in-corpus answer) to test CRAG and Agentic RAG in their designed setting
- **Embedding model ablation**: `text-embedding-3-small` vs `all-MiniLM-L6-v2` — better embeddings may change which retrieval strategies win
- **Domain transfer**: 200 Arxiv abstracts (cs.CL/cs.IR, 2023–2025) to test on technical document retrieval

---

## References

1. Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.
2. Yang et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-Hop Question Answering. *EMNLP 2018*.
3. Gao et al. (2023). Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE). *ACL 2023*.
4. Shi et al. (2023). REPLUG: Retrieval-Augmented Language Model Pre-Training.
5. Yan et al. (2024). Corrective Retrieval Augmented Generation (CRAG). *arXiv 2401.15884*.
6. Edge et al. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. *arXiv 2404.16130*.
7. Yao et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*.
8. Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *arXiv 2309.15217*.
9. Robertson & Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in IR*.
10. Johnson et al. (2019). Billion-scale similarity search with GPUs (FAISS). *IEEE TBBD*.
