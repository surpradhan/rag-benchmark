# RAG Architecture Benchmark

A reproducible, open-source benchmark comparing **11 RAG architectures + 2 baselines** on the same dataset, embedding model, and LLM. The only variable that changes between runs is the retrieval strategy.

---

## Key Findings

- **Re-ranking RAG** achieves the best Recall@5 (0.706, +15.9% vs. Basic RAG, p<0.001) and highest Faithfulness (0.725) — the only pattern with a statistically meaningful retrieval improvement
- **Hybrid RAG** (BM25 + vector + RRF) wins Recall@10 (0.745) and Answer Relevance (0.481) with near-identical latency to Basic RAG — the best default starting point
- **Multi-query, HyDE, Parent-Child, Corrective RAG** show no statistically meaningful retrieval gain over Basic RAG despite 1.5–2.9× the latency/tokens
- **Agentic RAG** underperforms Basic RAG on retrieval (−18.2%, p<0.001) — the ReAct loop compounds retrieval errors at 8B model scale
- **Tree RAG** (vectorless, PageIndex-inspired) achieves Recall@5=0.562 (−7.7%) with a 47.8s p50 latency — 7× slower than Basic RAG; the multi-LLM-call pipeline is poorly suited to short-paragraph corpora
- **Graph RAG** has the weakest Recall@5 (0.348) — designed for dataset-level synthesis, not factoid retrieval
- **Faithfulness** is discriminated by retrieval quality: Re-ranking (0.725) > Hybrid (0.693) > Basic (0.571), confirming better retrieval → more faithful answers

---

## Results

> 📊 **[Interactive Dashboard](results/dashboard.html)** — sortable table, charts, and significance badges for all 13 patterns.

Evaluated on **500 questions** from HotpotQA distractor dev set, 3 runs each (`seed=42`, `temperature=0`).

| Pattern | Recall@5 | Recall@10 | Precision@5 | Faithfulness | Ans. Relevance | p50 Latency |
|---|---|---|---|---|---|---|
| Oracle *(upper bound)* | **1.000** | **1.000** | **0.400** | 0.135 | 0.379 | 2,210 ms |
| Re-ranking RAG | **0.706** | 0.732 | **0.284** | **0.725** | 0.427 | 7,947 ms |
| Hybrid RAG | 0.651 | **0.745** | 0.262 | 0.693 | **0.481** | 6,829 ms |
| Basic RAG | 0.609 | 0.684 | 0.245 | 0.571 | 0.444 | 6,774 ms |
| Multi-query RAG | 0.607 | 0.685 | 0.244 | 0.647 | 0.429 | 10,000 ms |
| HyDE | 0.602 | 0.679 | 0.242 | 0.623 | 0.407 | 14,046 ms |
| Parent-Child RAG | 0.605 | 0.679 | 0.243 | 0.632 | 0.395 | 6,707 ms |
| Corrective RAG | 0.604 | 0.681 | 0.243 | 0.572 | **0.485** | 15,596 ms |
| Self-Query RAG | 0.579 | 0.645 | 0.233 | 0.600 | 0.439 | 8,763 ms |
| Agentic RAG | 0.498 | 0.508 | 0.200 | 0.541 | 0.480 | 12,206 ms |
| Tree RAG | 0.562 | 0.562 | 0.226 | 0.567 | 0.317 | 47,818 ms |
| Graph RAG | 0.348 | 0.559 | 0.140 | 0.557 | 0.451 | 5,964 ms |
| Zero-retrieval *(lower bound)* | 0.000 | 0.000 | 0.000 | 0.370 | 0.034 | 568 ms |

*Faithfulness and Answer Relevance via RAGAS (50-question sample per pattern, Llama 3.1 8B judge).*

---

## Charts

### % Improvement over Basic RAG Baseline
![Improvement over Baseline](results/charts/7_improvement_over_baseline.png)

### Accuracy vs. Latency
![Accuracy vs Latency](results/charts/3_accuracy_vs_latency.png)

### Radar: Top-5 Patterns
![Radar Chart](results/charts/6_radar.png)

### Recall@K Curves
![Recall@K Curves](results/charts/4_recall_at_k.png)

### Latency Distribution per Pattern
![Latency Distribution](results/charts/5_latency_distribution.png)

### Recall@5 by Question Type
![Error Heatmap](results/charts/8_error_heatmap.png)

---

## Controlled Variables

Every run uses identical values for all variables except the retrieval strategy:

| Variable | Value |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` (384-dim) |
| Chunk size | 512 tokens, 50 overlap |
| Splitter | `RecursiveCharacterTextSplitter` |
| Vector store | FAISS `IndexFlatIP` (cosine via L2-normalized inner product) |
| LLM | `llama3.1:8b-instruct-q8_0` via Ollama |
| Temperature | 0.0 |
| Seed | 42 |
| Eval dataset | HotpotQA distractor dev — 500 dev / 2,000 test questions |

---

## Patterns

| ID | Pattern | Description |
|---|---|---|
| B0 | Zero-retrieval | LLM parametric knowledge only — lower bound |
| B1 | Basic RAG | Standard vector similarity retrieval |
| B2 | Oracle | Gold supporting facts injected directly — upper bound |
| P2 | Hybrid RAG | BM25 + vector with Reciprocal Rank Fusion (k=60) |
| P3 | Re-ranking RAG | FAISS top-20 → cross-encoder (`ms-marco-MiniLM-L-6-v2`) → top-5 |
| P4 | Multi-query RAG | LLM generates 3 query variants → union of results |
| P5 | HyDE | LLM generates hypothetical doc → embed → search |
| P6 | Parent-Child RAG | Retrieve child chunks (256 tok), return parent context (1024 tok) |
| P7 | Self-Query RAG | LLM extracts metadata filters from natural language query |
| P8 | Corrective RAG | Re-retrieves if <50% of chunks rated RELEVANT (up to 2 attempts) |
| P9 | Agentic RAG | ReAct loop with `search`, `lookup`, `answer` tools (max 5 iterations) |
| P10 | Graph RAG | Entity extraction → knowledge graph → community summaries |
| P11 | Tree RAG | BM25 shortlist (15) → LLM article selection (top 5) → LLM node navigation (top 3 paragraphs/article) — no embeddings at query time |

---

## Setup

**Requirements:** Python 3.11, [Ollama](https://ollama.com) running locally.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull the LLM
ollama pull llama3.1:8b-instruct-q8_0

# 3. (Optional) Set API keys for cloud LLM fallback
cp .env.example .env

# 4. Download dataset and build indexes
python scripts/prepare_dataset.py
python scripts/build_indexes.py

# 5. Verify setup
python scripts/verify_setup.py

# 6. Run a single pattern (dev mode = 50 questions)
python evaluation/run_eval.py --pattern basic_rag --run-id 1 --dev

# 7. Run the full benchmark
bash scripts/run_all.sh
```

**Fallback:** If local GPU is insufficient for Llama 8B, set `TOGETHER_API_KEY` in `.env` and update `config/config.yaml` to use `provider: together`.

---

## Repository Structure

```
config/           # config.yaml (all hyperparameters) + prompt templates
data/
  processed/      # test_questions.json, dev_questions.json (version-controlled)
  raw/            # original downloads (gitignored)
  faiss_index/    # FAISS vector store (gitignored, rebuilt by build_indexes.py)
  tree_index/     # Tree RAG JSON index (gitignored, rebuilt by build_indexes.py)
rag_patterns/     # one file per RAG pattern, all implementing BaseRAG
evaluation/       # metrics.py, run_eval.py, logger.py
results/
  aggregated/     # comparison.csv (mean ± std across runs)
  charts/         # PNG (300 DPI) + SVG
scripts/          # data prep, index building, chart generation, run_all.sh
report/           # research report and blog draft
```

---

## Reproducing the Results

```bash
# Run all 13 patterns × 3 runs
bash scripts/run_all.sh

# Or run a specific pattern
python evaluation/run_eval.py --pattern reranking_rag --run-id 1 --config config/config.yaml

# Aggregate and regenerate charts
python scripts/aggregate_results.py --input results/raw/ --output results/aggregated/
python scripts/generate_charts.py --input results/aggregated/ --output results/charts/
```

All results are deterministic: same config → same output (`temperature=0`, `seed=42`).

---

## License

MIT
