# Empirical Benchmark Report: RAG Retrieval Performance

**Project:** EY Agentic Financial & Corporate Tax Assistant (`Enterprise_bot`)  
**Evaluation Scope:** Standalone Benchmark of Dense Vector Retrieval Pipeline  
**Evaluator:** Senior AI Engineer  
**Date of Evaluation:** November 19, 2026  
**Artifacts Generated:**
- Benchmark Script: `benchmarks/rag_benchmark.py`
- Raw Execution Logs: `benchmarks/rag_benchmark_results.json`

---

## 1. Executive Summary

This benchmark rigorously evaluates the Retrieval-Augmented Generation (RAG) vector retrieval component of the **EY Agentic Tax Consultant Assistant**. 

The evaluation was performed **without modifying the production RAG pipeline, the source document, or the existing ChromaDB vector database**. 

### Key Empirical Findings:
1. **Zero Query Failures:** 100% of the 25 evaluated tax statutory queries executed cleanly across the dense vector store with 0 runtime exceptions.
2. **Sub-10ms Vector Retrieval Latency:**
   - **Average Retrieval Latency:** **9.96 ms**
   - **P50 (Median) Latency:** **8.99 ms**
   - **P95 Latency:** **14.84 ms**
   - **Range:** [7.83 ms, 19.00 ms]
3. **Database Vector Duplication Factor (30.5×):**
   - The persisted ChromaDB store (`./enterprise_vector_db`) contains **47,070 total embedding vectors**, but only **1,541 unique document chunks**.
   - Root Cause: In `tools/rag_tool.py`, `vectorstore_persisted.add_documents(semantic_chunks)` executes upon module import without an idempotency guard, accumulating ~1,541 new UUIDs per server reload.
4. **Retrieval Accuracy:**
   - **Production Pipeline Hit@1, Hit@3, and Hit@5:** **52.0% (13/25)**. In the production retriever (`k=5`), all 5 slots return duplicate copies of the rank-1 chunk (`Uniq/Top5 = 1/5`).
   - **True Deduplicated Semantic Hit@3 / Hit@5:** **64.0% (16/25)**. Once vector redundancy is bypassed to evaluate distinct semantic chunks, retrieval accuracy increases by +12.0 percentage points.

---

## 2. Methodology & Benchmark Configuration

### A. Environment & Hardware
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
- **Vector Database:** ChromaDB (Local SQLite + HNSW Index, 77.0 MB `data_level0.bin`)
- **Execution Device:** NVIDIA CUDA GPU
- **Distance Metric:** Cosine / Squared L2 Distance in HNSW graph

### B. Ground-Truth Dataset (25 Curated Tax Questions)
The benchmark uses 25 domain-specific tax questions spanning major international jurisdictions in the *EY Worldwide Corporate Tax Guide*. Each question is mapped to strict, manually verifiable passage criteria extracted from the indexed corpus:
- **Coverage:** Germany, Canada, France, Ireland, United Kingdom, Japan, Singapore, Brazil, Mexico, South Africa, Netherlands, Sweden, Austria, Belgium, Finland, India, United States, Australia, Switzerland, Spain, Luxembourg, Denmark.
- **Topics:** Corporate statutory income tax rates, solidarity surcharges, territoriality principles, trading income definitions, loss carryforwards, controlled foreign company (CFC) anti-avoidance rules, transfer pricing compliance, pioneer incentives, interest on net equity (JCP), dividend imputation / franking credits, participation exemptions, and related-party withholding taxes.

### C. Evaluation Criteria
- **Hit@K:** Defined as $\mathbb{I}(\text{at least one retrieved chunk in top-}K \text{ satisfies the verifiable ground-truth passage})$.
- **Warm-Up Guard:** A pre-benchmark query was executed to ensure PyTorch and CUDA runtime kernels were warmed up prior to latency recording.

---

## 3. Detailed Results & Query Breakdown

| ID | Jurisdiction | Topic | Prod Hit@1 | Prod Hit@3 | Prod Hit@5 | Dedup Hit@3 | Dedup Hit@5 | Latency (ms) | Top-5 Unique |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q01** | Germany | Corp Tax Rate & Solidarity Surcharge | NO | NO | NO | NO | NO | 13.19 | 1/5 |
| **Q02** | Canada | Corporate Taxation of Resident Corps | **YES** | **YES** | **YES** | **YES** | **YES** | 19.00 | 1/5 |
| **Q03** | France | Territorial Principle of Corp Tax | NO | NO | NO | **YES** | **YES** | 12.60 | 1/5 |
| **Q04** | Ireland | Corporation Tax on Trading Income | NO | NO | NO | NO | NO | 8.91 | 1/5 |
| **Q05** | United Kingdom | Controlled Foreign Company (CFC) Rules | **YES** | **YES** | **YES** | **YES** | **YES** | 15.26 | 1/5 |
| **Q06** | Japan | Transfer Pricing Documentation | **YES** | **YES** | **YES** | **YES** | **YES** | 9.14 | 1/5 |
| **Q07** | Singapore | Pioneer Certificate Incentive | NO | NO | NO | NO | NO | 9.27 | 1/5 |
| **Q08** | Brazil | Interest on Net Equity (JCP) | NO | NO | NO | NO | NO | 8.26 | 1/5 |
| **Q09** | Mexico | Worldwide Income Taxation | **YES** | **YES** | **YES** | **YES** | **YES** | 8.15 | 1/5 |
| **Q10** | South Africa | Basis of Corporate Taxation | NO | NO | NO | **YES** | **YES** | 8.55 | 1/5 |
| **Q11** | Netherlands | Participation Exemption Rules | NO | NO | NO | **YES** | **YES** | 8.89 | 1/5 |
| **Q12** | Sweden | Business Income Aggregation | **YES** | **YES** | **YES** | **YES** | **YES** | 8.31 | 1/5 |
| **Q13** | Austria | Taxation of Resident & Foreign Corps | **YES** | **YES** | **YES** | **YES** | **YES** | 8.01 | 1/5 |
| **Q14** | Belgium | Worldwide Income Principle | **YES** | **YES** | **YES** | **YES** | **YES** | 9.03 | 1/5 |
| **Q15** | Finland | Worldwide Corporate Taxation | **YES** | **YES** | **YES** | **YES** | **YES** | 8.66 | 1/5 |
| **Q16** | India | Domestic Company Definition | **YES** | **YES** | **YES** | **YES** | **YES** | 9.53 | 1/5 |
| **Q17** | Germany | Loss Carryforward & Minimum Tax | NO | NO | NO | NO | NO | 10.61 | 1/5 |
| **Q18** | United States | Federal Corporate Tax Rate (21%) | NO | NO | NO | NO | NO | 8.79 | 1/5 |
| **Q19** | Singapore | Development & Expansion Incentive | NO | NO | NO | NO | NO | 8.53 | 1/5 |
| **Q20** | Australia | Dividend Imputation & Franking | **YES** | **YES** | **YES** | **YES** | **YES** | 10.97 | 1/5 |
| **Q21** | Switzerland | Federal Corporate Tax Rate | NO | NO | NO | NO | NO | 8.99 | 1/5 |
| **Q22** | Canada | Non-Resident Branch Tax | **YES** | **YES** | **YES** | **YES** | **YES** | 9.59 | 1/5 |
| **Q23** | Spain | Capital Gains Standard Rate | **YES** | **YES** | **YES** | **YES** | **YES** | 7.83 | 1/5 |
| **Q24** | Luxembourg | Corporate Income Tax Standard Rate | NO | NO | NO | NO | NO | 8.39 | 1/5 |
| **Q25** | Denmark | Related-Party Withholding Tax | **YES** | **YES** | **YES** | **YES** | **YES** | 10.50 | 1/5 |

---

## 4. Aggregate Performance Metrics

| Metric Category | Performance Indicator | Measured Result |
| :--- | :--- | :--- |
| **Query Robustness** | Evaluated Queries | **25 / 25 (100%)** |
| | Failed Queries | **0 (0.0%)** |
| **Production Accuracy ($k=5$)** | Production Hit@1 Rate | **52.0% (13/25)** |
| | Production Hit@3 Rate | **52.0% (13/25)** |
| | Production Hit@5 Rate | **52.0% (13/25)** |
| **Deduplicated Accuracy** | True Semantic Hit@3 Rate | **64.0% (16/25)** |
| | True Semantic Hit@5 Rate | **64.0% (16/25)** |
| **Retrieval Latency** | Mean (Average) Latency | **9.96 ms** |
| | P50 (Median) Latency | **8.99 ms** |
| | P95 Latency | **14.84 ms** |
| | Min Latency | **7.83 ms** |
| | Max Latency | **19.00 ms** |

---

## 5. Architectural Insights & Technical Limitations

1. **Impact of Chunk Duplication on Effective Context Window:**
   Because each top-5 retrieval in production returns 5 duplicate copies of the same text segment, the synthesis LLM receives redundant context rather than 5 diverse chunks from complementary sections. In queries such as France (Q03), South Africa (Q10), and Netherlands (Q11), the true relevant passage ranked 2nd or 3rd in semantic distance, but was completely crowded out in production because all 5 returned slots were filled by duplicates of the 1st-place chunk.
2. **Dense Bi-Encoder Limitations:**
   The `all-MiniLM-L6-v2` bi-encoder is tuned on general semantic similarity. In tax queries with exact statutory rates (e.g., German 15% rate vs. German general introduction), dense search sometimes ranks generic country overviews ahead of specific tax tables. Introducing BM25 keyword matching or a cross-encoder re-ranker (e.g., `bge-reranker-base`) would directly resolve this.
3. **Chunk Boundary Truncation:**
   Semantic chunking with `langchain-experimental` produced chunks ranging from 2 characters up to 55,649 characters (averaging 3,595 characters). Extremely long chunks dilute embedding density, while tiny chunks lose paragraph context. Fixed-size recursive chunking (e.g., 1,000 characters with 150-character overlap) provides more uniform retrieval granularity.
