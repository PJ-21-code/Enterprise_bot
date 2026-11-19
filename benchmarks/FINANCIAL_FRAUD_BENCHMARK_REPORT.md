# Empirical Benchmark Report: SQL Analytics & Forensic Fraud Detection

**Project:** EY Agentic Financial & Corporate Tax Assistant (`Enterprise_bot`)  
**Evaluation Scope:** Standalone Benchmark of SQL Analytics, Split-Transaction Detection, and Benford's Law Forensic Modules  
**Evaluator:** Senior AI Engineer  
**Date of Evaluation:** November 19, 2026  
**Artifacts Generated:**
- Database Generator: `benchmarks/generate_synthetic_database.py`
- Synthetic SQLite Database: `benchmarks/synthetic_finance_1000.db` (1,000 transactions across 50 accounts)
- Ground Truth Labels: `benchmarks/ground_truth_splits.json`
- Benchmark Script: `benchmarks/financial_fraud_benchmark.py`
- Raw Execution Logs: `benchmarks/financial_fraud_benchmark_results.json`

---

## 1. Executive Summary

This benchmark rigorously evaluates the relational financial analytics (SQL) and forensic fraud detection tools of the **EY Agentic Financial & Tax Assistant** on a 1,000-transaction synthetic financial ledger.

The evaluation was executed **without modifying the original database (`enterprise_finance.db`), source code, or production tools**. All benchmark scripts and generated databases are isolated in the `benchmarks/` directory.

### Key Measured Outcomes:
1. **100% SQL Execution Success Rate (15/15 Queries):**
   - Sub-millisecond SQLite execution latency across diverse aggregations, windowing, entity joins, and threshold screenings.
   - **Average SQL Latency:** **0.61 ms** (P50: **0.51 ms**, P95: **1.14 ms**).
2. **Deterministic Split-Transaction Forensic Performance:**
   - Evaluated against 12 planted ground-truth approval-evasion clusters ($10,000 threshold, 36-hour sliding window) and 5 controlled near-misses.
   - **Precision:** **100.0%** (12 / 12)
   - **Recall:** **100.0%** (12 / 12)
   - **F1-Score:** **1.0000**
   - **False Positive Rate:** **0.0%** (All 5 negative control near-misses rejected).
   - **Execution Latency:** **12.45 ms** average over 1,000 records (P50: **8.94 ms**, P95: **19.15 ms**).
3. **Robust Benford's Law Anomaly Classification (100% Accuracy):**
   - Successfully distinguished conforming log-normal financial distributions (Discrepancy: **0.0677** vs. 0.15 threshold) from fabricated uniform (Discrepancy: **0.5901**) and psychologically clustered distributions (Discrepancy: **1.4964**).
   - **Execution Latency:** **~1.48 ms** per 1,000 transactions.

---

## 2. Dataset Synthesis & Ground-Truth Design

### A. Database Characteristics (`synthetic_finance_1000.db`)
- **Total Records:** 1,000 financial transactions
- **Account Entities:** 50 customer accounts (`customer_id` 101 to 150)
- **Time Span:** January 1, 2026 to June 30, 2026 (6-month period)
- **Transaction Types:**
  - `debit` (476 records): Operational expenses, SaaS, travel, utility fees.
  - `credit` (257 records): Invoices, advisory retainers, bank interest.
  - `transfer` (267 records): Vendor disbursements, milestone wires, capital leases.
- **Value Distribution:** Log-normal distribution spanning $15.00 to $48,500.00 (median ~$600, mean ~$1,500).

### B. Planted Split-Transaction Clusters (12 Positive Cases)
12 multi-tranche clusters were planted under a $10,000 approval limit rule (Candidate window: $8,000 to $10,000; Sliding window: $\le 36$ hours):
- 2-tranche clusters: 8 accounts (e.g., $8,500 + $9,200 within 14h; $9,950 + $9,800 within 6h).
- 3-tranche clusters: 3 accounts (e.g., $8,100 + $8,900 + $9,500 within 28h).
- 4-tranche clusters: 1 account ($8,600 + $8,800 + $9,100 + $8,450 within 32h).

### C. Controlled Negative Test Cases (5 Near-Misses)
To rigorously test for false positives, 5 specific near-miss scenarios were planted:
1. *Isolated Candidate:* Customer 108 had a single transaction of $9,150.00 (in range, but single txn $\rightarrow$ rejected).
2. *Exceeded Window:* Customer 115 had two candidate transactions ($8,500 and $8,800) spaced 54 hours apart ($> 36\text{h} \rightarrow$ rejected).
3. *Sub-Threshold Near-Miss:* Customer 120 had two transactions within 8h, but one was $7,200.00 ($< \$8,000 \rightarrow$ rejected).
4. *Over-Threshold Near-Miss:* Customer 125 had two transactions within 10h, but one was $10,500.00 ($> \$10,000 \rightarrow$ rejected).
5. *Inter-Account Isolation:* Unrelated accounts with transactions at identical timestamps ($\rightarrow$ rejected).

---

## 3. SQL Execution Benchmark Results

15 enterprise analytical queries were evaluated over 5 repetitions each:

| Query ID | Analytical Category | Query Intent | Rows | Median Latency (ms) | Status |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **SQL01** | Basic Aggregation | Total record count | 1 | 0.158 | SUCCESS |
| **SQL02** | Basic Aggregation | Total financial volume | 1 | 0.190 | SUCCESS |
| **SQL03** | Group Aggregation | Volume breakdown by type | 3 | 0.966 | SUCCESS |
| **SQL04** | Ranking & Sorting | Top 5 debit spending accounts | 5 | 0.368 | SUCCESS |
| **SQL05** | Time Series Analysis | Monthly volume and count trend | 6 | 0.899 | SUCCESS |
| **SQL06** | Threshold Filtering | High-value transactions (> $10k) | 1 | 0.193 | SUCCESS |
| **SQL07** | Range Screening | Candidate transactions ($8k - $10k) | 1 | 0.511 | SUCCESS |
| **SQL08** | Entity Having Filter | Accounts with $\ge 2$ candidate txns | 13 | 0.507 | SUCCESS |
| **SQL09** | Statistical Summary | Min, max, and average amounts | 1 | 0.825 | SUCCESS |
| **SQL10** | Date Range by Type | Earliest/latest dates per category | 3 | 1.501 | SUCCESS |
| **SQL11** | Entity Cardinality | Distinct customer account count | 1 | 0.982 | SUCCESS |
| **SQL12** | Deep Investigation | Account 105 candidate history | 2 | 0.533 | SUCCESS |
| **SQL13** | Conditional Aggregation| Net debit vs. credit balance per entity | 10 | 0.451 | SUCCESS |
| **SQL14** | Text Categorization | Top 5 recurring transaction descriptions | 5 | 0.437 | SUCCESS |
| **SQL15** | Subquery Analytics | Average daily aggregate volume | 1 | 0.631 | SUCCESS |

### Aggregate SQL Performance:
- **Success Rate:** **100.0% (15/15)**
- **Average Latency:** **0.61 ms**
- **P50 (Median) Latency:** **0.51 ms**
- **P95 Latency:** **1.14 ms**
- **Min / Max Latency:** **0.16 ms / 1.50 ms**

---

## 4. Split-Transaction Forensic Evaluation

| Forensic Metric | Planted / Ground Truth | Detected by Tool | Performance Score |
| :--- | :---: | :---: | :---: |
| **Candidate Transactions in Range** | 34 transactions | 34 transactions | 100.0% coverage |
| **Suspicious Clusters** | 12 clusters | 12 clusters | - |
| **True Positives (TP)** | 12 | 12 | - |
| **False Positives (FP)** | 0 | 0 | - |
| **False Negatives (FN)** | 0 | 0 | - |
| **Precision** | - | - | **100.00%** |
| **Recall** | - | - | **100.00%** |
| **F1-Score** | - | - | **1.0000** |
| **Negative Control Rejection** | 5 near-misses | 5 rejected | **100.0% (0 false alerts)** |
| **Execution Latency (1,000 txns)** | - | - | **12.45 ms** (P50: **8.94 ms**, P95: **19.15 ms**) |

---

## 5. Benford's Law Forensic Evaluation

The `benford_finance_fraud_detector` was tested across four 1,000-sample financial distributions against the theoretical first-digit distribution:
$$P(d) = \log_{10}\left(1 + \frac{1}{d}\right), \quad d \in \{1, \dots, 9\}$$

| Dataset Tested | Distribution Description | Total Variation Discrepancy | Tool Threshold | Anomaly Flag | Expected Classification | Benchmark Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Dataset A** | Natural Log-Normal Expenses | **0.0677** | 0.15 | **False** | Conforming | **CORRECT** |
| **Dataset B** | Fabricated Uniform Expenses | **0.5901** | 0.15 | **True** | Non-conforming | **CORRECT** |
| **Dataset C** | Psychological Clustered Values | **1.4964** | 0.15 | **True** | Non-conforming | **CORRECT** |
| **Dataset D** | Synthetic 1,000-Record Database | **0.1020** | 0.15 | **False** | Conforming | **CORRECT** |

### Execution Performance:
- **Average Analysis Time (1,000 items):** **1.48 ms**
- **Classification Accuracy:** **100.0% (4/4 datasets)**

---

## 6. Architectural Insights & Technical Limitations

1. **Greedy Windowing Semantics in Split Detection:**
   The sliding window implementation in `detect_split_transactions` measures the 36-hour span strictly from the *first* transaction in a cluster: `records[end]["timestamp"] - records[start]["timestamp"] <= window`. Once a cluster of $\ge 2$ items is closed, `start` jumps to `end`. While this eliminates overlapping redundant cluster reports, an adversary spacing transactions across overlapping 36-hour boundaries could partition across windows.
2. **Heuristic vs. Chi-Square Test in Benford's Law:**
   The current Benford implementation uses a Total Variation Distance (TVD) heuristic ($\sum |O_d - E_d|$) with a fixed threshold of `0.15`. While this performs with 100% accuracy on large synthetic sample sizes ($N=1,000$), TVD threshold sensitivity scales with $N$. For small sample sizes ($N < 50$), statistical variance can naturally exceed $0.15$ even for benign ledgers. Integrating a formal Pearson Chi-Square ($\chi^2$) goodness-of-fit test with sample-size-adjusted $p$-values is recommended for enterprise audits.
3. **In-Memory DataFrame Conversions:**
   The split detection tool accepts `list[dict]` and converts it into a pandas `DataFrame` in-memory. While processing 1,000 transactions takes only **12.45 ms**, scaling to millions of enterprise transactions would benefit from pushing candidate filtering (`WHERE amount BETWEEN 8000 AND 10000`) directly into SQLite before DataFrame instantiation.
