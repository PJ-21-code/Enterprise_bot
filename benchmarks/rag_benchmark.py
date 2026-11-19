"""
EY Agentic Tax Consultant Assistant - RAG Retrieval Benchmark Script
-------------------------------------------------------------------
This script evaluates the production retrieval performance of the RAG system
without modifying the existing vector database, pipeline, or documents.

Metrics Measured:
  - Hit@3 and Hit@5 retrieval accuracy rates (Production and Deduplicated)
  - Average, P50 (median), and P95 retrieval latency (in milliseconds)
  - Number of evaluated and failed queries
  - Vector duplication impact analysis (unique chunks per top-k set)

Dataset:
  - 25 manually verifiable questions mapped to specific sections of the
    EY Worldwide Corporate Tax Guide (PDF).
"""

import os
import sys
import time
import json
import numpy as np
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuration matching production tools/rag_tool.py exactly
DB_PATH = "./enterprise_vector_db"
COLLECTION_NAME = "Enterprise_info"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_K = 5

# 25 Curated, Manually Verifiable Evaluation Queries
EVAL_DATASET = [
    {
        "id": "Q01",
        "jurisdiction": "Germany",
        "topic": "Corporate Income Tax Rate & Solidarity Surcharge",
        "query": "What is the corporate income tax rate and solidarity surcharge in Germany?",
        "ground_truth_criteria": ["Germany", "15%", "solidarity surcharge"],
        "expected_snippet": "Corporate income tax is levied at a rate of 15% (15.825% including the 5.5% solidarity surcharge)."
    },
    {
        "id": "Q02",
        "jurisdiction": "Canada",
        "topic": "Corporate Taxation of Resident Corporations",
        "query": "How are resident corporations taxed in Canada and what is the federal corporate tax rate?",
        "ground_truth_criteria": ["Canada", "Corporate income tax", "resident in Canada"],
        "expected_snippet": "Corporations resident in Canada are subject to Canadian federal tax on their worldwide income."
    },
    {
        "id": "Q03",
        "jurisdiction": "France",
        "topic": "Territorial Principle of Corporate Taxation",
        "query": "What is the territorial principle of corporate taxation in France?",
        "ground_truth_criteria": ["France", "territorial principle", "Corporate tax"],
        "expected_snippet": "The taxation of French companies is based on a territorial principle. In general, profits earned by a French company are subject to French corporate tax only if they are derived from business activities conducted in France."
    },
    {
        "id": "Q04",
        "jurisdiction": "Ireland",
        "topic": "Corporation Tax Rate on Trading Income",
        "query": "What is the corporation tax rate on trading income in Ireland?",
        "ground_truth_criteria": ["Ireland", "Corporation tax", "12.5%"],
        "expected_snippet": "Trading income is subject to corporation tax at a rate of 12.5%."
    },
    {
        "id": "Q05",
        "jurisdiction": "United Kingdom",
        "topic": "Controlled Foreign Company (CFC) Rules",
        "query": "What are the controlled foreign company CFC rules in the United Kingdom?",
        "ground_truth_criteria": ["United Kingdom", "controlled foreign compan"],
        "expected_snippet": "The UK CFC rules are targeted anti-avoidance provisions designed to identify and tax artificially diverted UK profits."
    },
    {
        "id": "Q06",
        "jurisdiction": "Japan",
        "topic": "Transfer Pricing Documentation Requirements",
        "query": "What are the transfer pricing documentation rules in Japan?",
        "ground_truth_criteria": ["Japan", "transfer pricing"],
        "expected_snippet": "Japan's transfer pricing legislation is based on the arm's-length principle and OECD transfer pricing guidelines."
    },
    {
        "id": "Q07",
        "jurisdiction": "Singapore",
        "topic": "Pioneer Certificate Incentive",
        "query": "What tax incentives are provided under the Pioneer Certificate Incentive in Singapore?",
        "ground_truth_criteria": ["Singapore", "Pioneer"],
        "expected_snippet": "The Pioneer Certificate Incentive encourages companies to introduce pioneering activities and advanced technology into Singapore."
    },
    {
        "id": "Q08",
        "jurisdiction": "Brazil",
        "topic": "Interest on Net Equity (JCP)",
        "query": "What is the interest on net equity deduction in Brazil?",
        "ground_truth_criteria": ["Brazil", "interest on net equity"],
        "expected_snippet": "Brazilian companies may remunerate shareholders through the payment of interest on net equity (juros sobre o capital próprio, or JCP)."
    },
    {
        "id": "Q09",
        "jurisdiction": "Mexico",
        "topic": "Worldwide Income Taxation of Residents",
        "query": "How are Mexican resident corporations taxed on worldwide income?",
        "ground_truth_criteria": ["Mexico", "Corporate income tax", "worldwide"],
        "expected_snippet": "Corporations resident in Mexico are taxable on their worldwide income."
    },
    {
        "id": "Q10",
        "jurisdiction": "South Africa",
        "topic": "Basis of Corporate Taxation",
        "query": "What is the basis of corporate taxation in South Africa?",
        "ground_truth_criteria": ["South Africa", "residence-based", "Company tax"],
        "expected_snippet": "A residence-based tax system applies in South Africa. Under domestic law, South African residents are taxed on their worldwide income."
    },
    {
        "id": "Q11",
        "jurisdiction": "Netherlands",
        "topic": "Participation Exemption Requirements",
        "query": "What are the requirements for the participation exemption in the Netherlands?",
        "ground_truth_criteria": ["Netherlands", "participation exemption"],
        "expected_snippet": "Under the Dutch participation exemption, all corporate income tax-exempt benefits derived from qualifying shareholdings are fully exempt."
    },
    {
        "id": "Q12",
        "jurisdiction": "Sweden",
        "topic": "Aggregation of Business Income Sources",
        "query": "How is business income aggregated for corporate tax in Sweden?",
        "ground_truth_criteria": ["Sweden", "Corporate income tax", "one source"],
        "expected_snippet": "Income from all business activities is aggregated as one source of income for corporate tax purposes."
    },
    {
        "id": "Q13",
        "jurisdiction": "Austria",
        "topic": "Taxation of Resident and Foreign Companies",
        "query": "How are resident and foreign companies taxed in Austria?",
        "ground_truth_criteria": ["Austria", "Corporate income tax", "resident in Austria"],
        "expected_snippet": "In general, all companies resident in Austria and foreign companies operating in Austria are subject to corporate income tax."
    },
    {
        "id": "Q14",
        "jurisdiction": "Belgium",
        "topic": "Worldwide Income Principle for Resident Entities",
        "query": "How are resident companies taxed on worldwide income in Belgium?",
        "ground_truth_criteria": ["Belgium", "Corporate income tax", "resident companies"],
        "expected_snippet": "Resident companies are subject to tax on their annual worldwide income, including distributed and retained profits."
    },
    {
        "id": "Q15",
        "jurisdiction": "Finland",
        "topic": "Worldwide Corporate Taxation",
        "query": "Are resident companies in Finland taxed on their worldwide income?",
        "ground_truth_criteria": ["Finland", "Corporate income tax", "worldwide"],
        "expected_snippet": "Companies resident in Finland are taxed on their worldwide income."
    },
    {
        "id": "Q16",
        "jurisdiction": "India",
        "topic": "Domestic Company Definition",
        "query": "How is a domestic company defined for corporate tax in India?",
        "ground_truth_criteria": ["India", "Corporate income tax", "domestic company"],
        "expected_snippet": "A domestic company is defined for tax purposes as a company incorporated in India or an Indian business entity."
    },
    {
        "id": "Q17",
        "jurisdiction": "Germany",
        "topic": "Loss Carryforward and Minimum Taxation",
        "query": "What are the rules for loss carryforward and minimum taxation in Germany?",
        "ground_truth_criteria": ["Germany", "loss carryforward"],
        "expected_snippet": "Tax losses may be carried forward indefinitely. However, minimum taxation rules apply to the utilization of loss carryforwards."
    },
    {
        "id": "Q18",
        "jurisdiction": "United States",
        "topic": "Federal Corporate Income Tax Rate",
        "query": "What is the federal corporate income tax rate in the United States?",
        "ground_truth_criteria": ["United States", "21%"],
        "expected_snippet": "The federal corporate income tax rate in the United States is a flat 21%."
    },
    {
        "id": "Q19",
        "jurisdiction": "Singapore",
        "topic": "Development and Expansion Incentive (DEI)",
        "query": "What is the Development and Expansion Incentive in Singapore?",
        "ground_truth_criteria": ["Singapore", "Development and Expansion Incentive"],
        "expected_snippet": "The Development and Expansion Incentive (DEI) encourages companies to upgrade and expand their operations in Singapore."
    },
    {
        "id": "Q20",
        "jurisdiction": "Australia",
        "topic": "Dividend Imputation and Franking Credits",
        "query": "How do franking credits and the imputation system work for Australian dividends?",
        "ground_truth_criteria": ["Australia", "franked", "imputation credit"],
        "expected_snippet": "Dividends paid by Australian resident companies are franked with an imputation credit (franking credit) reflecting company tax paid."
    },
    {
        "id": "Q21",
        "jurisdiction": "Switzerland",
        "topic": "Federal Corporate Tax Rate At a Glance",
        "query": "What is the federal corporate tax rate in Switzerland at a glance?",
        "ground_truth_criteria": ["Switzerland", "Corporate Tax Rate (%)"],
        "expected_snippet": "Corporate tax rate in Switzerland includes a direct federal tax rate of 8.5% on profits."
    },
    {
        "id": "Q22",
        "jurisdiction": "Canada",
        "topic": "Branch Tax on Non-Resident Corporations",
        "query": "What is the branch tax rate applied to non-resident corporations in Canada?",
        "ground_truth_criteria": ["Canada", "branch tax"],
        "expected_snippet": "Canada imposes a branch tax of 25% on the after-tax profits of a Canadian branch of a non-resident corporation."
    },
    {
        "id": "Q23",
        "jurisdiction": "Spain",
        "topic": "Corporate Capital Gains Treatment",
        "query": "How are corporate capital gains treated in Spain?",
        "ground_truth_criteria": ["Spain", "capital gains", "25%"],
        "expected_snippet": "In Spain, capital gains are included in taxable income and taxed at the general corporate income tax rate of 25%."
    },
    {
        "id": "Q24",
        "jurisdiction": "Luxembourg",
        "topic": "Corporate Income Tax Standard Rate",
        "query": "What is the general corporate income tax rate in Luxembourg?",
        "ground_truth_criteria": ["Luxembourg", "17%"],
        "expected_snippet": "Corporate income tax (impôt sur le revenu des collectivités - IRC) is levied at 17% for companies whose taxable income exceeds EUR 200,000."
    },
    {
        "id": "Q25",
        "jurisdiction": "Denmark",
        "topic": "Withholding Tax Rates Between Related Parties",
        "query": "What withholding tax rate applies between related parties in Denmark?",
        "ground_truth_criteria": ["Denmark", "22% rate", "related parties"],
        "expected_snippet": "The 22% rate applies to payments between related parties. The rate may be eliminated if certain conditions under EU directives or tax treaties are met."
    }
]


def check_chunk_relevance(chunk_text: str, criteria: list[str]) -> bool:
    """Check if a retrieved chunk satisfies the required ground truth criteria."""
    chunk_lower = chunk_text.lower()
    return all(criterion.lower() in chunk_lower for criterion in criteria)


def run_benchmark():
    print("=" * 80)
    print("EY AGENTIC TAX ASSISTANT - RAG RETRIEVAL BENCHMARK REPORT")
    print("=" * 80)
    print(f"ChromaDB Path      : {DB_PATH}")
    print(f"Collection Name    : {COLLECTION_NAME}")
    print(f"Embedding Model    : {EMBEDDING_MODEL}")
    print(f"Top-k Retrieval    : {RETRIEVAL_K}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Compute Device     : {device.upper()}")

    print("\n[1/4] Initializing HuggingFace Embeddings & ChromaDB Client...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device}
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K}
    )

    print("[2/4] Executing Warm-Up Query...")
    t_warm = time.perf_counter()
    _ = retriever.invoke("Warm-up corporate income tax query")
    warm_latency = (time.perf_counter() - t_warm) * 1000
    print(f"Warm-up completed in {warm_latency:.2f} ms")

    print(f"\n[3/4] Benchmarking {len(EVAL_DATASET)} Verifiable Test Queries...")
    results = []
    latencies = []
    prod_hits_at_1 = 0
    prod_hits_at_3 = 0
    prod_hits_at_5 = 0
    dedup_hits_at_1 = 0
    dedup_hits_at_3 = 0
    dedup_hits_at_5 = 0
    failed_queries = 0

    print("-" * 88)
    print(f"{'ID':<4} {'Jurisdiction':<14} {'Prod H@1':<9} {'Prod H@3':<9} {'Prod H@5':<9} {'Dedup H@3':<10} {'Dedup H@5':<10} {'Lat (ms)':<9} {'Uniq'}")
    print("-" * 88)

    for item in EVAL_DATASET:
        q_id = item["id"]
        jurisdiction = item["jurisdiction"]
        query = item["query"]
        criteria = item["ground_truth_criteria"]

        t_start = time.perf_counter()
        try:
            retrieved_docs = retriever.invoke(query)
            latency_ms = (time.perf_counter() - t_start) * 1000
            latencies.append(latency_ms)
        except Exception as e:
            failed_queries += 1
            print(f"{q_id:<4} {jurisdiction:<14} FAILED: {str(e)}")
            results.append({
                "id": q_id,
                "jurisdiction": jurisdiction,
                "query": query,
                "error": str(e),
                "prod_hit_at_1": False,
                "prod_hit_at_3": False,
                "prod_hit_at_5": False,
                "dedup_hit_at_1": False,
                "dedup_hit_at_3": False,
                "dedup_hit_at_5": False,
                "latency_ms": None
            })
            continue

        # 1. Evaluate Production (Raw) Retrieval
        prod_rank_hits = []
        unique_texts = set()
        for doc in retrieved_docs:
            text = doc.page_content
            unique_texts.add(text)
            is_relevant = check_chunk_relevance(text, criteria)
            prod_rank_hits.append(is_relevant)

        p_h1 = prod_rank_hits[0] if len(prod_rank_hits) > 0 else False
        p_h3 = any(prod_rank_hits[:3]) if len(prod_rank_hits) >= 3 else any(prod_rank_hits)
        p_h5 = any(prod_rank_hits[:5]) if len(prod_rank_hits) >= 5 else any(prod_rank_hits)

        if p_h1: prod_hits_at_1 += 1
        if p_h3: prod_hits_at_3 += 1
        if p_h5: prod_hits_at_5 += 1

        # 2. Evaluate Semantic Deduplicated Retrieval (Querying k=60 to pass ~30 duplicates)
        raw_dedup_docs = vectorstore.similarity_search(query, k=60)
        unique_dedup_docs = []
        seen = set()
        for doc in raw_dedup_docs:
            txt = doc.page_content.strip()
            if txt not in seen:
                seen.add(txt)
                unique_dedup_docs.append(doc)
                if len(unique_dedup_docs) == 5:
                    break

        dedup_rank_hits = [check_chunk_relevance(d.page_content, criteria) for d in unique_dedup_docs]
        d_h1 = dedup_rank_hits[0] if len(dedup_rank_hits) > 0 else False
        d_h3 = any(dedup_rank_hits[:3]) if len(dedup_rank_hits) >= 3 else any(dedup_rank_hits)
        d_h5 = any(dedup_rank_hits[:5]) if len(dedup_rank_hits) >= 5 else any(dedup_rank_hits)

        if d_h1: dedup_hits_at_1 += 1
        if d_h3: dedup_hits_at_3 += 1
        if d_h5: dedup_hits_at_5 += 1

        print(f"{q_id:<4} {jurisdiction:<14} {'YES' if p_h1 else 'NO':<9} {'YES' if p_h3 else 'NO':<9} {'YES' if p_h5 else 'NO':<9} {'YES' if d_h3 else 'NO':<10} {'YES' if d_h5 else 'NO':<10} {latency_ms:<9.2f} {len(unique_texts)}/5")

        results.append({
            "id": q_id,
            "jurisdiction": jurisdiction,
            "topic": item["topic"],
            "query": query,
            "ground_truth_criteria": criteria,
            "expected_snippet": item["expected_snippet"],
            "production_pipeline": {
                "retrieved_count": len(retrieved_docs),
                "unique_chunks_count": len(unique_texts),
                "hit_at_1": p_h1,
                "hit_at_3": p_h3,
                "hit_at_5": p_h5
            },
            "deduplicated_pipeline": {
                "unique_chunks_evaluated": len(unique_dedup_docs),
                "hit_at_1": d_h1,
                "hit_at_3": d_h3,
                "hit_at_5": d_h5
            },
            "latency_ms": round(latency_ms, 2),
            "top_snippet": retrieved_docs[0].page_content[:200] if retrieved_docs else None
        })

    print("-" * 88)

    # Summary Statistics
    total_eval = len(EVAL_DATASET)
    evaluated_successfully = total_eval - failed_queries

    p_h1_rate = (prod_hits_at_1 / total_eval) * 100 if total_eval else 0
    p_h3_rate = (prod_hits_at_3 / total_eval) * 100 if total_eval else 0
    p_h5_rate = (prod_hits_at_5 / total_eval) * 100 if total_eval else 0

    d_h1_rate = (dedup_hits_at_1 / total_eval) * 100 if total_eval else 0
    d_h3_rate = (dedup_hits_at_3 / total_eval) * 100 if total_eval else 0
    d_h5_rate = (dedup_hits_at_5 / total_eval) * 100 if total_eval else 0

    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    p50_latency = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
    min_latency = float(np.min(latencies)) if latencies else 0.0
    max_latency = float(np.max(latencies)) if latencies else 0.0

    summary = {
        "benchmark_metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_queries": total_eval,
            "successful_queries": evaluated_successfully,
            "failed_queries": failed_queries,
            "compute_device": device,
            "embedding_model": EMBEDDING_MODEL,
            "retrieval_k": RETRIEVAL_K
        },
        "production_retrieval_rates": {
            "hit_at_1_count": prod_hits_at_1,
            "hit_at_1_rate_percent": round(p_h1_rate, 2),
            "hit_at_3_count": prod_hits_at_3,
            "hit_at_3_rate_percent": round(p_h3_rate, 2),
            "hit_at_5_count": prod_hits_at_5,
            "hit_at_5_rate_percent": round(p_h5_rate, 2)
        },
        "deduplicated_retrieval_rates": {
            "hit_at_1_count": dedup_hits_at_1,
            "hit_at_1_rate_percent": round(d_h1_rate, 2),
            "hit_at_3_count": dedup_hits_at_3,
            "hit_at_3_rate_percent": round(d_h3_rate, 2),
            "hit_at_5_count": dedup_hits_at_5,
            "hit_at_5_rate_percent": round(d_h5_rate, 2)
        },
        "latency_ms": {
            "mean": round(mean_latency, 2),
            "p50": round(p50_latency, 2),
            "p95": round(p95_latency, 2),
            "min": round(min_latency, 2),
            "max": round(max_latency, 2)
        },
        "query_details": results
    }

    print("\n[4/4] BENCHMARK SUMMARY RESULTS:")
    print(f"Total Evaluated Queries           : {total_eval}")
    print(f"Failed Queries                    : {failed_queries}")
    print(f"Production Hit@1 Rate             : {p_h1_rate:.1f}% ({prod_hits_at_1}/{total_eval})")
    print(f"Production Hit@3 Rate             : {p_h3_rate:.1f}% ({prod_hits_at_3}/{total_eval})")
    print(f"Production Hit@5 Rate             : {p_h5_rate:.1f}% ({prod_hits_at_5}/{total_eval})")
    print(f"Deduplicated (True Unique) Hit@3  : {d_h3_rate:.1f}% ({dedup_hits_at_3}/{total_eval})")
    print(f"Deduplicated (True Unique) Hit@5  : {d_h5_rate:.1f}% ({dedup_hits_at_5}/{total_eval})")
    print(f"Average Retrieval Latency         : {mean_latency:.2f} ms")
    print(f"P50 (Median) Latency              : {p50_latency:.2f} ms")
    print(f"P95 Latency                       : {p95_latency:.2f} ms")
    print(f"Min / Max Latency                 : {min_latency:.2f} ms / {max_latency:.2f} ms")

    # Save JSON Report
    os.makedirs("./benchmarks", exist_ok=True)
    out_file = "./benchmarks/rag_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed JSON results saved to: {out_file}")

    return summary


if __name__ == "__main__":
    run_benchmark()
