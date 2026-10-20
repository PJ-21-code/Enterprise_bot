
from decimal import Decimal, InvalidOperation
from collections import Counter
from datetime import timedelta

import numpy as np
import pandas as pd
from langchain_core.tools import tool


@tool
def calculate_variance(actual: list[float],budgeted_per_transaction: float) -> dict:
    """Compare actual transaction amounts against a budget per transaction.
    Returns total actual, total budget, difference, percentage change,
    and a high-risk flag when absolute variance is at least 10 percent.
    """
    if not actual:
        return {
            "error": "No actual transaction amounts provided."
        }

    try:
        amounts = [float(x) for x in actual]
        budget = float(budgeted_per_transaction)
    except (TypeError, ValueError):
        return {"error": "Amounts and budget must be numeric."}

    if not np.isfinite(budget) or not all(np.isfinite(x) for x in amounts):
        return {"error": "Amounts and budget must be finite numbers."}

    actual_total = sum(amounts)
    budget_total = budget * len(amounts)
    difference = actual_total - budget_total

    percentage = (
        difference / abs(budget_total) * 100
        if budget_total != 0
        else None
    )

    high_risk = (
        abs(percentage) >= 10
        if percentage is not None
        else None
    )

    return {
        "transaction_count": len(amounts),
        "actual_total": round(actual_total, 2),
        "budget_total": round(budget_total, 2),
        "difference": round(difference, 2),
        "percentage_change": (
            round(percentage, 2)
            if percentage is not None else None
        ),
        "high_risk_flag": (
            "Yes" if high_risk else "No"
        ) if high_risk is not None else "Cannot determine"
    }


@tool
def benford_finance_fraud_detector(transaction_amounts: list[float]) -> dict:
    """Analyze the first significant digit of financial amounts against
    Benford's Law. Returns observed and expected distributions and a
    heuristic anomaly flag. A flag is not proof of fraud.
    """
    if not transaction_amounts:
        return {"error": "No transaction amounts provided."}

    digit_counts = Counter()
    skipped_zero = 0

    for amount in transaction_amounts:
        try:
            value = Decimal(str(amount))
        except (InvalidOperation, ValueError):
            return {"error": f"Invalid transaction amount: {amount}"}

        if not value.is_finite():
            return {"error": "Amounts must be finite numbers."}

        value = abs(value)

        if value == 0:
            skipped_zero += 1
            continue

        # Decimal normalization handles small decimals and scientific notation.
        first_digit = int(value.normalize().as_tuple().digits[0])

        if 1 <= first_digit <= 9:
            digit_counts[first_digit] += 1

    total = sum(digit_counts.values())

    if total == 0:
        return {
            "error": "No non-zero transaction amounts to analyze.",
            "zero_amounts_skipped": skipped_zero
        }

    expected = {
        d: float(np.log10(1 + 1 / d))
        for d in range(1, 10)
    }

    observed = {
        d: digit_counts[d] / total
        for d in range(1, 10)
    }

    discrepancy = sum(
        abs(observed[d] - expected[d])
        for d in range(1, 10)
    )

    # This is a configurable heuristic, not a statistical significance test.
    threshold = 0.15

    return {
        "total_transactions_analyzed": total,
        "zero_amounts_skipped": skipped_zero,
        "discrepancy_score": round(discrepancy, 4),
        "heuristic_threshold": threshold,
        "anomaly_flag": discrepancy > threshold,
        "observed_distribution": {
            d: round(observed[d], 4) for d in range(1, 10)
        },
        "expected_distribution": {
            d: round(expected[d], 4) for d in range(1, 10)
        },
        "interpretation": (
            "Potential digit-distribution anomaly; investigate further."
            if discrepancy > threshold
            else "No anomaly detected by this heuristic."
        )
    }


@tool
def detect_split_transactions(transactions: list[dict],threshold: float = 10000.0,window_hours: int = 36) -> dict:
    """Flag groups of two or more transactions from the same vendor,
    each between 80% and 100% of the approval threshold, occurring
    within the specified time window. Input records must contain
    vendor_id, amount, and timestamp. This is a screening heuristic.
    """
    required = {"vendor_id", "amount", "timestamp"}

    if not transactions:
        return {
            "total_suspicious_clusters_found": 0,
            "details": [],
            "message": "No transactions provided."
        }

    if not np.isfinite(threshold) or threshold <= 0:
        return {"error": "Threshold must be a positive finite number."}

    if window_hours <= 0:
        return {"error": "window_hours must be positive."}

    df = pd.DataFrame(transactions)

    missing = required - set(df.columns)
    if missing:
        return {
            "error": f"Missing required fields: {sorted(missing)}"
        }

    try:
        df["amount"] = pd.to_numeric(df["amount"], errors="raise")
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="raise", utc=True
        )
    except (ValueError, TypeError, OverflowError):
        return {"error": "Invalid amount or timestamp in transactions."}

    if not np.isfinite(df["amount"].to_numpy(dtype=float)).all():
        return {"error": "Transaction amounts must be finite."}

    # Keep amounts in the range commonly examined for possible splitting.
    lower_bound = threshold * 0.80

    candidates = df[
        (df["amount"] >= lower_bound)
        & (df["amount"] <= threshold)
    ].copy()

    candidates = candidates.sort_values("timestamp")
    flagged_clusters = []
    window = timedelta(hours=window_hours)

    for vendor_id, group in candidates.groupby(
        "vendor_id", dropna=False, sort=False
    ):
        records = group.to_dict("records")
        start = 0

        # Form non-overlapping time windows, measured from the first
        # transaction in each window, not from the preceding transaction.
        while start < len(records):
            end = start + 1

            while (
                end < len(records)
                and records[end]["timestamp"]
                - records[start]["timestamp"] <= window
            ):
                end += 1

            cluster = records[start:end]

            if len(cluster) >= 2:
                flagged_clusters.append({
                    "vendor_id": (
                        None if pd.isna(vendor_id) else str(vendor_id)
                    ),
                    "transaction_count": len(cluster),
                    "total_amount": round(
                        sum(float(r["amount"]) for r in cluster), 2
                    ),
                    "individual_amounts": [
                        float(r["amount"]) for r in cluster
                    ],
                    "timestamps": [
                        r["timestamp"].isoformat() for r in cluster
                    ],
                    "approval_threshold": threshold,
                    "requires_review": True
                })

            start = end

    return {
        "candidate_transactions_analyzed": len(candidates),
        "total_suspicious_clusters_found": len(flagged_clusters),
        "details": flagged_clusters,
        "limitation": (
            "Flags are screening indicators, not proof of intentional "
            "approval-limit evasion. Review all vendor transactions."
        )
    }
