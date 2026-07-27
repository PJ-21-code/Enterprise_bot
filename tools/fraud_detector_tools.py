import numpy as np
import pandas as pd
from langchain_core.tools import tool

@tool
def calculate_variance(actual: list, budgeted: float) -> dict:
    """Calculates variance statistics for financial audit analysis."""
    actual_total= np.sum(actual)
    budgeted_total= budgeted*len(actual)
    diff= actual_total- budgeted_total
    percent= (diff/budgeted_total)*100 if budgeted_total!=0 else 0
    is_high_risk= abs(percent) >= 10

    return {
        "difference": round(diff,2),
        "percentage change": round(percent,2),
        "High risk flag": "Yes" if is_high_risk  else "No"
    }


@tool
def benford_finance_fraud_detector(transaction_amounts: list) ->dict:
    leading_digits = [int(str(abs(amt)).strip("0.")[0]) for amt in transaction_amounts if amt != 0]

    df = pd.DataFrame({"digit": leading_digits})
    counts = df["digit"].value_counts().sort_index()
    total = len(leading_digits)

    actual_proportion = (counts / total).to_dict()

    expected_proportion = {
      d: np.log10(1 + 1 / d) for d in range(1, 10)
     }

    discrepancy = sum(
      abs(actual_proportion.get(d, 0) - expected_proportion[d]) for d in range(1, 10)
     )

    is_suspicious = (discrepancy > 0.15)  

    return {
      "total_transactions_analyzed": total,
      "discrepancy_score": round(discrepancy, 4),
      "fraud_risk_flag": is_suspicious,
      "observed_distribution": actual_proportion,
    }
    

@tool
def second_fraud_detector():
    pass