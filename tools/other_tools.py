import numpy as np
import pandas as pd
from langchain_core.tools import tool

@tool
def calculate_variance():
    pass

@tool
def benford_finance_fraud_detector(transaction_amounts: list):
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