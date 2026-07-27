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
def detect_split_transactions(transaction_df= pd.DataFrame, threshold :float= 10000.0, window_hours: int=36):
    """Detects potential split transactions designed to bypass approval limits.

    Parameters:
    - transactions_df: DataFrame with columns ['vendor_id', 'amount',
    'timestamp']
    - threshold: The company policy limit that requires approval (e.g., $10,000)
    - window_hours: Time window to look for clustered transactions (e.g., 48
    hours)
    """

    transaction_df['timestamp']= pd.to_datetime(transaction_df['timestamp'])
    lower_bound= threshold*0.80

    suspicious_slice= transaction_df[(transaction_df['amount']>= lower_bound) & (transaction_df['amount']<= threshold)].copy()

    flagged_clusters=[]

    for vendor_id, group in suspicious_slice.groupby("vendor_id"):

        group = group.sort_values("timestamp")

        group['time_diff']= group['timestamp'].diff().dt.total_seconds()/3600

        cluster= group[(group['time_diff']<= window_hours) | (group['time_diff'].isnull())]

        if(len(cluster)>=2):
            total_split_amount= cluster['amount'].sum()

            flagged_clusters.append({
                "vendor_id": vendor_id,
                "transaction_count": len(cluster),
                "total_amount": round(total_split_amount, 2),
                "individual_amounts": cluster["amount"].tolist(),
                "timestamps": cluster["timestamp"].astype(str).tolist(),
            })

    return{
        "total_suspicious_clusters_found": len(flagged_clusters),
        "details": flagged_clusters,
    }