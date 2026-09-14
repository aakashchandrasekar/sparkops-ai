import argparse
import joblib
import pandas as pd

from src.optimizer.resource_optimizer import recommend_resources


MODEL_PATH = "models/oom_model.pkl"


FEATURES = [
    "input_data_gb",
    "output_data_gb",
    "executor_count",
    "executor_memory_gb",
    "executor_cores",
    "shuffle_gb",
    "peak_memory_gb",
    "is_campaign_day",
    "expected_customers",
    "expected_transactions",
    "transaction_per_customer",
]


def predict(campaign_id):

    model = joblib.load(MODEL_PATH)

    campaigns = pd.read_csv("data/sample_campaigns.csv")
    jobs = pd.read_csv("data/sample_spark_jobs.csv")

    campaign = campaigns[
        campaigns["campaign_id"] == campaign_id
    ]

    if campaign.empty:
        raise ValueError(
            f"Campaign {campaign_id} not found"
        )

    campaign = campaign.iloc[0]

    historical_jobs = jobs[
        jobs["campaign_id"] == campaign_id
    ]

    if historical_jobs.empty:

        # If we have no historical job for this campaign,
        # use historical average workload as a baseline.

        input_data_gb = jobs["input_data_gb"].mean()
        output_data_gb = jobs["output_data_gb"].mean()
        shuffle_gb = jobs["shuffle_gb"].mean()
        peak_memory_gb = jobs["peak_memory_gb"].mean()

        executor_count = 16
        executor_memory_gb = 64
        executor_cores = 4

    else:

        latest = historical_jobs.iloc[-1]

        input_data_gb = latest["input_data_gb"]
        output_data_gb = latest["output_data_gb"]
        shuffle_gb = latest["shuffle_gb"]
        peak_memory_gb = latest["peak_memory_gb"]

        executor_count = latest["executor_count"]
        executor_memory_gb = latest["executor_memory_gb"]
        executor_cores = latest["executor_cores"]

    expected_customers = campaign["expected_customers"]
    expected_transactions = campaign["expected_transactions"]

    transaction_per_customer = (
        expected_transactions
        / max(expected_customers, 1)
    )

    features = pd.DataFrame(
        [{
            "input_data_gb": input_data_gb,
            "output_data_gb": output_data_gb,
            "executor_count": executor_count,
            "executor_memory_gb": executor_memory_gb,
            "executor_cores": executor_cores,
            "shuffle_gb": shuffle_gb,
            "peak_memory_gb": peak_memory_gb,
            "is_campaign_day": 1,
            "expected_customers": expected_customers,
            "expected_transactions": expected_transactions,
            "transaction_per_customer": transaction_per_customer,
        }]
    )

    probability = model.predict_proba(features)[0][1]

    recommendation = recommend_resources(probability)

    print("\n" + "=" * 55)
    print("                 SPARKOPS AI")
    print("       Predictive Resource Advisor")
    print("=" * 55)

    print(f"\nCampaign: {campaign['campaign_name']}")
    print(f"Campaign ID: {campaign_id}")

    print(
        f"\nExpected Transactions: "
        f"{expected_transactions:,.0f}"
    )

    print(
        f"Expected Customers: "
        f"{expected_customers:,.0f}"
    )

    print(
        f"\nOOM Probability: "
        f"{probability * 100:.1f}%"
    )

    print(
        f"Risk: "
        f"{recommendation['risk']}"
    )

    print("\n" + "-" * 55)
    print("RECOMMENDED SPARK CONFIGURATION")
    print("-" * 55)

    print(
        f"Executors          : "
        f"{recommendation['executor_count']}"
    )

    print(
        f"Executor Memory    : "
        f"{recommendation['executor_memory_gb']} GB"
    )

    print(
        f"Executor Cores     : "
        f"{recommendation['executor_cores']}"
    )

    print(
        f"Shuffle Partitions : "
        f"{recommendation['shuffle_partitions']}"
    )

    print("=" * 55)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--campaign",
        required=True
    )

    args = parser.parse_args()

    predict(args.campaign)
