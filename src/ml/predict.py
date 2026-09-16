import joblib
import pandas as pd

from src.optimizer.resource_optimizer import (
    recommend_resources,
    print_recommendation,
)

from src.ml.workload_estimator import estimate_input_data


MODEL_PATH = "models/oom_model.pkl"
CAMPAIGN_PATH = "data/sample_campaigns.csv"

FEATURES = [
    "input_data_gb",
    "executor_count",
    "executor_memory_gb",
    "executor_cores",
    "is_campaign_day",
    "expected_customers",
    "expected_transactions",
    "transaction_per_customer",
    "campaign_type",
    "priority",
]


def predict(campaign_id):

    print("=" * 70)
    print("SPARKOPS AI - PREDICTIVE RESOURCE ADVISOR")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Load ML model
    # ------------------------------------------------------------

    print("\nLoading ML model...")

    model = joblib.load(MODEL_PATH)

    # ------------------------------------------------------------
    # 2. Load campaign data
    # ------------------------------------------------------------

    campaigns = pd.read_csv(CAMPAIGN_PATH)

    campaign = campaigns[
        campaigns["campaign_id"] == campaign_id
    ]

    if campaign.empty:
        raise ValueError(
            f"Campaign {campaign_id} not found"
        )

    campaign = campaign.iloc[0]

    # ------------------------------------------------------------
    # 3. Extract campaign information
    # ------------------------------------------------------------

    expected_customers = int(
        campaign["expected_customers"]
    )

    expected_transactions = int(
        campaign["expected_transactions"]
    )

    transaction_per_customer = (
        expected_transactions
        / max(expected_customers, 1)
    )

    # ------------------------------------------------------------
    # 4. Estimate workload BEFORE the Spark job runs
    # ------------------------------------------------------------

    input_data_gb, estimation_source = (
        estimate_input_data(
            campaign_id=campaign_id,
            campaign_type=campaign["campaign_type"],
            expected_transactions=expected_transactions,
        )
    )

    # ------------------------------------------------------------
    # 5. Baseline Spark configuration
    #
    # This represents the configuration we want to
    # evaluate for OOM risk.
    # ------------------------------------------------------------

    executor_count = 16
    executor_memory_gb = 64
    executor_cores = 4

    # ------------------------------------------------------------
    # 6. Build ML features
    # ------------------------------------------------------------

    features = pd.DataFrame(
        [
            {
                "input_data_gb": input_data_gb,
                "executor_count": executor_count,
                "executor_memory_gb": executor_memory_gb,
                "executor_cores": executor_cores,
                "is_campaign_day": 1,
                "expected_customers": expected_customers,
                "expected_transactions": expected_transactions,
                "transaction_per_customer": transaction_per_customer,
                "campaign_type": campaign["campaign_type"],
                "priority": campaign["priority"],
            }
        ]
    )

    features = features[FEATURES]

    # ------------------------------------------------------------
    # 7. Predict OOM probability
    # ------------------------------------------------------------

    probability = model.predict_proba(
        features
    )[0][1]

    # ------------------------------------------------------------
    # 8. Determine risk level
    # ------------------------------------------------------------

    if probability >= 0.75:
        risk = "HIGH"
    elif probability >= 0.40:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # ------------------------------------------------------------
    # 9. Ask historical optimizer for resource evidence
    # ------------------------------------------------------------

    recommendation = recommend_resources(
        estimated_input_data_gb=input_data_gb
    )

    # ------------------------------------------------------------
    # 10. Display prediction
    # ------------------------------------------------------------

    print(
        f"\nCampaign              : "
        f"{campaign['campaign_name']}"
    )

    print(
        f"Campaign ID           : "
        f"{campaign_id}"
    )

    print(
        f"\nExpected Customers    : "
        f"{expected_customers:,.0f}"
    )

    print(
        f"Expected Transactions : "
        f"{expected_transactions:,.0f}"
    )

    print(
        f"Estimated Input       : "
        f"{input_data_gb:,.2f} GB"
    )

    print(
        f"Estimation Source     : "
        f"{estimation_source}"
    )

    print(
        f"\nOOM Probability       : "
        f"{probability * 100:.2f}%"
    )

    print(
        f"Risk                  : "
        f"{risk}"
    )

    # ------------------------------------------------------------
    # 11. Display historical resource recommendation
    # ------------------------------------------------------------

    print_recommendation(
        recommendation,
        input_data_gb
    )


if __name__ == "__main__":

    predict("C003")
