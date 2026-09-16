import json
import os

import joblib
import pandas as pd

from src.ml.workload_estimator import estimate_input_data
from src.optimizer.resource_optimizer import recommend_resources


MODEL_PATH = "models/oom_model.pkl"
CAMPAIGN_PATH = "data/sample_campaigns.csv"
OUTPUT_PATH = "predictions/C003_prediction.json"

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


def create_prediction(campaign_id):

    # Load model
    model = joblib.load(MODEL_PATH)

    # Load campaign
    campaigns = pd.read_csv(CAMPAIGN_PATH)

    campaign = campaigns[
        campaigns["campaign_id"] == campaign_id
    ]

    if campaign.empty:
        raise ValueError(
            f"Campaign {campaign_id} not found"
        )

    campaign = campaign.iloc[0]

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

    # Estimate workload
    input_data_gb, estimation_source = (
        estimate_input_data(
            campaign_id=campaign_id,
            campaign_type=campaign["campaign_type"],
            expected_transactions=expected_transactions,
        )
    )

    # Baseline configuration used for ML risk prediction
    features = pd.DataFrame(
        [
            {
                "input_data_gb": input_data_gb,
                "executor_count": 16,
                "executor_memory_gb": 64,
                "executor_cores": 4,
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

    # ML prediction
    probability = float(
        model.predict_proba(features)[0][1]
    )

    if probability >= 0.75:
        risk = "HIGH"
    elif probability >= 0.40:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # Historical resource recommendation
    recommendation = recommend_resources(
        estimated_input_data_gb=input_data_gb
    )

    # Build output
    prediction = {
        "campaign_id": campaign_id,
        "campaign_name": campaign["campaign_name"],
        "expected_customers": expected_customers,
        "expected_transactions": expected_transactions,
        "estimated_input_data_gb": round(
            input_data_gb, 2
        ),
        "estimation_source": estimation_source,
        "oom_probability": round(
            probability, 4
        ),
        "risk": risk,
        "recommendation_available": recommendation[
            "recommendation_available"
        ],
        "recommendation_reason": recommendation[
            "reason"
        ],
        "recommended_executor_count": recommendation[
            "executor_count"
        ],
        "recommended_executor_memory_gb": recommendation[
            "executor_memory_gb"
        ],
        "recommended_executor_cores": recommendation[
            "executor_cores"
        ],
        "recommended_shuffle_partitions": recommendation[
            "shuffle_partitions"
        ],
        "historical_max_successful_input_gb": recommendation[
            "max_successful_input_gb"
        ],
        "historical_successful_runs": recommendation.get(
            "successful_runs"
        ),
    }

    return prediction


def main():

    prediction = create_prediction("C003")

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w"
    ) as file:

        json.dump(
            prediction,
            file,
            indent=2
        )

    print("=" * 70)
    print("SPARKOPS AI - PREDICTION ARTIFACT")
    print("=" * 70)

    print(
        json.dumps(
            prediction,
            indent=2
        )
    )

    print("=" * 70)
    print(
        f"Prediction saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
