import pandas as pd


CAMPAIGN_PATH = "data/sample_campaigns.csv"
SPARK_JOB_PATH = "data/sample_spark_jobs.csv"
OUTPUT_PATH = "data/training_features.csv"


def build_training_data():
    campaigns = pd.read_csv(CAMPAIGN_PATH)
    jobs = pd.read_csv(SPARK_JOB_PATH)

    df = jobs.merge(
        campaigns[
            [
                "campaign_id",
                "campaign_type",
                "expected_customers",
                "expected_transactions",
                "priority",
            ]
        ],
        on="campaign_id",
        how="left",
    )

    df["is_campaign_day"] = (
        df["campaign_id"].notna().astype(int)
    )

    df["expected_customers"] = (
        df["expected_customers"].fillna(0)
    )

    df["expected_transactions"] = (
        df["expected_transactions"].fillna(0)
    )

    df["transaction_per_customer"] = (
        df["expected_transactions"]
        / df["expected_customers"].replace(0, 1)
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Training dataset created: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    build_training_data()
