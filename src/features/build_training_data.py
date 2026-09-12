import pandas as pd

CAMPAIGN_PATH = "data/sample_campaigns.csv"
SPARK_JOB_PATH = "data/sample_spark_jobs.csv"
OUTPUT_PATH = "data/training_features.csv"


def build_training_data():

    # Load source data
    campaigns = pd.read_csv(CAMPAIGN_PATH)
    jobs = pd.read_csv(SPARK_JOB_PATH)

    # Join Spark jobs with campaign information
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

    # Identify campaign days
    df["is_campaign_day"] = df["campaign_id"].notna().astype(int)

    # Fill missing campaign values for normal days
    df["expected_customers"] = df["expected_customers"].fillna(0)
    df["expected_transactions"] = df["expected_transactions"].fillna(0)
    df["campaign_type"] = df["campaign_type"].fillna("NONE")
    df["priority"] = df["priority"].fillna("NONE")

    # Calculate transaction intensity
    df["transaction_per_customer"] = (
        df["expected_transactions"]
        / df["expected_customers"].replace(0, 1)
    )

    # Save training dataset
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Training dataset created: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    build_training_data()