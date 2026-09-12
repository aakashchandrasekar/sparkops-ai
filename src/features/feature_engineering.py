import pandas as pd


def load_data(
    campaign_path: str,
    spark_job_path: str
):
    campaigns = pd.read_csv(campaign_path)
    jobs = pd.read_csv(spark_job_path)

    campaigns["start_date"] = pd.to_datetime(campaigns["start_date"])
    campaigns["end_date"] = pd.to_datetime(campaigns["end_date"])
    jobs["run_date"] = pd.to_datetime(jobs["run_date"])

    return campaigns, jobs


def create_features(campaigns, jobs):

    df = jobs.merge(
        campaigns[
            [
                "campaign_id",
                "campaign_type",
                "expected_customers",
                "expected_transactions",
                "priority"
            ]
        ],
        on="campaign_id",
        how="left"
    )

    df["is_campaign_day"] = df["campaign_id"].notna().astype(int)

    df["expected_transactions"] = (
        df["expected_transactions"].fillna(0)
    )

    df["expected_customers"] = (
        df["expected_customers"].fillna(0)
    )

    df["campaign_type"] = (
        df["campaign_type"]
        .fillna("NONE")
    )

    df["priority"] = (
        df["priority"]
        .fillna("NONE")
    )

    df["transaction_per_customer"] = (
        df["expected_transactions"]
        / df["expected_customers"].replace(0, 1)
    )

    return df


if __name__ == "__main__":

    campaigns, jobs = load_data(
        "data/sample_campaigns.csv",
        "data/sample_spark_jobs.csv"
    )

    features = create_features(campaigns, jobs)

    print(features.head())