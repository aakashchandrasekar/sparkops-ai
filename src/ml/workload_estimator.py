import pyarrow.parquet as pq
import glob


TRAINING_DATA_PATH = "/tmp/training_features_v2"


def load_historical_data():

    files = glob.glob(
        f"{TRAINING_DATA_PATH}/*.parquet"
    )

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {TRAINING_DATA_PATH}"
        )

    tables = [
        pq.read_table(file)
        for file in files
    ]

    table = tables[0]

    if len(tables) > 1:
        import pyarrow as pa
        table = pa.concat_tables(tables)

    return table.to_pandas()


def estimate_input_data(
    campaign_id,
    campaign_type,
    expected_transactions,
):

    df = load_historical_data()

    # --------------------------------------------------------
    # 1. Campaign-specific history
    # --------------------------------------------------------

    campaign_history = df[
        (df["campaign_id"] == campaign_id)
        & (df["expected_transactions"] > 0)
        & (df["input_data_gb"] > 0)
    ]

    if len(campaign_history) >= 5:

        median_gb_per_transaction = (
            campaign_history["input_data_gb"]
            / campaign_history["expected_transactions"]
        ).median()

        estimated_gb = (
            expected_transactions
            * median_gb_per_transaction
        )

        return estimated_gb, "campaign_history"


    # --------------------------------------------------------
    # 2. Campaign-type history
    # --------------------------------------------------------

    type_history = df[
        (df["campaign_type"] == campaign_type)
        & (df["expected_transactions"] > 0)
        & (df["input_data_gb"] > 0)
    ]

    if len(type_history) >= 5:

        median_gb_per_transaction = (
            type_history["input_data_gb"]
            / type_history["expected_transactions"]
        ).median()

        estimated_gb = (
            expected_transactions
            * median_gb_per_transaction
        )

        return estimated_gb, "campaign_type_history"


    # --------------------------------------------------------
    # 3. Overall historical fallback
    # --------------------------------------------------------

    valid_history = df[
        (df["expected_transactions"] > 0)
        & (df["input_data_gb"] > 0)
    ]

    median_gb_per_transaction = (
        valid_history["input_data_gb"]
        / valid_history["expected_transactions"]
    ).median()

    estimated_gb = (
        expected_transactions
        * median_gb_per_transaction
    )

    return estimated_gb, "overall_history"


if __name__ == "__main__":

    estimated_gb, source = estimate_input_data(
        campaign_id="C003",
        campaign_type="CASHBACK",
        expected_transactions=25_000_000,
    )

    print("=" * 60)
    print("SPARKOPS AI - WORKLOAD ESTIMATOR")
    print("=" * 60)

    print(f"\nCampaign: C003")
    print(f"Expected transactions: 25,000,000")

    print(
        f"\nEstimated input data: "
        f"{estimated_gb:,.2f} GB"
    )

    print(
        f"Estimation source: {source}"
    )
