import pyarrow.parquet as pq
import glob


TRAINING_DATA_PATH = "/tmp/training_features_v2"


def load_historical_data():
    files = glob.glob(f"{TRAINING_DATA_PATH}/*.parquet")

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {TRAINING_DATA_PATH}"
        )

    tables = [pq.read_table(file) for file in files]

    table = tables[0]

    if len(tables) > 1:
        import pyarrow as pa
        table = pa.concat_tables(tables)

    return table.to_pandas()


def calculate_resource_cost(config):
    """
    Simple resource-cost proxy.

    Higher executor count and executor memory
    means a larger Spark resource footprint.
    """

    return (
        config["executor_count"]
        * config["executor_memory_gb"]
    )


def build_successful_configurations(df):
    """
    Build a catalog of Spark configurations that
    have historically completed without OOM.
    """

    successful = df[df["oom"] == 0].copy()

    configurations = (
        successful
        .groupby(
            [
                "executor_count",
                "executor_memory_gb",
                "executor_cores",
            ]
        )
        .agg(
            successful_runs=("oom", "count"),
            max_successful_input_gb=("input_data_gb", "max"),
            avg_successful_input_gb=("input_data_gb", "mean"),
        )
        .reset_index()
    )

    configurations["resource_cost"] = configurations.apply(
        calculate_resource_cost,
        axis=1
    )

    return configurations


def recommend_resources(
    estimated_input_data_gb,
    target_oom_probability=0.20,
):
    """
    Recommend a configuration using historical
    successful Spark executions.

    The recommendation is considered validated only
    when historical successful workload covers the
    estimated workload.
    """

    df = load_historical_data()

    configurations = build_successful_configurations(df)

    # Only configurations with historical evidence
    # covering the estimated workload are considered.
    feasible = configurations[
        configurations["max_successful_input_gb"]
        >= estimated_input_data_gb
    ].copy()

    if feasible.empty:

        return {
            "recommendation_available": False,
            "reason": (
                "No historically successful Spark configuration "
                "has demonstrated capacity for the estimated workload."
            ),
            "executor_count": None,
            "executor_memory_gb": None,
            "executor_cores": None,
            "shuffle_partitions": None,
            "resource_cost": None,
            "max_successful_input_gb": None,
        }

    # Select the lowest resource footprint among
    # configurations that historically handled
    # the required workload.
    selected = feasible.sort_values(
        [
            "resource_cost",
            "executor_count",
            "executor_memory_gb",
        ]
    ).iloc[0]

    # Derive shuffle partitions from workload size.
    #
    # This is a heuristic rather than a learned value.
    shuffle_partitions = max(
        200,
        int(estimated_input_data_gb / 5)
    )

    return {
        "recommendation_available": True,
        "reason": (
            "Selected the lowest-resource configuration "
            "with historical successful workload coverage."
        ),
        "executor_count": int(selected["executor_count"]),
        "executor_memory_gb": int(selected["executor_memory_gb"]),
        "executor_cores": int(selected["executor_cores"]),
        "shuffle_partitions": shuffle_partitions,
        "resource_cost": int(selected["resource_cost"]),
        "max_successful_input_gb": float(
            selected["max_successful_input_gb"]
        ),
        "successful_runs": int(
            selected["successful_runs"]
        ),
    }


def print_recommendation(
    recommendation,
    estimated_input_data_gb,
):
    print("\n" + "=" * 70)
    print("SPARKOPS AI - HISTORICAL RESOURCE ADVISOR")
    print("=" * 70)

    print(
        f"\nEstimated Input Workload : "
        f"{estimated_input_data_gb:,.2f} GB"
    )

    if not recommendation["recommendation_available"]:

        print("\nRecommendation: NOT AVAILABLE")

        print(
            "\nReason:"
            f"\n{recommendation['reason']}"
        )

        print(
            "\nThe system will not extrapolate beyond "
            "historical evidence."
        )

        print("=" * 70)

        return

    print("\nValidated Historical Configuration")
    print("-" * 70)

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

    print(
        f"Resource Cost      : "
        f"{recommendation['resource_cost']}"
    )

    print(
        f"Successful Runs    : "
        f"{recommendation['successful_runs']}"
    )

    print(
        f"Max Successful Workload : "
        f"{recommendation['max_successful_input_gb']:,.2f} GB"
    )

    print(
        f"\nReason:"
        f"\n{recommendation['reason']}"
    )

    print("=" * 70)


if __name__ == "__main__":

    workload = 10351.6

    recommendation = recommend_resources(
        estimated_input_data_gb=workload
    )

    print_recommendation(
        recommendation,
        workload
    )
