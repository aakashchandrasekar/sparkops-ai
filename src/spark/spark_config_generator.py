import json
import os


OUTPUT_DIR = "generated"


def generate_spark_config(prediction):
    """
    Convert a validated SparkOps AI recommendation
    into Spark configuration properties.

    Configuration is generated only when historical
    evidence is available.
    """

    if not prediction["recommendation_available"]:
        return {
            "config_available": False,
            "reason": (
                "No validated resource configuration is "
                "available for this workload."
            ),
            "spark_config": None,
        }

    config = {
        "spark.executor.instances": str(
            prediction["recommended_executor_count"]
        ),
        "spark.executor.memory": (
            f"{prediction['recommended_executor_memory_gb']}g"
        ),
        "spark.executor.cores": str(
            prediction["recommended_executor_cores"]
        ),
        "spark.sql.shuffle.partitions": str(
            prediction["recommended_shuffle_partitions"]
        ),
    }

    return {
        "config_available": True,
        "reason": (
            "Spark configuration generated from a "
            "historically validated resource recommendation."
        ),
        "spark_config": config,
    }


def save_spark_config(result, prediction):
    """
    Save the validated Spark configuration as:

    1. spark-defaults.conf
    2. spark_config.json
    """

    if not result["config_available"]:
        print(
            "\nNo configuration will be written "
            "because validation failed."
        )
        return

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # 1. Write spark-defaults.conf
    # ------------------------------------------------------------

    defaults_path = os.path.join(
        OUTPUT_DIR,
        "spark-defaults.conf"
    )

    with open(
        defaults_path,
        "w"
    ) as file:

        for key, value in result["spark_config"].items():
            file.write(
                f"{key}={value}\n"
            )

    # ------------------------------------------------------------
    # 2. Write machine-readable JSON
    # ------------------------------------------------------------

    json_path = os.path.join(
        OUTPUT_DIR,
        "spark_config.json"
    )

    config_artifact = {
        "campaign_id": prediction["campaign_id"],
        "campaign_name": prediction["campaign_name"],
        "estimated_input_data_gb": prediction[
            "estimated_input_data_gb"
        ],
        "oom_probability": prediction[
            "oom_probability"
        ],
        "risk": prediction["risk"],
        "configuration": result["spark_config"],
        "reason": result["reason"],
    }

    with open(
        json_path,
        "w"
    ) as file:

        json.dump(
            config_artifact,
            file,
            indent=2
        )

    print("\nConfiguration files created:")
    print(f"  {defaults_path}")
    print(f"  {json_path}")


def print_spark_config(result):

    print("\n" + "=" * 70)
    print("SPARKOPS AI - SPARK CONFIGURATION GENERATOR")
    print("=" * 70)

    if not result["config_available"]:

        print("\nConfiguration: NOT AVAILABLE")

        print(
            f"\nReason:\n{result['reason']}"
        )

        print(
            "\nSpark job should NOT be automatically launched."
        )

        print("=" * 70)

        return

    print("\nGenerated Spark Configuration")
    print("-" * 70)

    for key, value in result["spark_config"].items():
        print(
            f"{key} = {value}"
        )

    print("\nReason:")
    print(result["reason"])

    print("=" * 70)


if __name__ == "__main__":

    example_prediction = {
        "campaign_id": "C004",
        "campaign_name": "Loan Festival",
        "estimated_input_data_gb": 10351.6,
        "oom_probability": 0.30,
        "risk": "MEDIUM",
        "recommendation_available": True,
        "recommended_executor_count": 40,
        "recommended_executor_memory_gb": 64,
        "recommended_executor_cores": 5,
        "recommended_shuffle_partitions": 2070,
    }

    result = generate_spark_config(
        example_prediction
    )

    print_spark_config(result)

    save_spark_config(
        result,
        example_prediction
    )
