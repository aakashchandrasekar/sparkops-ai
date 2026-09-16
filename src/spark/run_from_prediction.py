import json
import sys

from src.spark.spark_config_generator import (
    generate_spark_config,
    print_spark_config,
    save_spark_config,
)


def load_prediction(path):

    with open(path, "r") as file:
        return json.load(file)


def main():

    if len(sys.argv) != 2:

        print(
            "Usage: python3 -m src.spark.run_from_prediction "
            "<prediction_json>"
        )

        sys.exit(1)

    prediction_path = sys.argv[1]

    print("=" * 70)
    print("SPARKOPS AI - PREDICTION TO SPARK CONFIGURATION")
    print("=" * 70)

    print(
        f"\nLoading prediction: {prediction_path}"
    )

    prediction = load_prediction(
        prediction_path
    )

    print(
        f"Campaign: "
        f"{prediction['campaign_id']} - "
        f"{prediction['campaign_name']}"
    )

    print(
        f"Estimated workload: "
        f"{prediction['estimated_input_data_gb']:,.2f} GB"
    )

    print(
        f"OOM probability: "
        f"{prediction['oom_probability'] * 100:.2f}%"
    )

    print(
        f"Risk: {prediction['risk']}"
    )

    result = generate_spark_config(
        prediction
    )

    print_spark_config(
        result
    )

    # Only create configuration files when
    # historical validation exists.
    save_spark_config(
        result,
        prediction
    )


if __name__ == "__main__":
    main()
