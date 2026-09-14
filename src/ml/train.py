import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


DATA_PATH = "data/training_features.csv"
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


def train_model():

    print("Loading training data...")

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df["oom"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print("Training model...")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nModel Accuracy:")
    print(accuracy_score(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
