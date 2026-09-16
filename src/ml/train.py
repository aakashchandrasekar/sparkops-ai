import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "/tmp/training_features_v2"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "oom_model.pkl")


# Features available BEFORE the Spark job runs
NUMERIC_FEATURES = [
    "input_data_gb",
    "executor_count",
    "executor_memory_gb",
    "executor_cores",
    "is_campaign_day",
    "expected_customers",
    "expected_transactions",
    "transaction_per_customer",
]

CATEGORICAL_FEATURES = [
    "campaign_type",
    "priority",
]

TARGET = "oom"


# ============================================================
# LOAD DATA
# ============================================================

def load_training_data():

    print("=" * 70)
    print("SPARKOPS AI - ML TRAINING")
    print("=" * 70)

    print("\nLoading Parquet data...")

    # Read all Parquet files in the directory
    import pyarrow.parquet as pq
    import glob

    files = glob.glob(
        os.path.join(INPUT_PATH, "*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {INPUT_PATH}"
        )

    print(f"Parquet files found: {len(files)}")

    tables = [
        pq.read_table(file)
        for file in files
    ]

    table = tables[0]

    for additional_table in tables[1:]:
        table = __import__("pyarrow").concat_tables(
            [table, additional_table]
        )

    df = table.to_pandas()

    print(f"Rows loaded: {len(df)}")

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    print("\nPreparing training data...")

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X = df[features].copy()
    y = df[TARGET].astype(int)

    # Make sure categorical columns don't contain nulls
    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].fillna("NONE")

    # Make sure numeric columns don't contain nulls
    for column in NUMERIC_FEATURES:
        X[column] = X[column].fillna(0)

    print("\nFeatures:")
    for feature in features:
        print(f"  - {feature}")

    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    return X, y


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    print("\nBuilding ML pipeline...")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    return model


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    df = load_training_data()

    X, y = prepare_data(df)

    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows : {len(X_test)}")

    model = build_model()

    print("\nTraining Random Forest...")

    model.fit(
        X_train,
        y_train,
    )

    print("Training completed.")


    # ========================================================
    # EVALUATION
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print("=" * 70)
    print(f"Model saved to: {MODEL_PATH}")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train_model()
