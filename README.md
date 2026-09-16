# SparkOps AI

## AI-Powered Spark Resource Optimization & OOM Risk Prediction

SparkOps AI is a portfolio project that combines machine learning with historical Apache Spark execution evidence to predict Out-Of-Memory (OOM) risk before workload execution and validate Spark resource configurations for expected workloads.

The project uses banking campaign workloads as a realistic business scenario where promotional campaigns can create sudden increases in transaction volume, input data, shuffle volume, execution time, and memory pressure.

> **Status:** Portfolio / demonstration implementation. Training data is synthetic, and resource recommendations are intentionally limited to configurations supported by historical evidence.

---

## Problem

Large banking campaigns can create sudden increases in:

- Customer activity
- Transaction volume
- Input data
- Shuffle data
- Spark execution time
- Memory pressure

A Spark configuration that works for normal workloads may fail when a major campaign generates significantly larger volumes.

SparkOps AI aims to move from:

> Run the Spark job and react when it fails

to:

> Predict workload risk and validate resources before execution.

---

## Solution

SparkOps AI combines:

1. Historical workload analysis
2. Feature engineering
3. Machine learning
4. OOM risk prediction
5. Historical resource validation
6. Spark configuration generation
7. Infrastructure safety guardrails

The system deliberately separates **ML risk prediction** from **resource validation**.

The ML model estimates OOM probability. The resource advisor only generates a Spark configuration when historical execution evidence supports the expected workload.

---

## Architecture

```text
                    Banking Campaign
                           |
                           v
                    Campaign Data
                           |
                           v
                  Workload Estimator
                           |
                           v
                  Expected Input Size
                           |
                           v
                   ML OOM Risk Model
                           |
                           v
                     OOM Probability
                           |
                           v
              Historical Resource Advisor
                    /                                  /                         Evidence exists          No evidence
               |                     |
               v                     v
     Generate Spark Config           STOP
               |              Prevent unsupported
               v                extrapolation
      spark-defaults.conf
               |
               v
           Spark Job
               |
               v
        Actual Execution
               |
               v
        Future Feedback
```

---

## Key Design Principle

### 1. ML Risk Prediction

The machine-learning model estimates the probability that a workload will encounter an OOM condition.

Output:

```text
P(OOM)
```

### 2. Resource Validation

The system does **not** assume that the ML model can accurately simulate arbitrary Spark resource configurations.

Instead, it searches historical successful executions.

A configuration is eligible when:

```text
historically successful workload >= estimated workload
```

If no historical configuration covers the workload, SparkOps AI does not invent a resource recommendation.

---

# Example: High-Risk Campaign

## C003 - Diwali Cashback

| Attribute | Value |
|---|---:|
| Expected customers | 5,000,000 |
| Expected transactions | 25,000,000 |
| Estimated input workload | 26,539.70 GB |
| Estimation source | campaign_history |
| OOM probability | 88% |
| Risk | HIGH |
| Historical validated configuration | Not available |

The system returns:

```text
Configuration: NOT AVAILABLE
Spark job should NOT be automatically launched.
```

This is intentional. The system refuses to extrapolate beyond the historical evidence available to it.

---

# Example: Validated Large Workload

A historical workload of:

```text
10,351.60 GB
```

successfully completed using:

```text
Executors       : 40
Executor Memory : 64 GB
Executor Cores  : 5
```

SparkOps AI can therefore generate:

```properties
spark.executor.instances=40
spark.executor.memory=64g
spark.executor.cores=5
spark.sql.shuffle.partitions=2070
```

The configuration is based on historical successful workload evidence.

---

# Machine Learning

The current implementation uses a **Random Forest classifier** with a Scikit-learn preprocessing pipeline.

### Numeric features

```text
input_data_gb
executor_count
executor_memory_gb
executor_cores
is_campaign_day
expected_customers
expected_transactions
transaction_per_customer
```

### Categorical features

```text
campaign_type
priority
```

### Target

```text
oom
```

---

## Model Evaluation

The current synthetic demonstration dataset produced:

| Metric | Result |
|---|---:|
| Accuracy | 91.50% |
| Precision | 77.14% |
| Recall | 75.00% |
| F1 Score | 76.06% |
| ROC-AUC | 96.30% |

> These metrics are based on synthetic demonstration data and should not be interpreted as production model performance.

---

# Workload Estimation

SparkOps AI estimates expected input data before the Spark workload executes.

The estimator uses historical workload behavior:

```text
Campaign history
       |
       v
Input data / transaction relationship
       |
       v
Historical median
       |
       v
Expected transactions
       |
       v
Estimated input workload
```

For C003, historical campaign data estimates:

```text
26,539.70 GB
```

The estimator first looks for sufficient campaign-specific history and can fall back to campaign-type or overall historical behavior.

---

# Resource Optimization

The resource advisor builds a catalog of historically successful Spark configurations.

For each configuration it records:

- Executor count
- Executor memory
- Executor cores
- Number of successful runs
- Maximum successful workload
- Average successful workload
- Resource cost proxy

The system searches for configurations that historically handled the estimated workload.

Among eligible configurations, it selects the lowest resource-cost configuration according to the current simplified cost proxy:

```text
executor_count × executor_memory_gb
```

This is a portfolio/demo heuristic rather than a production cloud-cost model.

---

# Safety Guardrail

If no historically successful configuration covers the estimated workload:

```text
recommendation_available = false
```

The Spark configuration generator returns:

```text
Configuration: NOT AVAILABLE
```

and:

```text
Spark job should NOT be automatically launched.
```

This prevents unsupported infrastructure recommendations.

---

# Spark Configuration Generation

When a validated configuration exists, SparkOps AI generates:

### `spark-defaults.conf`

```properties
spark.executor.instances=40
spark.executor.memory=64g
spark.executor.cores=5
spark.sql.shuffle.partitions=2070
```

### `spark_config.json`

A machine-readable artifact containing:

- Campaign
- Estimated workload
- OOM probability
- Risk
- Spark configuration
- Validation reason

---

# AWS Implementation

The AWS implementation uses Amazon S3 as the data lake layer and AWS Glue for feature engineering.

```text
s3://sparkops-ai-univ/

├── raw/
│   ├── campaigns/
│   └── spark_jobs/
│
├── processed/
│   └── training_features_v2/
│
├── models/
│
└── predictions/
    └── C003_prediction.json
```

### AWS pipeline

```text
Campaign Data
      |
      v
Amazon S3
      |
      v
AWS Glue
      |
      v
Feature Engineering
      |
      v
Processed Parquet
      |
      v
ML Training / Prediction
      |
      v
OOM Risk
      |
      v
Historical Resource Validation
      |
      +-------------------------+
      |                         |
      v                         v
Validated                  Not Validated
      |                         |
      v                         v
Generate Config              STOP
      |                  No unsupported
      v                   recommendation
Spark Configuration
```

---

# Project Structure

```text
sparkops-ai/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── sample_campaigns.csv
│   ├── sample_spark_jobs.csv
│   └── training_features.csv
│
├── src/
│   ├── ingestion/
│   ├── features/
│   │   ├── feature_engineering.py
│   │   └── build_training_data.py
│   ├── ml/
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── save_prediction.py
│   │   └── workload_estimator.py
│   ├── optimizer/
│   │   └── resource_optimizer.py
│   ├── spark/
│   │   ├── spark_config_generator.py
│   │   └── run_from_prediction.py
│   └── api/
│
├── notebooks/
├── tests/
├── config/
└── architecture/
```

---

# Technology Stack

### Data Engineering

- Python
- Apache Spark
- PySpark
- AWS Glue
- Amazon S3
- Parquet
- PyArrow

### Machine Learning

- Pandas
- Scikit-learn
- Random Forest
- Joblib

### AWS

- Amazon S3
- AWS Glue
- AWS IAM
- AWS CloudShell

### Development

- Git
- GitHub

---

# Pre-Run Modeling and Data Leakage

The model is designed as a **pre-run risk prediction** workflow.

Post-run metrics such as actual peak memory, actual shuffle volume, and execution time are useful for feedback and retraining, but they should not be used as direct features for a true pre-run prediction because those values are unavailable before execution.

This separation helps avoid data leakage and keeps the prediction workflow aligned with its intended use.

---

# Current Limitations

This is a portfolio/demo implementation.

1. Training data is synthetic.
2. Historical resource configurations are limited.
3. The ML model should not be treated as a causal Spark resource simulator.
4. Some future workloads may fall outside historical coverage.
5. Resource cost is currently a simplified proxy.
6. Production deployment would require real Spark execution metrics.
7. More historical data is required for reliable large-workload recommendations.
8. The workload estimator is based on historical relationships rather than a production forecasting model.

---

# Future Improvements

Potential enhancements include:

- Real-time Spark metrics ingestion
- AWS CloudWatch integration
- Automated feedback collection
- Model retraining
- Runtime prediction
- Cost-aware resource optimization
- AWS Glue job integration
- REST API
- Monitoring dashboard
- Configuration confidence scoring
- Workload similarity matching
- Model registry
- Automated rollback
- Production data integration
- Continuous model monitoring

---

# Portfolio Value

SparkOps AI demonstrates practical experience across:

### Data Engineering

PySpark, data pipelines, feature engineering, Parquet, Amazon S3, and AWS Glue.

### Machine Learning

Classification, Random Forest, model evaluation, feature preprocessing, and probability-based risk prediction.

### Cloud Engineering

AWS, IAM, S3, Glue, and CloudShell.

### Infrastructure Optimization

Spark resource configuration, executor sizing, memory allocation, shuffle partition configuration, and historical resource validation.

### ML Engineering

Model pipelines, prediction artifacts, configuration generation, and safety guardrails.

---

# Author

Built as a data engineering and machine learning portfolio project demonstrating how machine learning can be combined with historical infrastructure evidence to improve Apache Spark workload preparation and resource decisioning.
