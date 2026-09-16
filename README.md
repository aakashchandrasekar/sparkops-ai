# SparkOps AI

## AI-Powered Spark Resource Optimization & OOM Risk Prediction

SparkOps AI is a machine-learning-driven system that predicts Apache Spark Out-Of-Memory (OOM) risk before workload execution and uses historical execution evidence to determine whether a Spark resource configuration is sufficiently validated for the expected workload.

The project uses banking campaign workloads as a realistic business scenario where promotional campaigns can create sudden increases in transaction volume, data processing requirements, shuffle volume, and Spark resource consumption.

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

The system intentionally separates ML prediction from resource validation.

The ML model predicts OOM risk, while the resource advisor only generates a Spark configuration when historical execution evidence supports the expected workload.

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
                           |
                  +--------+--------+
                  |                 |
           Evidence exists     No evidence
                  |                 |
                  v                 v
        Generate Spark Config      STOP
                  |            Prevent unsupported
                  v              extrapolation
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

### 3. Verify this section

```bash
tail -10 README.md
You should see:
Actual Execution
        |
        v
Future Feedback
and nothing should be cut off.
4. Append the next section
cat >> README.md <<''

## Key Design Principle

SparkOps AI deliberately separates two decisions.

### 1. ML Risk Prediction

The machine-learning model estimates the probability that a workload will encounter an OOM condition.

The output is:

```text
P(OOM)
This answers:
How risky does this workload look under the evaluated baseline configuration?
2. Resource Validation
The system does not assume that the ML model can accurately simulate arbitrary Spark resource configurations.
Instead, it searches historical successful executions.
A configuration is considered eligible when:
historically successful workload
            >=
estimated workload
If no historical configuration covers the workload, SparkOps AI does not invent a resource recommendation.
________________________________________
Example: High-Risk Campaign
C003 - Diwali Cashback
Expected customers:
5,000,000
Expected transactions:
25,000,000
Estimated input workload:
26,539.70 GB
Estimation source:
campaign_history
ML prediction:
OOM Probability: 88%
Risk: HIGH
Historical validation:
No successful historical configuration
covers the estimated workload.
Decision:
DO NOT AUTOMATICALLY LAUNCH
This is intentional.
The system refuses to extrapolate beyond the historical evidence available to it.
________________________________________
Example: Validated Large Workload
A historical workload of:
10,351.60 GB
successfully completed using:
Executors          : 40
Executor Memory    : 64 GB
Executor Cores     : 5
Shuffle Partitions : 2070
SparkOps AI can therefore generate:
spark.executor.instances=40
spark.executor.memory=64g
spark.executor.cores=5
spark.sql.shuffle.partitions=2070
The configuration is based on historical successful workload evidence.
________________________________________


### 5. Append the ML section

```bash
cat >> README.md <<''

# Machine Learning

The current implementation uses a Random Forest classifier with a Scikit-learn preprocessing pipeline.

## Numeric Features

```text
input_data_gb
executor_count
executor_memory_gb
executor_cores
is_campaign_day
expected_customers
expected_transactions
transaction_per_customer
Categorical Features
campaign_type
priority
Target
oom
The model produces an OOM probability:
P(OOM)
________________________________________
Model Evaluation
The current synthetic demonstration dataset produced:
Metric	Result
Accuracy	91.50%
Precision	77.14%
Recall	75.00%
F1 Score	76.06%
ROC-AUC	96.30%
These metrics are based on synthetic demonstration data and should not be interpreted as production model performance.
________________________________________
Workload Estimation
SparkOps AI estimates expected input data before the Spark workload executes.
The estimator uses historical workload behavior:
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
For C003, historical campaign data estimates:
26,539.70 GB
________________________________________
Resource Optimization
The resource advisor builds a catalog of historically successful Spark configurations.
For each configuration it records:
•	Executor count 
•	Executor memory 
•	Executor cores 
•	Number of successful runs 
•	Maximum successful workload 
•	Average successful workload 
•	Resource cost proxy 
The system searches for configurations that historically handled the estimated workload.
Among eligible configurations, it selects the lowest resource-cost configuration.
________________________________________
Safety Guardrail
If no historically successful configuration covers the estimated workload:
recommendation_available = false
The Spark configuration generator then returns:
Configuration: NOT AVAILABLE
and:
Spark job should NOT be automatically launched.
This prevents unsupported infrastructure recommendations.
________________________________________


### 6. Append the final sections

```bash
cat >> README.md <<''

# Spark Configuration Generation

When a validated configuration exists, SparkOps AI generates two artifacts.

## spark-defaults.conf

```properties
spark.executor.instances=40
spark.executor.memory=64g
spark.executor.cores=5
spark.sql.shuffle.partitions=2070
spark_config.json
A machine-readable artifact containing:
•	Campaign 
•	Estimated workload 
•	OOM probability 
•	Risk 
•	Spark configuration 
•	Validation reason 
________________________________________
Project Structure
sparkops-ai/
|
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
|
├── data/
│   ├── sample_campaigns.csv
│   ├── sample_spark_jobs.csv
│   └── training_features.csv
|
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
|
├── notebooks/
├── tests/
├── config/
└── architecture/
________________________________________
Technology Stack
Data Engineering
•	Python 
•	Apache Spark 
•	PySpark 
•	AWS Glue 
•	Amazon S3 
•	Parquet 
Machine Learning
•	Pandas 
•	Scikit-learn 
•	Random Forest 
•	PyArrow 
•	Joblib 
AWS
•	Amazon S3 
•	AWS Glue 
•	AWS IAM 
•	AWS CloudShell 
Development
•	Git 
•	GitHub 
________________________________________
AWS Data Architecture
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
________________________________________
Current Pipeline
Campaign
    |
    v
Feature Engineering
    |
    v
Historical Workload Estimation
    |
    v
ML OOM Prediction
    |
    v
Historical Resource Validation
    |
    +----------------------+
    |                      |
    v                      v
Validated             Not Validated
    |                      |
    v                      v
Generate Config          STOP
    |                No unsupported
    v                 recommendation
Spark Configuration
________________________________________
Current Limitations
This is a portfolio/demo implementation.
Current limitations include:
1.	Training data is synthetic. 
2.	Historical resource configurations are limited. 
3.	The ML model should not be treated as a causal Spark resource simulator. 
4.	Some future workloads may fall outside historical coverage. 
5.	Resource cost is currently a simplified proxy. 
6.	Production deployment would require real Spark execution metrics. 
7.	More historical data is required for reliable large-workload recommendations. 
8.	The current workload estimator is based on historical relationships rather than a production forecasting model. 
________________________________________
Future Improvements
Potential future enhancements include:
•	Real-time Spark metrics ingestion 
•	AWS CloudWatch integration 
•	Automated feedback collection 
•	Model retraining 
•	Runtime prediction 
•	Cost-aware resource optimization 
•	AWS Glue job integration 
•	REST API 
•	Monitoring dashboard 
•	Configuration confidence scoring 
•	Workload similarity matching 
•	Model registry 
•	Automated rollback 
•	Production data integration 
•	Continuous model monitoring 
________________________________________
Portfolio Value
SparkOps AI demonstrates practical experience across:
Data Engineering
•	PySpark 
•	Data pipelines 
•	Feature engineering 
•	Parquet 
•	AWS S3 
•	AWS Glue 
Machine Learning
•	Classification 
•	Random Forest 
•	Model evaluation 
•	Feature preprocessing 
•	Probability-based risk prediction 
Cloud Engineering
•	AWS 
•	IAM 
•	S3 
•	Glue 
•	CloudShell 
Infrastructure Optimization
•	Spark resource configuration 
•	Executor sizing 
•	Memory allocation 
•	Shuffle partition configuration 
•	Historical resource validation 
ML Engineering
•	Model pipeline 
•	Prediction artifacts 
•	Configuration generation 
•	Safety guardrails 
________________________________________
Author
Built as a data engineering and machine learning portfolio project demonstrating how machine learning can be combined with historical infrastructure evidence to improve Apache Spark workload preparation and resource decisioning.


