# SparkOps AI Architecture

```mermaid
flowchart TD
    A[Campaign / CRM Data] --> B[Amazon S3]
    C[Historical Spark Job Data] --> B

    B --> D[AWS Glue]
    D --> E[Feature Engineering]

    E --> F[ML Training]
    F --> G[OOM Risk Prediction]

    G --> H[Historical Workload Estimator]
    H --> I[Resource Optimizer]

    I --> J[Spark Configuration Generator]
    J --> K[Apache Spark Job]

    K --> L[Execution Metrics]
    L --> B

    B --> F
EOD
