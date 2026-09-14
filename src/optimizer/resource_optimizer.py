def recommend_resources(oom_probability: float):

    if oom_probability >= 0.75:

        return {
            "risk": "HIGH",
            "executor_count": 32,
            "executor_memory_gb": 96,
            "executor_cores": 4,
            "shuffle_partitions": 5000,
        }

    elif oom_probability >= 0.40:

        return {
            "risk": "MEDIUM",
            "executor_count": 24,
            "executor_memory_gb": 64,
            "executor_cores": 4,
            "shuffle_partitions": 3500,
        }

    else:

        return {
            "risk": "LOW",
            "executor_count": 16,
            "executor_memory_gb": 64,
            "executor_cores": 4,
            "shuffle_partitions": 2500,
        }
