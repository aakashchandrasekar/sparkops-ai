def recommend_resources(
    input_data_gb,
    oom_probability
):

    if oom_probability >= 0.75:

        return {
            "executor_count": 32,
            "executor_memory_gb": 96,
            "executor_cores": 4,
            "shuffle_partitions": 5000
        }

    elif oom_probability >= 0.40:

        return {
            "executor_count": 24,
            "executor_memory_gb": 64,
            "executor_cores": 4,
            "shuffle_partitions": 3500
        }

    return {
        "executor_count": 16,
        "executor_memory_gb": 64,
        "executor_cores": 4,
        "shuffle_partitions": 2500
    }