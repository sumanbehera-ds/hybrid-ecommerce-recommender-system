import pandas as pd

try:
    import mlflow
except ModuleNotFoundError:
    print("mlflow not installed. Run: pip install mlflow")
    raise SystemExit(1)


EXPERIMENT_NAME = "ecommerce_recommender_system"


if __name__ == "__main__":
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError(f"Experiment not found: {EXPERIMENT_NAME}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )

    columns = [
        "tags.mlflow.runName",
        "metrics.hitrate_at_10",
        "metrics.test_mse",
        "metrics.train_loss",
        "metrics.final_train_loss",
        "metrics.final_train_mse",
        "params.model",
        "params.epochs",
        "params.batch_size"
    ]

    available_columns = [col for col in columns if col in runs.columns]

    comparison = runs[available_columns]

    print(comparison)
