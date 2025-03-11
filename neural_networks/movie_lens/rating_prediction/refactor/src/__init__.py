import mlflow

__all__ = ["data_preprocessing", "train", "evaluate", "utils"]


def init_mlflow_tracking(experiment_name):
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(experiment_name)
