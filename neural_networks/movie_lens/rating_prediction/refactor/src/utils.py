def load_data(file_path):
    """Load data from a specified file path."""
    import pandas as pd

    return pd.read_csv(file_path)


def save_model(model, file_path):
    """Save the trained model to a specified file path."""
    import joblib

    joblib.dump(model, file_path)


def load_model(file_path):
    """Load a trained model from a specified file path."""
    import joblib

    return joblib.load(file_path)


def log_metrics(metrics):
    """Log metrics to MLflow."""
    import mlflow

    for key, value in metrics.items():
        mlflow.log_metric(key, value)
