import argparse
import pathlib
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import numpy as np
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dataloaders import get_dataloaders


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }


def evaluate(model, test_loader, device):
    """Evaluate the model on the test set."""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for user_ids, movie_ids, ratings in test_loader:
            user_ids, movie_ids, ratings = (
                user_ids.to(device),
                movie_ids.to(device),
                ratings.to(device),
            )
            predictions = model(user_ids, movie_ids)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(ratings.cpu().numpy())

    metrics = calculate_metrics(all_targets, all_predictions)
    return metrics, all_predictions, all_targets


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration from YAML file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set up MLflow tracking
    tracking_path = pathlib.Path(__file__).absolute().parents[1] / "experiments"
    mlflow.set_tracking_uri(str(tracking_path))
    mlflow.set_experiment(config["logging"]["experiment_name"])

    # Load model from MLflow
    print(f"Loading model from: {config['evaluation']['mlflow_model_uri']}")
    model = mlflow.pytorch.load_model(config["evaluation"]["mlflow_model_uri"]).to(
        device
    )

    # Load test data
    print("Loading test data...")
    _, test_loader = get_dataloaders(
        name=config["dataset"]["name"],
        batch_size=config["dataset"]["batch_size"],
        test_size=config["dataset"]["test_size"],
        subset_ratio=config["dataset"]["subset_ratio"],
    )

    # Evaluate model
    print("Evaluating model...")
    metrics, predictions, targets = evaluate(model, test_loader, device)

    # Print metrics
    print("\nEvaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.upper()}: {metric_value:.4f}")

    # Log metrics to MLflow (optional)
    if args.log_metrics:
        with mlflow.start_run():
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            mlflow.log_param("model_uri", config["evaluation"]["mlflow_model_uri"])

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config.yaml file"
    )
    parser.add_argument(
        "--log_metrics", action="store_true", help="Whether to log metrics to MLflow"
    )

    args = parser.parse_args()
    main(args)
