import argparse
import mlflow
import mlflow.pytorch
import torch
import yaml
from models import get_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataloaders import get_dataloaders


def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total
    return accuracy

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted")
    recall = recall_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")

    # Log metrics with MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration from YAML file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load model from MLflow
    mlflow.set_tracking_uri("../experiments")
    mlflow.set_experiment(config["logging"]["experiment_name"])

    model = mlflow.pytorch.load_model(config["evaluation"]["mlflow_model_uri"]).to(
        device
    )

    # Load dataset
    test_loader = get_dataloaders(
        config["dataset"], batch_size=config["batch_size"], train=False
    )

    # Evaluate model
    accuracy = evaluate(model, test_loader, device)
    mlflow.log_metric("test_accuracy", accuracy)

    print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config.yaml file"
    )

    args = parser.parse_args()
    main(args)
