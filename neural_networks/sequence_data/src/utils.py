import mlflow
import numpy as np
import os
import torch

from src.models import ModelProtocol, get_model

def load_learned_weights_from_mlflow(run_id: str) -> np.ndarray:
    """
    Loads the learned weights artifact from a specified MLflow run.

    Args:
        run_id (str): The MLflow Run ID from which to load the weights.

    Returns:
        np.ndarray: A NumPy array containing the learned weights.

    Raises:
        FileNotFoundError: If the 'learned_weights.txt' artifact is not found.
        Exception: For other MLflow-related errors.
    """
    client = mlflow.tracking.MlflowClient()
    try:
        # Create a temporary directory to download the artifact
        local_path = client.download_artifacts(run_id=run_id, path="learned_weights.txt")
        print(f"Downloaded artifact to: {local_path}")

        # Read the weights from the downloaded file
        weights = np.loadtxt(local_path)

        # Clean up the downloaded file (optional, but good practice)
        os.remove(local_path)

        return weights
    except Exception as e:
        raise Exception(f"Error loading weights from MLflow run {run_id}: {e}")

def load_pytorch_model_from_mlflow(run_id: str) -> ModelProtocol:
    """
    Loads a PyTorch model from a specified MLflow run.

    Args:
        run_id (str): The MLflow Run ID from which to load the model.

    Returns:
        ModelProtocol: The loaded PyTorch model.

    Raises:
        Exception: For MLflow-related errors during model loading.
    """
    try:
        # MLflow autologging saves the model in a 'model' artifact directory
        model_uri = f"runs:/{run_id}/model"
        loaded_model = mlflow.pytorch.load_model(model_uri)
        return loaded_model
    except Exception as e:
        raise Exception(f"Error loading PyTorch model from MLflow run {run_id}: {e}")
