"""
Evaluation script for multi-task learning models.

This script provides functionality to evaluate trained models with various metrics:
1. Accuracy (for binary classification tasks)
2. AUC (Area Under ROC Curve) for binary classification tasks
3. F1 Score for binary classification tasks
4. Average Precision (Area under Precision-Recall curve)
5. MSE (Mean Squared Error) for regression tasks

It supports both synthetic datasets and the UCI Census Income dataset.
Results are logged to MLflow, extending the same run used during training.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import argparse
import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score, 
    f1_score,
    roc_curve,
    confusion_matrix,
    mean_squared_error
)

from src.data_sources import (
    load_uci_census_dataset,
    create_synthetic_dataset,
    generate_mmoe_synthetic_data,
)
from src.torch_datasets import create_train_test_dataloaders
from src.models import ModelProtocol, get_model
from src.train import MultiTaskLightningModule, prepare_uci_census_data, prepare_synthetic_data


def get_model_path_from_mlflow(
    run_id: str = None,
    experiment_name: str = None,
    run_name: str = None,
    model_name: str = "model",
):
    """
    Get the path to a model artifact from MLflow using run ID, experiment name, or run name.
    
    Args:
        run_id (str, optional): MLflow run ID
        experiment_name (str, optional): MLflow experiment name
        run_name (str, optional): MLflow run name
        model_name (str, optional): Name of the model artifact (default: "model")
        
    Returns:
        str: Path to the model artifact
        
    Raises:
        ValueError: If the run or model artifact cannot be found
    """
    client = MlflowClient()
    
    print(f"Looking for model with run_id={run_id}, experiment_name={experiment_name}, run_name={run_name}")
    
    # Find run by ID, experiment name + run name, or just run name
    if run_id:
        run = client.get_run(run_id)
    elif experiment_name and run_name:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'"
        )
        if not runs:
            raise ValueError(f"Run '{run_name}' not found in experiment '{experiment_name}'")
        run = runs[0]
    elif run_name:
        # Search across all experiments
        experiments = client.search_experiments()
        for experiment in experiments:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.mlflow.runName = '{run_name}'"
            )
            if runs:
                run = runs[0]
                break
        else:
            raise ValueError(f"Run '{run_name}' not found in any experiment")
    else:
        raise ValueError("Must provide run_id, experiment_name + run_name, or run_name")
    
    print(f"Found run with ID: {run.info.run_id}")
    print(f"Run artifact URI: {run.info.artifact_uri}")
    
    # Get the artifact URI for the model
    artifact_uri = run.info.artifact_uri
    
    # Check if the model exists in the artifacts
    artifacts = client.list_artifacts(run.info.run_id)
    print("Listing artifacts:")
    for artifact in artifacts:
        print(f"- {artifact.path} ({'dir' if artifact.is_dir else 'file'})")
    
    # First, check if there's a PyTorch model saved with mlflow.pytorch.log_model()
    pytorch_model_path = None
    for artifact in artifacts:
        if artifact.path == model_name and artifact.is_dir:
            # Found a PyTorch model directory
            pytorch_model_path = os.path.join(artifact_uri, model_name)
            print(f"Found PyTorch model at: {pytorch_model_path}")
            
            # Check if the model has the expected files
            model_artifacts = client.list_artifacts(run.info.run_id, model_name)
            for model_file in model_artifacts:
                print(f"  - {model_file.path}")
                
            # Look for MLmodel file which indicates a proper mlflow.pytorch.log_model() artifact
            if any(a.path == f"{model_name}/MLmodel" for a in model_artifacts):
                print(f"Found valid PyTorch model with MLmodel file")
                return pytorch_model_path
    
    # If no PyTorch model found, look for checkpoint files
    checkpoint_artifacts = []
    
    def list_artifacts_recursive(run_id, path=""):
        artifacts = client.list_artifacts(run_id, path)
        for artifact in artifacts:
            if artifact.is_dir:
                list_artifacts_recursive(run_id, artifact.path)
            elif artifact.path.endswith(".ckpt"):
                checkpoint_artifacts.append(artifact)
                print(f"Found checkpoint: {artifact.path}")
    
    list_artifacts_recursive(run.info.run_id)
    
    if checkpoint_artifacts:
        # Use the first checkpoint file
        checkpoint_path = os.path.join(artifact_uri, checkpoint_artifacts[0].path)
        print(f"Using checkpoint file: {checkpoint_path}")
        return checkpoint_path
    elif pytorch_model_path:
        # Fall back to the PyTorch model path even if it doesn't have MLmodel
        print(f"Falling back to PyTorch model directory: {pytorch_model_path}")
        return pytorch_model_path
    else:
        raise ValueError(f"No model artifacts or checkpoints found for run {run.info.run_id}")


def evaluate_model(
    model_path: str,
    data_config: Dict[str, Any],
    metrics: List[str] = ["accuracy", "auc", "f1", "average_precision"],
    output_path: Optional[str] = None,
    generate_plots: bool = False,
    plots_dir: Optional[str] = None,
    log_to_mlflow: bool = True,
):
    """
    Evaluate a trained model on test data with multiple metrics.
    
    Args:
        model_path (str): Path to the trained model checkpoint or MLflow model
        data_config (Dict): Configuration for the dataset
        metrics (List[str]): List of metrics to compute
        output_path (Optional[str]): Path to save evaluation results
        generate_plots (bool): Whether to generate ROC and PR curves
        plots_dir (Optional[str]): Directory to save plots
        log_to_mlflow (bool): Whether to log metrics and plots to MLflow
        
    Returns:
        Dict: Dictionary of evaluation metrics
    """
    # Load the model - handle both PyTorch Lightning checkpoints and MLflow PyTorch models
    try:
        # First try loading as a PyTorch Lightning checkpoint
        print(f"Attempting to load model from path: {model_path}")
        
        # Extract run information from MLflow to get model parameters
        client = MlflowClient()
        run_id = None
        
        # Try to extract run_id from the checkpoint path
        if "mlruns" in model_path:
            parts = model_path.split("mlruns")
            if len(parts) > 1:
                run_parts = parts[1].split("/")
                if len(run_parts) > 2:
                    run_id = run_parts[2]
                    print(f"Extracted run_id from path: {run_id}")
        
        # If we have a run_id, get the model parameters from MLflow
        model_params = None
        task_names = None
        task_types = None
        feature_dim = None
        model_name = None
        
        if run_id:
            try:
                run = client.get_run(run_id)
                params = run.data.params
                
                # Extract model parameters
                model_name = params.get("model_name")
                print(f"Model name from MLflow: {model_name}")
                
                # Extract task names and types
                if "task_names" in params:
                    task_names = params["task_names"].split(",")
                    print(f"Task names from MLflow: {task_names}")
                
                if "task_types" in params:
                    # Parse the string representation of the dictionary
                    task_types_str = params["task_types"]
                    # Remove curly braces and split by comma
                    task_types_items = task_types_str.strip("{}").split(",")
                    task_types = {}
                    for item in task_types_items:
                        if ":" in item:
                            key, value = item.split(":")
                            key = key.strip().strip("'\"")
                            value = value.strip().strip("'\"")
                            task_types[key] = value
                    print(f"Task types from MLflow: {task_types}")
                
                # Extract feature dimension
                if "feature_dim" in params:
                    feature_dim = int(params["feature_dim"])
                    print(f"Feature dimension from MLflow: {feature_dim}")
                
                # Extract model-specific parameters
                model_params = {}
                for key, value in params.items():
                    if key.startswith("model_"):
                        param_name = key[6:]  # Remove "model_" prefix
                        # Try to convert to appropriate type
                        if value.lower() == "true":
                            model_params[param_name] = True
                        elif value.lower() == "false":
                            model_params[param_name] = False
                        elif value.isdigit():
                            model_params[param_name] = int(value)
                        elif value.replace(".", "", 1).isdigit():
                            model_params[param_name] = float(value)
                        elif value.startswith('[') and value.endswith(']'):
                            # Handle list parameters like hidden_dims
                            try:
                                # Parse string representation of list
                                list_str = value.strip('[]')
                                if list_str:
                                    # Split by comma and convert each element
                                    elements = list_str.split(',')
                                    parsed_list = []
                                    for elem in elements:
                                        elem = elem.strip()
                                        if elem.isdigit():
                                            parsed_list.append(int(elem))
                                        elif elem.replace(".", "", 1).isdigit():
                                            parsed_list.append(float(elem))
                                        else:
                                            parsed_list.append(elem)
                                    model_params[param_name] = parsed_list
                                else:
                                    model_params[param_name] = []
                            except Exception as e:
                                print(f"Error parsing list parameter {param_name}: {e}")
                                model_params[param_name] = value
                        else:
                            model_params[param_name] = value
                
                print(f"Model parameters from MLflow: {model_params}")
            except Exception as e:
                print(f"Error getting run information from MLflow: {e}")
        
        # If we couldn't get the model parameters from MLflow, use the ones from data_config
        if task_names is None:
            task_names = data_config["task_names"]
        if task_types is None:
            task_types = data_config["task_types"]
        if feature_dim is None:
            feature_dim = data_config["feature_dim"]
        
        # Create the model instance
        if model_name and feature_dim and task_names and task_types:
            from src.models import get_model
            
            # Create the model instance
            model_instance = get_model(
                model_name=model_name,
                num_tabular_features=feature_dim,
                task_names=task_names,
                task_types=task_types,
                model_params=model_params,
            )
            
            # Create the Lightning module
            lightning_model = MultiTaskLightningModule(
                model=model_instance,
                learning_rate=1e-3,  # Default value, will be overridden by checkpoint
            )
            
            # Load the checkpoint
            if model_path.endswith('.ckpt'):
                print("Loading weights from PyTorch Lightning checkpoint")
                # Remove file:// prefix if present
                local_path = model_path
                if local_path.startswith('file://'):
                    local_path = local_path[7:]  # Remove 'file://' prefix
                print(f"Using local path: {local_path}")
                checkpoint = torch.load(local_path)
                lightning_model.load_state_dict(checkpoint["state_dict"])
            else:
                # Try loading as an MLflow PyTorch model
                print("Loading as MLflow PyTorch model")
                try:
                    # Check if it's a directory containing an MLflow model
                    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "MLmodel")):
                        loaded_model = mlflow.pytorch.load_model(model_path)
                        
                        # If it's a MultiTaskLightningModule, use it directly
                        if isinstance(loaded_model, MultiTaskLightningModule):
                            lightning_model = loaded_model
                        else:
                            # If it's just the inner model, update our lightning_model
                            lightning_model.model = loaded_model
                except Exception as e:
                    print(f"Error loading as MLflow model: {e}")
                    raise
            
            model = lightning_model
        else:
            raise ValueError("Could not get model parameters from MLflow or data_config")
            
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Could not load model from {model_path}: {e}")
    
    model.eval()
    print(f"Successfully loaded model: {type(model).__name__}")
    
    # Get the test dataloader
    test_dataloader = data_config["dataloaders"]["test"]
    
    # Initialize metrics storage
    all_metrics = {}
    all_predictions = {}
    all_targets = {}
    
    # Collect predictions and targets
    with torch.no_grad():
        for batch in test_dataloader:
            features = batch["features"]
            
            # Get model predictions
            outputs = model(features)
            
            # Store predictions and targets for each task
            for task_name in model.model.task_names:
                # Skip tasks that aren't in the outputs (for SingleTaskModel)
                if task_name not in outputs:
                    continue
                
                # Get predictions and targets
                output = outputs[task_name]
                target = batch[task_name]
                
                # Ensure outputs and targets have the same shape
                if output.shape != target.shape:
                    output = output.squeeze()
                
                # Initialize storage for this task if needed
                if task_name not in all_predictions:
                    all_predictions[task_name] = []
                    all_targets[task_name] = []
                
                # Store predictions and targets
                all_predictions[task_name].append(output.cpu())
                all_targets[task_name].append(target.cpu())
    
    # Concatenate predictions and targets for each task
    for task_name in all_predictions:
        all_predictions[task_name] = torch.cat(all_predictions[task_name]).numpy()
        all_targets[task_name] = torch.cat(all_targets[task_name]).numpy()
    
    # Create plots directory if needed
    if generate_plots and plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
    
    # Get run ID from the checkpoint if available
    run_id = None
    if log_to_mlflow:
        # Try to extract run_id from the checkpoint path
        # PyTorch Lightning checkpoints often have the run_id in their path
        checkpoint_path = Path(model_path)
        if "mlruns" in str(checkpoint_path):
            # Extract run_id from path like "mlruns/0/run_id/artifacts/model.ckpt"
            parts = str(checkpoint_path).split("mlruns")
            if len(parts) > 1:
                run_parts = parts[1].split("/")
                if len(run_parts) > 2:
                    run_id = run_parts[2]
    
    # Start or resume MLflow run if logging is enabled
    if log_to_mlflow:
        if run_id:
            # Resume existing run
            mlflow.start_run(run_id=run_id)
            print(f"Resuming MLflow run: {run_id}")
        else:
            # Start a new run
            mlflow.start_run()
            print("Starting new MLflow run for evaluation")
    
    try:
        # Compute metrics for each task
        for task_name in all_predictions:
            task_metrics = {}
            predictions = all_predictions[task_name]
            targets = all_targets[task_name]
            
            # Only compute classification metrics for binary tasks
            if model.model.task_types[task_name] == "binary":
                # Compute accuracy
                if "accuracy" in metrics:
                    binary_preds = (predictions > 0.5).astype(float)
                    accuracy = float((binary_preds == targets).mean())
                    task_metrics["accuracy"] = accuracy
                    if log_to_mlflow:
                        mlflow.log_metric(f"test_{task_name}_accuracy", accuracy)
                
                # Compute AUC
                if "auc" in metrics:
                    auc = float(roc_auc_score(targets, predictions))
                    task_metrics["auc"] = auc
                    if log_to_mlflow:
                        mlflow.log_metric(f"test_{task_name}_auc", auc)
                
                # Compute F1 score
                if "f1" in metrics:
                    binary_preds = (predictions > 0.5).astype(float)
                    f1 = float(f1_score(targets, binary_preds))
                    task_metrics["f1"] = f1
                    if log_to_mlflow:
                        mlflow.log_metric(f"test_{task_name}_f1", f1)
                
                # Compute average precision (area under PR curve)
                if "average_precision" in metrics:
                    ap = float(average_precision_score(targets, predictions))
                    task_metrics["average_precision"] = ap
                    if log_to_mlflow:
                        mlflow.log_metric(f"test_{task_name}_average_precision", ap)
                
                # Compute confusion matrix
                if "confusion_matrix" in metrics:
                    binary_preds = (predictions > 0.5).astype(float)
                    cm = confusion_matrix(targets, binary_preds)
                    task_metrics["confusion_matrix"] = cm.tolist()
                    
                    # Log confusion matrix as a figure to MLflow
                    if log_to_mlflow:
                        plt.figure(figsize=(8, 6))
                        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                        plt.title(f'Confusion Matrix - {task_name}')
                        plt.colorbar()
                        tick_marks = np.arange(2)
                        plt.xticks(tick_marks, ['Negative', 'Positive'])
                        plt.yticks(tick_marks, ['Negative', 'Positive'])
                        
                        # Add text annotations
                        thresh = cm.max() / 2.
                        for i in range(cm.shape[0]):
                            for j in range(cm.shape[1]):
                                plt.text(j, i, format(cm[i, j], 'd'),
                                        horizontalalignment="center",
                                        color="white" if cm[i, j] > thresh else "black")
                        
                        plt.ylabel('True label')
                        plt.xlabel('Predicted label')
                        plt.tight_layout()
                        
                        # Save to temp file and log to MLflow
                        cm_path = os.path.join(plots_dir if plots_dir else '.', f'{task_name}_confusion_matrix.png')
                        plt.savefig(cm_path)
                        if log_to_mlflow:
                            mlflow.log_artifact(cm_path, f"evaluation/plots")
                        plt.close()
                
                # Generate plots if requested
                if generate_plots:
                    # ROC curve
                    fpr, tpr, _ = roc_curve(targets, predictions)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, label=f'AUC = {task_metrics["auc"]:.4f}')
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curve - {task_name}')
                    plt.legend()
                    
                    # Save to file
                    roc_path = os.path.join(plots_dir if plots_dir else '.', f'{task_name}_roc_curve.png')
                    plt.savefig(roc_path)
                    if log_to_mlflow:
                        mlflow.log_artifact(roc_path, f"evaluation/plots")
                    plt.close()
                    
                    # Precision-Recall curve
                    precision, recall, _ = precision_recall_curve(targets, predictions)
                    plt.figure(figsize=(8, 6))
                    plt.plot(recall, precision, label=f'AP = {task_metrics["average_precision"]:.4f}')
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.title(f'Precision-Recall Curve - {task_name}')
                    plt.legend()
                    
                    # Save to file
                    pr_path = os.path.join(plots_dir if plots_dir else '.', f'{task_name}_pr_curve.png')
                    plt.savefig(pr_path)
                    if log_to_mlflow:
                        mlflow.log_artifact(pr_path, f"evaluation/plots")
                    plt.close()
            
            # For regression tasks, compute regression metrics
            else:
                # Compute MSE
                if "mse" in metrics:
                    mse = float(mean_squared_error(targets, predictions))
                    task_metrics["mse"] = mse
                    if log_to_mlflow:
                        mlflow.log_metric(f"test_{task_name}_mse", mse)
                
                # Compute RMSE
                if "rmse" in metrics:
                    rmse = float(np.sqrt(mean_squared_error(targets, predictions)))
                    task_metrics["rmse"] = rmse
                    if log_to_mlflow:
                        mlflow.log_metric(f"test_{task_name}_rmse", rmse)
                
                # Compute MAE
                if "mae" in metrics:
                    mae = float(np.mean(np.abs(targets - predictions)))
                    task_metrics["mae"] = mae
                    if log_to_mlflow:
                        mlflow.log_metric(f"test_{task_name}_mae", mae)
            
            # Store metrics for this task
            all_metrics[task_name] = task_metrics
        
        # Log the metrics summary as a JSON artifact
        if log_to_mlflow:
            metrics_json_path = os.path.join(plots_dir if plots_dir else '.', 'evaluation_metrics.json')
            with open(metrics_json_path, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            mlflow.log_artifact(metrics_json_path, "evaluation")
        
        # Print metrics
        print("\nEvaluation Results:")
        for task_name, task_metrics in all_metrics.items():
            print(f"\nTask: {task_name}")
            for metric_name, value in task_metrics.items():
                if metric_name != "confusion_matrix":  # Skip printing confusion matrix
                    print(f"  {metric_name}: {value:.4f}")
        
        # Save metrics to file if output path is provided
        if output_path:
            with open(output_path, "w") as f:
                json.dump(all_metrics, f, indent=2)
            print(f"\nResults saved to {output_path}")
    
    finally:
        # End MLflow run if we started one
        if log_to_mlflow:
            mlflow.end_run()
    
    return all_metrics


def evaluate_multiple_models(
    model_dir: str,
    data_config: Dict[str, Any],
    metrics: List[str] = ["accuracy", "auc", "f1", "average_precision"],
    output_dir: Optional[str] = None,
    generate_plots: bool = False,
    log_to_mlflow: bool = True,
):
    """
    Evaluate multiple trained models in a directory.
    
    Args:
        model_dir (str): Directory containing model checkpoints
        data_config (Dict): Configuration for the dataset
        metrics (List[str]): List of metrics to compute
        output_dir (Optional[str]): Directory to save evaluation results
        generate_plots (bool): Whether to generate ROC and PR curves
        log_to_mlflow (bool): Whether to log metrics and plots to MLflow
        
    Returns:
        Dict: Dictionary mapping model names to evaluation metrics
    """
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all checkpoint files
    checkpoint_files = list(Path(model_dir).glob("**/*.ckpt"))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {model_dir}")
        return {}
    
    # Evaluate each model
    all_results = {}
    for checkpoint_file in checkpoint_files:
        model_name = checkpoint_file.stem
        print(f"\nEvaluating model: {model_name}")
        
        # Create plots directory for this model
        plots_dir = os.path.join(output_dir, model_name, "plots") if output_dir and generate_plots else None
        
        # Evaluate model
        try:
            model_metrics = evaluate_model(
                model_path=str(checkpoint_file),
                data_config=data_config,
                metrics=metrics,
                output_path=os.path.join(output_dir, f"{model_name}_metrics.json") if output_dir else None,
                generate_plots=generate_plots,
                plots_dir=plots_dir,
                log_to_mlflow=log_to_mlflow,
            )
            
            all_results[model_name] = model_metrics
        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")
    
    # Save combined results
    if output_dir:
        with open(os.path.join(output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)
    
    return all_results


def compare_models(
    results: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Optional[str] = None,
    log_to_mlflow: bool = True,
):
    """
    Compare multiple models based on their evaluation metrics.
    
    Args:
        results (Dict): Dictionary mapping model names to evaluation metrics
        output_path (Optional[str]): Path to save comparison results
        log_to_mlflow (bool): Whether to log comparison results to MLflow
        
    Returns:
        Dict: Dictionary of comparison results
    """
    # Extract all tasks and metrics
    all_tasks = set()
    all_metrics = set()
    
    for model_results in results.values():
        for task_name, task_metrics in model_results.items():
            all_tasks.add(task_name)
            for metric_name in task_metrics:
                if metric_name != "confusion_matrix":  # Skip confusion matrix for comparison
                    all_metrics.add(metric_name)
    
    # Create comparison table
    comparison = {}
    for task_name in all_tasks:
        task_comparison = {}
        for metric_name in all_metrics:
            metric_values = {}
            for model_name, model_results in results.items():
                if task_name in model_results and metric_name in model_results[task_name]:
                    metric_values[model_name] = model_results[task_name][metric_name]
            
            if metric_values:
                # Find best model for this metric
                if metric_name in ["accuracy", "auc", "f1", "average_precision"]:
                    best_model = max(metric_values, key=metric_values.get)
                else:  # For metrics where lower is better (MSE, RMSE, MAE)
                    best_model = min(metric_values, key=metric_values.get)
                
                task_comparison[metric_name] = {
                    "values": metric_values,
                    "best_model": best_model,
                    "best_value": metric_values[best_model],
                }
        
        comparison[task_name] = task_comparison
    
    # Print comparison
    print("\nModel Comparison:")
    for task_name, task_comparison in comparison.items():
        print(f"\nTask: {task_name}")
        for metric_name, metric_comparison in task_comparison.items():
            print(f"  {metric_name}:")
            for model_name, value in metric_comparison["values"].items():
                is_best = model_name == metric_comparison["best_model"]
                best_marker = " (BEST)" if is_best else ""
                print(f"    {model_name}: {value:.4f}{best_marker}")
    
    # Save comparison to file if output path is provided
    if output_path:
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to {output_path}")
    
    # Log comparison to MLflow
    if log_to_mlflow:
        # Start a new run for the comparison
        with mlflow.start_run(run_name="Model Comparison"):
            # Log the comparison as a JSON artifact
            comparison_path = output_path or "model_comparison.json"
            if not os.path.exists(comparison_path):
                with open(comparison_path, "w") as f:
                    json.dump(comparison, f, indent=2)
            mlflow.log_artifact(comparison_path)
            
            # Create and log comparison plots
            for task_name, task_comparison in comparison.items():
                for metric_name, metric_comparison in task_comparison.items():
                    # Create bar chart
                    plt.figure(figsize=(10, 6))
                    models = list(metric_comparison["values"].keys())
                    values = list(metric_comparison["values"].values())
                    
                    # Sort by value
                    if metric_name in ["accuracy", "auc", "f1", "average_precision"]:
                        # Higher is better, sort descending
                        sorted_indices = np.argsort(values)[::-1]
                    else:
                        # Lower is better, sort ascending
                        sorted_indices = np.argsort(values)
                    
                    sorted_models = [models[i] for i in sorted_indices]
                    sorted_values = [values[i] for i in sorted_indices]
                    
                    # Create bar chart
                    bars = plt.bar(sorted_models, sorted_values)
                    
                    # Highlight best model
                    best_idx = sorted_models.index(metric_comparison["best_model"])
                    bars[best_idx].set_color('green')
                    
                    plt.title(f'{task_name} - {metric_name} Comparison')
                    plt.xlabel('Model')
                    plt.ylabel(metric_name)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    # Save and log
                    comparison_plot_path = f"{task_name}_{metric_name}_comparison.png"
                    plt.savefig(comparison_plot_path)
                    mlflow.log_artifact(comparison_plot_path, "comparison_plots")
                    plt.close()
    
    return comparison


def main():
    """Main function to parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate multi-task learning models.")
    
    # Model options
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model_path",
        type=str,
        help="Path to a single trained model checkpoint",
    )
    model_group.add_argument(
        "--model_dir",
        type=str,
        help="Directory containing multiple model checkpoints",
    )
    model_group.add_argument(
        "--mlflow_run_id",
        type=str,
        help="MLflow run ID to load model from",
    )
    model_group.add_argument(
        "--mlflow_run_name",
        type=str,
        help="MLflow run name to load model from",
    )
    
    # MLflow experiment options
    parser.add_argument(
        "--mlflow_experiment_name",
        type=str,
        default=None,
        help="MLflow experiment name (used with --mlflow_run_name)",
    )
    parser.add_argument(
        "--mlflow_model_name",
        type=str,
        default="model",
        help="Name of the model artifact in MLflow",
    )
    
    # Data options
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "uci_census"],
        help="Dataset to use for evaluation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    
    # Synthetic data options
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples for synthetic dataset",
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=20,
        help="Number of features for synthetic dataset",
    )
    parser.add_argument(
        "--task_correlation",
        type=float,
        default=0.5,
        help="Correlation between tasks for synthetic dataset",
    )
    parser.add_argument(
        "--use_mmoe_synthetic",
        action="store_true",
        help="Use MMoE synthetic data generation",
    )
    
    # Metrics options
    parser.add_argument(
        "--metrics",
        type=str,
        default="accuracy,auc,f1,average_precision",
        help="Comma-separated list of metrics to compute",
    )
    
    # Output options
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save evaluation results for a single model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results for multiple models",
    )
    
    # Visualization options
    parser.add_argument(
        "--generate_plots",
        action="store_true",
        help="Generate ROC and PR curves",
    )
    
    # MLflow options
    parser.add_argument(
        "--no_mlflow",
        action="store_true",
        help="Disable logging to MLflow",
    )
    parser.add_argument(
        "--mlflow_logging_run_id",
        type=str,
        default=None,
        help="Specific MLflow run ID to log evaluation results to (if not provided, will try to extract from checkpoint path)",
    )
    
    args = parser.parse_args()
    
    # Parse metrics
    metrics = args.metrics.split(",")
    
    # Prepare data
    if args.dataset == "uci_census":
        data_config = prepare_uci_census_data(batch_size=args.batch_size)
    else:  # synthetic
        data_config = prepare_synthetic_data(
            num_samples=args.num_samples,
            num_features=args.num_features,
            task_correlation=args.task_correlation,
            batch_size=args.batch_size,
            use_mmoe_synthetic=args.use_mmoe_synthetic,
        )
    
    # Set MLflow run ID for logging if provided
    if args.mlflow_logging_run_id:
        os.environ["MLFLOW_RUN_ID"] = args.mlflow_logging_run_id
    
    # Get model path from MLflow if specified
    if args.mlflow_run_id or args.mlflow_run_name:
        try:
            model_path = get_model_path_from_mlflow(
                run_id=args.mlflow_run_id,
                experiment_name=args.mlflow_experiment_name,
                run_name=args.mlflow_run_name,
                model_name=args.mlflow_model_name,
            )
            print(f"Loaded model path from MLflow: {model_path}")
            args.model_path = model_path
        except Exception as e:
            print(f"Error loading model from MLflow: {e}")
            return
    
    # Evaluate single model or multiple models
    if args.model_path:
        # Evaluate single model
        evaluate_model(
            model_path=args.model_path,
            data_config=data_config,
            metrics=metrics,
            output_path=args.output_path,
            generate_plots=args.generate_plots,
            plots_dir=os.path.join(os.path.dirname(args.output_path), "plots") if args.output_path else None,
            log_to_mlflow=not args.no_mlflow,
        )
    else:
        # Evaluate multiple models
        results = evaluate_multiple_models(
            model_dir=args.model_dir,
            data_config=data_config,
            metrics=metrics,
            output_dir=args.output_dir,
            generate_plots=args.generate_plots,
            log_to_mlflow=not args.no_mlflow,
        )
        
        # Compare models
        if len(results) > 1:
            compare_models(
                results=results,
                output_path=os.path.join(args.output_dir, "comparison.json") if args.output_dir else None,
                log_to_mlflow=not args.no_mlflow,
            )


if __name__ == "__main__":
    main()
