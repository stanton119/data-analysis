"""
Evaluation script for multi-task learning models.

This script provides functionality to evaluate trained models with various metrics:
1. Accuracy (for binary classification tasks)
2. AUC (Area Under ROC Curve) for binary classification tasks
3. F1 Score for binary classification tasks
4. Average Precision (Area under Precision-Recall curve)
5. MSE (Mean Squared Error) for regression tasks

It supports both synthetic datasets and the UCI Census Income dataset.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import mlflow
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


def evaluate_model(
    model_path: str,
    data_config: Dict[str, Any],
    metrics: List[str] = ["accuracy", "auc", "f1", "average_precision"],
    output_path: Optional[str] = None,
    generate_plots: bool = False,
    plots_dir: Optional[str] = None,
):
    """
    Evaluate a trained model on test data with multiple metrics.
    
    Args:
        model_path (str): Path to the trained model checkpoint
        data_config (Dict): Configuration for the dataset
        metrics (List[str]): List of metrics to compute
        output_path (Optional[str]): Path to save evaluation results
        generate_plots (bool): Whether to generate ROC and PR curves
        plots_dir (Optional[str]): Directory to save plots
        
    Returns:
        Dict: Dictionary of evaluation metrics
    """
    # Load the model
    model = MultiTaskLightningModule.load_from_checkpoint(model_path)
    model.eval()
    
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
                task_metrics["accuracy"] = float((binary_preds == targets).mean())
            
            # Compute AUC
            if "auc" in metrics:
                task_metrics["auc"] = float(roc_auc_score(targets, predictions))
            
            # Compute F1 score
            if "f1" in metrics:
                binary_preds = (predictions > 0.5).astype(float)
                task_metrics["f1"] = float(f1_score(targets, binary_preds))
            
            # Compute average precision (area under PR curve)
            if "average_precision" in metrics:
                task_metrics["average_precision"] = float(average_precision_score(targets, predictions))
            
            # Compute confusion matrix
            if "confusion_matrix" in metrics:
                binary_preds = (predictions > 0.5).astype(float)
                cm = confusion_matrix(targets, binary_preds)
                task_metrics["confusion_matrix"] = cm.tolist()
            
            # Generate plots if requested
            if generate_plots and plots_dir:
                # ROC curve
                fpr, tpr, _ = roc_curve(targets, predictions)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'AUC = {task_metrics["auc"]:.4f}')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {task_name}')
                plt.legend()
                plt.savefig(os.path.join(plots_dir, f'{task_name}_roc_curve.png'))
                plt.close()
                
                # Precision-Recall curve
                precision, recall, _ = precision_recall_curve(targets, predictions)
                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, label=f'AP = {task_metrics["average_precision"]:.4f}')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall Curve - {task_name}')
                plt.legend()
                plt.savefig(os.path.join(plots_dir, f'{task_name}_pr_curve.png'))
                plt.close()
        
        # For regression tasks, compute regression metrics
        else:
            # Compute MSE
            if "mse" in metrics:
                task_metrics["mse"] = float(mean_squared_error(targets, predictions))
            
            # Compute RMSE
            if "rmse" in metrics:
                task_metrics["rmse"] = float(np.sqrt(mean_squared_error(targets, predictions)))
            
            # Compute MAE
            if "mae" in metrics:
                task_metrics["mae"] = float(np.mean(np.abs(targets - predictions)))
        
        # Store metrics for this task
        all_metrics[task_name] = task_metrics
    
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
    
    return all_metrics


def evaluate_multiple_models(
    model_dir: str,
    data_config: Dict[str, Any],
    metrics: List[str] = ["accuracy", "auc", "f1", "average_precision"],
    output_dir: Optional[str] = None,
    generate_plots: bool = False,
):
    """
    Evaluate multiple trained models in a directory.
    
    Args:
        model_dir (str): Directory containing model checkpoints
        data_config (Dict): Configuration for the dataset
        metrics (List[str]): List of metrics to compute
        output_dir (Optional[str]): Directory to save evaluation results
        generate_plots (bool): Whether to generate ROC and PR curves
        
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
):
    """
    Compare multiple models based on their evaluation metrics.
    
    Args:
        results (Dict): Dictionary mapping model names to evaluation metrics
        output_path (Optional[str]): Path to save comparison results
        
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
        )
    else:
        # Evaluate multiple models
        results = evaluate_multiple_models(
            model_dir=args.model_dir,
            data_config=data_config,
            metrics=metrics,
            output_dir=args.output_dir,
            generate_plots=args.generate_plots,
        )
        
        # Compare models
        if len(results) > 1:
            compare_models(
                results=results,
                output_path=os.path.join(args.output_dir, "comparison.json") if args.output_dir else None,
            )


if __name__ == "__main__":
    main()
