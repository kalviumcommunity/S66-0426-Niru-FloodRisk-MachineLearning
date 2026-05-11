"""
Evaluation module for the Flood Risk Prediction system.
Handles model evaluation and performance metrics.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from typing import Dict, Tuple


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true (pd.Series): True labels
        y_pred (np.ndarray): Predicted labels
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    return metrics


def print_metrics(
    metrics: Dict[str, float],
    set_name: str = "Test Set"
) -> None:
    """
    Print metrics in a readable format.
    
    Args:
        metrics (Dict[str, float]): Metrics dictionary
        set_name (str): Name of the dataset (e.g., "Test Set", "Training Set")
    """
    print(f"\n{set_name} Metrics:")
    print("-" * 40)
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name.upper():12} : {metric_value:.4f}")
    print("-" * 40)


def get_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray
) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        y_true (pd.Series): True labels
        y_pred (np.ndarray): Predicted labels
        
    Returns:
        np.ndarray: Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def print_confusion_matrix(
    cm: np.ndarray,
    class_names: list = ["No Flood", "Flood"]
) -> None:
    """
    Print confusion matrix in a readable format.
    
    Args:
        cm (np.ndarray): Confusion matrix
        class_names (list): Names of the classes
    """
    print("\nConfusion Matrix:")
    print("-" * 40)
    print(f"{'':15} {class_names[0]:>10} {class_names[1]:>10}")
    print(f"{class_names[0]:15} {cm[0,0]:>10} {cm[0,1]:>10}")
    print(f"{class_names[1]:15} {cm[1,0]:>10} {cm[1,1]:>10}")
    print("-" * 40)


def print_classification_report(
    y_true: pd.Series,
    y_pred: np.ndarray
) -> None:
    """
    Print detailed classification report.
    
    Args:
        y_true (pd.Series): True labels
        y_pred (np.ndarray): Predicted labels
    """
    print("\nDetailed Classification Report:")
    print("-" * 40)
    print(classification_report(y_true, y_pred, target_names=["No Flood", "Flood"]))


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Dict:
    """
    Complete evaluation pipeline: calculate metrics for both train and test sets.
    
    Args:
        model: Trained model
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        y_train (pd.Series): Training target
        y_test (pd.Series): Testing target
        
    Returns:
        Dict: Dictionary containing all evaluation results
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    # Print metrics
    print_metrics(train_metrics, "Training Set")
    print_metrics(test_metrics, "Test Set")
    
    # Get and print confusion matrix
    cm_test = get_confusion_matrix(y_test, y_test_pred)
    print_confusion_matrix(cm_test)
    
    # Print classification report
    print_classification_report(y_test, y_test_pred)
    
    # Prepare results
    results = {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "confusion_matrix": cm_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
    }
    
    print("="*50)
    print("✓ Evaluation completed successfully!\n")
    
    return results


def save_evaluation_report(
    results: Dict,
    file_path: str = "reports/evaluation_report.txt"
) -> None:
    """
    Save evaluation report to a text file.
    
    Args:
        results (Dict): Evaluation results dictionary
        file_path (str): Path to save the report
    """
    with open(file_path, "w") as f:
        f.write("FLOOD PREDICTION MODEL - EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("TRAINING METRICS:\n")
        for metric, value in results["train_metrics"].items():
            f.write(f"  {metric}: {value:.4f}\n")
        
        f.write("\nTEST METRICS:\n")
        for metric, value in results["test_metrics"].items():
            f.write(f"  {metric}: {value:.4f}\n")
    
    print(f"✓ Evaluation report saved to: {file_path}")
