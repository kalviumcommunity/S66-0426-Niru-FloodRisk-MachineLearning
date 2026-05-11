"""
Training module for the Flood Risk Prediction system.
Handles model training and saving.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from typing import Tuple
from . import config


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = config.N_ESTIMATORS,
    max_depth: int = config.RANDOM_FOREST_MAX_DEPTH,
    random_state: int = config.RANDOM_STATE
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier for flood prediction.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        n_estimators (int): Number of trees in the forest
        max_depth (int): Maximum depth of each tree
        random_state (int): Random seed for reproducibility
        
    Returns:
        RandomForestClassifier: Trained model
    """
    print(f"\nTraining Random Forest Classifier...")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  random_state: {random_state}")
    
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    print(f"✓ Model training completed!")
    print(f"  Model classes: {model.classes_}")
    
    return model


def get_feature_importance(
    model: RandomForestClassifier,
    feature_names: list,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Get feature importance scores from the trained model.
    
    Args:
        model (RandomForestClassifier): Trained model
        feature_names (list): Names of features
        top_n (int): Number of top features to return
        
    Returns:
        pd.DataFrame: Feature importance dataframe sorted by importance
    """
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print(f"\nTop {top_n} Most Important Features:")
    print(feature_importance.head(top_n).to_string(index=False))
    
    return feature_importance


def save_model(
    model: RandomForestClassifier,
    file_path: str = config.MODEL_PATH
) -> None:
    """
    Save the trained model to disk using joblib.
    
    Args:
        model (RandomForestClassifier): Trained model
        file_path (str): Path to save the model
    """
    joblib.dump(model, file_path)
    print(f"✓ Model saved to: {file_path}")


def train_and_save(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_path: str = config.MODEL_PATH
) -> Tuple[RandomForestClassifier, pd.DataFrame]:
    """
    Complete training pipeline: train model, get feature importance, and save.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        model_path (str): Path to save the model
        
    Returns:
        Tuple: Trained model and feature importance dataframe
    """
    print("\n" + "="*50)
    print("MODEL TRAINING PIPELINE")
    print("="*50)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Get feature importance
    feature_importance = get_feature_importance(model, X_train.columns.tolist())
    
    # Save model
    save_model(model, model_path)
    
    print("="*50)
    print("✓ Training pipeline completed successfully!\n")
    
    return model, feature_importance
