"""
Feature engineering module for the Flood Risk Prediction system.
Contains functions for creating and transforming features.
"""

import pandas as pd
import numpy as np
from typing import List


def create_interaction_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between key variables.
    
    Example: rainfall * humidity might be more predictive than individual features.
    
    Args:
        data (pd.DataFrame): Input data with original features
        
    Returns:
        pd.DataFrame: Data with additional interaction features
    """
    data = data.copy()
    
    # Example interaction features
    if "rainfall" in data.columns and "humidity" in data.columns:
        data["rainfall_humidity_interaction"] = data["rainfall"] * data["humidity"]
    
    if "temperature" in data.columns and "river_level" in data.columns:
        data["temp_river_interaction"] = data["temperature"] * data["river_level"]
    
    print("✓ Interaction features created")
    return data


def create_polynomial_features(
    data: pd.DataFrame,
    columns: List[str],
    degree: int = 2
) -> pd.DataFrame:
    """
    Create polynomial features for specified columns.
    
    Args:
        data (pd.DataFrame): Input data
        columns (List[str]): Columns to create polynomial features for
        degree (int): Degree of polynomial (default: 2 for squared terms)
        
    Returns:
        pd.DataFrame: Data with polynomial features added
    """
    data = data.copy()
    
    for col in columns:
        if col in data.columns:
            for d in range(2, degree + 1):
                data[f"{col}_squared"] = data[col] ** d
    
    print(f"✓ Polynomial features created (degree={degree})")
    return data


def create_statistical_features(
    data: pd.DataFrame,
    window_size: int = 3
) -> pd.DataFrame:
    """
    Create statistical features like rolling mean and standard deviation.
    Note: Only use this if data has temporal ordering.
    
    Args:
        data (pd.DataFrame): Input data
        window_size (int): Window size for rolling statistics
        
    Returns:
        pd.DataFrame: Data with statistical features
    """
    data = data.copy()
    
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        # Rolling mean
        data[f"{col}_rolling_mean"] = data[col].rolling(
            window=window_size, 
            min_periods=1
        ).mean()
        
        # Rolling standard deviation
        data[f"{col}_rolling_std"] = data[col].rolling(
            window=window_size,
            min_periods=1
        ).std()
    
    print(f"✓ Statistical features created (window={window_size})")
    return data


def normalize_feature(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize a feature to 0-1 range (min-max scaling).
    
    Args:
        data (pd.DataFrame): Input data
        column (str): Column to normalize
        
    Returns:
        pd.DataFrame: Data with normalized column
    """
    data = data.copy()
    
    if column in data.columns:
        min_val = data[column].min()
        max_val = data[column].max()
        data[f"{column}_normalized"] = (data[column] - min_val) / (max_val - min_val)
    
    return data


def select_features(
    data: pd.DataFrame,
    feature_list: List[str]
) -> pd.DataFrame:
    """
    Select specific features from the dataset.
    Useful for feature selection after analysis.
    
    Args:
        data (pd.DataFrame): Input data
        feature_list (List[str]): List of features to keep
        
    Returns:
        pd.DataFrame: Data with selected features only
    """
    available_features = [f for f in feature_list if f in data.columns]
    missing_features = [f for f in feature_list if f not in data.columns]
    
    if missing_features:
        print(f"⚠ Warning: These features are not available: {missing_features}")
    
    print(f"✓ Selected {len(available_features)} features")
    return data[available_features]
