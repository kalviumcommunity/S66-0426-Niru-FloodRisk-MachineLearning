"""
Data preprocessing module for the Flood Risk Prediction system.
Handles data loading, cleaning, and train-test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Tuple, Optional
from . import config


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data from the specified file path.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    try:
        data = pd.read_csv(file_path)
        print(f"✓ Data loaded successfully from {file_path}")
        print(f"  Shape: {data.shape}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values by dropping columns with too many missing values
    and filling remaining missing values with the mean.
    
    Args:
        data (pd.DataFrame): Input data
        
    Returns:
        pd.DataFrame: Data with missing values handled
    """
    # Check for missing values
    missing_ratio = data.isnull().sum() / len(data)
    
    # Drop columns with missing values above threshold
    cols_to_drop = missing_ratio[missing_ratio > config.MISSING_VALUE_THRESHOLD].index
    if len(cols_to_drop) > 0:
        print(f"⚠ Dropping columns with > {config.MISSING_VALUE_THRESHOLD*100}% missing values: {list(cols_to_drop)}")
        data = data.drop(columns=cols_to_drop)
    
    # Fill remaining missing values with mean (for numerical columns)
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if data[col].isnull().sum() > 0:
            mean_value = data[col].mean()
            data[col].fillna(mean_value, inplace=True)
            print(f"  Filled {col} missing values with mean: {mean_value:.2f}")
    
    print(f"✓ Missing values handled")
    return data


def split_features_target(
    data: pd.DataFrame,
    target_column: str = config.TARGET_COLUMN
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) and target (y) from the dataset.
    
    Args:
        data (pd.DataFrame): Complete dataset
        target_column (str): Name of the target column
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    print(f"✓ Features and target separated")
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    return X, y


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    IMPORTANT: This is done BEFORE scaling to prevent data leakage.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )
    
    print(f"✓ Train-test split completed ({(1-test_size)*100:.0f}-{test_size*100:.0f})")
    print(f"  Training set: {X_train.shape}")
    print(f"  Testing set: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numerical_features: Optional[list] = None,
    scaler_path: str = config.SCALER_PATH
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale numerical features using StandardScaler.
    IMPORTANT: Fit the scaler on training data ONLY, then transform both train and test data.
    This prevents data leakage.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        numerical_features (list): List of numerical columns to scale
        scaler_path (str): Path to save the scaler
        
    Returns:
        Tuple: Scaled X_train, Scaled X_test, Fitted scaler
    """
    if numerical_features is None:
        numerical_features = config.NUMERICAL_FEATURES
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # FIT scaler on training data ONLY
    scaler.fit(X_train[numerical_features])
    
    # TRANSFORM both train and test data
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_features] = scaler.transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
    
    # Save scaler for later use during prediction
    joblib.dump(scaler, scaler_path)
    
    print(f"✓ Features scaled successfully")
    print(f"  Scaler saved to: {scaler_path}")
    return X_train_scaled, X_test_scaled, scaler


def preprocess_data(
    file_path: str = config.RAW_DATA_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Complete preprocessing pipeline: load, clean, split, and scale data.
    
    Args:
        file_path (str): Path to raw data CSV
        
    Returns:
        Tuple: X_train_scaled, X_test_scaled, y_train, y_test
    """
    print("\n" + "="*50)
    print("DATA PREPROCESSING PIPELINE")
    print("="*50)
    
    # Load data
    data = load_data(file_path)
    
    # Handle missing values
    data = handle_missing_values(data)
    
    # Split features and target
    X, y = split_features_target(data)
    
    # Train-test split (BEFORE scaling to prevent data leakage)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    
    # Scale features (fit on training data only)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    print("="*50)
    print("✓ Preprocessing completed successfully!\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test
