"""
Prediction module for the Flood Risk Prediction system.
Handles loading the model and making predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Union, List, Tuple
from . import config


def load_model(model_path: str = config.MODEL_PATH):
    """
    Load the trained model from disk.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        RandomForestClassifier: Loaded model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    try:
        model = joblib.load(model_path)
        print(f"✓ Model loaded from: {model_path}")
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_path}")


def load_scaler(scaler_path: str = config.SCALER_PATH):
    """
    Load the fitted scaler from disk.
    
    Args:
        scaler_path (str): Path to the saved scaler
        
    Returns:
        StandardScaler: Loaded scaler
        
    Raises:
        FileNotFoundError: If scaler file doesn't exist
    """
    try:
        scaler = joblib.load(scaler_path)
        print(f"✓ Scaler loaded from: {scaler_path}")
        return scaler
    except FileNotFoundError:
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")


def preprocess_prediction_data(
    data: pd.DataFrame,
    scaler,
    numerical_features: List[str] = None
) -> pd.DataFrame:
    """
    Preprocess new data for prediction.
    Uses the saved scaler (fitted on training data).
    
    Args:
        data (pd.DataFrame): New data for prediction
        scaler: Fitted scaler object
        numerical_features (List[str]): List of numerical columns to scale
        
    Returns:
        pd.DataFrame: Preprocessed data ready for prediction
    """
    if numerical_features is None:
        numerical_features = config.NUMERICAL_FEATURES
    
    data = data.copy()
    
    # Scale numerical features using the saved scaler
    data[numerical_features] = scaler.transform(data[numerical_features])
    
    print(f"✓ Prediction data preprocessed and scaled")
    return data


def predict(
    model,
    X: pd.DataFrame,
    scaler = None,
    numerical_features: List[str] = None
) -> np.ndarray:
    """
    Make predictions on new data.
    
    Args:
        model: Trained model
        X (pd.DataFrame): Features for prediction
        scaler: Fitted scaler (optional, if data needs scaling)
        numerical_features (List[str]): Numerical columns to scale
        
    Returns:
        np.ndarray: Predictions (0 = No Flood, 1 = Flood)
    """
    # Scale data if scaler is provided
    if scaler is not None:
        X = preprocess_prediction_data(X, scaler, numerical_features)
    
    # Make predictions
    predictions = model.predict(X)
    
    return predictions


def predict_with_probabilities(
    model,
    X: pd.DataFrame,
    scaler = None,
    numerical_features: List[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions and get probability scores.
    
    Args:
        model: Trained model
        X (pd.DataFrame): Features for prediction
        scaler: Fitted scaler (optional)
        numerical_features (List[str]): Numerical columns to scale
        
    Returns:
        Tuple: Predictions and probability scores
    """
    # Scale data if scaler is provided
    if scaler is not None:
        X = preprocess_prediction_data(X, scaler, numerical_features)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Get prediction probabilities
    probabilities = model.predict_proba(X)
    
    return predictions, probabilities


def predict_single_sample(
    model,
    sample: dict,
    scaler = None,
    numerical_features: List[str] = None
) -> Tuple[int, float]:
    """
    Make prediction for a single sample.
    
    Args:
        model: Trained model
        sample (dict): Dictionary with feature values
        scaler: Fitted scaler (optional)
        numerical_features (List[str]): Numerical columns to scale
        
    Returns:
        Tuple: (prediction, probability of flood)
    """
    # Convert dict to DataFrame
    X = pd.DataFrame([sample])
    
    # Get predictions with probabilities
    predictions, probabilities = predict_with_probabilities(
        model, X, scaler, numerical_features
    )
    
    prediction = predictions[0]
    flood_probability = probabilities[0][1]  # Probability of flood class
    
    return prediction, flood_probability


def batch_predict(
    model,
    data_path: str,
    scaler = None,
    numerical_features: List[str] = None
) -> pd.DataFrame:
    """
    Make predictions on a batch of data from a CSV file.
    
    Args:
        model: Trained model
        data_path (str): Path to CSV file with data
        scaler: Fitted scaler (optional)
        numerical_features (List[str]): Numerical columns to scale
        
    Returns:
        pd.DataFrame: Original data with predictions added
    """
    # Load data
    data = pd.read_csv(data_path)
    
    # Make predictions
    predictions, probabilities = predict_with_probabilities(
        model, data, scaler, numerical_features
    )
    
    # Add predictions to dataframe
    data["prediction"] = predictions
    data["flood_probability"] = probabilities[:, 1]
    data["no_flood_probability"] = probabilities[:, 0]
    
    print(f"✓ Batch predictions completed for {len(data)} samples")
    
    return data


def classify_prediction(prediction: int, probability: float) -> str:
    """
    Convert numerical prediction to human-readable classification.
    
    Args:
        prediction (int): Model prediction (0 or 1)
        probability (float): Probability score
        
    Returns:
        str: Human-readable classification
    """
    if prediction == 1:
        return f"FLOOD RISK (Probability: {probability:.2%})"
    else:
        return f"NO FLOOD (Probability: {probability:.2%})"
