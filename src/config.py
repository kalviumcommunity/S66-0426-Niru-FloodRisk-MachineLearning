"""
Configuration file for the Flood Risk Prediction system.
This file contains all hardcoded values and settings used throughout the project.
"""

# File paths
RAW_DATA_PATH = "data/raw/flood_data.csv"
PROCESSED_DATA_PATH = "data/processed/flood_data_processed.csv"
MODEL_PATH = "models/flood_prediction_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Model hyperparameters
RANDOM_STATE = 42
N_ESTIMATORS = 100  # Number of trees in RandomForest
TEST_SIZE = 0.2  # 80-20 train-test split
RANDOM_FOREST_MAX_DEPTH = 10

# Data processing
MISSING_VALUE_THRESHOLD = 0.5  # Drop columns with > 50% missing values
TEST_TRAIN_SPLIT_RANDOM_STATE = RANDOM_STATE

# Target column
TARGET_COLUMN = "flood_risk"

# Feature column names (adjust based on your data)
FEATURE_COLUMNS = [
    "rainfall",
    "temperature",
    "humidity",
    "river_level",
    "soil_moisture",
]

# Numerical columns to scale
NUMERICAL_FEATURES = [
    "rainfall",
    "temperature",
    "humidity",
    "river_level",
    "soil_moisture",
]

# Random seed for reproducibility
ML_SEED = RANDOM_STATE
