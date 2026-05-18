# Project Overview

This project is a machine learning workflow for predicting flood risk from environmental conditions stored in CSV files.

## What The Project Does

- Loads raw flood data from `data/raw/flood_data.csv`.
- Cleans the data by handling missing values.
- Separates input features from the target column.
- Splits the dataset into training and testing sets.
- Scales numeric features using statistics learned only from the training data.
- Trains a Random Forest classification model.
- Evaluates the model using common classification metrics.
- Saves the trained model, scaler, processed data, feature importance, and evaluation report.
- Loads the saved model later for single-row or batch prediction.

## Input Features Used In The Main Pipeline

- `rainfall`
- `temperature`
- `humidity`
- `river_level`
- `soil_moisture`

## Target Used

- `flood_risk`

This is a binary classification target where `0` means no flood and `1` means flood risk.

## Concepts Used

- Supervised learning
- Binary classification
- Train-test split
- Stratified sampling
- Feature scaling
- Missing value handling
- Model persistence
- Batch inference
- Feature importance
- Baseline comparison
- Regression demo workflow
- Modular ML workflow design

## Algorithms And Models Used

### Main model

- `RandomForestClassifier`

This is the actual model used by the main training pipeline.

### Baseline model

- `DummyClassifier` with `most_frequent`

This predicts the majority class and is useful for checking whether the real model is better than a trivial baseline.

### Regression demo model

- `LinearRegression`
- `DummyRegressor` with `mean`

These are included as learning/demo utilities and are not part of the main flood classification pipeline.

## Preprocessing And Training Methods Used

### Data loading

- Uses `pandas.read_csv()` to load the dataset.

### Missing value handling

- Drops columns whose missing-value ratio is above the configured threshold.
- Fills remaining numeric missing values with the column mean.

### Feature-target split

- Selects the five flood-related feature columns as `X`.
- Selects `flood_risk` as `y`.

### Train-test split

- Uses `train_test_split()`.
- Uses `stratify=y` to preserve class balance.
- Uses `test_size=0.2`, which means 80 percent training and 20 percent testing.

### Feature scaling

- Uses `StandardScaler` in the main classification workflow.
- Fits the scaler only on training data to avoid data leakage.
- Applies the trained scaler to both train and test sets.

### Alternate preprocessing demo

- Uses `ColumnTransformer`.
- Uses `SimpleImputer` for numeric and categorical imputation.
- Uses `MinMaxScaler` for numeric scaling.
- Uses `OneHotEncoder` for categorical encoding.

This alternate preprocessing pipeline exists as a reusable example and is not required by the main flood model.

## Feature Engineering Utilities Included

- Interaction features such as `rainfall * humidity`
- Interaction features such as `temperature * river_level`
- Polynomial features such as squared terms
- Rolling mean features
- Rolling standard deviation features
- Min-max normalization for a selected column
- Manual feature selection helper

These utilities exist in the codebase for experimentation and learning, but the current main pipeline does not automatically apply them during training.

## Evaluation Metrics Used

### Main classification metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report

### Regression demo metrics

- RMSE
- MAE
- R2 score

## Output Files Produced By The Main Pipeline

- `data/processed/flood_data_processed.csv`
- `models/flood_prediction_model.pkl`
- `models/scaler.pkl`
- `reports/feature_importance.csv`
- `reports/evaluation_report.txt`

## Prediction Capabilities

The project supports:

- Predicting one sample from a Python dictionary
- Predicting many rows from a CSV file
- Returning class labels and probabilities
- Saving batch prediction results to a new CSV file

## Extra Learning Material In The Repository

The repository also contains many Jupyter notebooks that explain:

- Jupyter setup
- Data exploration
- Preprocessing
- Model training
- Model evaluation
- Prediction workflows
- Precision and recall
- F1 score
- Confusion matrix
- Logistic regression demo
- KNN demo
- Linear regression demo
- Bias-variance ideas
- Data leakage prevention

## Current Structure After Consolidation

The runnable code is now centered in one file:

- `main.py`

That single file now contains:

- configuration constants
- preprocessing helpers
- feature engineering helpers
- main classifier training logic
- evaluation helpers
- prediction helpers
- baseline classifier demo
- regression demo helper

## Main Goal Of The Project

The main goal is to show a complete beginner-friendly machine learning pipeline for flood risk prediction while also including extra educational utilities for experimentation and learning.
