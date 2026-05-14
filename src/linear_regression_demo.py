import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def evaluate_regression(y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


def train_linear_regression(df):

    # Example regression setup
    X = df[['Rainfall', 'Humidity', 'Temperature']]
    y = df['RiverLevel']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Baseline
    baseline = DummyRegressor(strategy='mean')

    baseline.fit(X_train, y_train)

    baseline_preds = baseline.predict(X_test)

    baseline_metrics = evaluate_regression(
        y_test,
        baseline_preds
    )

    # Linear Regression Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    model_metrics = evaluate_regression(
        y_test,
        predictions
    )

    print("\nBaseline Metrics")
    print(baseline_metrics)

    print("\nLinear Regression Metrics")
    print(model_metrics)

    return pipeline