import os
import warnings
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# These constants control paths, features, and model settings.
RAW_DATA_PATH = "data/raw/flood_data.csv"
PROCESSED_DATA_PATH = "data/processed/flood_data_processed.csv"
MODEL_PATH = "models/flood_prediction_model.pkl"
SCALER_PATH = "models/scaler.pkl"
EVALUATION_REPORT_PATH = "reports/evaluation_report.txt"
FEATURE_IMPORTANCE_PATH = "reports/feature_importance.csv"
RANDOM_STATE = 42
N_ESTIMATORS = 100
TEST_SIZE = 0.2
RANDOM_FOREST_MAX_DEPTH = 10
MISSING_VALUE_THRESHOLD = 0.5
TARGET_COLUMN = "flood_risk"
FEATURE_COLUMNS = ["rainfall", "temperature", "humidity", "river_level", "soil_moisture"]
NUMERICAL_FEATURES = ["rainfall", "temperature", "humidity", "river_level", "soil_moisture"]


# This ensures folders exist before saving outputs.
def ensure_directories() -> None:
    for directory in ["data/processed", "models", "reports"]:
        os.makedirs(directory, exist_ok=True)


# This loads the CSV dataset from disk.
def load_data(file_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    data = pd.read_csv(file_path)
    print(f"Loaded data from {file_path} with shape {data.shape}")
    return data


# This fills missing numeric values and drops very sparse columns.
def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    missing_ratio = data.isnull().sum() / len(data)
    cols_to_drop = missing_ratio[missing_ratio > MISSING_VALUE_THRESHOLD].index.tolist()
    if cols_to_drop:
        data = data.drop(columns=cols_to_drop)
        print(f"Dropped sparse columns: {cols_to_drop}")
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if data[col].isnull().any():
            data[col] = data[col].fillna(data[col].mean())
    print("Handled missing values")
    return data


# This validates that the required columns exist.
def validate_columns(data: pd.DataFrame, required_columns: List[str]) -> None:
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


# This separates the target column from the feature columns.
def split_features_target(data: pd.DataFrame, target_column: str = TARGET_COLUMN) -> Tuple[pd.DataFrame, pd.Series]:
    validate_columns(data, FEATURE_COLUMNS + [target_column])
    X = data[FEATURE_COLUMNS].copy()
    y = data[target_column].copy()
    print(f"Prepared features {X.shape} and target {y.shape}")
    return X, y


# This splits the dataset into train and test partitions.
def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"Split data into train {X_train.shape} and test {X_test.shape}")
    return X_train, X_test, y_train, y_test


# This scales numeric features using training statistics only.
def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numerical_features: Optional[List[str]] = None,
    scaler_path: str = SCALER_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    numerical_features = numerical_features or NUMERICAL_FEATURES
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    scaler.fit(X_train[numerical_features])
    X_train_scaled[numerical_features] = scaler.transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
    joblib.dump(scaler, scaler_path)
    print(f"Scaled features and saved scaler to {scaler_path}")
    return X_train_scaled, X_test_scaled, scaler


# This saves a cleaned copy of the dataset for inspection.
def save_processed_data(data: pd.DataFrame, file_path: str = PROCESSED_DATA_PATH) -> None:
    data.to_csv(file_path, index=False)
    print(f"Saved processed data to {file_path}")


# This runs the full preprocessing pipeline.
def preprocess_data(file_path: str = RAW_DATA_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data = load_data(file_path)
    data = handle_missing_values(data)
    save_processed_data(data)
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


# This creates interaction features between related variables.
def create_interaction_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "rainfall" in data.columns and "humidity" in data.columns:
        data["rainfall_humidity_interaction"] = data["rainfall"] * data["humidity"]
    if "temperature" in data.columns and "river_level" in data.columns:
        data["temp_river_interaction"] = data["temperature"] * data["river_level"]
    return data


# This creates squared or higher-power versions of selected features.
def create_polynomial_features(data: pd.DataFrame, columns: List[str], degree: int = 2) -> pd.DataFrame:
    data = data.copy()
    for col in columns:
        if col in data.columns:
            for power in range(2, degree + 1):
                suffix = "squared" if power == 2 else f"power_{power}"
                data[f"{col}_{suffix}"] = data[col] ** power
    return data


# This creates rolling mean and standard deviation features.
def create_statistical_features(data: pd.DataFrame, window_size: int = 3) -> pd.DataFrame:
    data = data.copy()
    for col in data.select_dtypes(include=[np.number]).columns:
        data[f"{col}_rolling_mean"] = data[col].rolling(window=window_size, min_periods=1).mean()
        data[f"{col}_rolling_std"] = data[col].rolling(window=window_size, min_periods=1).std().fillna(0)
    return data


# This normalizes one feature into the 0-to-1 range.
def normalize_feature(data: pd.DataFrame, column: str) -> pd.DataFrame:
    data = data.copy()
    if column in data.columns:
        min_val = data[column].min()
        max_val = data[column].max()
        if max_val != min_val:
            data[f"{column}_normalized"] = (data[column] - min_val) / (max_val - min_val)
    return data


# This selects only the requested columns that exist.
def select_features(data: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    available_features = [feature for feature in feature_list if feature in data.columns]
    return data[available_features].copy()


# This builds a mixed numeric and categorical preprocessing pipeline.
def create_preprocessor(numerical_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])


# This trains the main Random Forest classifier.
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = N_ESTIMATORS,
    max_depth: int = RANDOM_FOREST_MAX_DEPTH,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("Trained Random Forest classifier")
    return model


# This computes ranked feature importance from the trained model.
def get_feature_importance(model: RandomForestClassifier, feature_names: List[str], top_n: int = 10) -> pd.DataFrame:
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(feature_importance.head(top_n).to_string(index=False))
    return feature_importance


# This writes the trained model to disk.
def save_model(model: RandomForestClassifier, file_path: str = MODEL_PATH) -> None:
    joblib.dump(model, file_path)
    print(f"Saved model to {file_path}")


# This runs training and persists the model artifact.
def train_and_save(X_train: pd.DataFrame, y_train: pd.Series, model_path: str = MODEL_PATH) -> Tuple[RandomForestClassifier, pd.DataFrame]:
    model = train_model(X_train, y_train)
    feature_importance = get_feature_importance(model, X_train.columns.tolist())
    save_model(model, model_path)
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    print(f"Saved feature importance to {FEATURE_IMPORTANCE_PATH}")
    return model, feature_importance


# This computes standard classification metrics.
def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# This prints metrics in a compact readable format.
def print_metrics(metrics: Dict[str, float], set_name: str) -> None:
    print(f"\n{set_name} Metrics")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


# This computes the confusion matrix for predictions.
def get_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


# This prints the confusion matrix in table form.
def print_confusion_matrix(cm: np.ndarray) -> None:
    print("\nConfusion Matrix")
    print(cm)


# This prints the detailed classification report.
def print_classification_details(y_true: pd.Series, y_pred: np.ndarray) -> None:
    print("\nClassification Report")
    print(classification_report(y_true, y_pred, target_names=["No Flood", "Flood"]))


# This evaluates the classifier on train and test data.
def evaluate_model(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, object]:
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    cm_test = get_confusion_matrix(y_test, y_test_pred)
    print_metrics(train_metrics, "Training Set")
    print_metrics(test_metrics, "Test Set")
    print_confusion_matrix(cm_test)
    print_classification_details(y_test, y_test_pred)
    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "confusion_matrix": cm_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
    }


# This saves a text report with the evaluation results.
def save_evaluation_report(results: Dict[str, object], file_path: str = EVALUATION_REPORT_PATH) -> None:
    with open(file_path, "w") as report_file:
        report_file.write("FLOOD PREDICTION MODEL - EVALUATION REPORT\n")
        report_file.write("=" * 50 + "\n\n")
        report_file.write("TRAINING METRICS\n")
        for metric, value in results["train_metrics"].items():
            report_file.write(f"{metric}: {value:.4f}\n")
        report_file.write("\nTEST METRICS\n")
        for metric, value in results["test_metrics"].items():
            report_file.write(f"{metric}: {value:.4f}\n")
        report_file.write("\nCONFUSION MATRIX\n")
        report_file.write(np.array2string(results["confusion_matrix"]) + "\n")
    print(f"Saved evaluation report to {file_path}")


# This loads the saved classifier for later predictions.
def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)


# This loads the saved scaler for prediction preprocessing.
def load_scaler(scaler_path: str = SCALER_PATH):
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    return joblib.load(scaler_path)


# This selects the training features and scales them for inference.
def preprocess_prediction_data(data: pd.DataFrame, scaler, numerical_features: Optional[List[str]] = None) -> pd.DataFrame:
    numerical_features = numerical_features or NUMERICAL_FEATURES
    validate_columns(data, FEATURE_COLUMNS)
    prepared_data = data[FEATURE_COLUMNS].copy()
    prepared_data[numerical_features] = scaler.transform(prepared_data[numerical_features])
    return prepared_data


# This generates class predictions from a model.
def predict(model, X: pd.DataFrame, scaler=None, numerical_features: Optional[List[str]] = None) -> np.ndarray:
    if scaler is not None:
        X = preprocess_prediction_data(X, scaler, numerical_features)
    return model.predict(X)


# This generates both class labels and probabilities.
def predict_with_probabilities(model, X: pd.DataFrame, scaler=None, numerical_features: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    if scaler is not None:
        X = preprocess_prediction_data(X, scaler, numerical_features)
    return model.predict(X), model.predict_proba(X)


# This predicts one sample provided as a dictionary.
def predict_single_sample(model, sample: Dict[str, float], scaler=None, numerical_features: Optional[List[str]] = None) -> Tuple[int, float]:
    X = pd.DataFrame([sample])
    predictions, probabilities = predict_with_probabilities(model, X, scaler, numerical_features)
    return int(predictions[0]), float(probabilities[0][1])


# This predicts every row in a CSV file and appends outputs.
def batch_predict(model, data_path: str, scaler=None, numerical_features: Optional[List[str]] = None) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    predictions, probabilities = predict_with_probabilities(model, data, scaler, numerical_features)
    result = data.copy()
    result["prediction"] = predictions
    result["flood_probability"] = probabilities[:, 1]
    result["no_flood_probability"] = probabilities[:, 0]
    print(f"Generated predictions for {len(result)} rows")
    return result


# This converts a numeric prediction into readable text.
def classify_prediction(prediction: int, probability: float) -> str:
    return f"FLOOD RISK (Probability: {probability:.2%})" if prediction == 1 else f"NO FLOOD (Probability: {probability:.2%})"


# This evaluates a majority-class baseline classifier.
def evaluate_baseline(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Tuple[DummyClassifier, Dict[str, float]]:
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    predictions = baseline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
    }
    print("\nBaseline Metrics")
    print(metrics)
    print("\nBaseline Classification Report")
    print(classification_report(y_test, predictions))
    return baseline, metrics


# This computes regression metrics for demo experiments.
def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "RMSE": float(np.sqrt(mse)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


# This trains a simple linear regression demo on alternate column names.
def train_linear_regression(df: pd.DataFrame):
    required_columns = ["Rainfall", "Humidity", "Temperature", "RiverLevel"]
    validate_columns(df, required_columns)
    X = df[["Rainfall", "Humidity", "Temperature"]]
    y = df["RiverLevel"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate_regression(y_test, baseline.predict(X_test))
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])
    pipeline.fit(X_train, y_train)
    model_metrics = evaluate_regression(y_test, pipeline.predict(X_test))
    print("\nRegression Baseline Metrics")
    print(baseline_metrics)
    print("\nLinear Regression Metrics")
    print(model_metrics)
    return pipeline


# This runs the full classification workflow end to end.
def main() -> Tuple[Optional[RandomForestClassifier], Optional[Dict[str, object]]]:
    print("\n" + "=" * 70)
    print("FLOOD RISK PREDICTION - MACHINE LEARNING PIPELINE")
    print("=" * 70)
    try:
        ensure_directories()
        X_train, X_test, y_train, y_test = preprocess_data(RAW_DATA_PATH)
        model, _ = train_and_save(X_train, y_train, MODEL_PATH)
        results = evaluate_model(model, X_train, X_test, y_train, y_test)
        save_evaluation_report(results, EVALUATION_REPORT_PATH)
        print("\nPipeline completed successfully")
        print(f"Model saved at {MODEL_PATH}")
        print(f"Scaler saved at {SCALER_PATH}")
        print(f"Report saved at {EVALUATION_REPORT_PATH}")
        return model, results
    except Exception as error:
        print(f"\nPipeline failed: {error}")
        return None, None


# This loads saved artifacts and predicts on a new CSV file.
def predict_on_new_data(input_csv_path: str) -> Optional[pd.DataFrame]:
    try:
        model = load_model(MODEL_PATH)
        scaler = load_scaler(SCALER_PATH)
        predictions_df = batch_predict(model, input_csv_path, scaler, NUMERICAL_FEATURES)
        output_path = input_csv_path.replace(".csv", "_predictions.csv")
        predictions_df.to_csv(output_path, index=False)
        print(f"Saved predictions to {output_path}")
        return predictions_df
    except Exception as error:
        print(f"Prediction failed: {error}")
        return None


if __name__ == "__main__":
    main()
