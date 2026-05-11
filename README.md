# Flood Risk Prediction - Machine Learning Project

A beginner-friendly machine learning project for predicting flood risk using Python, scikit-learn, and pandas.

## 📋 Project Overview

This project implements a complete machine learning pipeline to predict flood risk based on environmental factors such as rainfall, temperature, humidity, river level, and soil moisture.

### Key Features:
- ✅ Clean, modular code structure
- ✅ Complete ML pipeline from data preprocessing to prediction
- ✅ Proper train/test split with no data leakage
- ✅ Feature scaling with StandardScaler
- ✅ Random Forest classifier for classification
- ✅ Model persistence using joblib
- ✅ Comprehensive evaluation metrics
- ✅ Batch prediction capabilities

## 📁 Project Structure

```
project-root/
├── data/
│   ├── raw/                          # Raw CSV data
│   └── processed/                    # Processed data
├── models/                           # Saved models and scalers
├── reports/                          # Evaluation reports and metrics
├── src/
│   ├── __init__.py
│   ├── config.py                     # Configuration and constants
│   ├── data_preprocessing.py         # Data loading and cleaning
│   ├── feature_engineering.py        # Feature creation
│   ├── train.py                      # Model training
│   ├── evaluate.py                   # Model evaluation
│   ├── predict.py                    # Prediction functions
│   └── main.py                       # Pipeline orchestration
├── main.py                           # Entry point
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your flood data CSV file in the `data/raw/` directory:
- File name: `flood_data.csv`
- Expected columns:
  - `rainfall` (float)
  - `temperature` (float)
  - `humidity` (float)
  - `river_level` (float)
  - `soil_moisture` (float)
  - `flood_risk` (int, 0=No Flood, 1=Flood) - **target column**

Example CSV format:
```csv
rainfall,temperature,humidity,river_level,soil_moisture,flood_risk
45.2,25.3,65.0,2.5,0.45,0
52.1,28.1,72.0,3.2,0.52,1
```

### 3. Run the Pipeline

```bash
python main.py
```

This will:
1. Load and preprocess your data
2. Train a Random Forest classifier
3. Evaluate the model
4. Save the trained model and scaler

## 📚 Module Guide

### `config.py`
Contains all configuration variables and constants:
- File paths
- Model hyperparameters
- Feature column names
- Random state for reproducibility

### `data_preprocessing.py`
Handles data loading and preparation:
- Load CSV data
- Handle missing values
- Split train/test sets (80-20 split)
- Scale numerical features

**Important**: Train-test split is performed BEFORE scaling to prevent data leakage!

### `feature_engineering.py`
Functions to create and transform features:
- Interaction features
- Polynomial features
- Statistical features (rolling mean, std)
- Feature selection

### `train.py`
Model training functions:
- Train Random Forest classifier
- Extract feature importance
- Save trained model

### `evaluate.py`
Comprehensive model evaluation:
- Calculate metrics (accuracy, precision, recall, F1)
- Confusion matrix
- Classification report
- Save evaluation report

### `predict.py`
Prediction functions:
- Load trained model
- Preprocess new data
- Make single predictions
- Batch predictions on CSV files

### `main.py`
Main pipeline orchestrator:
- `main()`: Runs complete pipeline
- `predict_on_new_data()`: Make predictions on new data

## 🔧 Usage Examples

### Running the Full Pipeline

```python
from src.main import main

# Train and evaluate the model
model, results = main()
```

### Making Predictions on New Data

```python
from src.main import predict_on_new_data

# Predict on a new CSV file
predictions = predict_on_new_data("data/new_flood_data.csv")
```

### Programmatic Prediction

```python
from src.predict import load_model, load_scaler, predict_single_sample
from src import config

# Load trained components
model = load_model(config.MODEL_PATH)
scaler = load_scaler(config.SCALER_PATH)

# Predict for a single sample
sample = {
    "rainfall": 45.2,
    "temperature": 25.3,
    "humidity": 65.0,
    "river_level": 2.5,
    "soil_moisture": 0.45
}

prediction, probability = predict_single_sample(model, sample, scaler, config.NUMERICAL_FEATURES)
print(f"Prediction: {prediction}, Flood Probability: {probability:.2%}")
```

## 📊 Output Files

After running the pipeline, you'll get:

- `models/flood_prediction_model.pkl` - Trained Random Forest model
- `models/scaler.pkl` - Fitted StandardScaler
- `reports/evaluation_report.txt` - Model performance metrics
- `reports/feature_importance.csv` - Feature importance rankings

## 🎯 ML Best Practices Implemented

1. **Separation of Concerns**: Each module has a single responsibility
2. **No Data Leakage**: Train-test split happens before scaling
3. **Proper Scaling**: Scaler is fit only on training data
4. **Reproducibility**: Fixed `RANDOM_STATE = 42` throughout
5. **Modular Functions**: Reusable functions with proper documentation
6. **Type Hints**: All functions include type annotations
7. **Configuration Management**: No hardcoded values
8. **Error Handling**: Graceful error messages

## 📈 Expected Results

With a well-prepared flood dataset, you can expect:

- **Accuracy**: 80-95% depending on data quality
- **Precision**: High precision for identifying flood events
- **Recall**: Good recall to catch potential flood risks
- **F1-Score**: Balanced performance metric

## 🔍 Troubleshooting

### "Data file not found"
- Ensure `data/raw/flood_data.csv` exists
- Check file name matches `config.RAW_DATA_PATH`

### "Model file not found"
- Run the training pipeline first: `python main.py`
- Ensure `models/` directory exists

### Missing columns error
- Verify your CSV has all required columns
- Check column names match `config.FEATURE_COLUMNS`

## 💡 Next Steps to Extend

1. **Cross-Validation**: Implement k-fold cross-validation for better evaluation
2. **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV
3. **More Models**: Try XGBoost, LightGBM, or Neural Networks
4. **Feature Selection**: Implement feature selection algorithms
5. **Data Visualization**: Add plots for EDA (Exploratory Data Analysis)
6. **API Deployment**: Create a Flask/FastAPI endpoint for predictions
7. **Ensemble Methods**: Combine multiple models

## 📝 License

This is an educational project. Feel free to use and modify for learning purposes.

## 👥 Contributing

Suggestions and improvements are welcome! Please feel free to modify and extend this project.

---

**Happy Learning!** 🎓

For questions or issues, check the code comments and docstrings in each module.