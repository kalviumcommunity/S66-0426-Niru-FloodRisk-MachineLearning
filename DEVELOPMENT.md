# 🔧 Development Notes - Code Reference

Quick reference for working with this ML project.

## File Size Reference

| File | Lines | Purpose |
|------|-------|---------|
| `config.py` | ~45 | Configuration constants |
| `data_preprocessing.py` | ~200+ | Data pipeline |
| `train.py` | ~100 | Model training |
| `evaluate.py` | ~150+ | Evaluation metrics |
| `predict.py` | ~180+ | Prediction functions |
| `feature_engineering.py` | ~120+ | Feature creation |
| `main.py` (src) | ~100+ | Pipeline orchestration |

## Key Functions Quick Reference

### Data Processing
```python
from src.data_preprocessing import preprocess_data
X_train, X_test, y_train, y_test = preprocess_data("data/raw/flood_data.csv")
```

### Training
```python
from src.train import train_and_save
model, importance = train_and_save(X_train, y_train)
```

### Evaluation
```python
from src.evaluate import evaluate_model
results = evaluate_model(model, X_train, X_test, y_train, y_test)
```

### Prediction
```python
from src.predict import load_model, load_scaler, batch_predict
model = load_model()
scaler = load_scaler()
predictions = batch_predict(model, "data/new_data.csv", scaler)
```

## Configuration Quick Edit

Most settings are in `src/config.py`:

```python
# Change these for different behavior:
N_ESTIMATORS = 100              # Number of trees
RANDOM_FOREST_MAX_DEPTH = 10    # Tree depth
TEST_SIZE = 0.2                 # Test split size
RANDOM_STATE = 42               # For reproducibility
```

## Expected Data Format

**Input CSV** (`data/raw/flood_data.csv`):
```
rainfall,temperature,humidity,river_level,soil_moisture,flood_risk
45.2,25.3,65.0,2.5,0.45,0
52.1,28.1,72.0,3.2,0.52,1
```

**Output CSV** (after predictions):
```
rainfall,temperature,humidity,river_level,soil_moisture,prediction,flood_probability,no_flood_probability
45.2,25.3,65.0,2.5,0.45,0,0.15,0.85
52.1,28.1,72.0,3.2,0.52,1,0.92,0.08
```

## Module Dependency Chain

```
main.py (entry)
  └─> config.py (settings)
  └─> data_preprocessing.py
      └─> pandas, sklearn
  └─> train.py
      └─> sklearn.ensemble.RandomForestClassifier
  └─> evaluate.py
      └─> sklearn.metrics
  └─> predict.py
      └─> joblib
```

## Type Hints Used

```python
from typing import Tuple, List, Optional, Dict
import pandas as pd
import numpy as np

# Examples from code:
def load_data(file_path: str) -> pd.DataFrame:
def split_features_target(data: pd.DataFrame, target_column: str = "flood_risk") -> Tuple[pd.DataFrame, pd.Series]:
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
```

## Error Handling

All functions include error handling:

```python
try:
    data = load_data(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"Error: {e}")
```

## Debugging Tips

### 1. Check Data Shape
```python
from src.data_preprocessing import load_data
data = load_data("data/raw/flood_data.csv")
print(f"Shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"\nFirst few rows:\n{data.head()}")
```

### 2. Check for Missing Values
```python
print(data.isnull().sum())  # Count missing per column
print(data.isnull().sum().sum())  # Total missing
```

### 3. Check Feature Statistics
```python
print(data.describe())  # Min, max, mean, std
print(data.dtypes)  # Data types
```

### 4. Check Model Performance
```python
from src.train import train_model
model = train_model(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")
```

### 5. Check Feature Importance
```python
import pandas as pd
importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
print(importance)
```

## Common Customizations

### Use Different Train/Test Split
```python
# In config.py:
TEST_SIZE = 0.3  # 70-30 split instead of 80-20
```

### Use Different RandomForest Parameters
```python
# In config.py:
N_ESTIMATORS = 200              # More trees
RANDOM_FOREST_MAX_DEPTH = 15    # Deeper trees
```

### Add Features Manually
```python
# In feature_engineering.py, add:
def create_custom_feature(data):
    data['new_feature'] = data['col1'] * data['col2']
    return data

# Then use in preprocessing
```

### Change Random State
```python
# In config.py - change RANDOM_STATE to any integer
RANDOM_STATE = 42  # Change this for different splits
```

## Testing New Data Format

```python
import pandas as pd

# Test with your data
data = pd.read_csv("data/raw/your_data.csv")

# Check required columns exist
required = ["rainfall", "temperature", "humidity", "river_level", "soil_moisture", "flood_risk"]
missing = [col for col in required if col not in data.columns]

if missing:
    print(f"Missing columns: {missing}")
else:
    print("✓ Data format is correct")

# Check data types
print(data.dtypes)

# Run preprocessing
from src.data_preprocessing import preprocess_data
X_train, X_test, y_train, y_test = preprocess_data()
```

## Performance Tips

### If model is overfitting:
```python
# In config.py:
RANDOM_FOREST_MAX_DEPTH = 7  # Reduce depth
N_ESTIMATORS = 50             # Reduce trees
```

### If model is underfitting:
```python
# In config.py:
RANDOM_FOREST_MAX_DEPTH = 15  # Increase depth
N_ESTIMATORS = 200            # Increase trees
```

### If training is slow:
```python
# Use fewer trees or limit depth
N_ESTIMATORS = 50
RANDOM_FOREST_MAX_DEPTH = 8
```

## Extending the Project

### Add a new function to preprocessing:
```python
# In data_preprocessing.py
def my_new_function(data: pd.DataFrame) -> pd.DataFrame:
    """Do something with data."""
    data_modified = data.copy()
    # ... do stuff ...
    return data_modified
```

### Add a new metric:
```python
# In evaluate.py
from sklearn.metrics import roc_auc_score

def calculate_metrics(y_true, y_pred):
    metrics = {
        # ... existing ...
        "roc_auc": roc_auc_score(y_true, y_pred),
    }
    return metrics
```

### Add a new feature:
```python
# In feature_engineering.py
def my_feature(data):
    """Create a new feature."""
    data = data.copy()
    data['my_new_feature'] = data['col1'] ** 2
    return data
```

## File Locations Reference

| Item | Path |
|------|------|
| Raw data | `data/raw/flood_data.csv` |
| Saved model | `models/flood_prediction_model.pkl` |
| Saved scaler | `models/scaler.pkl` |
| Evaluation report | `reports/evaluation_report.txt` |
| Feature importance | `reports/feature_importance.csv` |
| Configuration | `src/config.py` |

## Running Different Scenarios

### Scenario 1: Train with your data
```bash
# 1. Place CSV in data/raw/
# 2. Update src/config.py with file path
# 3. Run:
python main.py
```

### Scenario 2: Make predictions
```bash
# 1. Place new data in data/
# 2. Run in Python:
from src.main import predict_on_new_data
predictions = predict_on_new_data("data/new_data.csv")
```

### Scenario 3: Programmatic use
```python
from src.data_preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model

# Load and prepare
X_train, X_test, y_train, y_test = preprocess_data()

# Train
model = train_model(X_train, y_train)

# Evaluate
results = evaluate_model(model, X_train, X_test, y_train, y_test)

# Check performance
print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
```

---

*For more details, see README.md and code docstrings.*
