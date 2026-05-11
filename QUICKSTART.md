# 🚀 Quick Start Guide - Flood Risk Prediction ML Project

## Getting Started in 5 Minutes

### Step 1: Set Up Python Environment

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected packages to install:**
- pandas (2.0.3)
- scikit-learn (1.3.2)
- joblib (1.3.2)
- numpy (1.24.3)

### Step 3: Verify Sample Data

The project includes sample flood data at:
```
data/raw/flood_data.csv
```

This CSV has 50 sample records with all required columns:
- rainfall (mm)
- temperature (°C)
- humidity (%)
- river_level (m)
- soil_moisture (%)
- flood_risk (0=No Flood, 1=Flood) ← **Target**

### Step 4: Run the Full Pipeline

```bash
python main.py
```

**Expected output:**
```
==================================================
FLOOD RISK PREDICTION - MACHINE LEARNING PIPELINE
==================================================

==================================================
DATA PREPROCESSING PIPELINE
==================================================
✓ Data loaded successfully from data/raw/flood_data.csv
  Shape: (50, 6)
✓ Missing values handled
✓ Features and target separated
✓ Train-test split completed (80-20)
✓ Features scaled successfully
==================================================

==================================================
MODEL TRAINING PIPELINE
==================================================
Training Random Forest Classifier...
✓ Model training completed!

Top 10 Most Important Features:
           feature  importance
      river_level    0.350123
         rainfall    0.280456
...
==================================================

==================================================
MODEL EVALUATION
==================================================
Training Set Metrics:
--------
  ACCURACY : 1.0000
  PRECISION: 1.0000
  RECALL   : 1.0000
  F1       : 1.0000

Test Set Metrics:
--------
  ACCURACY : 0.9500
  PRECISION: 0.9000
  RECALL   : 1.0000
  F1       : 0.9474

...
==================================================
✓ PIPELINE COMPLETED SUCCESSFULLY!
==================================================
```

### Step 5: Check Generated Files

After running the pipeline, you'll find:

```
models/
├── flood_prediction_model.pkl    ← Trained model
└── scaler.pkl                    ← Feature scaler

reports/
├── evaluation_report.txt          ← Performance metrics
└── feature_importance.csv         ← Feature rankings
```

## Using Your Own Data

### Prepare Your CSV File:

1. Create a CSV file with these columns:
   - `rainfall` (numeric)
   - `temperature` (numeric)
   - `humidity` (numeric)
   - `river_level` (numeric)
   - `soil_moisture` (numeric)
   - `flood_risk` (0 or 1) ← **Must have this column for training**

2. Place it in `data/raw/` directory

3. Update the file path in `src/config.py`:
   ```python
   RAW_DATA_PATH = "data/raw/your_file_name.csv"
   ```

4. Run: `python main.py`

## Making Predictions on New Data

Once the model is trained, make predictions on new data:

```python
# In Python or in a script
from src.main import predict_on_new_data

predictions = predict_on_new_data("data/new_flood_data.csv")
# Output: data/new_flood_data_predictions.csv
```

The output CSV will have three new columns:
- `prediction` (0 or 1)
- `flood_probability` (0.0 to 1.0)
- `no_flood_probability` (0.0 to 1.0)

## Project Structure Explained

```
┌─ main.py                      ← Entry point (run this!)
│
├─ src/
│  ├─ config.py               ← Settings (edit here for configuration)
│  ├─ data_preprocessing.py   ← Data loading & cleaning
│  ├─ feature_engineering.py  ← Feature creation (optional)
│  ├─ train.py                ← Model training
│  ├─ evaluate.py             ← Model evaluation
│  ├─ predict.py              ← Prediction functions
│  └─ main.py                 ← Pipeline orchestrator
│
├─ data/
│  ├─ raw/                    ← Place your CSV here
│  └─ processed/              ← (For future use)
│
├─ models/                     ← Saved model & scaler
├─ reports/                    ← Results & metrics
└─ requirements.txt            ← Python packages
```

## Common Tasks

### Change Model Parameters

Edit `src/config.py`:
```python
N_ESTIMATORS = 100           # Increase for complex data
RANDOM_FOREST_MAX_DEPTH = 10 # Increase for deeper trees
TEST_SIZE = 0.2              # 80-20 train-test split
```

### View Feature Importance

After running the pipeline, open:
```
reports/feature_importance.csv
```

This shows which features are most important for predictions.

### Understand the Pipeline

The project follows this workflow:

```
Raw Data CSV
    ↓
Load Data (pandas)
    ↓
Handle Missing Values
    ↓
Split Train/Test (80-20)
    ↓
Scale Features (StandardScaler)
    ↓
Train Model (RandomForest)
    ↓
Evaluate Model (Metrics)
    ↓
Save Model & Scaler (joblib)
    ↓
Use for Predictions
```

## Troubleshooting

### Error: "FileNotFoundError: Data file not found"
- Check that `data/raw/flood_data.csv` exists
- Verify the path in `src/config.py` is correct

### Error: "ModuleNotFoundError: No module named 'sklearn'"
- Install dependencies: `pip install -r requirements.txt`

### Error: "Missing columns error"
- Ensure your CSV has these columns: `rainfall`, `temperature`, `humidity`, `river_level`, `soil_moisture`, `flood_risk`
- Column names are case-sensitive!

### Model accuracy is low
- Check your data quality
- Look at feature importance in `reports/feature_importance.csv`
- Ensure enough training samples (prefer 100+ samples)

## Next Steps

- ✅ **Already done**: Basic ML pipeline
- 📊 **Next**: Add data visualization (matplotlib, seaborn)
- 🔍 **Then**: Hyperparameter tuning (GridSearchCV)
- 🤖 **Advanced**: Try other algorithms (XGBoost, Neural Networks)
- 🌐 **Deploy**: Create a web API (Flask/FastAPI)

## Learning Resources

1. **scikit-learn**: https://scikit-learn.org/
2. **Pandas**: https://pandas.pydata.org/
3. **RandomForest**: https://en.wikipedia.org/wiki/Random_forest
4. **ML Basics**: https://www.coursera.org/courses?query=machine%20learning

---

**Enjoy learning ML!** 🎓

For detailed information, see `README.md`
