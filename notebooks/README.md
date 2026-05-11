# 📚 Jupyter Notebooks for Flood Risk Prediction

Complete set of interactive Jupyter notebooks for learning and using the flood risk prediction ML pipeline.

## 🚀 Quick Start

```bash
# Open any notebook
jupyter notebook 00_jupyter_setup.ipynb

# Or use JupyterLab (more modern)
jupyter lab
```

## 📋 Notebooks Overview

### **INDEX.ipynb** 📍 **START HERE**
- Guide to all notebooks
- Learning paths (quick, complete, practitioner)
- Tips and troubleshooting
- Quick reference guide

### **00_jupyter_setup.ipynb** ⚙️
- Install dependencies
- Configure Jupyter environment
- Setup matplotlib, pandas, numpy
- Enable auto-reload
- **Duration:** ~5 minutes

### **01_quick_demo.ipynb** ⚡
- Complete pipeline in one notebook
- Load → Train → Evaluate → Predict
- Quick overview of all steps
- **Duration:** ~10 minutes

### **02_data_exploration.ipynb** 📊
- Load and examine data
- Check for missing values
- Statistical analysis
- Distribution plots
- Correlation analysis
- Feature vs target relationships
- **Duration:** ~15 minutes

### **03_preprocessing_pipeline.ipynb** 🔄
- Step-by-step preprocessing
- Handle missing values
- Train-test splitting
- Feature scaling (NO data leakage!)
- Verify best practices
- Compare scaled vs unscaled
- **Duration:** ~20 minutes

### **04_model_training.ipynb** 🤖
- Initialize RandomForest classifier
- Train the model
- Extract feature importance
- Visualize importance
- Save model to disk
- Load and verify
- **Duration:** ~15 minutes

### **05_model_evaluation.ipynb** 📈
- Calculate metrics (accuracy, precision, recall, F1)
- Confusion matrix
- Classification report
- ROC curve
- Train vs test comparison
- Interpret results
- **Duration:** ~20 minutes

### **06_making_predictions.ipynb** 🎯
- Load trained model and scaler
- Single sample predictions
- Batch predictions from CSV
- Get probability scores
- Visualize predictions
- Export results
- **Duration:** ~20 minutes

## 📊 Learning Paths

### ⚡ Quick Overview (15 minutes)
```
00_jupyter_setup.ipynb
    ↓
01_quick_demo.ipynb
```

### 📚 Complete Learning (90 minutes)
```
00_jupyter_setup.ipynb
    ↓
02_data_exploration.ipynb
    ↓
03_preprocessing_pipeline.ipynb
    ↓
04_model_training.ipynb
    ↓
05_model_evaluation.ipynb
    ↓
06_making_predictions.ipynb
```

### 🛠️ Practitioner Focus (50 minutes)
```
00_jupyter_setup.ipynb
    ↓
03_preprocessing_pipeline.ipynb
    ↓
04_model_training.ipynb
    ↓
06_making_predictions.ipynb
```

## 🎯 What You'll Learn

### Data Science
- Exploratory Data Analysis (EDA)
- Feature engineering
- Data preprocessing
- Train-test splitting
- Feature scaling
- Model training and evaluation

### ML Best Practices
- Preventing data leakage
- Proper train-test splitting
- Scaler fitting strategy
- Reproducibility
- Modular code structure
- Model persistence

### Python Skills
- Pandas DataFrames
- NumPy arrays
- Matplotlib visualization
- Scikit-learn library
- Jupyter notebooks
- File I/O

## 🔧 Prerequisites

### Install Dependencies
```bash
pip install -r ../requirements.txt
```

**Required packages:**
- jupyter
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- joblib

### Sample Data
- Included: `../data/raw/flood_data.csv`
- 50 sample records for testing
- Ready to use immediately

## 💡 Tips

### Running Notebooks
1. **Run cells in order** - Each cell builds on previous ones
2. **Don't skip cells** - Setup is essential
3. **Experiment** - Modify code and see what changes
4. **Read comments** - Explanations in every cell

### If You Get Errors

**"Module not found"?**
- Run `00_jupyter_setup.ipynb` first
- Check dependencies are installed

**"Data file not found"?**
- Verify `../data/raw/flood_data.csv` exists
- Check file path in cell output

**"Model not found"?**
- Run training notebook first
- Check `../models/` directory

## 📁 Project Structure

```
notebooks/
├── INDEX.ipynb                    (start here)
├── 00_jupyter_setup.ipynb
├── 01_quick_demo.ipynb
├── 02_data_exploration.ipynb
├── 03_preprocessing_pipeline.ipynb
├── 04_model_training.ipynb
├── 05_model_evaluation.ipynb
├── 06_making_predictions.ipynb
└── README.md                      (this file)

../data/
├── raw/
│   └── flood_data.csv
└── processed/

../src/
├── config.py
├── data_preprocessing.py
├── train.py
├── evaluate.py
├── predict.py
└── main.py
```

## 🎓 Concepts Covered

### Preprocessing
- ✓ Data loading
- ✓ Missing value handling
- ✓ Train-test splitting
- ✓ Feature scaling
- ✓ Data leakage prevention

### Model Training
- ✓ RandomForest classifier
- ✓ Hyperparameter configuration
- ✓ Feature importance
- ✓ Model serialization

### Evaluation
- ✓ Accuracy, Precision, Recall, F1
- ✓ Confusion matrix
- ✓ Classification report
- ✓ ROC curve
- ✓ Train vs test analysis

### Prediction
- ✓ Single sample prediction
- ✓ Batch predictions
- ✓ Probability scores
- ✓ Result export

## 🚀 Next Steps After Notebooks

### Extend Your Skills
1. **Visualizations** - Add more plots and charts
2. **Hyperparameter tuning** - GridSearchCV, RandomizedSearchCV
3. **Other models** - Try XGBoost, LightGBM, Neural Networks
4. **Feature engineering** - Create new features
5. **Cross-validation** - K-fold validation
6. **API** - Deploy with Flask/FastAPI
7. **Deployment** - Docker, AWS, GCP

### Real-World Applications
- Use with your own data
- Adjust for different predictions
- Monitor model performance
- Retrain periodically
- A/B testing different models

## 📚 References

**In this project:**
- [README.md](../README.md) - Full documentation
- [QUICKSTART.md](../QUICKSTART.md) - Quick setup
- [DEVELOPMENT.md](../DEVELOPMENT.md) - Code reference

**External:**
- [Scikit-learn docs](https://scikit-learn.org/)
- [Pandas docs](https://pandas.pydata.org/)
- [Jupyter docs](https://jupyter.org/)
- [ML best practices](https://machinelearningmastery.com/)

## ❓ FAQ

**Q: Which notebook should I start with?**
A: Start with `INDEX.ipynb`, then choose your learning path.

**Q: Can I skip notebooks?**
A: No, run them in order. Each builds on previous ones.

**Q: How long does it take?**
A: Quick path: 15 min | Complete: 90 min | Practitioner: 50 min

**Q: Do I need to modify code?**
A: Yes! Experiment with parameters to learn better.

**Q: Can I use my own data?**
A: Yes, see QUICKSTART.md for data format requirements.

**Q: What if I get stuck?**
A: Check DEVELOPMENT.md for common issues and solutions.

---

**Ready to start? Open `INDEX.ipynb` now!** 🚀

For more information, see [README.md](../README.md)
