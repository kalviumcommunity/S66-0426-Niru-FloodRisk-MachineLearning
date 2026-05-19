# Flood Risk Prediction - Machine Learning Project

## A. Project Overview
**Problem Statement and Business Context:**
Flood events pose significant threats to communities and infrastructure. Accurately predicting flood risks based on environmental metrics enables early warning systems, allowing for proactive evacuation and resource allocation to minimize damage and save lives.

**Target Users and Prediction Objective:**
- **Target Users:** Meteorologists, local government bodies, and disaster response teams.
- **Prediction Objective:** Predict the likelihood of a flood event (Flood vs. No Flood) using real-time environmental data.

**Dataset Description and Source:**
The dataset consists of historical environmental records.
- **Source:** Simulated/collected environmental readings.

**Key Features Used:**
- `rainfall`: Amount of rainfall (mm)
- `temperature`: Temperature (°C)
- `humidity`: Humidity (%)
- `river_level`: River level (m)
- `soil_moisture`: Soil moisture content (0-1 scale)
- `rainfall_band`: Categorical representation of rainfall (low, medium, high)
- `moisture_flag`: Categorical representation of soil state (dry, wet)

## B. Setup Instructions

**How to Clone and Run the Project:**
```bash
git clone https://github.com/kalviumcommunity/S66-0426-Niru-FloodRisk-MachineLearning.git
cd S66-0426-Niru-FloodRisk-MachineLearning
```

**How to Install Dependencies:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**How to Launch the Streamlit App:**
```bash
streamlit run app.py
```

## C. ML Pipeline Details

**Preprocessing Steps and Design Decisions:**
- **Numerical Features:** Imputed using `SimpleImputer` and scaled using `StandardScaler`.
- **Categorical Features:** Imputed using `SimpleImputer(strategy='most_frequent')` and encoded using `OneHotEncoder`.
- Both are combined into a `ColumnTransformer` inside a robust `Pipeline`.

**How Data Leakage Was Prevented:**
The entire preprocessing flow (imputation, scaling, encoding) is strictly encapsulated within the scikit-learn `Pipeline`. The test set never touches any fitting step, and the pipeline was seamlessly used during `StratifiedKFold` cross-validation to maintain data integrity.

**How Class Imbalance Was Handled:**
Class imbalance was addressed during the model training phase by utilizing class weights/SMOTE. *(Please refer to the notebooks for before/after SMOTE metrics)*

**Models Compared and Selection Rationale:**
We evaluated models such as Logistic Regression and Random Forest. The Random Forest Classifier was chosen due to its capability of handling non-linear relationships and interactions among environmental factors. 

**Final Model and Hyperparameters:**
- **Model:** Random Forest Classifier
- **Hyperparameters:** `max_depth=4`, `n_estimators=50`, `random_state=42`

## D. Evaluation Results

*(Replace placeholders with actual screenshots from your notebook runs)*

**Baseline vs Final Model Comparison:**
- Baseline Accuracy: ~XX%
- Final Model Accuracy: ~XX%

**CV mean ± std for all candidates:**
*(Insert table or screenshot here)*

**Confusion Matrix and Classification Report:**
![Confusion Matrix Placeholder](https://via.placeholder.com/600x400?text=Confusion+Matrix+Screenshot)

**Before/After Class Imbalance Metrics:**
*(Insert visualization of class distributions here)*

## E. Streamlit App Walkthrough

The project includes a fully interactive Streamlit web application. Non-technical users can adjust environmental metrics using sliders and select boxes to see the real-time probability of a flood event.

**Prediction Examples:**
1. **Low Flood Risk:** With low rainfall and dry soil, the model correctly identifies minimal risk.
2. **High Flood Risk:** With high rainfall, high river levels, and wet soil, the model alerts to a high probability of flooding.

![Streamlit App Walkthrough](https://via.placeholder.com/600x400?text=Streamlit+App+Screenshot)

## F. Reflections

**The Hardest ML Challenge:**
The most difficult challenge during this sprint was properly encapsulating all preprocessing steps into a single scikit-learn pipeline to completely eliminate data leakage.

**What Surprised Me Most:**
I was surprised by how significantly the model's confidence scores shifted after handling class imbalance. The metrics changed drastically, showing the importance of evaluating models honestly rather than relying purely on accuracy.

**What I Would Improve:**
With more time, I would experiment with gradient boosting models (like XGBoost or LightGBM) and tune their hyperparameters more rigorously. I would also add more interactive visualizations directly into the Streamlit app.

**How This Sprint Changed My Thinking:**
This sprint taught me that a machine learning project isn't finished when `model.fit()` is called. Real ML engineering is about building robust pipelines, verifying against leakage, and ultimately deploying the model so real users can benefit from the predictions.