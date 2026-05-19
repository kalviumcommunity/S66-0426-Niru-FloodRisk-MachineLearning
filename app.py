import streamlit as st
import joblib
import pandas as pd

# ── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(page_title="Flood Risk Predictor", page_icon="🌊", layout="centered")
st.title("🌊 Flood Risk Predictor")
st.write("Enter environmental details below and click **Predict** to assess flood risk.")

# ── Load model (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("models/churn_pipeline.joblib")

pipeline = load_model()

# ── Input form ─────────────────────────────────────────────────────────────
st.header("Environmental Conditions")

col1, col2 = st.columns(2)

with col1:
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=35.0)
    temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=60.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0)
    rainfall_band = st.selectbox("Rainfall Band", ["low", "medium", "high"])

with col2:
    river_level = st.number_input("River Level (m)", min_value=0.0, max_value=20.0, value=2.5)
    soil_moisture = st.number_input("Soil Moisture (0-1)", min_value=0.0, max_value=1.0, value=0.45)
    moisture_flag = st.selectbox("Moisture Flag", ["dry", "wet"])

# ── Prediction ─────────────────────────────────────────────────────────────
if st.button("Predict Flood Risk", type="primary"):

    input_df = pd.DataFrame([{
        "rainfall": rainfall,
        "temperature": temperature,
        "humidity": humidity,
        "river_level": river_level,
        "soil_moisture": soil_moisture,
        "rainfall_band": rainfall_band,
        "moisture_flag": moisture_flag
    }])

    prediction = pipeline.predict(input_df)
    probability = pipeline.predict_proba(input_df)
    flood_label = prediction[0]
    flood_prob = probability[0][1]

    st.divider()
    st.subheader("Prediction Result")

    col_result, col_metric = st.columns([2, 1])

    with col_result:
        if flood_label == 1:
            st.error("⚠️ **High Flood Risk Detected.**")
        else:
            st.success("✅ **Low Flood Risk.**")

    with col_metric:
        st.metric("Flood Probability", f"{flood_prob:.1%}")

    st.progress(float(flood_prob))

    with st.expander("Show input summary"):
        st.dataframe(input_df)
