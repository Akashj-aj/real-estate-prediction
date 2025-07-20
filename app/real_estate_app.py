import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
MODEL_PATH = ROOT / "models" / "final_model.pkl"

@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        st.error("Model not found. Please run models/save_model.py first.")
        st.stop()
    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict):  # old model format
        st.error("Old model file detected. Retrain using models/save_model.py.")
        st.stop()
    return bundle["model"], bundle["feature_cols"], bundle["target_col"]

model, feature_cols, target_col = load_bundle()

st.title("🏠 Real Estate Price Predictor")
st.write("Enter property details to predict the estimated price per unit area.")

transaction_year = st.number_input("Transaction Year", min_value=2000, max_value=2030, value=2013, step=1)
house_age = st.number_input("House Age (years)", min_value=0, max_value=100, value=20, step=1)
distance_to_mrt = st.number_input("Distance to MRT Station (meters)", min_value=0, value=500, step=10)
num_stores = st.number_input("Number of Convenience Stores Nearby", min_value=0, max_value=20, value=5, step=1)
latitude = st.number_input("Latitude", format="%.6f", value=24.982212)
longitude = st.number_input("Longitude", format="%.6f", value=121.543500)

if st.button("Predict Price"):
    row = {
        "transaction_year": transaction_year,
        "house_age": house_age,
        "distance_to_mrt": distance_to_mrt,
        "num_convenience_stores": num_stores,
        "latitude": latitude,
        "longitude": longitude,
    }
    X_new = pd.DataFrame([row], columns=feature_cols)
    pred = model.predict(X_new)
    st.success(f"🏷️ Estimated Price per Unit Area: ${pred[0]:,.2f}")
