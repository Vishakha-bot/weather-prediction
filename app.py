# app.py
import streamlit as st
import pandas as pd
import pickle
import os
from fuzzy.rules import fuzzy_clothing

st.set_page_config(page_title="Weather Prediction & Clothing Recommendation", page_icon="🌦", layout="centered")
st.title("🌦 Weather Prediction & Clothing Recommendation")

# -----------------------------
# Load model & features
# -----------------------------
model_path = "models/temp_predictor.pkl"
features_path = "models/features.pkl"

if not os.path.exists(model_path) or not os.path.exists(features_path):
    st.error("Trained model or features not found! Run models/train_model.py first.")
    st.stop()

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(features_path, "rb") as f:
    model_features = pickle.load(f)

if not hasattr(model, "predict"):
    st.error("Loaded model is not a valid sklearn model. Re-train using train_model.py")
    st.stop()

st.success("✅ Trained model loaded successfully!")

# -----------------------------
# Input sliders
# -----------------------------
st.subheader("Interactive Prediction")
temp_input = st.slider("Select Temperature (°C)", -10.0, 40.0, 20.0)
humidity_input = st.slider("Select Humidity (%)", 0, 100, 50)

# -----------------------------
# Prediction
# -----------------------------
input_df = pd.DataFrame([[temp_input, humidity_input]], columns=model_features)
pred_temp = model.predict(input_df)[0]
clothing_score = fuzzy_clothing(pred_temp, humidity_input)

st.write(f"**Predicted Temperature:** {pred_temp:.2f} °C")
st.write(f"**Recommended Clothing Score:** {clothing_score:.2f} (0=light, 1=medium, 2=heavy)")

# -----------------------------
# Dataset preview & CSV export
# -----------------------------
st.subheader("Dataset Predictions")
dataset_file = "results/dataset_clothing_scores.csv"
if os.path.exists(dataset_file):
    df = pd.read_csv(dataset_file)
    st.dataframe(df.head(20))
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Full Dataset Results as CSV", data=csv, file_name="dataset_clothing_scores.csv")
else:
    st.warning(f"{dataset_file} not found. Run export_results.py to generate dataset predictions.")

st.info("✅ Adjust sliders to test different temperature & humidity values.")
