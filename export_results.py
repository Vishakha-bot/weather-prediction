import pandas as pd
import joblib
import os
from fuzzy.rules import fuzzy_clothing

# Paths
DATA_PATH = "data/Weather data.csv"
MODEL_PATH = "models/temp_predictor.pkl"
FEATURES_PATH = "models/features.pkl"
OUTPUT_PATH = "results/dataset_clothing_scores.csv"

# Load dataset
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Load trained model and features
print("🤖 Loading trained model & features...")
model = joblib.load(MODEL_PATH)
FEATURES = joblib.load(FEATURES_PATH)

if not all(f in df.columns for f in FEATURES):
    raise ValueError(f"Dataset is missing one of the required features: {FEATURES}")

X = df[FEATURES]
humidity = df['Rel Hum_%']

print(f"✅ Dataset shape: {df.shape}")
print("🔮 Predicting temperatures and clothing scores...")

# Predict temperatures
predicted_temps = model.predict(X)

# Apply fuzzy clothing logic
scores = []
for t, h in zip(predicted_temps, humidity):
    score = fuzzy_clothing(t, h)
    scores.append(score)

# Save results
df['Predicted_Temp_C'] = predicted_temps
df['Clothing_Score'] = scores

os.makedirs("results", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"🎉 Results saved to {OUTPUT_PATH}")
