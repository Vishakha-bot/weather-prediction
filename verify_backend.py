import os
import pandas as pd
import joblib
from fuzzy.rules import fuzzy_clothing

print("=== Backend Verification Script ===\n")

# --- Check dataset ---
data_path = "data/Weather data.csv"
if os.path.exists(data_path):
    print(f"Dataset found: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
else:
    print(f"Dataset NOT found at {data_path}")
    exit()

# --- Check trained model ---
model_path = "models/temp_predictor.pkl"
if os.path.exists(model_path):
    print(f"Model found: {model_path}")
    model = joblib.load(model_path)
else:
    print(f"Model NOT found at {model_path}")
    exit()

# --- Test prediction ---
X_test = pd.DataFrame([[df.iloc[0]['Rel Hum_%'], df.iloc[0]['Wind Speed_km/h'],
                        df.iloc[0]['Visibility_km'], df.iloc[0]['Press_kPa']]],
                      columns=['Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa'])
pred_temp = model.predict(X_test)[0]
print(f"Sample predicted temperature: {pred_temp:.2f} °C")

# --- Test fuzzy function ---
sample_humidity = df.iloc[0]['Rel Hum_%']
score = fuzzy_clothing(pred_temp, sample_humidity)
print(f"Sample clothing score: {score:.2f} (0=light, 1=medium, 2=heavy)")

# --- Check output CSV ---
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
output_csv = os.path.join(results_dir, "clothing_scores_sample.csv")
df['Clothing_Score'] = [fuzzy_clothing(model.predict(pd.DataFrame([[row['Rel Hum_%'], row['Wind Speed_km/h'], row['Visibility_km'], row['Press_kPa']]],
                                                                  columns=['Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa']))[0],
                                        row['Rel Hum_%']) for idx, row in df.iterrows()]
df.to_csv(output_csv, index=False)
print(f"Sample clothing scores saved at: {output_csv}")

print("\n=== Backend verification complete ===")
