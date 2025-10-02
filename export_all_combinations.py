import pandas as pd
import numpy as np
import joblib
from fuzzy.rules import fuzzy_clothing
import os

# Load model
model = joblib.load("models/temp_predictor.pkl")

# Create results directory if not exists
os.makedirs("results", exist_ok=True)

# Define slider ranges (same as app.py)
humidity_range = range(0, 101, 10)       # 0 to 100%, step 10
wind_range = range(0, 51, 5)             # 0 to 50 km/h, step 5
visibility_range = range(0, 21, 5)       # 0 to 20 km, step 5
pressure_range = range(90, 111, 5)       # 90 to 110 kPa, step 5

# Prepare dataframe to store results
results = []

for hum in humidity_range:
    for wind in wind_range:
        for vis in visibility_range:
            for pres in pressure_range:
                # Prepare input
                input_df = pd.DataFrame([[hum, wind, vis, pres]],
                                        columns=['Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa'])
                # Predict temperature
                pred_temp = model.predict(input_df)[0]
                # Compute clothing score
                clothing_score = fuzzy_clothing(pred_temp, hum)
                # Save row
                results.append([hum, wind, vis, pres, pred_temp, clothing_score])

# Convert to DataFrame
df_results = pd.DataFrame(results, columns=['Humidity', 'Wind Speed', 'Visibility', 'Pressure', 'Pred_Temp', 'Clothing_Score'])

# Save to CSV
output_file = "results/clothing_scores.csv"
df_results.to_csv(output_file, index=False)

print(f"All slider combinations exported to: {output_file}")
