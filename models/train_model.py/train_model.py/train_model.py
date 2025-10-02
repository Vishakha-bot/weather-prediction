# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import os

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/Weather data.csv")

# Use relevant features
features = ["Temp_C", "Rel Hum_%"]
target = "Temp_C"

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("models/temp_predictor.pkl", "wb") as f:
    pickle.dump(model, f)

# Save features used
with open("models/features.pkl", "wb") as f:
    pickle.dump(features, f)

# Print score
score = model.score(X_test, y_test)
print(f"Model R^2 score on test set: {score:.3f}")
print("Model and features saved in models folder.")
