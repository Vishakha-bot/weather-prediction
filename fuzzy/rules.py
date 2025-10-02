import numpy as np
import skfuzzy as fuzz

# -----------------------------
# Fuzzy logic function
# -----------------------------
def fuzzy_clothing(temp, humidity):
    """
    Predict clothing score based on temperature (°C) and humidity (%)
    Returns a score: 0 = light, 1 = medium, 2 = heavy
    """
    # Define fuzzy ranges
    clothing = np.arange(0, 2.01, 0.01)  # clothing score from 0 to 2

    # Fuzzy membership for clothing
    light = fuzz.trimf(clothing, [0, 0, 1])
    medium = fuzz.trimf(clothing, [0, 1, 2])
    heavy = fuzz.trimf(clothing, [1, 2, 2])

    # Temperature contribution (higher temp -> lighter clothing)
    temp_low = fuzz.trimf(np.arange(-10, 41, 1), [-10, 0, 20])
    temp_high = fuzz.trimf(np.arange(-10, 41, 1), [10, 30, 40])
    
    # Humidity contribution (higher humidity -> medium clothing)
    hum_low = fuzz.trimf(np.arange(0, 101, 1), [0, 20, 50])
    hum_high = fuzz.trimf(np.arange(0, 101, 1), [30, 60, 100])

    # Simplified logic (can improve with rules)
    light_level = max(0, min(1, (30 - temp) / 50))
    medium_level = max(0, min(1, (humidity / 100)))
    heavy_level = max(0, min(1, (temp - 20) / 50))

    # Aggregate
    aggregated = np.fmax(light_level*light, np.fmax(medium_level*medium, heavy_level*heavy))

    # Defuzzify
    clothing_score = fuzz.defuzz(clothing, aggregated, 'centroid')
    return clothing_score

# -----------------------------
# Test block (direct run)
# -----------------------------
if __name__ == "__main__":
    test_temp = 25
    test_humidity = 60
    score = fuzzy_clothing(test_temp, test_humidity)
    print(f"Test clothing score for Temp={test_temp}°C & Humidity={test_humidity}%: {score:.2f}")
