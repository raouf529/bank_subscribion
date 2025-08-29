import joblib
import json
import pandas as pd
from src.custom_transformers import AddDropFeatures  # make sure this is available

# === Load model ===
model = joblib.load("banking_prediction.pkl")

# === Load metadata (threshold, metrics, etc.) ===
with open("model_metadata.json", "r") as f:
    metadata = json.load(f)

threshold = metadata["optimal_threshold"]
print(f"Loaded model with threshold: {threshold:.3f}")

# === Example usage ===
# (replace with real test data)
sample_data = pd.DataFrame([{
    "age": 35,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1500,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "duration": 300,
    "campaign": 2,
    "pdays": 999,
    "previous": 0,
    "poutcome": "unknown"
}])

# get predicted probability
proba = model.predict_proba(sample_data)[:, 1]

# apply threshold
prediction = (proba >= threshold).astype(int)

print("Predicted probability:", proba[0])
print("Final prediction (with threshold):", prediction[0])
