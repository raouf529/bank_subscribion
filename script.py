import joblib
import json
import pandas as pd
#from costumetransform import AddDropFeatures  
from  costumetransform import AddDropFeatures
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
    "id": 482913,
    "age": 37,
    "job": "technician",
    "marital": "married",
    "education": "secondary",
    "default": "no",
    "balance": 1567,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 12,
    "month": "may",
    "duration": 321,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}
])

# get predicted probability
proba = model.predict(sample_data)

# apply threshold
prediction = (proba >= threshold).astype(int)
print("Final prediction (with threshold):", prediction[0])
