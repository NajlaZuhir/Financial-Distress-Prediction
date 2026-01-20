import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import joblib
import os

# Load artifacts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODEL_DIR, "xgb_final_model.pkl"))
final_features = joblib.load(os.path.join(MODEL_DIR, "final_features.pkl"))
threshold = joblib.load(os.path.join(MODEL_DIR, "optimal_threshold.pkl"))

app = FastAPI(title="Financial Distress Prediction API")


class CompanyFeatures(BaseModel):
    features: Dict[str, float]


@app.post("/predict")
def predict(data: CompanyFeatures):

    # Convert nested features dict → DataFrame
    df = pd.DataFrame([data.features])

    # Ensure numeric dtype
    df = df.astype(float)

    # Predict probability
    proba = model.predict_proba(df)[:, 1][0]

    # Apply business threshold
    prediction = int(proba >= threshold)

    return {
        "probability": round(float(proba), 4),
        "prediction": prediction  # 1 = distressed, 0 = healthy
    }

    