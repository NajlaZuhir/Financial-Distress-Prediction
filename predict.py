import joblib
import pandas as pd
from flask import Flask, request, jsonify

# =========================
# Load artifacts
# =========================
model = joblib.load("financial_distress_model.pkl")
scaler = joblib.load("scaler.pkl")
optimal_features = joblib.load("optimal_features.pkl")

app = Flask("DistressPredictor")


# =========================s
# Helper: transform input
# =========================
def transform_input(input_data: dict):
    df = pd.DataFrame([input_data])
    df = df[optimal_features]          # enforce feature order
    X_scaled = scaler.transform(df)
    return X_scaled


# =========================
# Prediction endpoint
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    X_scaled = transform_input(data)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    label = "Distressed" if prediction == 1 else "Not Distressed"

    result = {
        "prediction": int(prediction),
        "label": label,
        "probability_distressed": float(probability)
    }

    return jsonify(result)


# =========================
# Run apps
# =========================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
