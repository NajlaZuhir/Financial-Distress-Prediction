import requests
import random
import joblib
import os

# Load final feature list
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
final_features = joblib.load(os.path.join(MODEL_DIR, "final_features.pkl"))

url = "http://127.0.0.1:8001/predict"

# Generate ONE sample company
company_features = {f: random.random() for f in final_features}
payload = {
    "features": company_features
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    print("\nPrediction result:")
    print(response.json())
else:
    print("Error:", response.status_code, response.text)
