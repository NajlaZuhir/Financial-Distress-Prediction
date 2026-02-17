import pandas as pd
import joblib
import os
from sklearn.metrics import recall_score, f1_score
import random
import warnings
warnings.filterwarnings("ignore")


# Load artifacts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "artifacts")
DATA = "test_set.csv"

def load_artifacts():
    
    rf_model = joblib.load(os.path.join(MODEL_DIR, "Random_Forest.pkl"))
    xgb_model = joblib.load(os.path.join(MODEL_DIR, "XGBoost.pkl"))

    return rf_model, xgb_model


def data_sample(count=10):

    test_df = pd.read_csv(os.path.join(MODEL_DIR, DATA))
    samples = test_df.sample(n=count, random_state=random.randint(0, 1000))
    return samples


def predict(samples, model_name):
    rf_model, xgb_model = load_artifacts()

    model = rf_model if model_name == "Random Forest" else xgb_model
    X = samples.drop(columns=['Financial Distress'])
    y_pred = model.predict(X)

    return y_pred

def evaluate(y_true, y_pred):

    recall = round(recall_score(y_true, y_pred, pos_label=1), 4)
    f1     = round(f1_score(y_true, y_pred, pos_label=1), 4)

    print(f'Recall={recall:.4f}  F1={f1:.4f}')

    
if __name__ == "__main__":

    # User choices
    model_name = input("Choose model (Random Forest / XGBoost): ").strip()
    mode = input("Evaluate on (full / sample): ").strip().lower()

    if mode == "sample":
        count = int(input("How many samples? "))
        data = data_sample(count=count)
        values = data['Financial Distress'].values
    else:
        data = pd.read_csv(os.path.join(MODEL_DIR, DATA))
        values = data['Financial Distress'].value_counts()

    print("\nTarget (Financial Distress):")
    print(values)

    y_pred = predict(data, model_name)

    print("\nEvaluation:")
    evaluate(data['Financial Distress'], y_pred)
