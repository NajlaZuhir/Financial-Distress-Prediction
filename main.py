from fastapi import FastAPI
from sklearn.metrics import recall_score, f1_score
from src.predict import predict, data_sample
import pandas as pd

app = FastAPI(title="Financial Distress Predictor API")


@app.get("/")
def healthy():
    return {"message": "Financial Distress Predictor API is running!"}


@app.get("/predict")
def make_prediction(model_name: str = "Random Forest", mode: str = "full", count: int = 10):

    if mode == "sample":
        data = data_sample(count=count)
    else:
        data = pd.read_csv("artifacts/test_set.csv")

    y_true = data['Financial Distress'].values
    
    if y_true.sum() == 0:
        return {"warning": "No distressed companies in this sample — try a larger count or use mode=full"}

    y_pred = predict(data, model_name)

    return {
        "model"  : model_name,
        "mode"   : mode,
        "recall" : round(recall_score(y_true, y_pred, pos_label=1), 4),
        "f1"     : round(f1_score(y_true, y_pred, pos_label=1), 4)
    }