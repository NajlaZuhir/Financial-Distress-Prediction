import streamlit as st
import requests

st.title("Financial Distress Predictor")

model_name = st.selectbox("Choose Model", ["Random Forest", "XGBoost"])
mode       = st.radio("Evaluation Mode", ["sample", "full"])
count      = st.slider("Sample Size", 5, 100, 10) if mode == "sample" else None

if st.button("Predict"):
    params = {"model_name": model_name, "mode": mode}
    if count:
        params["count"] = count

    response = requests.get("http://127.0.0.1:8000/predict", params=params)
    result   = response.json()

    if "warning" in result:
        st.warning(result["warning"])
    else:
        st.metric("Recall", result["recall"])
        st.metric("F1 Score", result["f1"])