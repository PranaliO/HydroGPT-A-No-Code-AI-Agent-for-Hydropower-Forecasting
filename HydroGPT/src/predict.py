import joblib
import pandas as pd

def load_xgb():
    model = joblib.load("models/model_xgb.pkl")
    scaler = joblib.load("models/scaler_xgb.pkl")
    return model, scaler


def predict_xgb(model, scaler, df_input, feature_list):
    df_scaled = scaler.transform(df_input[feature_list])
    preds = model.predict(df_scaled)
    return preds