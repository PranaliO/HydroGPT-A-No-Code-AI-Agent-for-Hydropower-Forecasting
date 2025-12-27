import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib

def train_xgboost():

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Load dataset
    df = pd.read_csv("../data/merged_cleaned_dataset.csv", parse_dates=["date"], index_col="date")

    # Split dataset
    train = df.loc["2015":"2022"]
    val = df.loc["2023"]
    test = df.loc["2024":]

    FEATURES = [
        'upstream_outflow','rainfall_mm','temperature_c','reservoir_level',
        'inflow_lag1','inflow_lag3','inflow_lag7',
        'upstream_lag1','upstream_lag3',
        'rain_3day','rain_5day',
        'inflow_ma3','inflow_ma7',
        'month','dayofyear'
    ]
    TARGET = "inflow_cumecs"

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "models/scaler_xgb.pkl")

    # XGBoost model with eval_metric in constructor
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="rmse"
    )

    # Train model silently, with validation set
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False
    )

    # Save trained model
    joblib.dump(model, "models/model_xgb.pkl")

    # Access evaluation results for plotting RMSE
    evals_result = model.evals_result()
    
    # Predictions and metrics
    preds = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\n=== Model Evaluation ===")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R² Score: {r2:.3f}")

    # Plot RMSE over iterations
    plt.figure(figsize=(10,5))
    plt.plot(evals_result['validation_0']['rmse'], label='Validation RMSE', color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title("Validation RMSE over Iterations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("models/rmse_over_iterations.png")
    plt.show()

    # Plot Actual vs Predicted for test set
    plt.figure(figsize=(12,6))
    plt.plot(y_test.index, y_test, label="Actual", color='green')
    plt.plot(y_test.index, preds, label="Predicted", color='red')
    plt.xlabel("Date")
    plt.ylabel("Inflow (cumecs)")
    plt.title("Actual vs Predicted Inflow (2024 Test Set)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("models/actual_vs_predicted.png")
    plt.show()

if __name__ == "__main__":
    train_xgboost()
