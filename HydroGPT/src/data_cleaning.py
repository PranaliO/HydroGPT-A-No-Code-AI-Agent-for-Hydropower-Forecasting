import pandas as pd
import numpy as np
import os

def create_clean_dataset():

    # Load all raw datasets
    inflow = pd.read_csv("../data/inflow_data.csv", parse_dates=["date"])
    upstream = pd.read_csv("../data/upstream_outflow.csv", parse_dates=["date"])
    rainfall = pd.read_csv("../data/rainfall_data.csv", parse_dates=["date"])
    temp = pd.read_csv("../data/temperature_data.csv", parse_dates=["date"])
    reservoir = pd.read_csv("../data/reservoir_level.csv", parse_dates=["date"])

    # Merge datasets on date
    df = inflow.merge(upstream, on="date") \
               .merge(rainfall, on="date") \
               .merge(temp, on="date") \
               .merge(reservoir, on="date")

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # ================================
    # Feature Engineering
    # ================================

    # Lags for inflow
    df["inflow_lag1"] = df["inflow_cumecs"].shift(1)
    df["inflow_lag3"] = df["inflow_cumecs"].shift(3)
    df["inflow_lag7"] = df["inflow_cumecs"].shift(7)

    # Lags for upstream outflow
    df["upstream_lag1"] = df["upstream_outflow"].shift(1)
    df["upstream_lag3"] = df["upstream_outflow"].shift(3)

    # Rolling rainfall
    df["rain_3day"] = df["rainfall_mm"].rolling(3).sum()
    df["rain_5day"] = df["rainfall_mm"].rolling(5).sum()

    # Moving averages
    df["inflow_ma3"] = df["inflow_cumecs"].rolling(3).mean()
    df["inflow_ma7"] = df["inflow_cumecs"].rolling(7).mean()

    # Date features
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear

    # Remove first 7 rows (because lag and rolling create NaN)
    df = df.dropna().reset_index(drop=True)

    # Save output
    os.makedirs("../data", exist_ok=True)
    df.to_csv("../data/merged_cleaned_dataset.csv", index=False)

    print("merged_cleaned_dataset.csv created successfully!")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    create_clean_dataset()
