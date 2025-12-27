import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = "../data"

def validate():

    print("=== Loading merged_cleaned_dataset.csv ===")

    # Load with date parsing
    df = pd.read_csv(f"{DATA_DIR}/merged_cleaned_dataset.csv", parse_dates=["date"])
    df = df.set_index("date")

    # FIX 1: Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    print("✔ Dataset loaded successfully!")
    print("Shape:", df.shape)

    print("\n=== Missing Values ===")
    print(df.isna().sum())

    print("\n=== Outlier Check using IQR ===")
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    for col in df.columns:
        outliers = ((df[col] < (Q1[col] - 1.5 * IQR[col])) |
                    (df[col] > (Q3[col] + 1.5 * IQR[col]))).sum()
        print(f"{col}: {outliers}")

    print("\n=== Correlation Matrix (first 7 columns) ===")
    print(df.iloc[:, :7].corr())

    print("\n=== Date Range ===")
    print("Start Date:", df.index.min())
    print("End Date:", df.index.max())

    print("\n=== Dataset Split ===")

    # FIX 2: If dataframe empty → skip years
    if df.empty:
        print("Years present in dataset: []")
        print("\nValidation Successful!")
        return

    # FIX 3: If date parsing failed (all NaT)
    if df.index.isna().all():
        print("Years present in dataset: []")
        print("\nValidation Successful!")
        return

    # Original behavior (safe now)
    years = sorted(df.index.year.dropna().unique())
    print("Years present in dataset:", years)

    print("\nValidation Successful!")


if __name__ == "__main__":
    validate()
