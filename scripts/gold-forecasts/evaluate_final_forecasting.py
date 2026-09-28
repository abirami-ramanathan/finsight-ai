from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs"

df = pd.read_csv(
    OUTPUT_DIR / "final_forecasting_results.csv"
)

metrics = [
    "mae",
    "mse",
    "rmse",
    "r2",
    "mape"
]

summary = (
    df.groupby("asset_type")[metrics]
    .agg(["mean", "median", "min", "max"])
)

summary.to_csv(
    OUTPUT_DIR / "forecasting_asset_class_summary.csv"
)

top_r2 = (
    df.sort_values(
        "r2",
        ascending=False
    )
    .head(15)
)

top_r2.to_csv(
    OUTPUT_DIR / "forecasting_top_r2.csv",
    index=False
)

top_mape = (
    df.sort_values(
        "mape",
        ascending=True
    )
    .head(15)
)

top_mape.to_csv(
    OUTPUT_DIR / "forecasting_top_mape.csv",
    index=False
)

print("=" * 70)
print("FINAL MODULE B EVALUATION")
print("=" * 70)

print("\nExperiments:")
print(len(df))

print("\nAsset classes:")
print(
    df.groupby("asset_type")
    .size()
)

print("\nAverage metrics:")
print(
    df.groupby("asset_type")[metrics]
    .mean()
    .round(4)
)

print("\nOverall R²:")
print(
    f"Mean: {df['r2'].mean():.4f}"
)

print(
    f"Minimum: {df['r2'].min():.4f}"
)

print(
    f"Maximum: {df['r2'].max():.4f}"
)

print("\nAll R² values positive:")
print(
    bool((df["r2"] > 0).all())
)

print("\nSaved:")
print(
    OUTPUT_DIR /
    "forecasting_asset_class_summary.csv"
)

print(
    OUTPUT_DIR /
    "forecasting_top_r2.csv"
)

print(
    OUTPUT_DIR /
    "forecasting_top_mape.csv"
)