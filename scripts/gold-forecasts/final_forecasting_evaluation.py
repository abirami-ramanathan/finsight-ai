from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
frames = []

mapping = [
    ("stocks", "forecasting_all_stocks.csv", "stock"),
    ("crypto", "crypto_forecasting_results.csv", "crypto"),
    ("reit", "reit_forecasting_results.csv", "reit"),
    ("gold", "gold_forecasting_results.csv", "gold"),
    ("mutual_fund", "mutual_fund_forecasting_results.csv", "mutual_fund")
]

for _, filename, asset in mapping:
    p = OUT / filename
    if not p.exists():
        print("MISSING", p)
        continue
    df = pd.read_csv(p)
    df["asset"] = asset
    frames.append(df)

if not frames:
    raise RuntimeError("No forecasting result files found")

all_results = pd.concat(frames, ignore_index=True)
all_results.to_csv(OUT / "final_forecasting_results_all_assets.csv", index=False)

required = ["mae", "mse", "rmse", "r2", "mape"]
for c in required:
    if c not in all_results.columns:
        raise RuntimeError(f"Missing metric column: {c}")

best = all_results.loc[all_results.groupby(["asset", all_results.columns[0], all_results.columns[1] if len(all_results.columns) > 1 else all_results.columns[0]])["r2"].idxmax()]

asset_summary = all_results.groupby("asset").agg(
    models=("model", "nunique"),
    series=(all_results.columns[0], "nunique"),
    mean_r2=("r2", "mean"),
    best_r2=("r2", "max"),
    mean_rmse=("rmse", "mean"),
    mean_mape=("mape", "mean")
).reset_index()
asset_summary.to_csv(OUT / "final_forecasting_asset_summary.csv", index=False)

best_by_series = all_results.sort_values("r2", ascending=False).groupby(["asset", all_results.columns[0]], as_index=False).first()
best_by_series.to_csv(OUT / "final_forecasting_best_by_series.csv", index=False)

print("FINAL ASSET SUMMARY")
print(asset_summary.to_string(index=False))
print("\nBEST MODEL PER SERIES")
print(best_by_series.to_string(index=False))
