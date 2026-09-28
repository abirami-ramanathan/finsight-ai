from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "outputs" / "final_forecasting_results.csv"
OUTPUT_DIR = ROOT / "outputs" / "recommendation"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df = pd.read_csv(INPUT)

def get_asset_id(row):
    asset_type = str(row["asset_type"])

    if asset_type == "mutual_fund":
        identifier = str(row["scheme_code"])
    else:
        identifier = str(row["symbol"])

    return (
        asset_type
        + "_"
        + identifier
        + "_"
        + str(row["model"])
    )

df["asset_id"] = df.apply(
    get_asset_id,
    axis=1
)

df["risk_score"] = (
    1 - df["r2"].clip(0, 1)
)

df["forecast_quality"] = (
    df["r2"].clip(0, 1)
)

df["forecast_error"] = (
    df["mape"] / 100
).clip(0, 1)

horizon_map = {
    "short": 30,
    "medium": 90,
    "long": 365
}

rows = []

for horizon_name, horizon_days in horizon_map.items():

    temp = df.copy()

    temp["horizon_days"] = horizon_days
    temp["horizon_name"] = horizon_name

    rows.append(temp)

result = pd.concat(
    rows,
    ignore_index=True
)

result["horizon_normalized"] = (
    result["horizon_days"] / 365
)

output = (
    OUTPUT_DIR /
    "recommendation_assets.csv"
)

result.to_csv(
    output,
    index=False
)

print("=" * 70)
print("RECOMMENDATION DATASET CREATED")
print("=" * 70)

print(
    f"Forecasting experiments: {len(df)}"
)

print(
    f"Recommendation candidates: {len(result)}"
)

print("\nAsset classes:")
print(
    result.groupby("asset_type")
    .size()
)

print("\nSample asset IDs:")
print(
    result[
        ["asset_type", "asset_id"]
    ]
    .drop_duplicates()
    .head(15)
    .to_string(index=False)
)

print("\nSaved:")
print(output)