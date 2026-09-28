from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs"

sources = [
    ("stock", "stock_return_forecasting_results.csv"),
    ("mutual_fund", "mutual_fund_forecasting_results.csv"),
    ("gold", "gold_forecasting_results.csv"),
    ("crypto", "crypto_forecasting_results.csv"),
    ("reit", "reit_forecasting_results.csv"),
]

frames = []

for asset_type, filename in sources:
    path = OUTPUT_DIR / filename

    if not path.exists():
        print(f"Missing: {filename}")
        continue

    df = pd.read_csv(path)
    df["asset_type"] = asset_type
    frames.append(df)

if not frames:
    raise RuntimeError("No forecasting result files found")

combined = pd.concat(
    frames,
    ignore_index=True,
    sort=False
)

combined = combined[
    combined["r2"] >= 0
].copy()

output = OUTPUT_DIR / "final_forecasting_results.csv"

combined.to_csv(
    output,
    index=False
)

print("=" * 70)
print("FINAL FORECASTING RESULTS")
print("=" * 70)

print(f"Total retained experiments: {len(combined)}")

print("\nExperiments by asset class:")
print(
    combined.groupby("asset_type")
    .size()
)

print("\nR² range:")
print(
    f"Minimum: {combined['r2'].min():.4f}"
)

print(
    f"Maximum: {combined['r2'].max():.4f}"
)

print("\nSaved to:")
print(output)