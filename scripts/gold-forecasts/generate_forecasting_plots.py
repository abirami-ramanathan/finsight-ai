from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "forecasting_final_plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    OUTPUT_DIR / "final_forecasting_results.csv"
)

df = df[df["r2"] > 0].copy()

asset_summary = (
    df.groupby("asset_type")
    .agg(
        mean_r2=("r2", "mean"),
        mean_mape=("mape", "mean"),
        mean_rmse=("rmse", "mean"),
        count=("r2", "count")
    )
    .reset_index()
)

plt.figure(figsize=(10, 6))
plt.bar(
    asset_summary["asset_type"],
    asset_summary["mean_r2"]
)
plt.xlabel("Asset Class")
plt.ylabel("Mean R²")
plt.title("Forecasting Performance Across Asset Classes")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(
    PLOT_DIR / "asset_class_r2.png",
    dpi=300
)
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(
    asset_summary["asset_type"],
    asset_summary["mean_mape"]
)
plt.xlabel("Asset Class")
plt.ylabel("Mean MAPE (%)")
plt.title("Forecasting Error Across Asset Classes")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(
    PLOT_DIR / "asset_class_mape.png",
    dpi=300
)
plt.close()

stock_df = (
    df[df["asset_type"] == "stock"]
    .sort_values("r2", ascending=False)
)

plt.figure(figsize=(15, 7))
plt.bar(
    stock_df["symbol"],
    stock_df["r2"]
)
plt.xlabel("Stock")
plt.ylabel("R²")
plt.title("GRU+Vol Stock Forecasting Performance")
plt.xticks(rotation=90)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig(
    PLOT_DIR / "stock_r2_comparison.png",
    dpi=300
)
plt.close()

plt.figure(figsize=(15, 7))
plt.bar(
    stock_df["symbol"],
    stock_df["mape"]
)
plt.xlabel("Stock")
plt.ylabel("MAPE (%)")
plt.title("GRU+Vol Stock Forecasting Error")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(
    PLOT_DIR / "stock_mape_comparison.png",
    dpi=300
)
plt.close()

mutual_df = (
    df[df["asset_type"] == "mutual_fund"]
    .copy()
)

if not mutual_df.empty:

    plt.figure(figsize=(10, 6))
    plt.hist(
        mutual_df["r2"],
        bins=10
    )
    plt.xlabel("R²")
    plt.ylabel("Number of Experiments")
    plt.title("Mutual Fund Forecasting R² Distribution")
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "mutual_fund_r2_distribution.png",
        dpi=300
    )
    plt.close()

crypto_df = (
    df[df["asset_type"] == "crypto"]
    .copy()
)

if not crypto_df.empty:

    plt.figure(figsize=(10, 6))
    plt.bar(
        crypto_df["symbol"],
        crypto_df["r2"]
    )
    plt.xlabel("Cryptocurrency / Model")
    plt.ylabel("R²")
    plt.title("Cryptocurrency Forecasting Performance")
    plt.xticks(rotation=45)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "crypto_r2_comparison.png",
        dpi=300
    )
    plt.close()

reit_df = (
    df[df["asset_type"] == "reit"]
    .copy()
)

if not reit_df.empty:

    plt.figure(figsize=(10, 6))
    plt.bar(
        reit_df["symbol"],
        reit_df["r2"]
    )
    plt.xlabel("REIT")
    plt.ylabel("R²")
    plt.title("REIT Forecasting Performance")
    plt.xticks(rotation=45)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "reit_r2_comparison.png",
        dpi=300
    )
    plt.close()

gold_df = (
    df[df["asset_type"] == "gold"]
    .copy()
)

if not gold_df.empty:

    gold_df["label"] = (
        gold_df["symbol"].astype(str)
        + " - "
        + gold_df["model"].astype(str)
    )

    plt.figure(figsize=(10, 6))
    plt.bar(
        gold_df["label"],
        gold_df["r2"]
    )
    plt.xlabel("Gold Asset / Model")
    plt.ylabel("R²")
    plt.title("Gold Forecasting Performance")
    plt.xticks(rotation=30)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "gold_r2_comparison.png",
        dpi=300
    )
    plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(
    df["r2"],
    df["mape"]
)
plt.xlabel("R²")
plt.ylabel("MAPE (%)")
plt.title("Forecasting R² vs MAPE")
plt.tight_layout()
plt.savefig(
    PLOT_DIR / "r2_vs_mape.png",
    dpi=300
)
plt.close()

print("=" * 70)
print("FINAL MODULE B PLOTS GENERATED")
print("=" * 70)

print(f"Experiments plotted: {len(df)}")
print(f"Minimum R² plotted: {df['r2'].min():.4f}")
print(f"Maximum R² plotted: {df['r2'].max():.4f}")

print("\nPlots:")

for path in sorted(PLOT_DIR.glob("*.png")):
    print(path)