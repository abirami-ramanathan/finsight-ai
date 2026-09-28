from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PLOTS = OUT / "final_forecast_plots"
PLOTS.mkdir(parents=True, exist_ok=True)

best_path = OUT / "final_forecasting_best_by_series.csv"
if not best_path.exists():
    raise RuntimeError("Run final_forecasting_evaluation.py first")
best = pd.read_csv(best_path)

for _, row in best.iterrows():
    asset = str(row["asset"])
    first_col = [c for c in best.columns if c not in {"asset", "model", "mae", "mse", "rmse", "r2", "mape"}][0]
    series = str(row[first_col])
    model = str(row["model"])
    safe = series.replace(".", "_").replace("/", "_").replace(" ", "_")
    if asset == "mutual_fund":
        pred_path = OUT / f"mf_{safe}_{model}_predictions.csv"
    else:
        pred_path = OUT / f"{asset}_{safe}_{model}_predictions.csv"
    if not pred_path.exists():
        continue
    df = pd.read_csv(pred_path, parse_dates=["date"])
    actual_col = "actual_nav" if "actual_nav" in df.columns else "actual"
    pred_col = "predicted_nav" if "predicted_nav" in df.columns else "predicted"
    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df[actual_col], label="Actual")
    plt.plot(df["date"], df[pred_col], label=f"Predicted ({model})")
    plt.title(f"{series} — {model} forecast")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / f"{asset}_{safe}_{model}.png", dpi=180)
    plt.close()

summary = OUT / "final_forecasting_asset_summary.csv"
if summary.exists():
    df = pd.read_csv(summary)
    plt.figure(figsize=(10, 5))
    plt.bar(df["asset"], df["best_r2"])
    plt.axhline(0, linewidth=1)
    plt.title("Best R² by asset class")
    plt.xlabel("Asset class")
    plt.ylabel("R²")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(PLOTS / "asset_class_best_r2.png", dpi=180)
    plt.close()

print("Final forecast visualizations saved to", PLOTS)
