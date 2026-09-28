from pathlib import Path

import numpy as np
import pandas as pd

from finsight.data.split import prepare_data
from finsight.models.forecaster import SequenceForecaster
from finsight.data.features import feature_columns


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed" / "stock"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gru_vol"
LOOKBACK = 30
EPOCHS = 15
BATCH_SIZE = 64

columns = feature_columns("stock")

results = []

files = sorted(DATA_DIR.glob("*.csv"))

print(f"Stock datasets found: {len(files)}")
print(f"Model: {MODEL}")
print(f"Epochs: {EPOCHS}")
print("=" * 70)

for idx, path in enumerate(files, 1):

    symbol = path.stem

    print(f"\n[{idx}/{len(files)}] Training {symbol}")

    try:

        df = pd.read_csv(path)

        if len(df) < LOOKBACK + 100:
            print(f"Skipping {symbol}: insufficient rows")
            continue

        X_train, y_train, X_val, y_val, X_test, y_test, scaler, train_df, val_df, test_df = prepare_data(
            df,
            columns,
            lookback=LOOKBACK
        )

        model = SequenceForecaster(
            kind=MODEL,
            input_shape=(X_train.shape[1], X_train.shape[2])
        )

        model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )

        pred_scaled = model.predict(X_test)

        pred = scaler.inverse_transform(
            np.column_stack([
                pred_scaled,
                np.zeros((len(pred_scaled), len(columns) - 1))
            ])
        )[:, 0]

        actual = scaler.inverse_transform(
            np.column_stack([
                y_test,
                np.zeros((len(y_test), len(columns) - 1))
            ])
        )[:, 0]

        metrics = SequenceForecaster.metrics(
            actual,
            pred
        )

        results.append({
            "symbol": symbol,
            "model": MODEL,
            "lookback": LOOKBACK,
            "mae": metrics["mae"],
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "mape": metrics["mape"]
        })

        print(
            f"{symbol} | "
            f"R2={metrics['r2']:.4f} | "
            f"RMSE={metrics['rmse']:.4f} | "
            f"MAPE={metrics['mape']:.4f}"
        )

    except Exception as e:

        print(f"FAILED {symbol}: {e}")

results_df = pd.DataFrame(results)

results_path = OUTPUT_DIR / "expanded_stock_forecasting_results.csv"

results_df.to_csv(
    results_path,
    index=False
)

print("\n" + "=" * 70)
print("EXPANDED STOCK FORECASTING COMPLETE")
print("=" * 70)

print(f"Successful stocks: {len(results_df)}")
print(f"Results saved to: {results_path}")

if not results_df.empty:

    print("\nResults:")

    print(
        results_df
        .sort_values("r2", ascending=False)
        .to_string(index=False)
    )