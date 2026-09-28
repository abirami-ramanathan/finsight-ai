from pathlib import Path

import numpy as np
import pandas as pd

from finsight.data.split import prepare_return_data
from finsight.models.forecaster import SequenceForecaster
from finsight.data.features import feature_columns


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed" / "stock"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gru_vol"
LOOKBACK = 30
EPOCHS = 20
BATCH_SIZE = 64

columns = feature_columns("stock")

results = []

files = sorted(DATA_DIR.glob("*.csv"))

print(f"Stock datasets found: {len(files)}")
print(f"Model: {MODEL}")
print(f"Target: next-day log return")
print(f"Lookback: {LOOKBACK}")
print("=" * 70)

for idx, path in enumerate(files, 1):

    symbol = path.stem

    print(f"\n[{idx}/{len(files)}] Training {symbol}")

    try:

        df = pd.read_csv(path)

        X_train, y_train, X_val, y_val, X_test, y_test, scaler, train_df, val_df, test_df = prepare_return_data(
            df,
            columns,
            lookback=LOOKBACK
        )

        model = SequenceForecaster(
            kind=MODEL,
            input_shape=(
                X_train.shape[1],
                X_train.shape[2]
            )
        )

        model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )

        predicted_return = model.predict(X_test)

        actual_return = y_test

        previous_close = test_df["close"].iloc[:len(predicted_return)].values

        predicted_price = (
            previous_close *
            np.exp(predicted_return)
        )

        actual_price = (
            previous_close *
            np.exp(actual_return)
        )

        metrics = SequenceForecaster.metrics(
            actual_price,
            predicted_price
        )

        results.append({
            "symbol": symbol,
            "model": MODEL,
            "target": "next_day_log_return",
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

        print(
            f"FAILED {symbol}: {e}"
        )

results_df = pd.DataFrame(results)

results_path = (
    OUTPUT_DIR /
    "stock_return_forecasting_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\n" + "=" * 70)
print("RETURN-TARGET STOCK FORECASTING COMPLETE")
print("=" * 70)

print(
    f"Successful stocks: "
    f"{len(results_df)}"
)

print(
    f"Results saved to: "
    f"{results_path}"
)

if not results_df.empty:

    print("\nResults:")

    print(
        results_df
        .sort_values(
            "r2",
            ascending=False
        )
        .to_string(
            index=False
        )
    )