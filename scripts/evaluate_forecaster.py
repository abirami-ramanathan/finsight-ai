import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from config import PROCESSED_DIR, MODEL_DIR, OUTPUT_DIR, LOOKBACK
from finsight.data.features import feature_columns
from finsight.data.split import prepare_data
from finsight.models.forecaster import SequenceForecaster


parser = argparse.ArgumentParser()
parser.add_argument("--asset", required=True, choices=["stock", "crypto", "reit"])
parser.add_argument("--symbol", required=True)
parser.add_argument("--model", default="gru_vol")
args = parser.parse_args()


filename = args.symbol.replace(".", "_").replace("/", "_") + ".csv"
path = PROCESSED_DIR / args.asset / filename

if not path.exists():
    raise FileNotFoundError(f"Processed dataset not found: {path}")


df = pd.read_csv(path)
columns = feature_columns(args.asset)

(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    scaler,
    train_df,
    val_df,
    test_df
) = prepare_data(
    df,
    columns,
    lookback=LOOKBACK
)


model_folder = (
    MODEL_DIR
    / "forecasting"
    / f"{args.asset}_{args.symbol.replace('/', '_').replace('.', '_')}"
)

model, metadata = SequenceForecaster.load(model_folder)


pred_scaled = model.predict(X_test)


close_index = columns.index("close")

actual_dummy = np.zeros(
    (len(y_test), len(columns))
)

pred_dummy = np.zeros(
    (len(pred_scaled), len(columns))
)

actual_dummy[:, close_index] = y_test
pred_dummy[:, close_index] = pred_scaled


y_test_original = scaler.inverse_transform(
    actual_dummy
)[:, close_index]

pred_original = scaler.inverse_transform(
    pred_dummy
)[:, close_index]


metrics = SequenceForecaster.metrics(
    y_test_original,
    pred_original
)

metrics["asset"] = args.asset
metrics["symbol"] = args.symbol
metrics["model"] = args.model
metrics["test_samples"] = len(y_test_original)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_file = (
    OUTPUT_DIR
    / f"{args.asset}_{args.symbol.replace('.', '_')}_metrics.json"
)

output_file.write_text(
    json.dumps(metrics, indent=4),
    encoding="utf-8"
)


print("=" * 60)
print("FINsight AI — Forecast Evaluation")
print("=" * 60)
print(f"Asset       : {args.asset}")
print(f"Symbol      : {args.symbol}")
print(f"Model       : {args.model}")
print(f"Test samples: {len(y_test_original)}")
print("-" * 60)
print(f"MAE         : {metrics['mae']:.4f}")
print(f"MSE         : {metrics['mse']:.4f}")
print(f"RMSE        : {metrics['rmse']:.4f}")
print(f"R²          : {metrics['r2']:.4f}")
print(f"MAPE        : {metrics['mape']:.4f}%")
print("-" * 60)
print(f"Saved to    : {output_file}")
print("=" * 60)