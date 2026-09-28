import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import joblib

from config import PROCESSED_DIR, MODEL_DIR, LOOKBACK, EPOCHS, BATCH_SIZE
from finsight.data.features import feature_columns
from finsight.data.split import prepare_data
from finsight.models.forecaster import SequenceForecaster


parser = argparse.ArgumentParser()
parser.add_argument("--asset", required=True, choices=["stock", "crypto", "reit"])
parser.add_argument("--symbol", required=True)
parser.add_argument(
    "--model",
    default="gru_vol",
    choices=["lstm", "bilstm", "gru", "gru_vol"]
)
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


model = SequenceForecaster(
    kind=args.model,
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


model_folder = (
    MODEL_DIR
    / "forecasting"
    / f"{args.asset}_{args.symbol.replace('/', '_').replace('.', '_')}"
)

model_folder.mkdir(parents=True, exist_ok=True)


metadata = {
    "asset": args.asset,
    "symbol": args.symbol,
    "kind": args.model,
    "input_shape": list(X_train.shape[1:]),
    "learning_rate": 0.001,
    "lookback": LOOKBACK,
    "features": columns,
    "train_samples": len(X_train),
    "validation_samples": len(X_val),
    "test_samples": len(X_test)
}


model.save(
    model_folder,
    metadata
)


joblib.dump(
    scaler,
    model_folder / "scaler.pkl"
)


print("=" * 60)
print("FINsight AI — Forecast Training")
print("=" * 60)
print(f"Asset       : {args.asset}")
print(f"Symbol      : {args.symbol}")
print(f"Model       : {args.model}")
print(f"Features    : {len(columns)}")
print(f"Train       : {len(X_train)}")
print(f"Validation  : {len(X_val)}")
print(f"Test        : {len(X_test)}")
print(f"Scaler      : {model_folder / 'scaler.pkl'}")
print(f"Model saved : {model_folder}")
print("=" * 60)