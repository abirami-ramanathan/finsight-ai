import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from finsight.data.split import prepare_data
from finsight.data.features import feature_columns
from finsight.models.forecaster import SequenceForecaster


os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

symbol = "MINDSPACE.NS"
asset = "reit"
lookback = 60

path = Path(
    f"data/processed/reit/{symbol.replace('.', '_')}.csv"
)

df = pd.read_csv(path)

columns = feature_columns(asset)

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
    lookback=lookback
)

print("=" * 70)
print("MINDSPACE REIT - 60 DAY LOOKBACK")
print("=" * 70)

print(f"Rows: {len(df)}")
print(f"Features: {columns}")
print(f"Lookback: {lookback}")
print(f"Train sequences: {len(X_train)}")
print(f"Validation sequences: {len(X_val)}")
print(f"Test sequences: {len(X_test)}")

results = []

for model_name in ["lstm", "bilstm", "gru", "gru_vol"]:

    print(f"\nTraining {model_name}...")

    tf.keras.backend.clear_session()

    tf.random.set_seed(42)
    np.random.seed(42)

    model = SequenceForecaster(
        kind=model_name,
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
        epochs=50,
        batch_size=32
    )

    predictions = model.predict(X_test)

    pred_full = np.zeros(
        (len(predictions), len(columns))
    )

    true_full = np.zeros(
        (len(y_test), len(columns))
    )

    pred_full[:, 0] = predictions
    true_full[:, 0] = y_test

    predictions_original = scaler.inverse_transform(
        pred_full
    )[:, 0]

    true_original = scaler.inverse_transform(
        true_full
    )[:, 0]

    metrics = SequenceForecaster.metrics(
        true_original,
        predictions_original
    )

    results.append({
        "symbol": symbol,
        "lookback": lookback,
        "model": model_name,
        **metrics
    })

    model_dir = Path(
        f"models/forecasting/reit_{symbol.replace('.', '_')}_{model_name}_lb60"
    )

    model.save(
        model_dir,
        {
            "asset_type": asset,
            "symbol": symbol,
            "model": model_name,
            "features": columns,
            "lookback": lookback,
            "seed": 42
        }
    )

    joblib.dump(
        scaler,
        model_dir / "scaler.pkl"
    )

    print(
        f"MAE={metrics['mae']:.4f} | "
        f"RMSE={metrics['rmse']:.4f} | "
        f"R2={metrics['r2']:.4f} | "
        f"MAPE={metrics['mape']:.4f}"
    )

results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/mindspace_lookback60_results.csv",
    index=False
)

print("\n" + "=" * 70)
print("MINDSPACE 60-DAY RESULTS")
print("=" * 70)

print(
    results_df.to_string(index=False)
)

print("\nSaved to:")
print("outputs/mindspace_lookback60_results.csv")