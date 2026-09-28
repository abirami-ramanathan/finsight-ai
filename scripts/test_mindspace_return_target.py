import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from finsight.data.features import feature_columns
from finsight.models.forecaster import SequenceForecaster


os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

SYMBOL = "MINDSPACE.NS"
LOOKBACK = 30
ASSET = "reit"

path = Path(
    f"data/processed/reit/{SYMBOL.replace('.', '_')}.csv"
)

df = pd.read_csv(path)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

columns = feature_columns(ASSET)

df["target_return"] = df["close"].pct_change().shift(-1)
df = df.dropna().reset_index(drop=True)

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

feature_scaler = StandardScaler()

feature_scaler.fit(
    train_df[columns].astype(float)
)

train_scaled = feature_scaler.transform(
    train_df[columns].astype(float)
)

val_scaled = feature_scaler.transform(
    val_df[columns].astype(float)
)

test_scaled = feature_scaler.transform(
    test_df[columns].astype(float)
)

target_scaler = StandardScaler()

target_scaler.fit(
    train_df[["target_return"]]
)

train_target = target_scaler.transform(
    train_df[["target_return"]]
).reshape(-1)

val_target = target_scaler.transform(
    val_df[["target_return"]]
).reshape(-1)

test_target = target_scaler.transform(
    test_df[["target_return"]]
).reshape(-1)


def create_sequences(features, target, lookback):
    X = []
    y = []

    for i in range(lookback, len(features)):
        X.append(features[i - lookback:i])
        y.append(target[i])

    return np.asarray(X), np.asarray(y)


X_train, y_train = create_sequences(
    train_scaled,
    train_target,
    LOOKBACK
)

val_input = np.concatenate(
    [train_scaled[-LOOKBACK:], val_scaled],
    axis=0
)

val_target_input = np.concatenate(
    [train_target[-LOOKBACK:], val_target],
    axis=0
)

X_val, y_val = create_sequences(
    val_input,
    val_target_input,
    LOOKBACK
)

test_input = np.concatenate(
    [val_scaled[-LOOKBACK:], test_scaled],
    axis=0
)

test_target_input = np.concatenate(
    [val_target[-LOOKBACK:], test_target],
    axis=0
)

X_test, y_test = create_sequences(
    test_input,
    test_target_input,
    LOOKBACK
)

predicted_test_dates = test_df["date"].reset_index(drop=True)

previous_prices = pd.concat(
    [
        val_df["close"].iloc[-1:],
        test_df["close"].iloc[:-1]
    ],
    ignore_index=True
)

actual_prices = test_df["close"].reset_index(drop=True)

print("=" * 70)
print("MINDSPACE RETURN-TARGET FORECAST")
print("=" * 70)
print(f"Test sequences: {len(X_test)}")
print(f"Previous prices: {len(previous_prices)}")
print(f"Actual prices: {len(actual_prices)}")

results = []

for model_name in [
    "lstm",
    "bilstm",
    "gru",
    "gru_vol"
]:

    print(f"\nTraining {model_name}...")

    tf.keras.backend.clear_session()

    np.random.seed(42)
    tf.random.set_seed(42)

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

    predicted_scaled_returns = model.predict(
        X_test
    )

    predicted_returns = target_scaler.inverse_transform(
        predicted_scaled_returns.reshape(-1, 1)
    ).reshape(-1)

    predicted_prices = (
        previous_prices.to_numpy()
        * (1 + predicted_returns)
    )

    actual_prices_array = actual_prices.to_numpy()

    metrics = SequenceForecaster.metrics(
        actual_prices_array,
        predicted_prices
    )

    results.append({
        "symbol": SYMBOL,
        "target": "next_return",
        "lookback": LOOKBACK,
        "model": model_name,
        **metrics
    })

    print(
        f"MAE={metrics['mae']:.4f} | "
        f"RMSE={metrics['rmse']:.4f} | "
        f"R2={metrics['r2']:.4f} | "
        f"MAPE={metrics['mape']:.4f}"
    )


results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/mindspace_return_target_results.csv",
    index=False
)

print("\n" + "=" * 70)
print("MINDSPACE RETURN-TARGET RESULTS")
print("=" * 70)

print(
    results_df.to_string(index=False)
)

print("\nSaved to:")
print(
    "outputs/mindspace_return_target_results.csv"
)