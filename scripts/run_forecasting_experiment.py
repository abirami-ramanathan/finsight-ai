import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from config import (
    PROCESSED_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    LOOKBACK,
    EPOCHS,
    BATCH_SIZE
)

from finsight.data.features import feature_columns
from finsight.data.split import prepare_data
from finsight.models.forecaster import SequenceForecaster


ASSET = "stock"

SYMBOLS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS"
]

MODELS = [
    "lstm",
    "bilstm",
    "gru",
    "gru_vol"
]


def evaluate_model(model, X_test, y_test, scaler, columns):

    predictions = model.predict(X_test)

    close_index = columns.index("close")

    actual_dummy = np.zeros(
        (len(y_test), len(columns))
    )

    prediction_dummy = np.zeros(
        (len(predictions), len(columns))
    )

    actual_dummy[:, close_index] = y_test
    prediction_dummy[:, close_index] = predictions

    actual = scaler.inverse_transform(
        actual_dummy
    )[:, close_index]

    prediction = scaler.inverse_transform(
        prediction_dummy
    )[:, close_index]

    return SequenceForecaster.metrics(
        actual,
        prediction
    )


def train_stock(symbol):

    print()
    print("#" * 80)
    print(f"STOCK: {symbol}")
    print("#" * 80)

    filename = (
        symbol.replace(".", "_")
        .replace("/", "_")
        + ".csv"
    )

    data_path = (
        PROCESSED_DIR
        / ASSET
        / filename
    )

    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        return []

    df = pd.read_csv(data_path)

    columns = feature_columns(ASSET)

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

    results = []

    for model_name in MODELS:

        print()
        print("=" * 60)
        print(f"Training {symbol} — {model_name}")
        print("=" * 60)

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
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
            scaler,
            columns
        )

        metrics["symbol"] = symbol
        metrics["model"] = model_name

        results.append(metrics)

        model_folder = (
            MODEL_DIR
            / "forecasting"
            / f"stock_{symbol.replace('.', '_')}_{model_name}"
        )

        metadata = {
            "asset": ASSET,
            "symbol": symbol,
            "kind": model_name,
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

    return results


def main():

    all_results = []

    for symbol in SYMBOLS:

        results = train_stock(symbol)

        all_results.extend(results)

    if not all_results:
        raise RuntimeError(
            "No forecasting experiments were completed."
        )

    results_df = pd.DataFrame(
        all_results
    )

    results_df = results_df[
        [
            "symbol",
            "model",
            "mae",
            "mse",
            "rmse",
            "r2",
            "mape"
        ]
    ]

    results_df = results_df.sort_values(
        ["symbol", "model"]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "forecasting_all_stocks.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print()
    print("=" * 90)
    print("FINsight AI — All Stock Forecasting Results")
    print("=" * 90)
    print(
        results_df.to_string(
            index=False
        )
    )
    print("=" * 90)
    print(
        f"Saved to: {output_file}"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()