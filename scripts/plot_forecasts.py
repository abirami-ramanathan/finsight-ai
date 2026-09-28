import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import PROCESSED_DIR, MODEL_DIR, OUTPUT_DIR, LOOKBACK
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


def plot_stock(symbol):

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

    model_folder = (
        MODEL_DIR
        / "forecasting"
        / f"stock_{symbol.replace('.', '_')}_gru_vol"
    )

    model, metadata = SequenceForecaster.load(
        model_folder
    )

    predictions_scaled = model.predict(
        X_test
    )

    close_index = columns.index("close")

    actual_dummy = np.zeros(
        (len(y_test), len(columns))
    )

    prediction_dummy = np.zeros(
        (len(predictions_scaled), len(columns))
    )

    actual_dummy[:, close_index] = y_test
    prediction_dummy[:, close_index] = predictions_scaled

    actual = scaler.inverse_transform(
        actual_dummy
    )[:, close_index]

    predictions = scaler.inverse_transform(
        prediction_dummy
    )[:, close_index]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plot_dir = OUTPUT_DIR / "plots"
    plot_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    test_dates = test_df["date"].values

    test_dates = test_dates[
        :len(actual)
    ]

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        test_dates,
        actual,
        label="Actual"
    )

    plt.plot(
        test_dates,
        predictions,
        label="GRU + GARCH Predicted"
    )

    plt.title(
        f"{symbol} — Actual vs Predicted"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Closing Price"
    )

    plt.xticks(
        rotation=45
    )

    plt.legend()

    plt.tight_layout()

    output_file = (
        plot_dir
        / f"{symbol.replace('.', '_')}_gru_vol_forecast.png"
    )

    plt.savefig(
        output_file,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


def main():

    for symbol in SYMBOLS:

        try:
            plot_stock(symbol)

        except Exception as e:

            print(
                f"Failed {symbol}: {e}"
            )


if __name__ == "__main__":
    main()