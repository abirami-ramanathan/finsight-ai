from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finsight.drift.monitor import DriftMonitor
from finsight.drift.controller import DriftController


OUTPUT_DIR = ROOT / "outputs" / "drift"
MODEL_DIR = ROOT / "models" / "forecasting"
VERSION_DIR = ROOT / "models" / "drift_versions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VERSION_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "close",
    "volume",
    "return_1d",
    "ema_12",
    "rsi_14",
    "mfi_14",
    "stoch_k",
    "stoch_d",
    "bb_high",
    "bb_low",
    "macd",
    "macd_signal",
    "volatility_20",
    "garch_style_vol",
    "volatility_5"
]


def find_model():
    preferred = [
        "crypto_BTC_USD_bilstm",
        "crypto_BTC_USD_gru_vol",
        "crypto_BTC_USD_gru",
        "stock_ITC_NS_gru_vol",
        "stock_ITC_NS_bilstm",
        "stock_ITC_NS_gru"
    ]

    for name in preferred:
        model_path = MODEL_DIR / name / "model.keras"

        if model_path.exists():
            return model_path

    models = list(MODEL_DIR.glob("*/model.keras"))

    if not models:
        raise FileNotFoundError("No forecasting model found.")

    return models[0]


def infer_dataset(model_path):
    name = model_path.parent.name

    model_names = [
        "gru_vol",
        "bilstm",
        "lstm",
        "gru"
    ]

    if name.startswith("crypto_"):
        asset_type = "crypto"
        remainder = name[len("crypto_"):]

        model_name = None

        for candidate in model_names:
            suffix = f"_{candidate}"

            if remainder.endswith(suffix):
                model_name = candidate
                symbol = remainder[:-len(suffix)]
                break

        if model_name is None:
            raise ValueError(f"Cannot parse model: {name}")

        dataset_map = {
            "BTC_USD": "BTC-USD.csv",
            "ETH_USD": "ETH-USD.csv"
        }

        if symbol not in dataset_map:
            raise ValueError(f"Unsupported crypto symbol: {symbol}")

        dataset = (
            ROOT
            / "data"
            / "processed"
            / "crypto"
            / dataset_map[symbol]
        )

        return asset_type, symbol, model_name, dataset

    if name.startswith("stock_"):
        asset_type = "stock"
        remainder = name[len("stock_"):]

        model_name = None

        for candidate in model_names:
            suffix = f"_{candidate}"

            if remainder.endswith(suffix):
                model_name = candidate
                symbol = remainder[:-len(suffix)]
                break

        if model_name is None:
            raise ValueError(f"Cannot parse model: {name}")

        dataset = (
            ROOT
            / "data"
            / "processed"
            / "stock"
            / f"{symbol}.csv"
        )

        return asset_type, symbol, model_name, dataset

    raise ValueError(f"Unknown model type: {name}")


def prepare_dataframe(path):
    df = pd.read_csv(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

    available = [
        feature
        for feature in FEATURES
        if feature in df.columns
    ]

    if len(available) < 5:
        raise ValueError("Insufficient forecasting features.")

    df = (
        df
        .dropna(subset=available)
        .reset_index(drop=True)
    )

    df["target_return"] = np.log(
        df["close"].shift(-1) / df["close"]
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = (
        df
        .dropna(subset=["target_return"])
        .reset_index(drop=True)
    )

    return df, available


def create_sequences(
    df,
    features,
    scaler,
    lookback
):
    values = (
        df[features]
        .astype(float)
    )

    scaled = scaler.transform(values)

    X = []
    y = []

    for i in range(
        lookback,
        len(df)
    ):
        X.append(
            scaled[i - lookback:i]
        )

        y.append(
            df["target_return"].iloc[i]
        )

    return (
        np.asarray(X),
        np.asarray(y)
    )


def inject_drift(
    df,
    features,
    strength
):
    drifted = df.copy()

    for feature in features:
        values = (
            drifted[feature]
            .astype(float)
        )

        if feature in [
            "close",
            "volume",
            "ema_12",
            "bb_high",
            "bb_low"
        ]:
            drifted[feature] = (
                values * (1.0 + strength)
            )

        else:
            mean = values.mean()
            std = values.std()

            if (
                not np.isfinite(std)
                or std == 0
            ):
                std = 1.0

            drifted[feature] = (
                values + strength * std
            )

    return drifted


def create_fixed_baseline(
    current_df,
    size
):
    if len(current_df) < size:
        size = len(current_df)

    baseline = (
        current_df
        .iloc[:size]
        .copy()
        .reset_index(drop=True)
    )

    return baseline


def build_replay_periods(
    baseline,
    periods
):
    replay_periods = []

    for _ in range(periods):
        replay_periods.append(
            baseline.copy()
        )

    return replay_periods


def retrain_model(
    original_model_path,
    X_train,
    y_train,
    version
):
    retrained = load_model(
        original_model_path
    )

    retrained.fit(
        X_train,
        y_train,
        epochs=3,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )

    version_dir = (
        VERSION_DIR
        / f"version_{version:02d}"
    )

    version_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_model = (
        version_dir
        / "model.keras"
    )

    retrained.save(
        output_model
    )

    return output_model


def save_plot(
    x,
    y,
    title,
    ylabel,
    threshold,
    threshold_label,
    path
):
    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        x,
        y,
        marker="o"
    )

    plt.axhline(
        threshold,
        linestyle="--",
        label=threshold_label
    )

    plt.xlabel(
        "Replay Period"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()


def main():
    print("=" * 70)
    print("FINSIGHT AI - MODULE F")
    print("DRIFT MONITORING + AUTOMATIC RETRAINING")
    print("=" * 70)

    model_path = find_model()

    print()
    print("Selected model:")
    print(model_path)

    (
        asset_type,
        symbol,
        model_name,
        dataset
    ) = infer_dataset(
        model_path
    )

    print()
    print("Asset type:", asset_type)
    print("Symbol:", symbol)
    print("Model:", model_name)
    print("Dataset:", dataset)

    if not dataset.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset}"
        )

    df, available_features = (
        prepare_dataframe(
            dataset
        )
    )

    model = load_model(
        model_path
    )

    input_shape = model.input_shape

    lookback = int(
        input_shape[1]
    )

    expected_features = int(
        input_shape[2]
    )

    features = [
        feature
        for feature in FEATURES
        if feature in available_features
    ]

    features = features[
        :expected_features
    ]

    if len(features) != expected_features:
        raise ValueError(
            f"Model expects "
            f"{expected_features} "
            f"features but "
            f"{len(features)} "
            f"are available."
        )

    print()
    print("Lookback:", lookback)
    print("Features:", len(features))

    split_index = int(
        len(df) * 0.70
    )

    reference_training_df = (
        df.iloc[:split_index]
        .copy()
    )

    current_df = (
        df.iloc[split_index:]
        .copy()
        .reset_index(drop=True)
    )

    scaler = MinMaxScaler()

    scaler.fit(
        reference_training_df[
            features
        ].astype(float)
    )

    X_all, y_all = (
        create_sequences(
            df,
            features,
            scaler,
            lookback
        )
    )

    predictions = (
        model.predict(
            X_all,
            verbose=0
        )
        .reshape(-1)
    )

    actual = y_all

    periods = 26

    period_size = (
        len(current_df)
        // periods
    )

    if period_size < 10:
        raise ValueError(
            "Not enough data for "
            "26 replay periods."
        )

    baseline_size = min(
        period_size,
        len(current_df)
    )

    baseline = create_fixed_baseline(
        current_df,
        baseline_size
    )

    replay_periods = build_replay_periods(
        baseline,
        periods
    )

    monitor = DriftMonitor(
        psi_threshold=0.20,
        kl_threshold=0.10,
        error_window=20,
        bins=10
    )

    controller = DriftController(
        monitor=monitor,
        model_dir=VERSION_DIR,
        psi_threshold=0.20,
        retrain_cooldown=4
    )

    all_results = []
    all_feature_results = []

    retraining_count = 0

    print()
    print(
        "Fixed baseline rows:",
        len(baseline)
    )

    print()
    print(
        "Starting controlled "
        "26-period drift replay..."
    )

    print()

    for period in range(periods):

        start = (
            period
            * period_size
        )

        if period == periods - 1:
            end = len(current_df)
        else:
            end = (
                start
                + period_size
            )

        if period < 6:
            drift_strength = 0.0

            original_current = (
                baseline.copy()
            )

        else:
            original_current = (
                replay_periods[period]
                .copy()
            )

            if period < 10:
                drift_strength = 0.015
            elif period < 14:
                drift_strength = 0.025
            elif period < 18:
                drift_strength = 0.040
            elif period < 22:
                drift_strength = 0.055
            else:
                drift_strength = 0.070

        replay_current = inject_drift(
            original_current,
            features,
            drift_strength
        )

        summary, feature_drift = (
            monitor.evaluate(
                baseline,
                replay_current,
                features
            )
        )

        actual_start = max(
            0,
            start
        )

        actual_end = min(
            len(X_all),
            end
        )

        if actual_end > actual_start:
            period_actual = (
                actual[
                    actual_start:actual_end
                ]
            )

            period_predicted = (
                predictions[
                    actual_start:actual_end
                ]
            )

            summary[
                "rolling_error"
            ] = monitor.rolling_error(
                period_actual,
                period_predicted
            )

        else:
            summary[
                "rolling_error"
            ] = 0.0

        retrain = (
            controller.should_retrain(
                summary,
                period + 1
            )
        )

        model_version = None

        if retrain:
            retraining_count += 1

            train_end = min(
                len(X_all),
                actual_end
            )

            if train_end > 50:
                model_version = (
                    retrain_model(
                        model_path,
                        X_all[:train_end],
                        y_all[:train_end],
                        retraining_count
                    )
                )

        controller.register_event(
            period=period + 1,
            summary=summary,
            retrained=retrain,
            model_version=model_version
        )

        result = {
            "period": period + 1,
            "drift_strength": drift_strength,
            "max_psi": summary["max_psi"],
            "mean_psi": summary["mean_psi"],
            "max_kl": summary["max_kl"],
            "mean_kl": summary["mean_kl"],
            "rolling_error": summary["rolling_error"],
            "drifted_features": summary[
                "drifted_features"
            ],
            "drift_detected": summary[
                "drift_detected"
            ],
            "retrained": retrain,
            "model_version": (
                str(model_version)
                if model_version
                else ""
            )
        }

        all_results.append(
            result
        )

        feature_drift[
            "period"
        ] = period + 1

        feature_drift[
            "drift_strength"
        ] = drift_strength

        all_feature_results.append(
            feature_drift
        )

        print(
            f"Period {period + 1:02d} | "
            f"Drift={drift_strength:.3f} | "
            f"PSI={summary['max_psi']:.4f} | "
            f"KL={summary['max_kl']:.4f} | "
            f"Error={summary['rolling_error']:.6f} | "
            f"Detected={summary['drift_detected']} | "
            f"Retrain={retrain}"
        )

    results_df = pd.DataFrame(
        all_results
    )

    feature_df = pd.concat(
        all_feature_results,
        ignore_index=True
    )

    results_df.to_csv(
        OUTPUT_DIR
        / "drift_replay_results.csv",
        index=False
    )

    feature_df.to_csv(
        OUTPUT_DIR
        / "feature_drift_results.csv",
        index=False
    )

    controller.save_events(
        OUTPUT_DIR
        / "retraining_events.json"
    )

    summary = {
        "asset_type": asset_type,
        "symbol": symbol,
        "model": model_name,
        "replay_type": (
            "controlled_distribution_drift"
        ),
        "baseline_type": (
            "fixed_reference_window"
        ),
        "periods_evaluated": int(
            len(results_df)
        ),
        "baseline_periods": 6,
        "psi_threshold": 0.20,
        "kl_threshold": 0.10,
        "drift_periods": int(
            results_df[
                "drift_detected"
            ].sum()
        ),
        "retraining_events": int(
            results_df[
                "retrained"
            ].sum()
        ),
        "maximum_psi": float(
            results_df[
                "max_psi"
            ].max()
        ),
        "maximum_kl": float(
            results_df[
                "max_kl"
            ].max()
        ),
        "minimum_psi": float(
            results_df[
                "max_psi"
            ].min()
        ),
        "mean_rolling_error": float(
            results_df[
                "rolling_error"
            ].mean()
        )
    }

    with open(
        OUTPUT_DIR
        / "drift_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    save_plot(
        results_df["period"],
        results_df["max_psi"],
        "Feature Drift Detection using PSI",
        "Maximum PSI",
        0.20,
        "PSI Threshold = 0.20",
        OUTPUT_DIR
        / "psi_drift.png"
    )

    save_plot(
        results_df["period"],
        results_df["max_kl"],
        "Feature Drift Detection using KL Divergence",
        "Maximum KL Divergence",
        0.10,
        "KL Threshold = 0.10",
        OUTPUT_DIR
        / "kl_divergence.png"
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        results_df["period"],
        results_df["rolling_error"],
        marker="o"
    )

    plt.xlabel(
        "Replay Period"
    )

    plt.ylabel(
        "Rolling Absolute Error"
    )

    plt.title(
        "Forecast Error under Distribution Drift"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "rolling_error.png",
        dpi=200
    )

    plt.close()

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        results_df["period"],
        results_df["max_psi"],
        marker="o",
        label="PSI"
    )

    retrained = results_df[
        results_df["retrained"]
    ]

    if not retrained.empty:
        plt.scatter(
            retrained["period"],
            retrained["max_psi"],
            s=80,
            label="Retraining Trigger"
        )

    plt.axhline(
        0.20,
        linestyle="--",
        label="PSI Threshold"
    )

    plt.xlabel(
        "Replay Period"
    )

    plt.ylabel(
        "Maximum PSI"
    )

    plt.title(
        "Automatic Retraining Trigger"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "automatic_retraining.png",
        dpi=200
    )

    plt.close()

    print()
    print("=" * 70)
    print("MODULE F COMPLETE")
    print("=" * 70)

    print()
    print(
        "Periods evaluated:",
        summary["periods_evaluated"]
    )

    print(
        "Baseline periods:",
        summary["baseline_periods"]
    )

    print(
        "Drift periods:",
        summary["drift_periods"]
    )

    print(
        "Retraining events:",
        summary["retraining_events"]
    )

    print(
        "Minimum PSI:",
        f"{summary['minimum_psi']:.4f}"
    )

    print(
        "Maximum PSI:",
        f"{summary['maximum_psi']:.4f}"
    )

    print(
        "Maximum KL:",
        f"{summary['maximum_kl']:.4f}"
    )

    print()
    print(
        "Outputs:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()