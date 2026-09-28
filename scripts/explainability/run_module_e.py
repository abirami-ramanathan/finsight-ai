from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finsight.explainability.explanation_service import (
    ExplanationService
)


MODEL_DIR = ROOT / "models" / "forecasting"
PROCESSED_DIR = ROOT / "data" / "processed"


STOCK_FEATURES = [
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
    "garch_style_vol"
]


CRYPTO_FEATURES = STOCK_FEATURES + [
    "volatility_5"
]


REIT_FEATURES = STOCK_FEATURES + [
    "dividend_yield",
    "occupancy_rate"
]


GOLD_FEATURES = STOCK_FEATURES


PREFERRED_MODELS = [
    ("stock", "HDFCBANK_NS", "gru_vol"),
    ("stock", "RELIANCE_NS", "gru_vol"),
    ("stock", "TCS_NS", "gru_vol"),
    ("stock", "INFY_NS", "gru_vol"),
    ("crypto", "BTC_USD", "bilstm"),
    ("crypto", "BTC_USD", "gru_vol"),
    ("crypto", "ETH_USD", "gru_vol"),
    ("reit", "MINDSPACE_NS", "gru_vol"),
    ("reit", "EMBASSY_NS", "gru_vol"),
    ("reit", "BIRET_NS", "gru_vol"),
    ("gold", "GOLDBEES_NS", "gru_vol")
]


def discover_models():
    return sorted(
        MODEL_DIR.glob("*/model.keras")
    )


def parse_model_metadata(model_path):
    name = model_path.parent.name
    parts = name.split("_")

    if len(parts) < 3:
        raise ValueError(
            f"Cannot parse model name: {name}"
        )

    asset_type = parts[0]
    model_name = parts[-1]
    symbol = "_".join(parts[1:-1])

    return (
        asset_type,
        symbol,
        model_name
    )


def select_model(model_paths):
    metadata = []

    for path in model_paths:
        try:
            asset_type, symbol, model_name = (
                parse_model_metadata(path)
            )

            metadata.append(
                (
                    path,
                    asset_type,
                    symbol,
                    model_name
                )
            )

        except ValueError:
            continue

    for preferred in PREFERRED_MODELS:
        for item in metadata:
            (
                path,
                asset_type,
                symbol,
                model_name
            ) = item

            if (
                asset_type,
                symbol,
                model_name
            ) == preferred:
                return item

    if not metadata:
        raise FileNotFoundError(
            "No compatible forecasting models found."
        )

    return metadata[0]


def get_feature_names(asset_type):
    if asset_type == "crypto":
        return CRYPTO_FEATURES

    if asset_type == "reit":
        return REIT_FEATURES

    if asset_type == "gold":
        return GOLD_FEATURES

    return STOCK_FEATURES


def find_dataset(asset_type, symbol):
    if asset_type == "stock":
        directory = PROCESSED_DIR / "stock"

    elif asset_type == "crypto":
        directory = PROCESSED_DIR / "crypto"

    elif asset_type == "reit":
        directory = PROCESSED_DIR / "reit"

    elif asset_type == "gold":
        directory = PROCESSED_DIR / "gold"

    else:
        raise ValueError(
            f"Unsupported asset type: {asset_type}"
        )

    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {directory}"
        )

    normalized_symbol = (
        symbol.lower()
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
    )

    candidates = []

    for path in directory.glob("*.csv"):
        normalized_name = (
            path.stem.lower()
            .replace("_", "")
            .replace("-", "")
            .replace("/", "")
        )

        if normalized_symbol in normalized_name:
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No dataset found for {asset_type}/{symbol}"
        )

    return sorted(candidates)[0]


def chronological_split(df):
    n = len(df)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return (
        train_df,
        val_df,
        test_df
    )


def prepare_input(
    dataset_path,
    feature_names,
    lookback,
    expected_features
):
    df = pd.read_csv(dataset_path)

    missing = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing features in dataset: {missing}"
        )

    selected_features = (
        df[feature_names]
        .copy()
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
        .reset_index(drop=True)
    )

    if len(selected_features) < lookback:
        raise ValueError(
            "Dataset does not contain enough "
            "rows for the required lookback."
        )

    if len(feature_names) != expected_features:
        raise ValueError(
            f"Feature mismatch: dataset has "
            f"{len(feature_names)} features but "
            f"model expects {expected_features}."
        )

    train_df, val_df, test_df = (
        chronological_split(
            selected_features
        )
    )

    scaler = MinMaxScaler()

    scaler.fit(
        train_df[feature_names].astype(float)
    )

    scaled = scaler.transform(
        selected_features[feature_names].astype(float)
    )

    values = scaled[-lookback:]

    return np.expand_dims(
        values.astype(np.float32),
        axis=0
    )


def main():
    print("=" * 70)
    print("MODULE E — EXPLAINABLE AI")
    print("=" * 70)

    model_paths = discover_models()

    print(
        f"Saved forecasting models found: "
        f"{len(model_paths)}"
    )

    if not model_paths:
        raise FileNotFoundError(
            "No trained forecasting models found."
        )

    (
        model_path,
        asset_type,
        symbol,
        model_name
    ) = select_model(model_paths)

    print()
    print("Selected model:")
    print(model_path)

    print()
    print("Model metadata:")
    print(f"Asset type : {asset_type}")
    print(f"Symbol     : {symbol}")
    print(f"Model      : {model_name}")

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    print()
    print("Model loaded successfully.")

    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    expected_features = input_shape[-1]
    lookback = input_shape[1]

    print(
        f"Model input shape: {input_shape}"
    )

    print(
        f"Expected features: "
        f"{expected_features}"
    )

    print(
        f"Expected lookback: "
        f"{lookback}"
    )

    feature_names = get_feature_names(
        asset_type
    )

    dataset_path = find_dataset(
        asset_type,
        symbol
    )

    print()
    print("Matched dataset:")
    print(dataset_path)

    x = prepare_input(
        dataset_path=dataset_path,
        feature_names=feature_names,
        lookback=lookback,
        expected_features=expected_features
    )

    print()
    print(
        f"Prepared input shape: {x.shape}"
    )

    print(
        f"Features used: {feature_names}"
    )

    print()
    print(
        f"Input range: "
        f"{x.min():.6f} to {x.max():.6f}"
    )

    service = ExplanationService(
        output_dir=(
            ROOT /
            "outputs" /
            "explainability"
        )
    )

    result = service.explain_recurrent_prediction(
        model=model,
        x=x,
        feature_names=feature_names,
        asset_name=symbol,
        model_name=model_name
    )

    print()
    print("=" * 70)
    print("PREDICTION")
    print("=" * 70)

    print(
        f"{result['prediction']:.8f}"
    )

    print()
    print("BASELINE PREDICTION")

    print(
        f"{result['baseline_prediction']:.8f}"
    )

    print()
    print("PREDICTION DIFFERENCE")

    print(
        f"{result['prediction_difference']:.8f}"
    )

    print()
    print("TOP FEATURE ATTRIBUTIONS")

    for item in result[
        "feature_attributions"
    ][:10]:

        direction = (
            "POSITIVE"
            if item["attribution"] >= 0
            else "NEGATIVE"
        )

        print(
            f"{item['feature']:<25}"
            f"{item['attribution']:>12.8f} "
            f"{direction}"
        )

    print()
    print("TOP TEMPORAL CONTRIBUTIONS")

    for item in result[
        "temporal_attributions"
    ][:10]:

        print(
            f"Timestep {item['timestep']:<4}"
            f" Position {item['relative_position']:<5}"
            f" Attribution "
            f"{item['attribution']:>12.8f}"
        )

    print()
    print("COMPLETENESS CHECK")

    print(
        f"Attribution sum: "
        f"{result['completeness_sum']:.8f}"
    )

    print(
        f"Prediction difference: "
        f"{result['prediction_difference']:.8f}"
    )

    print(
        f"Completeness error: "
        f"{result['completeness_error']:.8f}"
    )

    print()
    print("WHY THIS FORECAST?")
    print(result["explanation"])

    print()
    print("=" * 70)
    print("MODULE E COMPLETED")
    print("=" * 70)

    output_dir = (
        ROOT /
        "outputs" /
        "explainability"
    )

    print()
    print("Results saved to:")
    print(output_dir)

    safe_asset = (
        str(symbol)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

    safe_model = (
        str(model_name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

    print()
    print("Generated files:")

    print(
        output_dir /
        f"{safe_asset}_{safe_model}_ig.json"
    )

    print(
        output_dir /
        f"{safe_asset}_{safe_model}_ig.png"
    )

    print(
        output_dir /
        f"{safe_asset}_{safe_model}_temporal_heatmap.png"
    )


if __name__ == "__main__":
    main()