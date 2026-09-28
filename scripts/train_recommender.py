from pathlib import Path
import sys

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

import numpy as np
import pandas as pd

from config import MODEL_DIR
from finsight.recommendation.recommender import InvestorRecommender


ROOT = Path(__file__).resolve().parents[1]

RESULTS_PATH = (
    ROOT
    / "outputs"
    / "final_forecasting_results.csv"
)

OUTPUT_PATH = (
    MODEL_DIR
    / "recommendation"
    / "recommender.joblib"
)


def normalize_series(series):
    values = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0.0)

    minimum = values.min()
    maximum = values.max()

    if maximum - minimum < 1e-12:
        return pd.Series(
            np.ones(len(values)),
            index=values.index
        )

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


def asset_risk(asset_type):
    mapping = {
        "stock": 0.55,
        "mutual_fund": 0.30,
        "gold": 0.25,
        "crypto": 0.95,
        "reit": 0.45
    }

    return mapping.get(
        str(asset_type).lower(),
        0.50
    )


def asset_horizon(asset_type):
    mapping = {
        "stock": 0.55,
        "mutual_fund": 0.80,
        "gold": 0.70,
        "crypto": 0.35,
        "reit": 0.75
    }

    return mapping.get(
        str(asset_type).lower(),
        0.60
    )


def build_item_features(df):
    result = df.copy()

    result["r2"] = pd.to_numeric(
        result["r2"],
        errors="coerce"
    ).fillna(0.0)

    result["mape"] = pd.to_numeric(
        result["mape"],
        errors="coerce"
    ).fillna(0.0)

    result["quality"] = normalize_series(
        result["r2"]
    )

    result["accuracy"] = (
        1.0
        / (
            1.0
            + result["mape"].clip(
                lower=0
            )
        )
    )

    result["accuracy"] = normalize_series(
        result["accuracy"]
    )

    result["risk"] = result[
        "asset_type"
    ].apply(
        asset_risk
    )

    result["horizon"] = result[
        "asset_type"
    ].apply(
        asset_horizon
    )

    result["model_quality"] = (
        0.6 * result["quality"]
        + 0.4 * result["accuracy"]
    )

    features = result[
        [
            "quality",
            "accuracy",
            "risk",
            "horizon",
            "model_quality"
        ]
    ].astype(float).values

    return (
        result,
        features
    )


def main():
    print("=" * 70)
    print("FINSIGHT AI - MODULE C")
    print("KNN + LinUCB RECOMMENDATION MODEL")
    print("=" * 70)

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Forecasting results not found: {RESULTS_PATH}"
        )

    df = pd.read_csv(
        RESULTS_PATH
    )

    required_columns = [
        "asset_type",
        "symbol",
        "model",
        "r2",
        "mape"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    df = df.dropna(
        subset=[
            "asset_type",
            "symbol",
            "model"
        ]
    ).reset_index(
        drop=True
    )

    print()
    print(
        f"Forecasting experiments loaded: {len(df)}"
    )

    prepared, item_features = (
        build_item_features(df)
    )

    item_names = []

    for _, row in prepared.iterrows():
        item_names.append(
            f"{row['asset_type']}|"
            f"{row['symbol']}|"
            f"{row['model']}"
        )

    recommender = InvestorRecommender(
        k=5,
        alpha=1.0
    )

    recommender.fit(
        item_features,
        item_names
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    recommender.save(
        OUTPUT_PATH
    )

    print()
    print(
        f"Recommendation candidates: "
        f"{len(item_names)}"
    )

    print(
        f"Feature dimensions: "
        f"{item_features.shape[1]}"
    )

    print()
    print(
        "Feature space:"
    )

    print(
        "1. Forecast quality"
    )

    print(
        "2. Forecast accuracy"
    )

    print(
        "3. Asset risk"
    )

    print(
        "4. Investment horizon"
    )

    print(
        "5. Overall model quality"
    )

    print()
    print(
        f"Saved recommender: {OUTPUT_PATH}"
    )

    print()
    print(
        "Module C training complete."
    )


if __name__ == "__main__":
    main()