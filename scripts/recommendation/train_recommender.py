from pathlib import Path
import numpy as np
import pandas as pd

from finsight.recommendation.recommender import InvestorRecommender


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT /
    "outputs" /
    "recommendation" /
    "recommendation_assets.csv"
)

OUTPUT_DIR = (
    ROOT /
    "outputs" /
    "recommendation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df = pd.read_csv(INPUT)

feature_columns = [
    "risk_score",
    "forecast_quality",
    "forecast_error",
    "horizon_normalized"
]

profiles = {
    "low_short": [0.20, 0.80, 0.20, 30 / 365],
    "low_medium": [0.20, 0.80, 0.20, 90 / 365],
    "low_long": [0.20, 0.80, 0.20, 365 / 365],
    "high_short": [0.80, 0.50, 0.50, 30 / 365],
    "high_medium": [0.80, 0.50, 0.50, 90 / 365],
    "high_long": [0.80, 0.50, 0.50, 365 / 365]
}

all_recommendations = []
training_rows = []

for horizon in [
    "short",
    "medium",
    "long"
]:

    subset = df[
        df["horizon_name"] == horizon
    ].copy()

    item_features = (
        subset[feature_columns]
        .astype(float)
        .values
    )

    item_names = (
        subset["asset_id"]
        .astype(str)
        .tolist()
    )

    recommender = InvestorRecommender(
        k=5,
        alpha=1.0
    )

    recommender.fit(
        item_features,
        item_names
    )

    for cluster, profile in profiles.items():

        if not cluster.endswith(horizon):
            continue

        profile_array = np.asarray(
            profile,
            dtype=float
        )

        candidates, distances = (
            recommender.retrieve(
                profile_array
            )
        )

        for arm in candidates:

            row = subset.iloc[int(arm)]

            risk_match = 1 - abs(
                float(row["risk_score"])
                -
                float(profile_array[0])
            )

            reward = (
                0.60 *
                float(row["forecast_quality"])
                +
                0.25 *
                max(risk_match, 0)
                +
                0.15 *
                (
                    1 -
                    float(row["forecast_error"])
                )
            )

            reward = float(
                np.clip(
                    reward,
                    0,
                    1
                )
            )

            recommender.update(
                profile_array,
                arm,
                reward
            )

            training_rows.append({
                "cluster": cluster,
                "horizon": horizon,
                "asset_id": row["asset_id"],
                "arm": int(arm),
                "reward": reward
            })

        recommendations = (
            recommender.recommend(
                profile_array
            )
        )

        for row in recommendations:

            row["cluster"] = cluster
            row["horizon"] = horizon

            all_recommendations.append(
                row
            )

recommendations_df = pd.DataFrame(
    all_recommendations
)

training_df = pd.DataFrame(
    training_rows
)

recommendation_output = (
    OUTPUT_DIR /
    "knn_linucb_recommendations.csv"
)

training_output = (
    OUTPUT_DIR /
    "linucb_offline_training.csv"
)

recommendations_df.to_csv(
    recommendation_output,
    index=False
)

training_df.to_csv(
    training_output,
    index=False
)

print("=" * 70)
print("MODULE C RECOMMENDER COMPLETE")
print("=" * 70)

print(
    f"Recommendation rows: "
    f"{len(recommendations_df)}"
)

print(
    f"Offline training rows: "
    f"{len(training_df)}"
)

print("\nRecommendations per cluster:")

print(
    recommendations_df
    .groupby("cluster")
    .size()
)

print("\nReward statistics:")

print(
    training_df["reward"]
    .describe()
    .round(4)
)

print("\nSaved:")
print(recommendation_output)
print(training_output)