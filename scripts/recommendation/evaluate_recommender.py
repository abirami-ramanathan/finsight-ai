from pathlib import Path
import numpy as np
import pandas as pd

from finsight.recommendation.recommender import InvestorRecommender


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "outputs" / "recommendation"

INPUT = DATA_DIR / "recommendation_assets.csv"

OUTPUT = DATA_DIR / "recommendation_evaluation.csv"

ROUNDS_OUTPUT = DATA_DIR / "linucb_2000_rounds.csv"

df = pd.read_csv(INPUT)

feature_columns = [
    "risk_score",
    "forecast_quality",
    "forecast_error",
    "horizon_normalized"
]

rng = np.random.default_rng(42)

profiles = {
    "low_short": [0.20, 0.80, 0.20, 30 / 365],
    "low_medium": [0.20, 0.80, 0.20, 90 / 365],
    "low_long": [0.20, 0.80, 0.20, 365 / 365],
    "high_short": [0.80, 0.50, 0.50, 30 / 365],
    "high_medium": [0.80, 0.50, 0.50, 90 / 365],
    "high_long": [0.80, 0.50, 0.50, 365 / 365]
}

round_counts = {
    "low_short": 334,
    "low_medium": 334,
    "low_long": 333,
    "high_short": 333,
    "high_medium": 333,
    "high_long": 333
}

round_rows = []

cluster_results = []

for cluster, base_profile in profiles.items():

    horizon = cluster.split("_")[1]

    subset = df[
        df["horizon_name"] == horizon
    ].copy().reset_index(drop=True)

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

    profile = np.asarray(
        base_profile,
        dtype=float
    )

    n_rounds = round_counts[cluster]

    for round_number in range(
        1,
        n_rounds + 1
    ):

        context = profile.copy()

        noise = rng.normal(
            0,
            0.03,
            size=context.shape
        )

        context = np.clip(
            context + noise,
            0,
            1
        )

        candidates, distances = (
            recommender.retrieve(context)
        )

        scores = []

        for arm in candidates:

            row = subset.iloc[int(arm)]

            risk_match = (
                1 -
                abs(
                    float(row["risk_score"])
                    -
                    float(context[0])
                )
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

            reward += float(
                rng.normal(
                    0,
                    0.02
                )
            )

            reward = float(
                np.clip(
                    reward,
                    0,
                    1
                )
            )

            scores.append(
                reward
            )

        bandit_scores = [
            recommender.bandit.score(
                context,
                int(arm)
            )
            for arm in candidates
        ]

        selected_position = int(
            np.argmax(
                bandit_scores
            )
        )

        selected_arm = int(
            candidates[selected_position]
        )

        observed_reward = float(
            scores[selected_position]
        )

        optimal_reward = float(
            max(scores)
        )

        regret = max(
            optimal_reward -
            observed_reward,
            0.0
        )

        recommender.update(
            context,
            selected_arm,
            observed_reward
        )

        round_rows.append({
            "cluster": cluster,
            "round": round_number,
            "selected_item": item_names[selected_arm],
            "reward": observed_reward,
            "optimal_reward": optimal_reward,
            "regret": regret
        })

    cluster_df = pd.DataFrame(
        [
            row
            for row in round_rows
            if row["cluster"] == cluster
        ]
    )

    cluster_results.append({
        "cluster": cluster,
        "rounds": len(cluster_df),
        "mean_reward": cluster_df[
            "reward"
        ].mean(),
        "cumulative_reward": cluster_df[
            "reward"
        ].sum(),
        "mean_regret": cluster_df[
            "regret"
        ].mean(),
        "cumulative_regret": cluster_df[
            "regret"
        ].sum()
    })

round_df = pd.DataFrame(
    round_rows
)

cluster_df = pd.DataFrame(
    cluster_results
)

total_rounds = len(round_df)

overall = pd.DataFrame([
    {
        "cluster": "overall",
        "rounds": total_rounds,
        "mean_reward": round_df[
            "reward"
        ].mean(),
        "cumulative_reward": round_df[
            "reward"
        ].sum(),
        "mean_regret": round_df[
            "regret"
        ].mean(),
        "cumulative_regret": round_df[
            "regret"
        ].sum()
    }
])

cluster_df = pd.concat(
    [
        cluster_df,
        overall
    ],
    ignore_index=True
)

round_df.to_csv(
    ROUNDS_OUTPUT,
    index=False
)

cluster_df.to_csv(
    OUTPUT,
    index=False
)

print("=" * 70)
print("MODULE C — 2000 ROUND BANDIT EVALUATION")
print("=" * 70)

print(
    f"Total rounds: {total_rounds}"
)

print(
    "\nExpected rounds: 2000"
)

print(
    f"Round count correct: "
    f"{total_rounds == 2000}"
)

print("\nCluster results:")

print(
    cluster_df.to_string(
        index=False
    )
)

print("\nOverall:")

print(
    overall.to_string(
        index=False
    )
)

print("\nSaved:")
print(OUTPUT)
print(ROUNDS_OUTPUT)