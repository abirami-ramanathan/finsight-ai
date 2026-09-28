from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

INPUT_EVAL = ROOT / "outputs" / "recommendation" / "recommendation_evaluation.csv"
INPUT_ROUNDS = ROOT / "outputs" / "recommendation" / "linucb_2000_rounds.csv"

OUTPUT_DIR = ROOT / "outputs" / "recommendation" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

eval_df = pd.read_csv(INPUT_EVAL)
rounds_df = pd.read_csv(INPUT_ROUNDS)

cluster_df = eval_df[eval_df["cluster"] != "overall"].copy()

plt.figure(figsize=(10, 6))
plt.bar(cluster_df["cluster"], cluster_df["mean_reward"])
plt.xlabel("Risk-Horizon Cluster")
plt.ylabel("Mean Reward")
plt.title("Mean Reward Across Risk-Horizon Clusters")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mean_reward_by_cluster.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(cluster_df["cluster"], cluster_df["mean_regret"])
plt.xlabel("Risk-Horizon Cluster")
plt.ylabel("Mean Regret")
plt.title("Mean Regret Across Risk-Horizon Clusters")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mean_regret_by_cluster.png", dpi=300)
plt.close()

rounds_df["cumulative_reward"] = (
    rounds_df.groupby("cluster")["reward"].cumsum()
)

overall_reward = (
    rounds_df.groupby("round")["reward"]
    .sum()
    .cumsum()
)

plt.figure(figsize=(10, 6))
plt.plot(
    overall_reward.index,
    overall_reward.values
)
plt.xlabel("Round")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward Over 2000 Bandit Rounds")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cumulative_reward.png", dpi=300)
plt.close()

overall_regret = (
    rounds_df.groupby("round")["regret"]
    .sum()
    .cumsum()
)

plt.figure(figsize=(10, 6))
plt.plot(
    overall_regret.index,
    overall_regret.values
)
plt.xlabel("Round")
plt.ylabel("Cumulative Regret")
plt.title("Cumulative Regret Over 2000 Bandit Rounds")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cumulative_regret.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))

for cluster in rounds_df["cluster"].unique():
    temp = rounds_df[
        rounds_df["cluster"] == cluster
    ].copy()

    temp["cumulative_reward"] = temp["reward"].cumsum()

    plt.plot(
        temp["round"],
        temp["cumulative_reward"],
        label=cluster
    )

plt.xlabel("Round")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward by Risk-Horizon Cluster")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cluster_cumulative_reward.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))

for cluster in rounds_df["cluster"].unique():
    temp = rounds_df[
        rounds_df["cluster"] == cluster
    ].copy()

    temp["cumulative_regret"] = temp["regret"].cumsum()

    plt.plot(
        temp["round"],
        temp["cumulative_regret"],
        label=cluster
    )

plt.xlabel("Round")
plt.ylabel("Cumulative Regret")
plt.title("Cumulative Regret by Risk-Horizon Cluster")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cluster_cumulative_regret.png", dpi=300)
plt.close()

print("=" * 70)
print("MODULE C — RECOMMENDATION PLOTS")
print("=" * 70)

print("\nGenerated plots:")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(file)

print("\nPlot directory:")
print(OUTPUT_DIR)