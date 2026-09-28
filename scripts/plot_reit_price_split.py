from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from finsight.data.split import chronological_split


symbol = "MINDSPACE.NS"

df = pd.read_csv(
    f"data/processed/reit/{symbol.replace('.', '_')}.csv"
)

train, val, test = chronological_split(df)

plt.figure(figsize=(12, 6))

plt.plot(
    pd.to_datetime(train["date"]),
    train["close"],
    label="Train"
)

plt.plot(
    pd.to_datetime(val["date"]),
    val["close"],
    label="Validation"
)

plt.plot(
    pd.to_datetime(test["date"]),
    test["close"],
    label="Test"
)

plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("Mindspace REIT: Chronological Train/Validation/Test Split")
plt.legend()
plt.grid(True)

Path("outputs/plots").mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    "outputs/plots/MINDSPACE_NS_data_split.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: outputs/plots/MINDSPACE_NS_data_split.png"
)