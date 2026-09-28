from pathlib import Path
import pandas as pd
import numpy as np
from finsight.data.features import build_features


REITS = {
    "EMBASSY.NS": {
        "occupancy": [
            ("2025-03-31", 0.93),
            ("2025-06-30", 0.93),
            ("2025-09-30", 0.93),
            ("2025-12-31", 0.92),
            ("2026-03-31", 0.90),
            ("2026-06-30", 0.90)
        ],
        "distribution": [
            ("2025-03-31", 5.68),
            ("2025-06-30", 5.80),
            ("2025-09-30", 6.51),
            ("2025-12-31", 6.47),
            ("2026-03-31", 6.50),
            ("2026-06-30", 6.31)
        ]
    },
    "MINDSPACE.NS": {
        "occupancy": [
            ("2025-03-31", 0.912),
            ("2025-06-30", 0.912),
            ("2025-09-30", 0.912),
            ("2025-12-31", 0.912),
            ("2026-03-31", 0.912),
            ("2026-06-30", 0.958)
        ],
        "distribution": [
            ("2025-03-31", 6.64),
            ("2025-06-30", 6.64),
            ("2025-09-30", 5.79),
            ("2025-12-31", 5.83),
            ("2026-03-31", 5.83),
            ("2026-06-30", 6.67)
        ]
    },
    "BIRET.NS": {
        "occupancy": [
            ("2025-03-31", 0.88),
            ("2025-06-30", 0.89),
            ("2025-09-30", 0.89),
            ("2025-12-31", 0.90),
            ("2026-03-31", 0.90),
            ("2026-06-30", 0.90)
        ],
        "distribution": [
            ("2025-03-31", 5.35),
            ("2025-06-30", 5.25),
            ("2025-09-30", 5.25),
            ("2025-12-31", 5.50),
            ("2026-03-31", 5.50),
            ("2026-06-30", 5.50)
        ]
    }
}


for symbol, fundamentals in REITS.items():

    raw_path = Path(
        f"data/raw/reit/{symbol.replace('.', '_')}.csv"
    )

    output_path = Path(
        f"data/processed/reit/{symbol.replace('.', '_')}.csv"
    )

    df = pd.read_csv(raw_path)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    occupancy = pd.DataFrame(
        fundamentals["occupancy"],
        columns=["date", "occupancy_rate"]
    )

    distribution = pd.DataFrame(
        fundamentals["distribution"],
        columns=["date", "distribution_per_unit"]
    )

    occupancy["date"] = pd.to_datetime(occupancy["date"])
    distribution["date"] = pd.to_datetime(distribution["date"])

    fundamentals_df = pd.merge(
        occupancy,
        distribution,
        on="date",
        how="outer"
    )

    df = pd.merge_asof(
        df,
        fundamentals_df.sort_values("date"),
        on="date",
        direction="backward"
    )

    df["dividend_yield"] = (
        df["distribution_per_unit"]
        / df["close"]
    )

    df["dividend_yield"] = (
        df["dividend_yield"]
        .replace([np.inf, -np.inf], np.nan)
    )

    df = build_features(
        df,
        asset_type="reit"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(f"{symbol}")
    print(f"Rows: {len(df)}")
    print(f"Saved: {output_path}")
    print(
        df[
            [
                "date",
                "close",
                "dividend_yield",
                "occupancy_rate"
            ]
        ].tail()
    )
    print()
    