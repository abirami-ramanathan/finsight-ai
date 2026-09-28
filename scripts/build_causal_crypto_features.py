from pathlib import Path
import pandas as pd
from finsight.data.features import build_features, add_causal_garch_volatility


SYMBOLS = ["BTC-USD", "ETH-USD"]


for symbol in SYMBOLS:
    input_path = Path(
        f"data/processed/crypto/{symbol}.csv"
    )

    output_path = Path(
        f"data/processed/crypto/{symbol}_causal.csv"
    )

    df = pd.read_csv(input_path)

    df = build_features(
        df,
        asset_type="crypto"
    )

    df = add_causal_garch_volatility(
        df,
        train_ratio=0.70
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(f"{symbol}:")
    print(f"Rows: {len(df)}")
    print(f"Saved: {output_path}")
    print(
        df[
            [
                "return_1d",
                "volatility_20",
                "garch_style_vol",
                "volatility_5"
            ]
        ].tail()
    )
    print()