from pathlib import Path
import pandas as pd

from finsight.data.features import build_features


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "stock"
PROCESSED_DIR = ROOT / "data" / "processed" / "stock"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(RAW_DIR.glob("*.csv"))

print(f"Stock datasets found: {len(files)}")
print("=" * 70)

successful = 0

for idx, path in enumerate(files, 1):

    symbol = path.stem

    print(f"[{idx}/{len(files)}] Processing {symbol}")

    try:
        df = pd.read_csv(path)

        if "date" not in df.columns:
            print(f"FAILED {symbol}: date column missing")
            continue

        df["date"] = pd.to_datetime(df["date"])

        df = df.sort_values("date").reset_index(drop=True)

        features = build_features(
            df,
            asset_type="stock"
        )

        output_path = PROCESSED_DIR / f"{symbol}.csv"

        features.to_csv(
            output_path,
            index=False
        )

        required = [
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

        missing = [
            col for col in required
            if col not in features.columns
        ]

        if missing:
            print(
                f"FAILED {symbol}: missing {missing}"
            )
            continue

        successful += 1

        print(
            f"OK {symbol}: "
            f"{len(features)} rows, "
            f"{len(features.columns)} columns"
        )

    except Exception as e:
        print(f"FAILED {symbol}: {e}")

print("\n" + "=" * 70)
print("STOCK FEATURE ENGINEERING COMPLETE")
print("=" * 70)
print(f"Successful stocks: {successful}")