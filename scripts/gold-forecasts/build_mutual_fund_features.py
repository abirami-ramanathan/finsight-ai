from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "mutual_fund"
OUT = ROOT / "data" / "processed" / "mutual_fund"
OUT.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(OUT / "mutual_fund_manifest.csv")
FEATURES = ["nav", "return_1d", "ema_12", "rsi_14", "bb_high", "bb_low", "macd", "macd_signal", "volatility_20"]

for _, row in manifest.iterrows():
    path = RAW / row["file"]
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    nav = df["nav"].astype(float)
    df["return_1d"] = nav.pct_change()
    df["ema_12"] = nav.ewm(span=12, adjust=False).mean()
    delta = nav.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    mid = nav.rolling(20).mean()
    std = nav.rolling(20).std()
    df["bb_high"] = mid + 2 * std
    df["bb_low"] = mid - 2 * std
    ema12 = nav.ewm(span=12, adjust=False).mean()
    ema26 = nav.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["volatility_20"] = df["return_1d"].rolling(20).std()
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna().reset_index(drop=True)
    code = str(int(row["scheme_code"]))
    df.to_csv(OUT / f"{code}_features.csv", index=False)

print("Mutual-fund feature files created:", len(manifest))
