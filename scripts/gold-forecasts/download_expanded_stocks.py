from pathlib import Path
import time
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "stock"
PROCESSED = ROOT / "data" / "processed" / "stock"
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "HINDUNILVR.NS",
    "AXISBANK.NS", "KOTAKBANK.NS", "MARUTI.NS", "M&M.NS", "SUNPHARMA.NS",
    "HCLTECH.NS", "WIPRO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "TITAN.NS", "ULTRACEMCO.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "ASIANPAINT.NS", "NESTLEIND.NS"
]

rows = []
for symbol in SYMBOLS:
    try:
        df = yf.download(symbol, period="10y", auto_adjust=False, progress=False, threads=False)
        if df.empty:
            rows.append((symbol, 0, "EMPTY"))
            continue
        if hasattr(df.columns, "levels"):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
        rename = {"adj_close": "adj_close"}
        df = df.rename(columns=rename)
        keep = [c for c in ["date", "open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
        df = df[keep].dropna(subset=["date", "close"])
        raw_path = RAW / f"{symbol.replace('.', '_')}.csv"
        proc_path = PROCESSED / f"{symbol.replace('.', '_')}.csv"
        df.to_csv(raw_path, index=False)
        df.to_csv(proc_path, index=False)
        rows.append((symbol, len(df), "OK"))
        print(symbol, len(df))
        time.sleep(0.5)
    except Exception as e:
        rows.append((symbol, 0, f"ERROR: {e}"))
        print(symbol, "ERROR", e)

import pandas as pd
pd.DataFrame(rows, columns=["symbol", "rows", "status"]).to_csv(ROOT / "outputs" / "expanded_stock_download_log.csv", index=False)
print("Saved expanded_stock_download_log.csv")
