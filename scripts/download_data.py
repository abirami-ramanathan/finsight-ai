from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_DIR
from finsight.data.ingestion import download_yfinance, save_raw

SYMBOLS = {
    "stock": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"],
    "crypto": ["BTC-USD", "ETH-USD"],
    "reit": ["EMBASSY.NS", "MINDSPACE.NS", "BIRET.NS"],
    "gold": ["GOLDBEES.NS"],
}

for asset, symbols in SYMBOLS.items():
    for symbol in symbols:
        try:
            df = download_yfinance(symbol)
            save_raw(df, RAW_DIR / asset / f"{symbol.replace('.', '_')}.csv")
            print("Downloaded", asset, symbol, len(df))
        except Exception as e:
            print("FAILED", asset, symbol, e)
