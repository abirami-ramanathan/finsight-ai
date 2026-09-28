from pathlib import Path
import pandas as pd
import yfinance as yf

def download_yfinance(symbol: str, start: str = "2018-01-01", end: str | None = None) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")
    if hasattr(df.columns, "levels"):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    out = df.copy()
    for c in required - {"date"}:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    out = out.drop_duplicates("date")
    out["close"] = out["close"].clip(lower=0)
    out = out.ffill().bfill()
    return out.reset_index(drop=True)

def save_raw(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
