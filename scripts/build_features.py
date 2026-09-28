from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_DIR, PROCESSED_DIR
from finsight.data.ingestion import validate_ohlcv
from finsight.data.features import add_features

for asset_dir in RAW_DIR.iterdir():
    if not asset_dir.is_dir():
        continue
    for path in asset_dir.glob("*.csv"):
        asset = asset_dir.name
        df = validate_ohlcv(__import__("pandas").read_csv(path))
        out = add_features(df, "reit" if asset == "reit" else "crypto" if asset == "crypto" else "stock")
        dest = PROCESSED_DIR / asset / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(dest, index=False)
        print("Built", dest)
