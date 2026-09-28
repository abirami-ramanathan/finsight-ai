from pathlib import Path
import re
import time
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "mutual_fund"
RAW.mkdir(parents=True, exist_ok=True)

QUERIES = [
"Parag Parikh Flexi Cap Fund", "HDFC Flexi Cap Fund", "HDFC Top 100 Fund", "ICICI Prudential Bluechip Fund",
"SBI Bluechip Fund", "Axis Bluechip Fund", "Mirae Asset Large Cap Fund", "Canara Robeco Bluechip Equity Fund",
"Nippon India Large Cap Fund", "Kotak Bluechip Fund", "Motilal Oswal Midcap Fund", "HDFC Mid-Cap Opportunities Fund",
"SBI Magnum Midcap Fund", "Axis Midcap Fund", "Kotak Emerging Equity Fund", "Nippon India Growth Fund",
"HDFC Small Cap Fund", "SBI Small Cap Fund", "Nippon India Small Cap Fund", "Axis Small Cap Fund",
"Quant Small Cap Fund", "Tata Small Cap Fund", "Parag Parikh ELSS Tax Saver Fund", "Mirae Asset Tax Saver Fund",
"Quant ELSS Tax Saver Fund", "Canara Robeco Equity Tax Saver Fund", "UTI Nifty 50 Index Fund", "HDFC Index Fund Nifty 50 Plan",
"ICICI Prudential Nifty 50 Index Fund", "SBI Nifty Index Fund", "Nippon India Nifty 50 Index Fund", "Motilal Oswal Nasdaq 100 Fund",
"HDFC Balanced Advantage Fund", "ICICI Prudential Balanced Advantage Fund", "SBI Balanced Advantage Fund", "Aditya Birla Sun Life Balanced Advantage Fund",
"Kotak Equity Hybrid Fund", "HDFC Hybrid Equity Fund", "ICICI Prudential Equity & Debt Fund", "SBI Equity Hybrid Fund",
"HDFC Corporate Bond Fund", "ICICI Prudential Corporate Bond Fund", "Aditya Birla Sun Life Corporate Bond Fund", "SBI Corporate Bond Fund",
"HDFC Short Term Debt Fund", "ICICI Prudential Short Term Fund", "SBI Short Term Debt Fund", "Kotak Bond Short Term Fund",
"Nippon India Gold Savings Fund", "HDFC Gold Fund", "SBI Gold Fund", "Aditya Birla Sun Life Gold Fund"
]

BASE = "https://api.mfapi.in/mf/search"
s = requests.Session()
selected = []
seen = set()


def score(item, query):
    name = item.get("schemeName", "").lower()
    q = query.lower()
    value = 0
    if "direct" in name: value += 30
    if "growth" in name: value += 30
    if "idcw" in name or "dividend" in name: value -= 30
    if "bonus" in name: value -= 20
    value += sum(2 for token in re.findall(r"[a-z0-9]+", q) if token in name)
    return value

for query in QUERIES:
    try:
        r = s.get(BASE, params={"q": query}, timeout=30)
        r.raise_for_status()
        items = r.json()
        candidates = [x for x in items if x.get("schemeCode") not in seen]
        if not candidates:
            continue
        candidates.sort(key=lambda x: score(x, query), reverse=True)
        chosen = candidates[0]
        code = int(chosen["schemeCode"])
        meta_r = s.get(f"https://api.mfapi.in/mf/{code}", timeout=30)
        meta_r.raise_for_status()
        payload = meta_r.json()
        data = payload.get("data", [])
        meta = payload.get("meta", {})
        if len(data) < 500:
            continue
        rows = []
        for x in data:
            rows.append({"date": pd.to_datetime(x["date"], format="%d-%m-%Y"), "nav": float(x["nav"])})
        df = pd.DataFrame(rows).dropna().sort_values("date").drop_duplicates("date")
        if len(df) < 500:
            continue
        safe = re.sub(r"[^A-Za-z0-9]+", "_", str(code)).strip("_")
        df.to_csv(RAW / f"{safe}.csv", index=False)
        selected.append({
            "scheme_code": code,
            "query": query,
            "scheme_name": meta.get("scheme_name", chosen.get("schemeName", "")),
            "fund_house": meta.get("fund_house", ""),
            "scheme_category": meta.get("scheme_category", ""),
            "rows": len(df),
            "file": f"{safe}.csv"
        })
        seen.add(code)
        print(len(selected), code, meta.get("scheme_name", chosen.get("schemeName", "")), len(df))
        if len(selected) >= 52:
            break
        time.sleep(0.2)
    except Exception as e:
        print("FAILED", query, e)

manifest = pd.DataFrame(selected)
ROOT.joinpath("data", "processed", "mutual_fund").mkdir(parents=True, exist_ok=True)
manifest.to_csv(ROOT / "data" / "processed" / "mutual_fund" / "mutual_fund_manifest.csv", index=False)
print(f"Downloaded {len(manifest)} mutual funds")
if len(manifest) < 52:
    raise RuntimeError(f"Only {len(manifest)} mutual funds were downloaded; rerun to fill missing schemes.")
