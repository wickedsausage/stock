"""Merge per-stock parquet batches into unified ohlcv.parquet."""
import pandas as pd, numpy as np, time
from pathlib import Path

DATA = Path("C:/factor_data")
BATCH_DIR = DATA / "batches"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log(f"Scanning {BATCH_DIR}...")
files = sorted(BATCH_DIR.glob("*.parquet"))
log(f"{len(files)} batch files")

rows = []
bad = 0
t0 = time.time()

for i, bfile in enumerate(files):
    try:
        df = pd.read_parquet(bfile)
        sym = bfile.stem

        # Extract datetime from index or columns
        if isinstance(df.index, pd.DatetimeIndex):
            date_series = pd.to_datetime(df.index)
        elif "datetime" in df.columns:
            date_series = pd.to_datetime(df["datetime"])
        elif "date" in df.columns:
            date_series = pd.to_datetime(df["date"])
        elif all(c in df.columns for c in ["year","month","day"]):
            date_series = pd.to_datetime(df[["year","month","day"]])
        else:
            bad += 1
            continue

        # Select OHLCV columns
        out = {"stock": sym, "date": date_series}
        for c in ["open","high","low","close"]:
            if c in df.columns:
                out[c] = df[c]

        # Volume: use 'volume' if available, else 'vol'
        if "volume" in df.columns:
            out["volume"] = df["volume"]
        elif "vol" in df.columns:
            out["volume"] = df["vol"]

        # Amount
        if "amount" in df.columns:
            out["amount"] = df["amount"]

        sdf = pd.DataFrame(out)
        sdf = sdf.dropna(subset=["close"])
        sdf = sdf[sdf["close"] > 0]
        if len(sdf) > 0:
            rows.append(sdf)
    except Exception as e:
        bad += 1

    if (i + 1) % 1000 == 0:
        log(f"  {i+1}/{len(files)} ({len(rows)} ok, {bad} bad)")

if rows:
    log("Concatenating...")
    ohlcv = pd.concat(rows, ignore_index=True)
    ohlcv = ohlcv.drop_duplicates(subset=["stock","date"])
    ohlcv = ohlcv.set_index(["stock","date"]).sort_index()
    ohlcv.to_parquet(DATA / "ohlcv.parquet")

    n_s = ohlcv.index.get_level_values("stock").nunique()
    dts = ohlcv.index.get_level_values("date")
    log(f"DONE: {n_s} stocks, {len(ohlcv):,} rows")
    log(f"  Range: {dts.min().date()} ~ {dts.max().date()}")

    dates = sorted(dts.unique())
    pd.DataFrame({"date": dates}).to_parquet(DATA / "calendar.parquet")
    log(f"  Calendar: {len(dates)} days")

    size_mb = (DATA / "ohlcv.parquet").stat().st_size / 1024**2
    log(f"  File: {size_mb:.0f} MB")
else:
    log("NO DATA!")

log(f"Bad: {bad}, Time: {(time.time()-t0)/60:.1f}min")
