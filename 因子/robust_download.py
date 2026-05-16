"""Robust incremental download: mootdx OHLCV → parquet batches on disk immediately."""
import os, time, gc
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("C:/factor_data")
DATA.mkdir(parents=True, exist_ok=True)
BATCH_DIR = DATA / "batches"
BATCH_DIR.mkdir(exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# Load stock list
log("Loading stock list...")
import baostock as bs
bs.login()
sl = bs.query_stock_basic().get_data()
bs.logout()
codes = sl[sl["type"]=="1"]["code"].tolist()
log(f"{len(codes)} stocks")

# mootdx download with immediate disk write
log("mootdx OHLCV (incremental to disk)...")
from mootdx.quotes import Quotes
client = Quotes.factory(market="std")

BATCH_SIZE = 100
t0 = time.time()
done = 0
skipped = 0

for batch_i in range(0, len(codes), BATCH_SIZE):
    batch_codes = codes[batch_i:batch_i + BATCH_SIZE]
    bfile = BATCH_DIR / f"mootdx_{batch_i//BATCH_SIZE:04d}.parquet"

    if bfile.exists():
        skipped += len(batch_codes)
        done += len(batch_codes)
        continue

    rows = []
    for code_str in batch_codes:
        parts = code_str.split(".")
        mkt = 1 if parts[0] == "sh" else 0
        sym = parts[1]
        try:
            bars = client.bars(symbol=sym, market=mkt, frequency=4, start=0, count=800)
            if bars is not None and len(bars) > 0:
                df = pd.DataFrame(bars)
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                    if "index" in df.columns:
                        df["date"] = pd.to_datetime(df["index"])
                        df = df.drop(columns=["index"])
                elif "datetime" in df.columns:
                    df["date"] = pd.to_datetime(df["datetime"])
                elif all(c in df.columns for c in ["year","month","day"]):
                    df["date"] = pd.to_datetime(df[["year","month","day"]])
                else:
                    continue
                keep = ["date","stock"]
                for c in ["open","high","low","close","volume","amount"]:
                    if c in df.columns: keep.append(c)
                if "vol" in df.columns and "volume" not in df.columns:
                    df["volume"] = df["vol"]
                    keep.append("volume")
                df["stock"] = sym
                rows.append(df[[c for c in keep if c in df.columns]])
        except:
            pass

    if rows:
        bdf = pd.concat(rows, ignore_index=True)
        bdf.to_parquet(bfile, index=False)

    done += len(batch_codes)
    if done % 500 == 0 or done >= len(codes):
        rate = (done - skipped) / max(time.time() - t0, 1) * 60
        pct = done / len(codes) * 100
        log(f"  {done}/{len(codes)} ({pct:.0f}%) ~{rate:.0f} stocks/min, {skipped} cached")

# Merge all batches
log("Merging batches...")
parts = []
for f in sorted(BATCH_DIR.glob("mootdx_*.parquet")):
    parts.append(pd.read_parquet(f))
ohlcv = pd.concat(parts, ignore_index=True)
ohlcv = ohlcv.drop_duplicates(subset=["stock","date"])
ohlcv = ohlcv[ohlcv["close"] > 0]
ohlcv = ohlcv.set_index(["stock","date"]).sort_index()
ohlcv.to_parquet(DATA / "ohlcv.parquet")

n = ohlcv.index.get_level_values("stock").nunique()
d = ohlcv.index.get_level_values("date")
log(f"Saved: {n} stocks, {len(ohlcv):,} rows, {d.min().date()} ~ {d.max().date()}")
log(f"Time: {(time.time()-t0)/60:.1f} min")

# Calendar
dates = sorted(d.unique())
pd.DataFrame({"date": dates}).to_parquet(DATA / "calendar.parquet")
log(f"Calendar: {len(dates)} days ({dates[0].date()} ~ {dates[-1].date()})")
log("=== OHLCV DONE ===")
