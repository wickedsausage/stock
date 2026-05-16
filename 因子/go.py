"""One-shot: get stock list → download all OHLCV → save."""
import pandas as pd, numpy as np, time, sys, os
from pathlib import Path
from mootdx.quotes import Quotes

DATA = Path("C:/factor_data")
DATA.mkdir(parents=True, exist_ok=True)
BATCH_DIR = DATA / "batches"
BATCH_DIR.mkdir(exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# === Step 1: Stock list ===
log("Stock list from baostock...")
import baostock as bs
bs.login()
df_sl = bs.query_stock_basic().get_data()
bs.logout()
codes = df_sl[df_sl["type"]=="1"]["code"].tolist()
log(f"{len(codes)} stocks")

# Save stock list (ASCII path, baostock format)
df_sl.to_parquet(DATA / "stock_list.parquet")

# === Step 2: OHLCV via mootdx ===
log("mootdx download (incremental)...")
client = Quotes.factory(market="std")

t0 = time.time()
done = 0
SKIP = 0  # set >0 to resume from crash

for batch_i, code_str in enumerate(codes):
    if batch_i < SKIP:
        done += 1
        continue

    parts = code_str.split(".")
    mkt = 1 if parts[0] == "sh" else 0
    sym = parts[1]

    # Per-stock batch file
    bfile = BATCH_DIR / f"{sym}.parquet"
    if bfile.exists():
        done += 1
        continue

    try:
        bars = client.bars(symbol=sym, market=mkt, frequency=4, start=0, count=800)
        if bars is not None and len(bars) > 0:
            # Save raw bars immediately
            pd.DataFrame(bars).to_parquet(bfile)
    except:
        pass

    done += 1
    if done % 500 == 0:
        elapsed = time.time() - t0
        rate = (done - SKIP) / max(elapsed, 1) * 60
        pct = done / len(codes) * 100
        log(f"  {done}/{len(codes)} ({pct:.0f}%) {rate:.0f} stk/min")

# === Step 3: Merge ===
log("Merging...")
rows = []
bad = 0
for bfile in BATCH_DIR.glob("*.parquet"):
    try:
        df = pd.read_parquet(bfile)
        sym = bfile.stem
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df["date"] = pd.to_datetime(df["index"])
            df = df.drop(columns=["index"] if "index" in df.columns else [])
        elif "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"])
        keep = ["date"]
        for c in ["open","high","low","close","volume","amount"]:
            if c in df.columns: keep.append(c)
        if "vol" in df.columns and "volume" not in df.columns:
            df["volume"] = df["vol"]
            keep.append("volume")
        df["stock"] = sym
        rows.append(df[[c for c in keep if c in df.columns]])
    except:
        bad += 1

if rows:
    ohlcv = pd.concat(rows, ignore_index=True)
    ohlcv = ohlcv.drop_duplicates(subset=["stock","date"])
    ohlcv = ohlcv[ohlcv["close"] > 0]
    ohlcv = ohlcv.set_index(["stock","date"]).sort_index()
    ohlcv.to_parquet(DATA / "ohlcv.parquet")
    n_s = ohlcv.index.get_level_values("stock").nunique()
    dts = ohlcv.index.get_level_values("date")
    log(f"DONE: {n_s} stocks, {len(ohlcv):,} rows, {dts.min().date()}~{dts.max().date()}")

    dates = sorted(dts.unique())
    pd.DataFrame({"date": dates}).to_parquet(DATA / "calendar.parquet")
    log(f"Calendar: {len(dates)} days")

else:
    log("NO DATA MERGED!")

log(f"Bad batches: {bad}, elapsed: {(time.time()-t0)/60:.1f}min")
