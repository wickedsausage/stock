"""Merge in 500-stock chunks to avoid memory crash."""
import pandas as pd, numpy as np, time
from pathlib import Path

DATA = Path("C:/factor_data")
BATCH_DIR = DATA / "batches"
files = sorted(BATCH_DIR.glob("*.parquet"))
print(f"{len(files)} files", flush=True)

CHUNK = 500
chunks_done = []
t0 = time.time()

for ci in range(0, len(files), CHUNK):
    chunk = files[ci:ci+CHUNK]
    cfile = DATA / f"chunk_{ci//CHUNK:03d}.parquet"

    if cfile.exists():
        chunks_done.append(cfile)
        continue

    rows = []
    for bfile in chunk:
        try:
            df = pd.read_parquet(bfile)
            sym = bfile.stem
            if isinstance(df.index, pd.DatetimeIndex):
                dates = pd.to_datetime(df.index)
            elif "datetime" in df.columns:
                dates = pd.to_datetime(df["datetime"])
            else:
                continue
            out = {"stock": sym, "date": dates}
            for c in ["open","high","low","close"]:
                if c in df.columns:
                    out[c] = df[c].values
            vcol = "volume" if "volume" in df.columns else ("vol" if "vol" in df.columns else None)
            if vcol:
                out["volume"] = df[vcol].values
            if "amount" in df.columns:
                out["amount"] = df["amount"].values
            sdf = pd.DataFrame(out)
            sdf = sdf.dropna(subset=["close"])
            sdf = sdf[sdf["close"] > 0]
            if len(sdf) > 0:
                rows.append(sdf)
        except Exception:
            pass

    if rows:
        cdf = pd.concat(rows, ignore_index=True)
        cdf.to_parquet(cfile, index=False)
        chunks_done.append(cfile)

    print(f"  Chunk {ci//CHUNK}: {len(rows)} stocks -> {cfile.name}", flush=True)

print(f"Merging {len(chunks_done)} chunks...", flush=True)
parts = [pd.read_parquet(f) for f in chunks_done]
ohlcv = pd.concat(parts, ignore_index=True)
ohlcv = ohlcv.drop_duplicates(subset=["stock","date"])
ohlcv = ohlcv.set_index(["stock","date"]).sort_index()
ohlcv.to_parquet(DATA / "ohlcv.parquet")

ns = ohlcv.index.get_level_values("stock").nunique()
dts = ohlcv.index.get_level_values("date")
print(f"DONE: {ns} stocks, {len(ohlcv):,} rows", flush=True)
print(f"Range: {dts.min().date()} ~ {dts.max().date()}", flush=True)

dates = sorted(dts.unique())
pd.DataFrame({"date": dates}).to_parquet(DATA / "calendar.parquet")
print(f"Calendar: {len(dates)} days", flush=True)
print(f"Time: {(time.time()-t0)/60:.1f}min", flush=True)
