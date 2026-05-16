"""
Download ALL A-share OHLCV via mootdx bars() (3 persistent workers).
bars() is MORE RELIABLE than get_k_data() -- no KeyError on certain stocks.
Per-worker persistent client, simple try/except, incremental checkpointing.
Output: C:/因子数据/ohlcv.parquet
"""
import pandas as pd, numpy as np, os, time, threading, socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from mootdx.quotes import Quotes
import baostock as bs

DATA_DIR = "C:/因子数据"
os.makedirs(DATA_DIR, exist_ok=True)
log_file = f"{DATA_DIR}/download_log.txt"
chk_dir = f"{DATA_DIR}/chunks"
os.makedirs(chk_dir, exist_ok=True)
NWORKERS = 3
CHUNK = 100

def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")

log("=== OHLCV Download (bars) ===")

# Get stock list (cached, with baostock timeout)
stock_cache = f"{DATA_DIR}/stock_list.parquet"
if os.path.exists(stock_cache):
    df_cache = pd.read_parquet(stock_cache)
    stocks = [(r["mkt"], r["sym"], r["full"]) for _, r in df_cache.iterrows()]
    log(f"Cached stock list: {len(stocks)} stocks")
else:
    stocks = None
    for attempt in range(3):
        def _bs_get():
            bs.login()
            try:
                return bs.query_stock_basic().get_data()
            finally:
                bs.logout()
        result = [None]
        t = threading.Thread(target=lambda: result.__setitem__(0, _bs_get()), daemon=True)
        t.start()
        t.join(60)
        all_s = result[0]
        if all_s is not None and len(all_s) > 0:
            stocks = []
            for _, row in all_s.iterrows():
                if row["type"] != "1": continue
                parts = row["code"].split(".")
                mkt_id = 1 if parts[0] == "sh" else 0
                stocks.append((mkt_id, parts[1], row["code"]))
            pd.DataFrame([{"mkt": s[0], "sym": s[1], "full": s[2]} for s in stocks]).to_parquet(stock_cache)
            log(f"{len(stocks)} A-share stocks")
            break
        else:
            log(f"baostock attempt {attempt+1} timed out or failed")
        time.sleep(3)
    if stocks is None:
        log("FATAL: Could not get stock list"); exit(1)

# Check existing chunks
done_stocks = set()
chunk_files = sorted([f for f in os.listdir(chk_dir) if f.endswith(".parquet")])
if chunk_files:
    for cf in chunk_files:
        df = pd.read_parquet(f"{chk_dir}/{cf}")
        if "stock" in df.columns:
            done_stocks.update(df["stock"].unique())
    log(f"Resume: {len(done_stocks)} stocks from {len(chunk_files)} chunks")

remaining = [s for s in stocks if s[2] not in done_stocks]
log(f"Remaining: {len(remaining)} stocks")

if len(remaining) == 0:
    log("All done!"); exit(0)

# Create NWORKERS batches
batches = [[] for _ in range(NWORKERS)]
for i, s in enumerate(remaining):
    batches[i % NWORKERS].append(s)
log(f"Batch sizes: {[len(b) for b in batches]}")

t0 = time.time()

def fetch_one(client, mkt, sym):
    """Try get_k_data() first (1 call, fast), fall back to bars() on fail."""
    try:
        df = client.get_k_data(sym, "2016-01-01", "2026-05-15")
        if df is not None and len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            if "vol" in df.columns and "volume" not in df.columns:
                df = df.rename(columns={"vol": "volume"})
            keep = [c for c in ["open","high","low","close","volume","amount","date"] if c in df.columns]
            return df[keep]
    except Exception:
        pass

    # Fallback: bars() -- 4 paginated calls
    try:
        all_parts = []
        for offset in [0, 800, 1600, 2400]:
            bdf = client.bars(sym, market=mkt, frequency=9, start=offset, count=800)
            if bdf is not None and len(bdf) > 0:
                all_parts.append(bdf)
        if all_parts:
            merged = pd.concat(all_parts)
            merged = merged[~merged.index.duplicated(keep="first")].sort_index()
            mask = (merged.index >= "2016-01-01") & (merged.index <= "2026-05-15")
            merged = merged[mask]
            if len(merged) > 0:
                out = pd.DataFrame({
                    "open": merged["open"].astype(float),
                    "high": merged["high"].astype(float),
                    "low": merged["low"].astype(float),
                    "close": merged["close"].astype(float),
                    "volume": merged["volume"].astype(float),
                    "amount": merged["amount"].astype(float),
                    "date": merged.index,
                })
                return out
    except Exception:
        pass
    return None

def dl_with_checkpoint(stock_batch, worker_id, chunk_idx):
    """Download stocks, save chunk every CHUNK."""
    socket.setdefaulttimeout(15)
    client = Quotes.factory(market="std")
    partial = []
    chunk_num = chunk_idx
    for i, (mkt, sym, full) in enumerate(stock_batch):
        df = fetch_one(client, mkt, sym)
        if df is not None:
            df["stock"] = full
            partial.append(df)

        if (i + 1) % CHUNK == 0 or i == len(stock_batch) - 1:
            if partial:
                chunk_df = pd.concat(partial, ignore_index=True)
                chunk_df.to_parquet(f"{chk_dir}/w{worker_id}_c{chunk_num:04d}.parquet")
                elapsed = time.time() - t0
                log(f"  W{worker_id} chunk {chunk_num}: {len(partial)}/{i+1} stocks ({elapsed:.0f}s)")
                chunk_num += 1
                partial = []
    return len(stock_batch)

# Run workers
log("Starting download...")
results = []
with ThreadPoolExecutor(max_workers=NWORKERS) as ex:
    futures = [ex.submit(dl_with_checkpoint, batches[wi], wi, wi * 1000) for wi in range(NWORKERS)]
    for f in as_completed(futures):
        try:
            results.append(f.result())
        except Exception as e:
            log(f"Worker error: {e}")

total = sum(results)
log(f"All workers done: {total} stocks in {time.time()-t0:.0f}s")

# Merge all chunks
log("Merging chunks...")
chunk_files = sorted([f for f in os.listdir(chk_dir) if f.endswith(".parquet")])
all_parts = [pd.read_parquet(f"{chk_dir}/{f}") for f in chunk_files]
ohlcv = pd.concat(all_parts, ignore_index=True)
ohlcv["date"] = pd.to_datetime(ohlcv["date"])
ohlcv = ohlcv.sort_values(["stock", "date"]).reset_index(drop=True)
ohlcv = ohlcv.set_index(["stock", "date"]).sort_index()
ohlcv = ohlcv[~ohlcv.index.duplicated(keep="first")]

n_stocks = ohlcv.index.get_level_values("stock").nunique()
n_rows = len(ohlcv)
log(f"OHLCV: {n_rows:,} rows, {n_stocks} stocks ({n_stocks/len(stocks)*100:.1f}% hit rate)")

ohlcv.to_parquet(f"{DATA_DIR}/ohlcv.parquet")
log("Saved ohlcv.parquet")

# Calendar
dates = sorted(ohlcv.index.get_level_values("date").unique())
pd.DataFrame({"date": dates}).to_parquet(f"{DATA_DIR}/calendar.parquet")
log(f"Calendar: {len(dates)} days")

log(f"Total: {(time.time()-t0)/60:.1f} min")
log("=== DONE ===")
