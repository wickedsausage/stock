"""
Download ALL A-share OHLCV via mootdx (3 persistent workers) + financials.
Output: C:/factor_data/ohlcv.parquet + calendar.parquet + industry.parquet + financials.parquet
"""
import pandas as pd, numpy as np, os, time
from concurrent.futures import ThreadPoolExecutor
from mootdx.quotes import Quotes
import baostock as bs

DATA_DIR = "C:/factor_data"
os.makedirs(DATA_DIR, exist_ok=True)
NWORKERS = 3
chk_path = f"{DATA_DIR}/ohlcv_checkpoint.parquet"

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def get_a_stocks():
    bs.login()
    all_s = bs.query_stock_basic().get_data()
    bs.logout()
    stocks = []
    for _, row in all_s.iterrows():
        if row["type"] != "1": continue
        parts = row["code"].split(".")
        mkt_id = 1 if parts[0] == "sh" else 0
        stocks.append((mkt_id, parts[1], row["code"]))
    return stocks

def dl_batch(stock_batch):
    """Worker: one persistent client for a batch of stocks."""
    client = Quotes.factory(market="std")
    results = []
    for mkt, sym, full in stock_batch:
        try:
            df = client.get_k_data(sym, "2016-01-01", "2026-05-15")
            if df is not None and len(df) > 0:
                df["stock"] = full
                df["date"] = pd.to_datetime(df["date"])
                if "vol" in df.columns and "volume" not in df.columns:
                    df = df.rename(columns={"vol": "volume"})
                keep = [c for c in ["open","high","low","close","volume","amount","stock","date"] if c in df.columns]
                results.append(df[keep])
        except:
            pass
    return results

if __name__ == "__main__":
    t0 = time.time()
    stocks = get_a_stocks()
    log(f"{len(stocks)} A-share stocks")

    # Resume from checkpoint?
    already_done = set()
    if os.path.exists(chk_path):
        cp = pd.read_parquet(chk_path)
        already_done = set(cp["stock"].unique()) if "stock" in cp.columns else set()
        log(f"Resume: {len(already_done)} stocks already downloaded")

    remaining = [s for s in stocks if s[2] not in already_done]
    if len(remaining) == 0:
        ohlcv = pd.read_parquet(chk_path)
        log(f"All done, loading from checkpoint")
    else:
        log(f"Stocks to download: {len(remaining)}")
        # Split remaining into NWORKERS batches
        batches = [[] for _ in range(NWORKERS)]
        for i, s in enumerate(remaining):
            batches[i % NWORKERS].append(s)

        all_parts = []
        dt = time.time()
        with ThreadPoolExecutor(max_workers=NWORKERS) as ex:
            for wi, batch_results in enumerate(ex.map(dl_batch, batches)):
                for r in batch_results:
                    all_parts.append(r)
                log(f"  Worker {wi+1} done: {len(batch_results)} stocks, "
                     f"{time.time()-dt:.0f}s")

        # Merge with checkpoint
        if already_done:
            all_parts.append(cp)

        log(f"Merging {len(all_parts)} parts...")
        ohlcv = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
        if len(ohlcv) == 0:
            log("ERROR: No data downloaded!")
            exit(1)
        ohlcv = ohlcv.sort_values(["stock", "date"]).reset_index(drop=True)
        ohlcv = ohlcv.set_index(["stock", "date"]).sort_index()
        ohlcv = ohlcv[~ohlcv.index.duplicated(keep="first")]

    log(f"OHLCV: {len(ohlcv):,} rows, "
         f"{ohlcv.index.get_level_values('stock').nunique()} stocks, "
         f"{ohlcv.index.get_level_values('date').min().date()} to "
         f"{ohlcv.index.get_level_values('date').max().date()}")
    log(f"Time: {(time.time()-t0)/60:.1f} min")

    # Save
    ohlcv.to_parquet(f"{DATA_DIR}/ohlcv.parquet")
    ohlcv.reset_index().to_parquet(chk_path)
    log("Saved ohlcv.parquet")

    # Calendar
    dates = sorted(ohlcv.index.get_level_values("date").unique())
    pd.DataFrame({"date": dates}).to_parquet(f"{DATA_DIR}/calendar.parquet")
    log(f"Calendar: {len(dates)} days")

    # Industry
    bs.login()
    ind = bs.query_stock_industry().get_data()
    if ind is not None and len(ind) > 0:
        ind.to_parquet(f"{DATA_DIR}/industry.parquet")
    bs.logout()
    log(f"Industry: {len(ind) if ind is not None else 0} rows")
