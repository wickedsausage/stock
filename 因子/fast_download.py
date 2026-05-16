"""Fast hybrid download: mootdx OHLCV (recent) + baostock OHLCV (history)."""
import os, sys, time, gc
import pandas as pd
import numpy as np

DATA = "C:/factor_data"
os.makedirs(DATA, exist_ok=True)
os.makedirs(f"{DATA}/batches", exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ===== STEP 1: Get stock list from saved baostock data =====
log("Loading stock list...")
try:
    sl = pd.read_parquet(f"{DATA}/stock_list.parquet")
    codes = sl[sl["type"]=="1"]["code"].tolist() if "type" in sl else sl["code"].tolist()
except:
    import baostock as bs
    bs.login()
    sl = bs.query_stock_basic().get_data()
    bs.logout()
    codes = sl[sl["type"]=="1"]["code"].tolist()
log(f"{len(codes)} A-share stocks")

# ===== STEP 2: Fast OHLCV via mootdx =====
log("Step 2: mootdx OHLCV download...")
from mootdx.quotes import Quotes
client = Quotes.factory(market="std")

t0 = time.time()
all_parts = []
done = 0

for code_str in codes:
    # Parse code: sh.600000 -> market=1, symbol=600000
    parts = code_str.split(".")
    mkt = 1 if parts[0] == "sh" else 0
    sym = parts[1]

    try:
        bars = client.bars(symbol=sym, market=mkt, frequency=4, start=0, count=800)
        if bars is not None and len(bars) > 0:
            df = pd.DataFrame(bars)
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df.rename(columns={"index": "datetime"}, inplace=True)
            if "datetime" in df.columns:
                df["date"] = pd.to_datetime(df["datetime"])
            elif "date" not in df.columns:
                # Try year/month/day columns
                if all(c in df.columns for c in ["year","month","day"]):
                    df["date"] = pd.to_datetime(df[["year","month","day"]])
                else:
                    continue

            # Get OHLCV columns
            cols = {"date", "stock"}
            for c in ["open","high","low","close","volume","amount"]:
                if c in df.columns:
                    cols.add(c)
            if "vol" in df.columns and "volume" not in df.columns:
                df["volume"] = df["vol"]
                cols.add("volume")

            df["stock"] = sym  # 6-digit code only
            keep = [c for c in cols if c in df.columns]
            all_parts.append(df[keep])
    except Exception:
        pass

    done += 1
    if done % 500 == 0:
        rate = done / (time.time() - t0) * 60
        log(f"  {done}/{len(codes)} ({rate:.0f} stocks/min)")

# Merge
log("Merging OHLCV...")
ohlcv = pd.concat(all_parts, ignore_index=True)
ohlcv = ohlcv.drop_duplicates(subset=["stock","date"])
ohlcv = ohlcv[ohlcv["close"] > 0]
ohlcv = ohlcv.set_index(["stock","date"]).sort_index()
n_stocks = ohlcv.index.get_level_values("stock").nunique()
n_dates = ohlcv.index.get_level_values("date").nunique()
log(f"OHLCV: {n_stocks} stocks, {len(ohlcv):,} rows, {n_dates} dates")
log(f"Range: {ohlcv.index.get_level_values('date').min().date()} ~ {ohlcv.index.get_level_values('date').max().date()}")
log(f"Time: {(time.time()-t0)/60:.1f} min")

# Save
ohlcv.to_parquet(f"{DATA}/ohlcv.parquet")
log("Saved ohlcv.parquet")

# Calendar
dates = sorted(ohlcv.index.get_level_values("date").unique())
pd.DataFrame({"date": dates}).to_parquet(f"{DATA}/calendar.parquet")
log(f"Calendar: {len(dates)} days")

# ===== STEP 3: Extend with baostock for older data =====
# Only if mootdx data starts after 2018 (mootdx limited to ~3 years)
oldest = dates[0]
if oldest > pd.Timestamp("2018-01-01"):
    log(f"Step 3: Extending with baostock history (oldest mootdx={oldest.date()})...")
    import baostock as bs

    # Only extend top stocks by liquidity (full market would take too long)
    stock_sizes = ohlcv.groupby("stock").size().sort_values(ascending=False)
    ext_stocks = stock_sizes.head(1000).index.tolist()

    bs.login()
    ext_parts = []
    ext_done = 0
    for sym in ext_stocks:
        try:
            # Determine prefix
            if sym.startswith(("6","9")):
                bs_code = f"sh.{sym}"
            else:
                bs_code = f"sz.{sym}"
            rs = bs.query_history_k_data_plus(bs_code, "date,open,high,low,close,volume,amount",
                                              "2016-01-01", oldest.strftime("%Y-%m-%d"), "d", "2")
            if rs.error_code == "0":
                df_h = rs.get_data()
                if df_h is not None and len(df_h) > 0:
                    for c in ["open","high","low","close","volume","amount"]:
                        df_h[c] = pd.to_numeric(df_h[c], errors="coerce")
                    df_h["stock"] = sym
                    df_h["date"] = pd.to_datetime(df_h["date"])
                    df_h = df_h[df_h["close"] > 0]
                    ext_parts.append(df_h[["stock","date","open","high","low","close","volume","amount"]])
        except:
            pass
        ext_done += 1
        if ext_done % 200 == 0:
            log(f"  history: {ext_done}/{len(ext_stocks)}")

    bs.logout()

    if ext_parts:
        hist = pd.concat(ext_parts, ignore_index=True)
        hist = hist.set_index(["stock","date"]).sort_index()
        # Merge with mootdx data
        combined = pd.concat([hist, ohlcv])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        combined.to_parquet(f"{DATA}/ohlcv.parquet")

        all_dates = sorted(combined.index.get_level_values("date").unique())
        pd.DataFrame({"date": all_dates}).to_parquet(f"{DATA}/calendar.parquet")
        log(f"Extended: {len(combined):,} rows, {len(all_dates)} dates ({all_dates[0].date()} ~ {all_dates[-1].date()})")

# ===== STEP 4: Financial data =====
log("Step 4: Financials from baostock (top 800)...")
try:
    import baostock as bs
    top800 = ohlcv.groupby("stock").size().sort_values(ascending=False).head(800).index.tolist()

    bs.login()
    funcs = {"p": bs.query_profit_data, "b": bs.query_balance_data,
             "c": bs.query_cash_flow_data, "o": bs.query_operation_data}
    recs = []
    for si, sym in enumerate(top800):
        if sym.startswith(("6","9")): bs_code = f"sh.{sym}"
        else: bs_code = f"sz.{sym}"
        for fk, fn in funcs.items():
            for y in range(2016, 2027):
                for q in [1,2,3,4]:
                    try:
                        d = fn(bs_code, y, q, "1").get_data()
                        if d is not None and len(d) > 0:
                            r = d.iloc[0].to_dict()
                            r["_t"], r["_y"], r["_q"] = fk, y, q
                            r["code"] = sym
                            recs.append(r)
                    except: pass
        if (si+1) % 200 == 0:
            log(f"  fin: {si+1}/{len(top800)} ({len(recs)} records)")
        time.sleep(0.01)
    bs.logout()

    if recs:
        fraw = pd.DataFrame(recs)
        fw = fraw.pivot_table(index=["code","_y","_q"], columns="_t", aggfunc="first")
        fw.columns = [f"{c[1]}_{c[0]}" for c in fw.columns]
        fw = fw.reset_index()
        fw.to_parquet(f"{DATA}/financials.parquet")
        log(f"Financials: {len(fw)} rows, {fw['code'].nunique()} stocks")
except Exception as e:
    log(f"Financials error: {e}")

log("=== ALL DONE ===")
