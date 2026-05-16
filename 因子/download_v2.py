"""
Phase 1: Download ALL A-share OHLCV + financials.
Uses baostock for OHLCV (10yr), mootdx for financials, baostock for calendar.
ASCII paths only (avoid CJK path encoding issues).
"""
import os, sys, time, gc
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("C:/factor_data")
DATA.mkdir(parents=True, exist_ok=True)
BATCH_DIR = DATA / "batches"
BATCH_DIR.mkdir(exist_ok=True)

START, END = "2016-01-01", "2026-05-15"
BATCH_N = 150  # stocks per batch

print("=" * 60, flush=True)
print("Phase 1A: Stock list from baostock", flush=True)

import baostock as bs
bs.login()
stock_df = bs.query_stock_basic().get_data()
stocks = stock_df[stock_df["type"] == "1"]["code"].tolist()  # type=1 = stock
print(f"A-share stocks: {len(stocks)}", flush=True)
stock_df.to_parquet(DATA / "stock_list.parquet")
bs.logout()

# ------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("Phase 1B: Trading calendar from baostock", flush=True)

bs.login()
cal_df = bs.query_trade_dates(start_date=START, end_date=END).get_data()
bs.logout()
cal = pd.to_datetime(cal_df[cal_df["is_trading_day"] == "1"]["calendar_date"])
cal = pd.DatetimeIndex(sorted(cal))
pd.DataFrame({"date": cal}).to_parquet(DATA / "calendar.parquet")
print(f"Trading days: {len(cal)} ({cal[0].date()} ~ {cal[-1].date()})", flush=True)

# ------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print(f"Phase 1C: OHLCV download ({len(stocks)} stocks)", flush=True)

FIELDS = "date,open,high,low,close,volume,amount,turn,peTTM,pbMRQ"
all_parts = []

for batch_idx in range(0, len(stocks), BATCH_N):
    batch_codes = stocks[batch_idx:batch_idx + BATCH_N]
    bfile = BATCH_DIR / f"b_{batch_idx//BATCH_N:03d}.parquet"

    if bfile.exists():
        all_parts.append(pd.read_parquet(bfile))
        pct = (batch_idx + len(batch_codes)) / len(stocks) * 100
        print(f"  Batch {batch_idx//BATCH_N}: cached ({pct:.0f}%)", flush=True)
        continue

    bs.login()
    batch_rows = []
    failed = 0
    for code in batch_codes:
        try:
            rs = bs.query_history_k_data_plus(code, FIELDS, START, END, "d", "2")
            if rs.error_code == "0":
                df_s = rs.get_data()
                if df_s is not None and len(df_s) > 0:
                    for c in ["open","high","low","close","volume","amount","turn","peTTM","pbMRQ"]:
                        df_s[c] = pd.to_numeric(df_s[c], errors="coerce")
                    df_s["stock"] = code.split(".")[-1]
                    df_s["date"] = pd.to_datetime(df_s["date"])
                    batch_rows.append(df_s)
        except Exception:
            failed += 1
    bs.logout()

    if batch_rows:
        bdf = pd.concat(batch_rows, ignore_index=True)
        bdf.to_parquet(bfile, index=False)
        all_parts.append(bdf)

    pct = (batch_idx + len(batch_codes)) / len(stocks) * 100
    print(f"  Batch {batch_idx//BATCH_N}: {len(batch_rows)} ok, {failed} fail ({pct:.0f}%)", flush=True)
    time.sleep(0.5)

# Merge
print("Merging batches...", flush=True)
ohlcv = pd.concat(all_parts, ignore_index=True)
ohlcv = ohlcv[ohlcv["close"] > 0]
ohlcv = ohlcv.dropna(subset=["close", "open", "high", "low"])
ohlcv = ohlcv.set_index(["stock", "date"]).sort_index()
ohlcv.to_parquet(DATA / "ohlcv.parquet")
n_stocks = ohlcv.index.get_level_values("stock").nunique()
n_rows = len(ohlcv)
print(f"OHLCV saved: {n_stocks} stocks, {n_rows:,} rows", flush=True)

# ------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("Phase 1D: Financials from mootdx (top 800 stocks by liquidity)", flush=True)

try:
    from mootdx.quotes import Quotes
    client = Quotes.factory(market='std')

    # Select top 800 stocks by data availability
    top_stocks = ohlcv.groupby("stock").size().sort_values(ascending=False).head(800).index.tolist()

    fin_records = []
    for i, code in enumerate(top_stocks):
        try:
            fin = client.finance(symbol=code)
            if fin is not None and len(fin) > 0:
                df_f = pd.DataFrame(fin)
                # mootdx finance returns dict-like; each key is a field
                row = {"stock": code}
                for k, v in fin.items() if isinstance(fin, dict) else df_f.iloc[0].to_dict().items():
                    try:
                        row[k] = float(v) if v else np.nan
                    except (ValueError, TypeError):
                        row[k] = v
                fin_records.append(row)
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f"  Financials: {i+1}/{len(top_stocks)}", flush=True)
        time.sleep(0.05)

    if fin_records:
        fin_df = pd.DataFrame(fin_records)
        fin_df.to_parquet(DATA / "financials.parquet")
        print(f"Financials saved: {len(fin_df)} stocks, {len(fin_df.columns)} fields", flush=True)
except Exception as e:
    print(f"Financials failed: {e}", flush=True)

# ------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("DOWNLOAD COMPLETE", flush=True)
print(f"  ohlcv:      {DATA / 'ohlcv.parquet'}", flush=True)
print(f"  financials: {DATA / 'financials.parquet'}", flush=True)
print(f"  calendar:   {DATA / 'calendar.parquet'}", flush=True)
print(f"  stock_list: {DATA / 'stock_list.parquet'}", flush=True)
