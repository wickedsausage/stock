"""
Download A-share financial data from baostock + akshare.
- peTTM, pbMRQ, turnover (baostock query_stock_operation)
- Financial indicators (akshare)

Output: C:/因子数据/financials.parquet
"""
import pandas as pd, numpy as np, os, time, threading
import baostock as bs
import akshare as ak

DATA_DIR = "C:/因子数据"
os.makedirs(DATA_DIR, exist_ok=True)
log_file = f"{DATA_DIR}/fin_download_log.txt"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")

log("=== Financial Data Download ===")

# Load stock list
stock_list_path = f"{DATA_DIR}/stock_list.parquet"
if os.path.exists(stock_list_path):
    df_cache = pd.read_parquet(stock_list_path)
    stock_codes = [r["full"] for _, r in df_cache.iterrows()]
    log(f"Stock list: {len(stock_codes)} stocks")
else:
    log("No stock list found - run download_ohlcv_only.py first")
    exit(1)

# ─────────────────────────────────────────────
# Baostock: peTTM, pbMRQ, turnover per stock
# ─────────────────────────────────────────────
def get_bs_operation(symbol):
    """Get operation data (pe, pb, turnover) for one stock."""
    try:
        bs.login()
        rs = bs.query_stock_operation(symbol, start_date="2016-01-01", end_date="2026-05-15")
        df = rs.get_data()
        return df
    except:
        return None
    finally:
        bs.logout()

def bs_fin_with_timeout(code):
    result = [None]
    t = threading.Thread(target=lambda: result.__setitem__(0, get_bs_operation(code)), daemon=True)
    t.start()
    t.join(30)
    return result[0]

# Process in batches
batch_size = 50
fin_parts = []
t0 = time.time()

for i in range(0, len(stock_codes), batch_size):
    batch = stock_codes[i:i+batch_size]
    for code in batch:
        df = bs_fin_with_timeout(code)
        if df is not None and len(df) > 0:
            # Normalize columns - peTTM, pbMRQ, turnover
            keep = [c for c in ["code", "pubDate", "statDate", "peTTM", "pbMRQ", "turnover", "petcf"] if c in df.columns]
            fin_parts.append(df[keep])
    if (i // batch_size) % 10 == 0 or i == 0:
        log(f"  Operation data: {min(i+batch_size, len(stock_codes))}/{len(stock_codes)} stocks")

    # Save checkpoint periodically
    if (i // batch_size) > 0 and (i // batch_size) % 20 == 0 and fin_parts:
        checkpoint = pd.concat(fin_parts, ignore_index=True)
        checkpoint.to_parquet(f"{DATA_DIR}/fin_checkpoint.parquet")
        log(f"  Checkpoint saved: {len(checkpoint)} rows")

if fin_parts:
    fin_ops = pd.concat(fin_parts, ignore_index=True)
else:
    fin_ops = pd.DataFrame()

log(f"Operation data: {len(fin_ops)} rows")

# ─────────────────────────────────────────────
# Akshare: financial indicators (ROE, ROA, etc.)
# ─────────────────────────────────────────────
log("Downloading akshare financial data...")
fin_ind_parts = []

# Test a few stocks
for code in stock_codes[:100]:
    try:
        df = ak.stock_financial_abstract(symbol=code, indicator="按年度")
        if df is not None and len(df) > 0:
            df["stock"] = code
            fin_ind_parts.append(df)
    except:
        pass

if fin_ind_parts:
    fin_ind = pd.concat(fin_ind_parts, ignore_index=True)
    fin_ind.to_parquet(f"{DATA_DIR}/financials_ak.parquet")
    log(f"Akshare financial data: {len(fin_ind)} rows")

# ─────────────────────────────────────────────
# Merge
# ─────────────────────────────────────────────
if fin_ops is not None and len(fin_ops) > 0:
    fin_ops.to_parquet(f"{DATA_DIR}/financials.parquet")
    log(f"Saved financials.parquet ({len(fin_ops)} rows)")

log(f"Total: {(time.time()-t0)/60:.1f} min")
log("=== DONE ===")
