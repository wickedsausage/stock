"""Worker process: download OHLCV for a chunk of A-share stocks via baostock.

Usage:
    python dual_worker.py <stock_list> <output_dir> <adjustflag> <worker_id> <total_workers>

Processes its assigned chunk of stocks with baostock, saving per-stock parquet files.
login/logout per batch. Skips existing batch files (resume).
Socket timeout prevents indefinite hangs.
"""
import os, sys, time, gc, socket
import pandas as pd
import baostock as bs

# Prevent hung connections from blocking indefinitely
socket.setdefaulttimeout(30)

BATCH_SIZE = 50  # smaller batches for faster progress feedback
DELAY_STOCK = 0.3
DELAY_BATCH = 0.3
FIELDS = "date,open,high,low,close,volume,amount"
NUM_COLS = ["open", "high", "low", "close", "volume", "amount"]


def log(msg):
    t = time.strftime('%H:%M:%S')
    print(f"[{t}] {msg}", flush=True)


def main():
    stock_list_path = sys.argv[1]
    output_dir = sys.argv[2]
    adjustflag = sys.argv[3]
    worker_id = int(sys.argv[4])
    total_workers = int(sys.argv[5])

    os.makedirs(output_dir, exist_ok=True)

    # Load stock list, filter A-shares
    stock_basic = pd.read_parquet(stock_list_path)
    all_codes = stock_basic[stock_basic["type"] == "1"]["code"].tolist()

    # Determine chunk
    chunk_size = (len(all_codes) + total_workers - 1) // total_workers
    start_idx = worker_id * chunk_size
    end_idx = min(start_idx + chunk_size, len(all_codes))
    my_codes = all_codes[start_idx:end_idx]

    log(f"Worker {worker_id}/{total_workers}: {len(my_codes)} stocks "
        f"(idx {start_idx}..{end_idx}), adjustflag={adjustflag}")

    # Check existing files for resume
    existing_set = set()
    if os.path.isdir(output_dir):
        for fn in os.listdir(output_dir):
            if fn.endswith(".parquet"):
                existing_set.add(fn.replace(".parquet", ""))

    to_dl = []
    for full_code in my_codes:
        sym = full_code.split(".")[1]
        if sym not in existing_set:
            to_dl.append((full_code, sym))

    log(f"Worker {worker_id}: {len(to_dl)} remaining after resume check "
        f"({len(existing_set)} already cached)")

    if not to_dl:
        log(f"Worker {worker_id}: nothing to do, exiting")
        return

    t0 = time.time()
    total_ok = 0

    for batch_start in range(0, len(to_dl), BATCH_SIZE):
        batch = to_dl[batch_start:batch_start + BATCH_SIZE]

        bs.login()
        for full_code, sym in batch:
            out_path = f"{output_dir}/{sym}.parquet"
            if os.path.exists(out_path):
                continue
            try:
                rs = bs.query_history_k_data_plus(
                    full_code, FIELDS,
                    "2016-01-01", "2026-05-15", "d", adjustflag,
                )
                if rs.error_code != "0":
                    time.sleep(DELAY_STOCK)
                    continue

                df = rs.get_data()
                if df is None or len(df) == 0:
                    time.sleep(DELAY_STOCK)
                    continue

                for c in NUM_COLS:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df[df["close"] > 0].copy()
                if len(df) == 0:
                    time.sleep(DELAY_STOCK)
                    continue

                df["stock"] = sym
                df["date"] = pd.to_datetime(df["date"])
                df = df[["stock", "date", "open", "high", "low", "close",
                         "volume", "amount"]]
                df.to_parquet(out_path, index=False)
                total_ok += 1
            except Exception as e:
                log(f"  W{worker_id} ERROR {full_code}: {e}")

            time.sleep(DELAY_STOCK)

        bs.logout()

        batch_num = batch_start // BATCH_SIZE + 1
        done_here = min(batch_start + BATCH_SIZE, len(to_dl))
        elapsed = time.time() - t0
        rate = done_here / elapsed * 60 if elapsed > 0 else 0
        log(f"  W{worker_id} batch {batch_num}: {done_here}/{len(to_dl)} "
            f"attempted, {total_ok} OK, {rate:.0f} stocks/min")

        time.sleep(DELAY_BATCH)
        gc.collect()

    elapsed_min = (time.time() - t0) / 60
    log(f"Worker {worker_id} done in {elapsed_min:.1f} min "
        f"({total_ok} new stocks)")


if __name__ == "__main__":
    main()
