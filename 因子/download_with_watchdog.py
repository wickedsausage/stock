"""
Baostock OHLCV downloader with watchdog auto-restart.

If workers stall (no new files for STALL_TIMEOUT seconds), the
watchdog kills and restarts them.  Per-stock parquet files mean
no data is lost on restart.

Usage: python download_with_watchdog.py
"""
import os, sys, time, subprocess, signal

DATA = "C:/factor_data"
RAW_DIR = f"{DATA}/raw_batches"
ADJ_DIR = f"{DATA}/adj_batches"
N_WORKERS = 2
BATCH_DIRS = {"1": RAW_DIR, "2": ADJ_DIR}
STALL_TIMEOUT = 300  # 5 minutes without new files = stall
CHECK_INTERVAL = 30   # check every 30 seconds

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(ADJ_DIR, exist_ok=True)

worker_script = os.path.join(os.path.dirname(__file__), "dual_worker.py")
stock_list = f"{DATA}/stock_list.parquet"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def count_files(directory):
    if not os.path.isdir(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.endswith(".parquet")])


def get_stock_list():
    """Ensure stock list exists."""
    if not os.path.exists(stock_list):
        import baostock as bs
        import pandas as pd
        log("Downloading stock list...")
        bs.login()
        stock_basic = bs.query_stock_basic().get_data()
        bs.logout()
        stock_basic.to_parquet(stock_list, index=False)
        log(f"Saved stock_list.parquet ({len(stock_basic)} rows)")


def spawn_workers(batch_dir, adjustflag):
    """Spawn N worker subprocesses."""
    procs = []
    for wid in range(N_WORKERS):
        p = subprocess.Popen(
            [sys.executable, worker_script,
             stock_list, batch_dir, adjustflag,
             str(wid), str(N_WORKERS)],
        )
        procs.append(p)
    return procs


def kill_processes(procs):
    for p in procs:
        if p.poll() is None:
            p.kill()
    for p in procs:
        p.wait()


def run_phase(adjustflag):
    batch_dir = BATCH_DIRS[adjustflag]
    label = "B" if adjustflag == "1" else "C"

    start_count = count_files(batch_dir)
    total_a = count_files(RAW_DIR) + count_files(ADJ_DIR)
    total_a -= count_files(ADJ_DIR) if adjustflag == "1" else 0

    log(f"Phase {label}: adjustflag={adjustflag}, "
        f"already have {start_count} files")

    procs = spawn_workers(batch_dir, adjustflag)
    last_count = start_count
    last_time = time.time()
    t0 = time.time()
    stall_warnings = 0

    while True:
        time.sleep(CHECK_INTERVAL)
        current_count = count_files(batch_dir)
        now = time.time()

        # Check if any progress
        if current_count > last_count:
            last_count = current_count
            last_time = now
            stall_warnings = 0

        elapsed_since_last = now - last_time

        # Check if all workers have exited (completed or crashed)
        alive = [p for p in procs if p.poll() is None]

        # If no workers alive and no more progress expected
        if not alive:
            # Check if there are remaining stocks to download
            # by comparing against full stock list
            import pandas as pd
            sl = pd.read_parquet(stock_list)
            all_codes = sl[sl["type"] == "1"]["code"].tolist()
            remaining = [c for c in all_codes
                         if c.split(".")[1] not in
                         set(f.replace(".parquet", "")
                             for f in os.listdir(batch_dir)
                             if f.endswith(".parquet"))]
            if not remaining:
                log(f"Phase {label}: ALL STOCKS COMPLETE!")
                break
            else:
                log(f"Phase {label}: Workers died, {len(remaining)} "
                    f"remaining. Restarting...")
                procs = spawn_workers(batch_dir, adjustflag)
                last_time = time.time()
                stall_warnings = 0
                continue

        # Check for stall
        if elapsed_since_last > STALL_TIMEOUT:
            stall_warnings += 1
            alive_count = len([p for p in procs if p.poll() is None])
            elapsed_total = now - t0
            rate = (current_count - start_count) / elapsed_total * 60 \
                if elapsed_total > 0 else 0
            log(f"STALLED ({elapsed_since_last:.0f}s no progress). "
                f"Killing {alive_count} workers and restarting. "
                f"Total: {current_count} files, {rate:.0f}/min")
            kill_processes(procs)
            time.sleep(2)
            procs = spawn_workers(batch_dir, adjustflag)
            last_count = count_files(batch_dir)
            last_time = time.time()
            stall_warnings = 0

    # Cleanup
    for p in procs:
        if p.poll() is None:
            p.kill()
            p.wait()

    elapsed = time.time() - t0
    final_count = count_files(batch_dir)
    log(f"Phase {label} done: {final_count} files in {elapsed/60:.1f} min")


def merge_phase(batch_dir, output_path):
    """Merge per-stock parquet files into single parquet."""
    import pandas as pd
    files = sorted(f for f in os.listdir(batch_dir)
                   if f.endswith(".parquet"))
    if not files:
        log(f"No files to merge in {batch_dir}")
        return None

    log(f"Merging {len(files)} files...")
    parts = []
    for fn in files:
        parts.append(pd.read_parquet(os.path.join(batch_dir, fn)))

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["stock", "date"])
    df = df.sort_values(["stock", "date"]).reset_index(drop=True)
    df.to_parquet(output_path, index=False)
    n_stock = df["stock"].nunique()
    n_date = df["date"].nunique()
    log(f"Saved {output_path}: {len(df):,} rows, "
        f"{n_stock} stocks, {n_date} dates, "
        f"{df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


if __name__ == "__main__":
    get_stock_list()

    # Phase B: Raw
    run_phase("1")
    df_raw = merge_phase(RAW_DIR, f"{DATA}/raw_ohlcv.parquet")

    # Phase C: Adjusted
    run_phase("2")
    df_adj = merge_phase(ADJ_DIR, f"{DATA}/adj_ohlcv.parquet")

    # Phase D: Calendar
    log("Phase D: Calendar...")
    if df_adj is not None and len(df_adj) > 0:
        dates = sorted(df_adj["date"].unique())
        pd.DataFrame({"date": dates}).to_parquet(
            f"{DATA}/calendar.parquet", index=False
        )
        log(f"Calendar: {len(dates)} days ({dates[0].date()} ~ {dates[-1].date()})")

    log("=== ALL DONE ===")
