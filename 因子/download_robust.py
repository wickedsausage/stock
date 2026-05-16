"""
Robust single-stock OHLCV downloader with per-stock subprocess timeout.

Each stock query runs in its own subprocess (60s timeout).
Failed stocks are retried. Progress is never lost (per-stock parquet).
Usage: python download_robust.py <adjustflag> [<stock_list>]
"""
import os, sys, time, subprocess, socket
import pandas as pd

DATA = "C:/factor_data"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download_one_stock(full_code, sym, adjustflag, output_dir):
    """Download one stock's OHLCV via baostock subprocess. Returns True on success."""
    out_path = f"{output_dir}/{sym}.parquet"
    if os.path.exists(out_path):
        return True  # already done

    code = f"""
import baostock as bs, pandas as pd, socket, sys
socket.setdefaulttimeout(30)
FIELDS = "date,open,high,low,close,volume,amount"
NUM_COLS = ["open", "high", "low", "close", "volume", "amount"]
bs.login()
rs = bs.query_history_k_data_plus("{full_code}", FIELDS,
    "2016-01-01", "2026-05-15", "d", "{adjustflag}")
if rs.error_code != "0":
    bs.logout()
    sys.exit(1)
df = rs.get_data()
if df is None or len(df) == 0:
    bs.logout()
    sys.exit(1)
for c in NUM_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df[df["close"] > 0]
if len(df) == 0:
    bs.logout()
    sys.exit(1)
df["stock"] = "{sym}"
df["date"] = pd.to_datetime(df["date"])
df = df[["stock","date","open","high","low","close","volume","amount"]]
df.to_parquet(r"{out_path}", index=False)
bs.logout()
"""
    try:
        p = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=60,
        )
        return p.returncode == 0 and os.path.exists(out_path)
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def run_phase(adjustflag, label):
    output_dir = f"{DATA}/{'raw' if adjustflag=='1' else 'adj'}_batches"
    os.makedirs(output_dir, exist_ok=True)

    # Load stock list
    sl = pd.read_parquet(f"{DATA}/stock_list.parquet")
    all_codes = sl[sl["type"] == "1"]["code"].tolist()

    # Check existing
    existing = set()
    for fn in os.listdir(output_dir):
        if fn.endswith(".parquet"):
            existing.add(fn.replace(".parquet", ""))

    to_dl = [(c, c.split(".")[1]) for c in all_codes
             if c.split(".")[1] not in existing]
    log(f"Phase {label} ({adjustflag}): {len(to_dl)} remaining, "
        f"{len(existing)} cached")

    if not to_dl:
        log(f"Phase {label}: nothing to do")
        return

    t0 = time.time()
    ok = 0
    fail = 0
    N = len(to_dl)

    for idx, (full_code, sym) in enumerate(to_dl):
        out_path = f"{output_dir}/{sym}.parquet"
        if os.path.exists(out_path):
            ok += 1
            continue

        success = download_one_stock(full_code, sym, adjustflag, output_dir)
        if success:
            ok += 1
        else:
            fail += 1
            # Write a marker for retry later
            log(f"  FAIL {full_code} ({fail}/{N})")

        if (idx + 1) % 200 == 0 or idx == N - 1:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed * 60 if elapsed > 0 else 0
            log(f"  {idx+1}/{N} attempted, {ok} OK, {fail} fail, "
                f"{rate:.0f} stocks/min")

    log(f"Phase {label}: {ok} OK, {fail} fail in "
        f"{(time.time()-t0)/60:.1f} min")

    # Retry failed stocks
    if fail > 0:
        log(f"Phase {label}: Retrying {fail} failed stocks...")
        retry_fail = 0
        for full_code, sym in to_dl:
            out_path = f"{output_dir}/{sym}.parquet"
            if not os.path.exists(out_path):
                success = download_one_stock(full_code, sym, adjustflag, output_dir)
                if not success:
                    retry_fail += 1
                    log(f"  RETRY FAIL {full_code}")
        log(f"Phase {label} retry: {fail - retry_fail} recovered, "
            f"{retry_fail} still failed")


if __name__ == "__main__":
    adjustflag = sys.argv[1]
    label = "B" if adjustflag == "1" else "C"
    run_phase(adjustflag, label)
