"""
单因子回测评估 (Single Factor Backtest)

Reads pre-computed factors from all_factors.parquet and OHLCV data from
ohlcv.parquet, then for each factor computes:
  - Cross-sectional IC / RankIC (daily)
  - 5-group equal-weight portfolio cumulative returns
  - Train/Valid period metrics

Inputs:
  C:/factor_data/all_factors.parquet   — MultiIndex(stock, date), columns = factor IDs
  C:/factor_data/ohlcv.parquet         — MultiIndex(stock, date), columns = [open, high, low, close, vol, vwap]

Outputs:
  C:/factor_data/single_factor_report.csv        — one row per factor, all metrics
  C:/factor_data/group_returns/[factor_id].parquet — 5-group cumulative returns (valid period)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
FACTOR_PATH = Path("C:/factor_data/all_factors.parquet")
OHLCV_PATH = Path("C:/factor_data/ohlcv.parquet")
OUTPUT_CSV = Path("C:/factor_data/single_factor_report.csv")
GROUP_DIR = Path("C:/factor_data/group_returns")

# ── Periods ────────────────────────────────────────────────────────────────
TRAIN_START = "2018-01-01"
TRAIN_END = "2023-12-31"
VALID_START = "2024-01-01"
VALID_END = "2026-05-15"

FWD_PERIODS = 20  # 20 trading days forward return
ANNUAL_TRADING_DAYS = 252


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load factor panel and OHLCV data, both with MultiIndex (stock, date)."""
    if not FACTOR_PATH.exists():
        print(f"ERROR: Factor data not found at {FACTOR_PATH}", flush=True)
        print("Run factor computation engine (Task #2) first to generate all_factors.parquet.", flush=True)
        print("See: C:/factor_data/", flush=True)
        sys.exit(1)

    if not OHLCV_PATH.exists():
        print(f"OHLCV data not found at {OHLCV_PATH}, bootstrapping from qlib ...", flush=True)
        bootstrap_ohlcv_from_qlib()

    print(f"Loading factors from {FACTOR_PATH} ...", flush=True)
    factors = pd.read_parquet(FACTOR_PATH)
    if not isinstance(factors.index, pd.MultiIndex):
        msg = "all_factors.parquet must have a MultiIndex (stock, date)"
        raise ValueError(msg)

    print(f"  Factors shape: {factors.shape}, columns: {len(factors.columns)}")
    print(f"  Index levels: {factors.index.names}")
    print(f"  Date range: {factors.index.get_level_values('date').min()} ~ "
          f"{factors.index.get_level_values('date').max()}")

    print(f"Loading OHLCV from {OHLCV_PATH} ...", flush=True)
    ohlcv = pd.read_parquet(OHLCV_PATH)
    print(f"  OHLCV columns: {list(ohlcv.columns)}")

    return factors, ohlcv


def _read_qlib_bin(filepath: Path) -> tuple[int, np.ndarray] | None:
    """Read a qlib-format .bin feature file.

    Format: [start_index:int32][value1:float32][value2:float32]...

    Returns (start_index, values_array) or None if file missing / empty.
    """
    if not filepath.exists():
        return None
    with open(filepath, "rb") as f:
        raw = f.read()
    if len(raw) < 8:  # header(4) + at least 1 value(4)
        return None
    start_index = int(np.frombuffer(raw[:4], dtype=np.int32)[0])
    values = np.frombuffer(raw[4:], dtype=np.float32)
    if len(values) == 0:
        return None
    return start_index, values


def bootstrap_ohlcv_from_qlib(
    target_path: Path = OHLCV_PATH,
    qlib_root: str = "C:/qlib_data",
) -> None:
    """Convert qlib .bin OHLCV files to a single parquet with MultiIndex (stock, date)."""
    cal_path = Path(qlib_root) / "calendars" / "day.txt"
    inst_path = Path(qlib_root) / "instruments" / "all.txt"
    feat_root = Path(qlib_root) / "features"

    if not cal_path.exists():
        print(f"Cannot bootstrap: calendar not found at {cal_path}", flush=True)
        return
    if not inst_path.exists():
        print(f"Cannot bootstrap: instruments not found at {inst_path}", flush=True)
        return

    # Read calendar
    cal = pd.read_csv(cal_path, header=None, names=["date"])
    cal["date"] = pd.to_datetime(cal["date"])

    # Read instruments
    inst = pd.read_csv(
        inst_path, sep="\t", header=None, names=["code", "start", "end"],
        dtype={"code": str},
    )

    fields = ["open", "high", "low", "close", "vol", "amount", "vwap"]
    records: list[dict] = []

    for code in inst["code"]:
        arrays: dict[str, np.ndarray | None] = {}
        start_idx = None

        for fld in fields:
            fpath = feat_root / code / f"{fld}.day.bin"
            result = _read_qlib_bin(fpath)
            if result is None:
                arrays[fld] = None
            else:
                idx, vals = result
                if start_idx is None:
                    start_idx = idx
                arrays[fld] = vals

        if start_idx is None:
            continue

        # Build aligned date index
        n = max(
            (len(v) for v in arrays.values() if v is not None), default=0
        )
        if n == 0:
            continue

        start_idx = int(start_idx)
        n = int(n)
        date_idx = cal.iloc[start_idx: start_idx + n]["date"].values
        for i in range(n):
            row: dict = {"stock": code, "date": date_idx[i]}
            for fld in fields:
                arr = arrays.get(fld)
                row[fld] = float(arr[i]) if arr is not None and i < len(arr) else np.nan
            records.append(row)

    if not records:
        print("No qlib data found; cannot bootstrap ohlcv.parquet", flush=True)
        return

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["stock", "date"]).sort_index()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target_path)
    print(f"Bootstrapped {len(df)} rows -> {target_path}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# Forward Returns
# ═══════════════════════════════════════════════════════════════════════════

def calc_forward_returns(close: pd.Series, periods: int = FWD_PERIODS) -> pd.Series:
    """Forward return: close.shift(-periods) / close - 1, within each stock."""
    return close.groupby(level="stock").shift(-periods) / close - 1


# ═══════════════════════════════════════════════════════════════════════════
# Single-factor Metrics
# ═══════════════════════════════════════════════════════════════════════════

def _safe_pearsonr(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    n = mask.sum()
    if n < 10:
        return np.nan
    r, _ = pearsonr(x[mask], y[mask])
    return r


def _safe_spearmanr(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    n = mask.sum()
    if n < 10:
        return np.nan
    r, _ = spearmanr(x[mask], y[mask])
    return r


def _group_returns(factor: pd.Series, fwd: pd.Series, n_groups: int = 5) -> pd.Series:
    """Sort cross-section by factor, split into n groups, return mean fwd ret per group."""
    mask = factor.notna() & fwd.notna()
    valid = mask.sum()
    if valid < n_groups:
        return pd.Series([np.nan] * n_groups, index=[f"G{i+1}" for i in range(n_groups)])

    order = np.argsort(factor[mask])
    n = valid
    size = n // n_groups
    rets = {}
    for g in range(n_groups):
        lo = g * size
        hi = n if g == n_groups - 1 else (g + 1) * size
        idx = order[lo:hi]
        rets[f"G{g+1}"] = float(fwd[mask].iloc[idx].mean())
    return pd.Series(rets)


def _monotonicity(groups: pd.Series) -> float:
    """1.0 if group means are strictly monotonic, else 0.0."""
    vals = groups.values
    asc = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    desc = all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    return 1.0 if (asc or desc) else 0.0


def _max_drawdown(cum_series: pd.Series) -> float:
    """Max drawdown from a cumulative return series."""
    peak = cum_series.expanding().max()
    dd = cum_series / peak - 1
    return float(dd.min())


def _factor_autocorr(factor: pd.Series, lag: int) -> float:
    """Cross-sectional average of time-series autocorrelation at given lag."""
    per_stock = (
        factor.groupby(level="stock")
        .apply(lambda s: s.autocorr(lag=lag), include_groups=False)
        .dropna()
    )
    return float(per_stock.mean()) if len(per_stock) > 0 else np.nan


def process_factor(
    fid: str,
    factor_series: pd.Series,
    fwd_ret: pd.Series,
) -> dict:
    """Compute all train/valid metrics for a single factor."""
    df = pd.DataFrame({"factor": factor_series, "fwd_ret": fwd_ret})
    dates = df.index.get_level_values("date")

    train_df = df[(dates >= TRAIN_START) & (dates <= TRAIN_END)]
    valid_df = df[(dates >= VALID_START) & (dates <= VALID_END)]

    result: dict = {"factor_id": fid}

    # Autocorrelation (turnover proxy) — full-sample
    result["autocorr_1d"] = _factor_autocorr(factor_series, lag=1)
    result["autocorr_5d"] = _factor_autocorr(factor_series, lag=5)

    for period_name, period_df in [("train", train_df), ("valid", valid_df)]:
        if period_df.empty:
            _fill_nan_metrics(result, period_name)
            continue

        coverage = float(period_df["factor"].notna().mean())
        result[f"{period_name}_coverage"] = coverage

        ic_vals: list[float] = []
        rankic_vals: list[float] = []
        group_ret_list: list[pd.Series] = []
        group_date_idx: list[pd.Timestamp] = []

        for dt, day_df in period_df.groupby(level="date"):
            f = day_df["factor"]
            r = day_df["fwd_ret"]

            ic_vals.append(_safe_pearsonr(f, r))
            rankic_vals.append(_safe_spearmanr(f, r))

            gr = _group_returns(f, r)
            group_ret_list.append(gr)
            group_date_idx.append(dt)  # type: ignore[arg-type]

        ic_arr = np.array(ic_vals, dtype=float)
        rankic_arr = np.array(rankic_vals, dtype=float)

        ic_mean = float(np.nanmean(ic_arr))
        ic_std = float(np.nanstd(ic_arr))
        result[f"{period_name}_IC_mean"] = ic_mean
        result[f"{period_name}_IC_std"] = ic_std
        result[f"{period_name}_ICIR"] = ic_mean / ic_std if ic_std != 0 else np.nan
        result[f"{period_name}_IC_win_rate"] = float(np.nanmean(ic_arr > 0))

        rankic_mean = float(np.nanmean(rankic_arr))
        rankic_std = float(np.nanstd(rankic_arr))
        result[f"{period_name}_RankIC_mean"] = rankic_mean
        result[f"{period_name}_RankIC_std"] = rankic_std
        result[f"{period_name}_RankICIR"] = (
            rankic_mean / rankic_std if rankic_std != 0 else np.nan
        )
        result[f"{period_name}_RankIC_win_rate"] = float(np.nanmean(rankic_arr > 0))

        # Group returns
        if not group_ret_list:
            for m in ["group_spread", "monotonicity", "max_dd_top", "annualized_top"]:
                result[f"{period_name}_{m}"] = np.nan
            continue

        group_df = pd.DataFrame(group_ret_list, index=group_date_idx)
        group_df.index.name = "date"
        mean_grp = group_df.mean()

        spread = mean_grp.get("G5", np.nan) - mean_grp.get("G1", np.nan)
        result[f"{period_name}_group_spread"] = spread
        result[f"{period_name}_monotonicity"] = _monotonicity(mean_grp)

        # Annualized return of G5 (top group)
        top_mean = mean_grp.get("G5", np.nan)
        if pd.notna(top_mean):
            result[f"{period_name}_annualized_top"] = (
                (1 + top_mean) ** (ANNUAL_TRADING_DAYS / FWD_PERIODS) - 1
            )
        else:
            result[f"{period_name}_annualized_top"] = np.nan

        # Max drawdown of top group cumulative returns
        if "G5" in group_df.columns:
            cum_g5 = (1 + group_df["G5"]).cumprod()
            result[f"{period_name}_max_dd_top"] = _max_drawdown(cum_g5)
        else:
            result[f"{period_name}_max_dd_top"] = np.nan

        # Save valid-period group returns to parquet
        if period_name == "valid":
            GROUP_DIR.mkdir(parents=True, exist_ok=True)
            # Cumulative returns
            cum_df = (1 + group_df).cumprod()
            cum_path = GROUP_DIR / f"{fid}.parquet"
            cum_df.to_parquet(cum_path)

    return result


def _fill_nan_metrics(result: dict, period_name: str) -> None:
    """Fill all metrics for a period with NaN if no data."""
    metrics = [
        "IC_mean", "IC_std", "ICIR", "IC_win_rate",
        "RankIC_mean", "RankIC_std", "RankICIR", "RankIC_win_rate",
        "group_spread", "monotonicity", "max_dd_top", "annualized_top", "coverage",
    ]
    for m in metrics:
        result[f"{period_name}_{m}"] = np.nan


def annualize_spread(daily_spread: float) -> float:
    """Annualize a daily mean spread (difference in 20d forward returns)."""
    return (1 + daily_spread) ** (ANNUAL_TRADING_DAYS / FWD_PERIODS) - 1


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    GROUP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    factors, ohlcv = load_data()

    # 2. Forward returns
    if "close" not in ohlcv.columns:
        print("ERROR: ohlcv.parquet must contain a 'close' column.", flush=True)
        sys.exit(1)

    print("Computing forward 20-day returns ...", flush=True)
    fwd_ret = calc_forward_returns(ohlcv["close"])
    print(f"  Forward return range: [{fwd_ret.min():.4f}, {fwd_ret.max():.4f}]")
    print(f"  Non-null rate: {fwd_ret.notna().mean():.2%}")

    # 3. Per-factor backtest
    factor_ids: list[str] = list(factors.columns)
    n = len(factor_ids)
    print(f"\nProcessing {n} factors ...", flush=True)

    all_results: list[dict] = []
    for i, fid in enumerate(factor_ids):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{n}] {fid}", flush=True)

        series = factors[fid]
        result = process_factor(fid, series, fwd_ret)
        all_results.append(result)

    # 4. Build report
    report = pd.DataFrame(all_results).set_index("factor_id")

    # Sort columns: train then valid
    train_cols = [c for c in report.columns if c.startswith("train_")]
    valid_cols = [c for c in report.columns if c.startswith("valid_")]
    other_cols = [c for c in report.columns if not c.startswith(("train_", "valid_"))]
    report = report[other_cols + train_cols + valid_cols]

    report.to_csv(OUTPUT_CSV, encoding="utf-8-sig")
    print(f"\nReport saved -> {OUTPUT_CSV}")
    print(f"  Factors: {len(report)}")
    print(f"  Metrics: {len(report.columns)}")

    # 5. Print top 20 by valid_RankIC_mean
    rank_col = "valid_RankIC_mean"
    spread_col = "valid_group_spread"
    if rank_col in report.columns:
        top20 = report.sort_values(rank_col, ascending=False).head(20)
        display_cols = [
            rank_col, "valid_RankICIR", "valid_IC_mean",
            "valid_ICIR", "valid_group_spread",
            "valid_annualized_top", "valid_monotonicity", "valid_coverage",
        ]
        display_cols = [c for c in display_cols if c in top20.columns]
        print(f"\n{'=' * 80}")
        print(f"TOP 20 FACTORS by {rank_col}")
        print(f"{'=' * 80}")
        pd.set_option("display.max_columns", 12)
        pd.set_option("display.width", 120)
        print(top20[display_cols].to_string(float_format=lambda x: f"{x:.6f}"))

    # 6. Also show top 20 by ICIR
    icir_col = "valid_ICIR"
    if icir_col in report.columns:
        top_icir = report.sort_values(icir_col, ascending=False).head(20)
        print(f"\n{'=' * 80}")
        print(f"TOP 20 FACTORS by {icir_col}")
        print(f"{'=' * 80}")
        print(top_icir[display_cols].to_string(float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
