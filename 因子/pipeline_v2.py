"""
Factor MVP Pipeline v2 — 5-part pipeline.

Part 1: adj_factor (raw/adj ratio)
Part 2: Three-layer universe (ranking/tradable/portfolio)
Part 3: Labels (forward returns for H in [1,5,10,20])
Part 4: Factor computation (~50 factors from adj_ohlcv)
Part 5: Single-factor backtest (IC, RankIC, reports)

All paths use C:/factor_data/ exclusively.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

DATA = Path("C:/factor_data")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _fix_ohlcv(df):
    """No-op — merge already produces correct (stock, date) index."""
    return df


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — adj_factor
# ═══════════════════════════════════════════════════════════════════════════

def part1_adj_factor() -> pd.DataFrame:
    """Compute adj_factor = raw_close / adj_close per stock per date."""
    log("=" * 50)
    log("PART 1: adj_factor")

    raw = (pd.read_parquet(DATA / "raw_ohlcv.parquet"))
    adj = (pd.read_parquet(DATA / "adj_ohlcv.parquet"))
    log(f"  raw: {len(raw):,} rows, adj: {len(adj):,} rows")

    # Align on common (stock, date) index
    common_idx = raw.index.intersection(adj.index)
    raw = raw.loc[common_idx]
    adj = adj.loc[common_idx]

    adj_factor = (raw["close"] / adj["close"]).to_frame("adj_factor")
    adj_factor.to_parquet(DATA / "adj_factor.parquet")
    log(f"  Saved adj_factor.parquet ({len(adj_factor):,} rows)")
    return adj_factor


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — Universe
# ═══════════════════════════════════════════════════════════════════════════

def _infer_board(code: str) -> str:
    """Infer board from stock code prefix."""
    if code.startswith("6"):
        return "SH"
    if code.startswith("68"):
        return "SH"
    if code.startswith("00"):
        return "SZ_GE"
    if code.startswith("30"):
        return "SZ_MA"
    if code.startswith("83"):
        return "BJ"
    # fallback
    return "SH"


def _infer_limit_pct(board: str) -> float:
    """Daily price limit by board."""
    if board in ("SH", "SZ_GE"):
        return 0.10
    if board in ("SZ_MA",):
        return 0.20
    if board == "BJ":
        return 0.30
    return 0.10


def part2_universe() -> pd.DataFrame:
    """Build three-layer universe and save to universe.parquet."""
    log("=" * 50)
    log("PART 2: Universe")

    raw = (pd.read_parquet(DATA / "raw_ohlcv.parquet"))
    close_raw = raw["close"].unstack(level=0)  # → (date x stock)
    vol_raw = raw["volume"].unstack(level=0)

    # Stock codes
    stocks = close_raw.columns.tolist()

    # Board & limit pct
    board_map = {s: _infer_board(s) for s in stocks}
    limit_map = {s: _infer_limit_pct(board_map[s]) for s in stocks}

    # Prev close for limit prices
    prev_close = close_raw.shift(1)

    # Limit up/down prices
    limit_up_prices = prev_close * (1 + pd.Series(limit_map))
    limit_down_prices = prev_close * (1 - pd.Series(limit_map))

    # is_listed: stock has valid close
    is_listed = close_raw.notna()

    # days_since_ipo
    ds_ipo = pd.DataFrame(
        np.maximum(0, np.arange(len(close_raw))[:, None] - close_raw.notna().idxmax(skipna=True).map(
            {s: i for i, s in enumerate(close_raw.columns)}
        ).values),
        index=close_raw.index,
        columns=close_raw.columns,
    )

    # is_suspended: volume is 0 or NaN
    is_suspended = vol_raw.fillna(0) == 0

    # Build universe panel per date
    dates = close_raw.index
    n = len(dates)
    n_stocks = len(stocks)

    ranking_mask = is_listed.values & (ds_ipo.values > 60) & (~is_suspended.values)
    # Tradable also excludes ST and limit-up (same as ranking for MVP since no ST data)
    tradable_mask = ranking_mask.copy()

    records = []
    for i, dt in enumerate(dates):
        for j, stock in enumerate(stocks):
            if not ranking_mask[i, j]:
                records.append({
                    "stock": stock, "date": dt,
                    "is_listed": False, "is_suspended": True,
                    "days_since_ipo": int(ds_ipo.values[i, j]) if not pd.isna(ds_ipo.values[i, j]) else 9999,
                    "limit_pct": limit_map[stock],
                    "limit_up_price": np.nan, "limit_down_price": np.nan,
                    "board": board_map[stock],
                    "is_st": False, "st_risk_unknown": True,
                    "ranking_universe": False, "tradable_universe": False,
                    "portfolio_universe": False,
                })
            else:
                lup = limit_up_prices.values[i, j]
                ldn = limit_down_prices.values[i, j]
                records.append({
                    "stock": stock, "date": dt,
                    "is_listed": True, "is_suspended": bool(is_suspended.values[i, j]),
                    "days_since_ipo": int(ds_ipo.values[i, j]) if not pd.isna(ds_ipo.values[i, j]) else 9999,
                    "limit_pct": limit_map[stock],
                    "limit_up_price": float(lup) if not np.isnan(lup) else np.nan,
                    "limit_down_price": float(ldn) if not np.isnan(ldn) else np.nan,
                    "board": board_map[stock],
                    "is_st": False, "st_risk_unknown": True,
                    "ranking_universe": True,
                    "tradable_universe": bool(tradable_mask[i, j]),
                    "portfolio_universe": bool(tradable_mask[i, j]),
                })

    uni = pd.DataFrame(records)
    uni = uni.set_index(["stock", "date"]).sort_index()
    uni.to_parquet(DATA / "universe.parquet")
    log(f"  Saved universe.parquet ({len(uni):,} rows)")
    log(f"  ranking_universe: {uni['ranking_universe'].sum():,}")
    log(f"  tradable_universe: {uni['tradable_universe'].sum():,}")
    return uni


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — Labels
# ═══════════════════════════════════════════════════════════════════════════

def _tradable_entry_mask(open_panel: pd.DataFrame, high_panel: pd.DataFrame,
                         low_panel: pd.DataFrame, close_panel: pd.DataFrame,
                         vol_panel: pd.DataFrame, limit_up: pd.DataFrame,
                         is_suspended: pd.DataFrame) -> pd.DataFrame:
    """Return bool panel: True if stock is tradable at entry."""
    # Not suspended
    ok = ~is_suspended
    # Not limit-up blocked (strict): open >= limit_up - 0.01 AND
    #   (vol==0 OR open==high==low==close)
    at_limit = open_panel >= (limit_up - 0.01)
    no_trade = (vol_panel == 0) | (
        (open_panel == high_panel) & (high_panel == low_panel) & (low_panel == close_panel)
    )
    limit_blocked = at_limit & no_trade
    ok = ok & ~limit_blocked
    return ok


def _tradable_exit_mask(open_panel: pd.DataFrame, close_panel: pd.DataFrame,
                        vol_panel: pd.DataFrame, limit_down: pd.DataFrame,
                        avg_vol_20: pd.DataFrame) -> pd.DataFrame:
    """Return bool panel: True if stock is tradable at exit (not limit-down blocked)."""
    near_limit = (open_panel <= limit_down * 1.005) & (close_panel <= limit_down * 1.005)
    low_vol = vol_panel < avg_vol_20 * 0.1
    return ~(near_limit & low_vol)


def part3_labels() -> pd.DataFrame:
    """Compute forward returns for H in [1,5,10,20] with tradable flags."""
    log("=" * 50)
    log("PART 3: Labels")

    raw = (pd.read_parquet(DATA / "raw_ohlcv.parquet"))
    uni = pd.read_parquet(DATA / "universe.parquet")
    adj_factor = pd.read_parquet(DATA / "adj_factor.parquet")

    # Unstack raw OHLCV panels
    open_ = raw["open"].unstack(level=0)
    high = raw["high"].unstack(level=0)
    low = raw["low"].unstack(level=0)
    close = raw["close"].unstack(level=0)
    vol = raw["volume"].unstack(level=0)

    # Universe panels
    is_suspended = ~uni["is_listed"].unstack(level=0).fillna(False)

    # Limit prices from universe
    limit_up = uni["limit_up_price"].unstack(level=0)
    limit_down = uni["limit_down_price"].unstack(level=0)

    # 20-day average volume for exit check
    avg_vol_20 = vol.rolling(20, min_periods=5).mean()

    # Suspended at entry (= exited previous day)
    # We need current-date is_suspended for entry checks
    # (is_suspended already has the right alignment)

    # Entry mask — precompute once
    entry_ok = _tradable_entry_mask(open_, high, low, close, vol, limit_up, is_suspended)

    all_labels = []
    horizons = [1, 5, 10, 20]

    for H in horizons:
        log(f"  Horizon {H:2d}D ...")
        # Shifted panels
        entry_price = open_.shift(-1)  # next day open
        exit_price = close.shift(-H)   # close H days later
        raw_return = exit_price / entry_price - 1
        # Market excess: demean cross-sectionally per date
        cs_mean = raw_return.mean(axis=1, skipna=True)
        market_excess = raw_return.sub(cs_mean, axis=0)
        # Industry neutral placeholder (no industry data) — same as market excess
        industry_neutral = market_excess.copy()

        # Exit tradable mask
        exit_ok = _tradable_exit_mask(
            open_.shift(-H), close.shift(-H),
            vol.shift(-H), limit_down.shift(-H) if H == 1 else limit_down.shift(-H),
            avg_vol_20
        )

        # For multi-day, recalc exit mask with proper shifts
        if H > 1:
            exit_ok = _tradable_exit_mask(
                open_.shift(-H), close.shift(-H),
                vol.shift(-H),
                limit_down.shift(-(H-1)),  # exit date limit down uses exit-1 prev close
                avg_vol_20
            )

        # Tradable flag: entry_ok AND exit_ok
        tradable = entry_ok & exit_ok

        # Stack to (stock, date) format
        col_map = {
            "raw_return": raw_return,
            "market_excess": market_excess,
            "industry_neutral": industry_neutral,
            "tradable": tradable,
        }
        for colname, panel in col_map.items():
            stacked = panel.stack(dropna=False)
            stacked.name = f"label_{H:02d}D_{colname}"
            all_labels.append(stacked)

    result = pd.concat(all_labels, axis=1)
    result.index.names = ["date", "stock"]
    result = result.swaplevel().sort_index()
    result.to_parquet(DATA / "labels.parquet")
    log(f"  Saved labels.parquet ({len(result):,} rows, {len(result.columns)} cols)")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — Factor Computation
# ═══════════════════════════════════════════════════════════════════════════

def _preprocess_panel(panel: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-date: inf->NaN, winsorize, fillna(cs_median), zscore within mask."""
    result = panel.copy()
    for dt in result.index:
        row = result.loc[dt]
        sel = mask.loc[dt] if mask is not None else pd.Series(True, index=row.index)
        # Restrict to valid mask + finite
        valid = sel & np.isfinite(row)
        vals = row[valid]
        if len(vals) < 10:
            result.loc[dt] = np.nan
            continue
        # Winsorize (in the full row context for consistency)
        q_lo = vals.quantile(0.005)
        q_hi = vals.quantile(0.995)
        clipped = row.clip(lower=q_lo, upper=q_hi)
        # Fill NaN with cross-sectional median (within mask)
        cs_median = clipped[valid].median()
        filled = clipped.fillna(cs_median)
        # Z-score
        mu = filled[sel].mean()
        sd = filled[sel].std()
        if sd > 1e-12:
            result.loc[dt] = (filled - mu) / sd
        else:
            result.loc[dt] = 0.0
    return result


def part4_factors(labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute 50+ factors from adj_ohlcv, with rank/zscore variants."""
    log("=" * 50)
    log("PART 4: Factor Computation")

    adj = (pd.read_parquet(DATA / "adj_ohlcv.parquet"))
    uni = pd.read_parquet(DATA / "universe.parquet")

    # Unstack (stock, date) -> (date x stock) panel
    close = adj["close"].unstack(level=0)
    high = adj["high"].unstack(level=0)
    low = adj["low"].unstack(level=0)
    open_ = adj["open"].unstack(level=0)
    vol = adj["volume"].unstack(level=0)
    amt = adj["amount"].unstack(level=0)

    # Ranking universe mask (date x stock)
    ranking_mask = uni["ranking_universe"].unstack(level=0).fillna(False).astype(bool)

    # Forward fill for suspension gaps (max 5 days)
    close_ff = close.ffill(limit=5)
    high_ff = high.ffill(limit=5)
    low_ff = low.ffill(limit=5)
    open_ff = open_.ffill(limit=5)
    vol_ff = vol.fillna(0)
    amt_ff = amt.fillna(0)

    ret = close_ff.pct_change()
    ret5 = close_ff.pct_change(5)
    ret20 = close_ff.pct_change(20)
    ret60 = close_ff.pct_change(60)
    ret120 = close_ff.pct_change(120)
    ret252 = close_ff.pct_change(252)

    log("  Computing raw factors...")
    t0 = time.time()

    # --- Momentum (6) ---
    mom_1M = ret20
    mom_3M = ret60
    mom_6M = ret120
    mom_12m1m = ret252 - ret20
    overnight_mom = (open_ - close.shift(1)) / close.shift(1)
    intraday_mom = (close - open_) / open_

    # --- Reversal (3) ---
    rev_1D = -ret
    rev_5D = -ret5
    rev_20D = -ret20

    # --- Volatility (6) ---
    vol_20 = ret.rolling(20).std()
    vol_60 = ret.rolling(60).std()
    downside_vol = ret.clip(upper=0).rolling(20).std()
    parkinson_vol = np.sqrt((np.log(high / low) ** 2 / (4 * np.log(2))).rolling(20).mean())
    # Idio vol: market-hedged residual std
    mkt_ret = ret.mean(axis=1)
    idio_vol = ret.sub(mkt_ret, axis=0).rolling(60).std()
    # Beta 60
    cov_rm = ret.rolling(60).cov(mkt_ret)
    var_m = mkt_ret.rolling(60).var()
    beta_60 = cov_rm.div(var_m.replace(0, np.nan), axis=0)

    # --- Liquidity (6) ---
    turnover_20 = vol_ff.rolling(20).mean()
    turnover_60 = vol_ff.rolling(60).mean()
    amihud = (ret.abs() / amt_ff.replace(0, np.nan)).rolling(20).mean()
    vol_shock = vol_ff / vol_ff.rolling(20).mean() - 1
    abnormal_turnover = vol_ff.rolling(20).mean() / vol_ff.rolling(120).mean() - 1
    avg_amount = amt_ff.rolling(20).mean()

    # --- Technical (8) ---
    ma5 = close_ff / close_ff.rolling(5).mean()
    ma20 = close_ff / close_ff.rolling(20).mean()
    ma60 = close_ff / close_ff.rolling(60).mean()
    std20 = close_ff.rolling(20).std() / close_ff
    l20 = low_ff.rolling(20).min()
    h20 = high_ff.rolling(20).max()
    rsv = (close_ff - l20) / (h20 - l20 + 1e-10)
    delta = close_ff.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi14 = 100 - 100 / (1 + rs)
    skew_20 = ret.rolling(20).skew()
    kurt_20 = ret.rolling(20).kurt()

    # --- Composite (4) ---
    # Precompute CS ranks for composites
    mom_1M_cs = mom_1M.rank(axis=1, pct=True)
    vol_20_cs = vol_20.rank(axis=1, pct=True)
    rev_5d_cs = rev_5D.rank(axis=1, pct=True)
    turnover_cs = turnover_20.rank(axis=1, pct=True)
    price_52wh = close_ff / close_ff.rolling(252).max()
    mom_52wh_cs = mom_1M_cs * price_52wh.rank(axis=1, pct=True)

    mom_lowvol = mom_1M_cs * (1 - vol_20_cs)
    rev_liq = rev_5d_cs * turnover_cs
    factor_mom_52wh = price_52wh
    vol_regime = vol_60.rank(axis=1, pct=True) - vol_20_cs

    # Collect all raw factor panels
    raw_factors = {
        # Momentum
        "mom_1M": mom_1M, "mom_3M": mom_3M, "mom_6M": mom_6M,
        "mom_12m1m": mom_12m1m, "overnight_mom": overnight_mom, "intraday_mom": intraday_mom,
        # Reversal
        "rev_1D": rev_1D, "rev_5D": rev_5D, "rev_20D": rev_20D,
        # Volatility
        "vol_20": vol_20, "vol_60": vol_60, "downside_vol": downside_vol,
        "parkinson_vol": parkinson_vol, "idio_vol": idio_vol, "beta_60": beta_60,
        # Liquidity
        "turnover_20": turnover_20, "turnover_60": turnover_60, "amihud": amihud,
        "vol_shock": vol_shock, "abnormal_turnover": abnormal_turnover, "avg_amount": avg_amount,
        # Technical
        "MA5": ma5, "MA20": ma20, "MA60": ma60, "STD20": std20,
        "RSV": rsv, "RSI14": rsi14, "SKEW_20": skew_20, "KURT_20": kurt_20,
        # Composite
        "mom_lowvol": mom_lowvol, "rev_liq": rev_liq, "mom_52WH": factor_mom_52wh,
        "vol_regime": vol_regime,
    }
    log(f"  Raw factor panels: {len(raw_factors)} ({time.time()-t0:.1f}s)")

    # Preprocess: per-date winsorize, fillna, zscore within ranking_universe
    log("  Preprocessing (winsorize -> fillna -> zscore)...")
    t1 = time.time()
    processed: dict[str, pd.DataFrame] = {}
    for fname, panel in raw_factors.items():
        pp = _preprocess_panel(panel, ranking_mask)
        processed[fname] = pp
    log(f"  Preprocessing done ({time.time()-t1:.1f}s)")

    # Cross-sectional rank and zscore versions
    log("  Computing rank and zscore variants...")
    t2 = time.time()
    out_panels: dict[str, pd.DataFrame] = {}
    for fname, panel in processed.items():
        # Raw (already z-scored)
        out_panels[fname] = panel
        # Rank (pct)
        out_panels[f"{fname}_rank"] = panel.rank(axis=1, pct=True)
        # zscore (already z-scored in preprocessing, but store explicit version)
        # The raw version IS the zscore already. Add _zscore alias.
        out_panels[f"{fname}_zscore"] = panel
    log(f"  Variants done ({time.time()-t2:.1f}s)")

    # Factor direction adjustment using train-period RankIC
    if labels is not None:
        log("  Adjusting factor directions using train RankIC...")
        label_20d = labels["label_20D_raw_return"] if "label_20D_raw_return" in labels.columns else None
        if label_20d is not None:
            # Align
            label_panel = label_20d.unstack(level=0)
            train_end = "2021-12-31"
            train_mask = label_panel.index <= train_end
            train_labels = label_panel[train_mask]

            # Adjust each factor by sign of train RankIC
            for fname in list(out_panels.keys()):
                if fname.endswith("_rank") or fname.endswith("_zscore"):
                    continue
                panel = out_panels[fname]
                common_dates = train_labels.index.intersection(panel.index)
                if len(common_dates) < 10:
                    continue

                # Compute daily RankIC over train period
                ic_vals = []
                for dt in common_dates:
                    f = panel.loc[dt]
                    l = train_labels.loc[dt]
                    valid = f.notna() & l.notna()
                    if valid.sum() < 30:
                        continue
                    x = f[valid].values
                    y = l[valid].values
                    if x.std() > 1e-12 and y.std() > 1e-12:
                        from scipy.stats import rankdata
                        ic_vals.append(np.corrcoef(rankdata(x), rankdata(y))[0, 1])

                if ic_vals:
                    mean_ic = np.mean(ic_vals)
                    sign = 1 if mean_ic >= 0 else -1
                    if sign == -1:
                        out_panels[fname] = -panel
                        log(f"    {fname}: flipped (train_RankIC={mean_ic:+.4f})")

    # Stack all factor panels to (stock, date) MultiIndex
    log("  Stacking to MultiIndex...")
    rows = []
    for fname, panel in out_panels.items():
        s = panel.stack(dropna=False)
        s.name = fname
        rows.append(s)

    result = pd.concat(rows, axis=1)
    result.index.names = ["date", "stock"]
    result = result.swaplevel().sort_index()
    result = result.dropna(how="all")
    log(f"  Final: {result.index.get_level_values('stock').nunique()} stocks, "
        f"{len(result):,} rows, {len(result.columns)} columns")

    result.to_parquet(DATA / "all_factors.parquet")
    log(f"  Saved all_factors.parquet")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — Single Factor Backtest
# ═══════════════════════════════════════════════════════════════════════════

def _safe_pearsonr(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    n = mask.sum()
    if n < 10:
        return np.nan
    r, _ = pearsonr(x[mask], y[mask])
    return r


def _daily_rankic(factor: pd.Series, fwd_ret: pd.Series) -> float:
    """Spearman RankIC between factor and forward return cross-section."""
    valid = factor.notna() & fwd_ret.notna()
    if valid.sum() < 30:
        return np.nan
    x = factor[valid].values
    y = fwd_ret[valid].values
    if x.std() < 1e-12 or y.std() < 1e-12:
        return np.nan
    from scipy.stats import rankdata
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def part5_backtest() -> None:
    """Single-factor backtest: RankIC for each factor and each horizon."""
    log("=" * 50)
    log("PART 5: Single Factor Backtest")

    factors = pd.read_parquet(DATA / "all_factors.parquet")
    labels = pd.read_parquet(DATA / "labels.parquet")
    log(f"  Factors: {factors.shape}, Labels: {labels.shape}")

    # Identify factor columns (non-label columns)
    factor_cols = [c for c in factors.columns
                   if not c.endswith("_rank") and not c.endswith("_zscore")]
    # Deduplicate: keep only the "raw" (zscore) versions
    base_cols = sorted(set(c.replace("_rank", "").replace("_zscore", "") for c in factor_cols))
    base_cols = [c for c in base_cols if c in factors.columns]

    # Label columns
    label_cols = [c for c in labels.columns if c.endswith("_raw_return")]
    horizon_tags = [c.replace("label_", "").replace("_raw_return", "") for c in label_cols]

    log(f"  Factors: {len(base_cols)}, Horizons: {len(horizon_tags)}")

    # Date splits
    train_start, train_end = "2016-01-01", "2021-12-31"
    valid_start, valid_end = "2022-01-01", "2023-12-31"
    test_start, test_end = "2024-01-01", "2026-12-31"

    all_rows = []
    t0 = time.time()

    for fi, fname in enumerate(base_cols):
        factor_series = factors[fname]
        factor_dates = factor_series.index.get_level_values("date")

        for hi, (lh, lc) in enumerate(zip(horizon_tags, label_cols)):
            label_series = labels[lc]
            tradable_col = lc.replace("_raw_return", "_tradable")
            tradable_series = labels[tradable_col] if tradable_col in labels.columns else None

            # Join factor + label + tradable
            combined = pd.DataFrame({
                "factor": factor_series,
                "label": label_series,
                "tradable": tradable_series if tradable_series is not None else True,
            }).dropna(subset=["factor", "label"])

            # Filter tradable
            tradable_mask = combined["tradable"] if tradable_series is not None else pd.Series(True, index=combined.index)
            combined = combined[tradable_mask & combined["factor"].notna() & combined["label"].notna()]

            if len(combined) < 1000:
                continue

            dates = combined.index.get_level_values("date")

            # Per-period metrics
            for period_name, p_start, p_end in [
                ("train", train_start, train_end),
                ("valid", valid_start, valid_end),
                ("test", test_start, test_end),
            ]:
                date_mask = (dates >= p_start) & (dates <= p_end)
                sub = combined[date_mask]
                if len(sub) < 100:
                    continue

                # Daily RankIC
                ic_vals = []
                n_days = 0
                for dt, day_df in sub.groupby(level="date"):
                    ic = _daily_rankic(day_df["factor"], day_df["label"])
                    if not np.isnan(ic):
                        ic_vals.append(ic)
                        n_days += 1

                if n_days < 5:
                    continue

                ic_arr = np.array(ic_vals)
                ic_mean = float(np.nanmean(ic_arr))
                ic_std = float(np.nanstd(ic_arr))
                icir = ic_mean / ic_std if ic_std > 1e-12 else 0.0
                win_rate = float(np.nanmean(ic_arr > 0))
                coverage = len(sub) / len(combined) if len(combined) > 0 else 0.0

                all_rows.append({
                    "factor": fname,
                    "horizon": lh,
                    "period": period_name,
                    "RankIC_mean": ic_mean,
                    "RankIC_std": ic_std,
                    "ICIR": icir,
                    "win_rate": win_rate,
                    "n_days": n_days,
                    "n_samples": len(sub),
                    "coverage": coverage,
                })

        if (fi + 1) % 10 == 0:
            rate = (fi + 1) / (time.time() - t0)
            log(f"  [{fi+1}/{len(base_cols)}] ({rate:.1f} factors/s)")

    if not all_rows:
        log("  WARNING: No results from backtest!")
        return

    report = pd.DataFrame(all_rows)
    report.to_csv(DATA / "single_factor_report.csv", index=False)
    log(f"  Saved single_factor_report.csv ({len(report):,} rows)")

    # Build pivot for summary: factor x horizon, best valid RankIC
    valid_report = report[report["period"] == "valid"].copy()
    if len(valid_report) > 0:
        pivot = valid_report.pivot_table(
            index="factor", columns="horizon", values="RankIC_mean"
        )
        pivot["avg_RankIC"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("avg_RankIC", ascending=False)
        pivot.to_csv(DATA / "single_factor_rankic_pivot.csv")
        log(f"  RankIC pivot saved ({len(pivot)} factors)")

        # Top 20
        print("\n" + "=" * 70)
        print("TOP 20 FACTORS (by avg valid RankIC)")
        print("=" * 70)
        for fname in pivot.index[:20]:
            row = pivot.loc[fname]
            avg = row["avg_RankIC"]
            details = "  ".join(f"{h}: {row[h]:+.4f}" for h in horizon_tags if h in row)
            print(f"  {fname:20s}  avg={avg:+.4f}  {details}")

    # --- Failure Reports ---
    # 1. Rejected factors (train ICIR < 0.1)
    train_report = report[report["period"] == "train"].copy()
    if len(train_report) > 0:
        rejected = train_report.groupby("factor")["ICIR"].mean().reset_index()
        rejected = rejected[rejected["ICIR"] < 0.1].sort_values("ICIR")
        rejected.to_csv(DATA / "rejected_factor_report.csv", index=False)
        log(f"  Rejected factors (train ICIR<0.1): {len(rejected)}")

    # 2. Low coverage factors (< 50%)
    if len(valid_report) > 0:
        low_cov = valid_report[valid_report["coverage"] < 0.5].copy()
        if len(low_cov) > 0:
            low_cov.to_csv(DATA / "low_coverage_factor_list.csv", index=False)
            log(f"  Low coverage factors (<50%): {len(low_cov)}")
        else:
            log("  No low-coverage factors detected")

    log(f"PART 5 done in {(time.time()-t0)/60:.1f}min")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run all 5 pipeline steps sequentially."""
    t_start = time.time()
    log("=" * 60)
    log("FACTOR MVP PIPELINE v2 START")
    log("=" * 60)

    # Validate input data
    required = ["raw_ohlcv.parquet", "adj_ohlcv.parquet", "calendar.parquet"]
    missing = [f for f in required if not (DATA / f).exists()]
    if missing:
        log(f"ERROR: Missing input files: {missing}")
        log("Run worker-1 pipeline first to generate input data.")
        sys.exit(1)

    # Step 1: adj_factor
    part1_adj_factor()

    # Step 2: Universe
    part2_universe()

    # Step 3: Labels
    labels = part3_labels()

    # Step 4: Factors (pass labels for direction adjustment)
    part4_factors(labels)

    # Step 5: Backtest
    part5_backtest()

    elapsed = time.time() - t_start
    log("=" * 60)
    log(f"PIPELINE COMPLETE in {elapsed/60:.1f}min")
    log("=" * 60)


if __name__ == "__main__":
    main()
