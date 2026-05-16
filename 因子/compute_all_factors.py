"""
Compute ALL factors from OHLCV + financial data.
~140+ factors across: Value, Quality, Momentum, Risk, Liquidity, Technical.

Output: C:/factor_data/all_factors.parquet
"""

import pandas as pd
import numpy as np
from scipy import stats
import os, gc
from datetime import datetime

DATA_DIR = "C:/因子数据"
os.makedirs(DATA_DIR, exist_ok=True)


def cs_zscore(s: pd.Series) -> pd.Series:
    return s.groupby("date", group_keys=False).apply(
        lambda x: (x - x.mean()) / x.std(ddof=0)
    )


def cs_rank_pct(s: pd.Series) -> pd.Series:
    return s.groupby("date", group_keys=False).rank(pct=True)


def winsorize(s: pd.Series, lo: float = 0.005, hi: float = 0.995) -> pd.Series:
    q_lo, q_hi = s.quantile(lo), s.quantile(hi)
    return s.clip(q_lo, q_hi)


def load_ohlcv():
    print("Loading OHLCV...")
    return pd.read_parquet(f"{DATA_DIR}/ohlcv.parquet")


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("stock")
    df["ret"] = g["close"].pct_change()
    df["logret"] = np.log(1 + df["ret"].clip(-0.3, 0.3))
    df["vwap"] = (df["amount"] / df["volume"].replace(0, np.nan))
    df["vwap"] = df["vwap"].replace([np.inf, -np.inf], np.nan)
    df["mkt_ret"] = df.groupby("date")["ret"].transform("mean")
    return df


# ══════════════════════════════════════════════════════════════════════════
# FACTOR GROUP: VALUE (F001-F020)
# ══════════════════════════════════════════════════════════════════════════

def group_value(df: pd.DataFrame) -> pd.DataFrame:
    fac = pd.DataFrame(index=df.index)
    fac["F001_BM"] = 1.0 / df["pbMRQ"].replace(0, np.nan)
    fac["F002_EP"] = (1.0 / df["peTTM"].replace(0, np.nan)).clip(-0.5, 0.5)
    return fac


# ══════════════════════════════════════════════════════════════════════════
# FACTOR GROUP: MOMENTUM (F081-F096)
# ══════════════════════════════════════════════════════════════════════════

def group_momentum(df: pd.DataFrame) -> pd.DataFrame:
    fac = pd.DataFrame(index=df.index)
    g = df.groupby("stock")
    r = df["ret"]

    r1 = g["ret"].transform(lambda x: (1 + x).rolling(21).prod() - 1)
    r3 = g["ret"].transform(lambda x: (1 + x).rolling(63).prod() - 1)
    r6 = g["ret"].transform(lambda x: (1 + x).rolling(126).prod() - 1)
    r12 = g["ret"].transform(lambda x: (1 + x).rolling(252).prod() - 1)

    fac["F081_Momentum_12m1m"] = r12 - r1
    fac["F082_Momentum_6M"] = r6
    fac["F083_Momentum_3M"] = r3
    fac["F084_Momentum_1M"] = r1
    fac["F090_Long_Reversal"] = -(r12 - r1)
    fac["F091_Short_Reversal"] = -r1
    fac["F092_Overnight_Mom"] = g["ret"].transform(lambda x: x.rolling(21).mean())

    # 52-week high/low
    h52 = g["high"].transform(lambda x: x.rolling(252).max())
    l52 = g["low"].transform(lambda x: x.rolling(252).min())
    fac["F088_52WH"] = df["close"] / h52.replace(0, np.nan)
    fac["F089_52WL"] = df["close"] / l52.replace(0, np.nan)

    # Intraday momentum
    fac["F093_Intraday_Mom"] = (df["high"] - df["low"]) / df["close"]

    return fac


# ══════════════════════════════════════════════════════════════════════════
# FACTOR GROUP: RISK / VOLATILITY (F097-F116)
# ══════════════════════════════════════════════════════════════════════════

def group_risk(df: pd.DataFrame) -> pd.DataFrame:
    fac = pd.DataFrame(index=df.index)
    g = df.groupby("stock")

    # Volatility
    fac["F100_Vol_20"] = g["ret"].transform(lambda x: x.rolling(20).std())
    fac["F101_Vol_60"] = g["ret"].transform(lambda x: x.rolling(60).std())

    # Parkinson Vol
    hp = np.log(df["high"] / df["low"]).clip(lower=1e-10) ** 2 / (4 * np.log(2))
    fac["F102_Parkinson_Vol"] = np.sqrt(
        g[hp].transform(lambda x: x.rolling(20).mean())
    )

    # Garman-Klass Vol
    hl2 = np.log(df["high"] / df["low"]).clip(lower=1e-10) ** 2
    co2 = np.log(df["close"] / df["open"]).clip(lower=1e-10) ** 2
    gk = 0.5 * hl2 - (2 * np.log(2) - 1) * co2
    fac["F103_GK_Vol"] = np.sqrt(g[gk].transform(lambda x: x.rolling(20).mean()))

    # ATR14
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - g["close"].shift(1)).abs(),
        (df["low"] - g["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = g[tr].transform(lambda x: x.rolling(14).mean())
    fac["F104_ATR14"] = atr14 / df["close"].replace(0, np.nan)

    # Downside semivariance
    neg = df["ret"].clip(upper=0)
    fac["F106_Downside_SemiVar"] = g[neg].transform(
        lambda x: (x ** 2).rolling(20).mean()
    )

    # Upside vol
    pos = df["ret"].clip(lower=0)
    fac["F107_Upside_Vol"] = g[pos].transform(lambda x: x.rolling(20).std())

    # Skew / Kurtosis
    fac["F108_Skew_20"] = g["ret"].transform(lambda x: x.rolling(20).skew())
    fac["F109_Kurtosis_20"] = g["ret"].transform(lambda x: x.rolling(20).kurt())

    # MAX effect
    fac["F112_MAX_20"] = g["ret"].transform(lambda x: x.rolling(20).max())

    # Beta 60
    def _beta(gb):
        r, m = gb["ret"].values, gb["mkt_ret"].values
        out = np.full(len(r), np.nan)
        for i in range(60, len(r)):
            sr, sm = r[i-60:i], m[i-60:i]
            if np.std(sm) > 1e-10:
                out[i] = np.cov(sr, sm)[0, 1] / np.var(sm)
        return pd.Series(out, index=gb.index)
    fac["F097_Beta"] = df.groupby("stock", group_keys=False).apply(_beta)

    # Idio vol
    def _idio(gb):
        r, m = gb["ret"].values, gb["mkt_ret"].values
        out = np.full(len(r), np.nan)
        for i in range(60, len(r)):
            sr, sm = r[i-60:i], m[i-60:i]
            if np.std(sm) > 1e-10 and np.var(sr) > 1e-15:
                slope, intercept = stats.linregress(sm, sr)[:2]
                out[i] = np.std(sr - (intercept + slope * sm))
        return pd.Series(out, index=gb.index)
    fac["F105_Idio_Vol"] = df.groupby("stock", group_keys=False).apply(_idio)

    # Down beta
    df_d = df.copy()
    df_d["mkt_neg"] = df_d["mkt_ret"].clip(upper=0)
    def _down_beta(gb):
        r, m = gb["ret"].values, gb["mkt_neg"].values
        out = np.full(len(r), np.nan)
        for i in range(60, len(r)):
            sr, sm = r[i-60:i], m[i-60:i]
            if np.std(sm) > 1e-10:
                out[i] = np.cov(sr, sm)[0, 1] / np.var(sm)
        return pd.Series(out, index=gb.index)
    fac["F098_Down_Beta"] = df_d.groupby("stock", group_keys=False).apply(_down_beta)

    # Coskewness
    def _coskew(gb):
        r, m = gb["ret"].values, gb["mkt_ret"].values
        out = np.full(len(r), np.nan)
        for i in range(60, len(r)):
            sr, sm = r[i-60:i], m[i-60:i]
            sr_s, sm_s = np.std(sr), np.std(sm)
            if sr_s > 1e-10 and sm_s > 1e-10:
                out[i] = (np.mean((sr - np.mean(sr)) * (sm - np.mean(sm))**2)
                          / (sr_s * sm_s**2))
        return pd.Series(out, index=gb.index)
    fac["F110_Coskew"] = df.groupby("stock", group_keys=False).apply(_coskew)

    # CVaR 95%
    def _cvar(gb):
        v = gb.values
        out = np.full(len(v), np.nan)
        for i in range(60, len(v)):
            w = v[i-60:i]
            th = np.percentile(w, 5)
            out[i] = w[w <= th].mean()
        return pd.Series(out, index=gb.index)
    fac["F113_CVaR_95"] = g["ret"].transform(_cvar)

    return fac


# ══════════════════════════════════════════════════════════════════════════
# FACTOR GROUP: LIQUIDITY (F117-F136)
# ══════════════════════════════════════════════════════════════════════════

def group_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    fac = pd.DataFrame(index=df.index)
    g = df.groupby("stock")
    to = df["turnover"] / 100.0

    fac["F117_TO_1M"] = g[to].transform(lambda x: x.rolling(21).mean())
    fac["F118_TO_3M"] = g[to].transform(lambda x: x.rolling(63).mean())
    fac["F119_Abnormal_TO"] = fac["F117_TO_1M"] - fac["F118_TO_3M"]

    illiq = df["ret"].abs() / df["amount"].replace(0, np.nan)
    illiq = illiq.replace([np.inf, -np.inf], np.nan)
    fac["F120_Amihud_20"] = g[illiq].transform(lambda x: x.rolling(20).mean())
    fac["F121_Amihud_60"] = g[illiq].transform(lambda x: x.rolling(60).mean())

    fac["F123_Avg_Amount_20"] = g["amount"].transform(lambda x: x.rolling(20).mean())
    fac["F124_Log_Price"] = np.log(df["close"].clip(lower=0.01))

    hl = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    fac["F125_HL_Spread"] = g[hl].transform(lambda x: x.rolling(20).mean())

    zt = (df["volume"] < 1000).astype(float)
    fac["F126_Zero_Trade_20"] = g[zt].transform(lambda x: x.rolling(20).mean())

    vol20 = g["volume"].transform(lambda x: x.rolling(20).mean())
    fac["F127_Volume_Shock"] = (df["volume"] / vol20.replace(0, np.nan)).clip(0, 10)
    fac["F129_VWAP_Deviation"] = (df["close"] / df["vwap"] - 1).clip(-0.1, 0.1)

    fac["F130_TO_Vol"] = g[to].transform(lambda x: x.rolling(20).std())

    fac["F132_ADV_20"] = g["volume"].transform(lambda x: x.rolling(20).mean())
    fac["F133_ADV_60"] = g["volume"].transform(lambda x: x.rolling(60).mean())

    return fac


# ══════════════════════════════════════════════════════════════════════════
# FACTOR GROUP: TECHNICAL (F157-F188)
# ══════════════════════════════════════════════════════════════════════════

def group_technical(df: pd.DataFrame) -> pd.DataFrame:
    fac = pd.DataFrame(index=df.index)
    g = df.groupby("stock")

    # MA deviations
    for p, fid in [(5, "F171_MA5"), (10, "F172_MA10"),
                    (20, "F173_MA20"), (60, "F174_MA60")]:
        ma = g["close"].transform(lambda x: x.rolling(p).mean())
        fac[fid] = (df["close"] / ma.replace(0, np.nan) - 1).clip(-0.3, 0.3)

    # RSV
    l9 = g["low"].transform(lambda x: x.rolling(9).min())
    h9 = g["high"].transform(lambda x: x.rolling(9).max())
    rng = (h9 - l9).replace(0, np.nan)
    fac["F180_RSV"] = ((df["close"] - l9) / rng).clip(0, 1)

    # Volume MA ratios
    for p, fid in [(5, "F187_Vol_MA5"), (20, "F188_Vol_MA20")]:
        vma = g["volume"].transform(lambda x: x.rolling(p).mean())
        fac[fid] = (df["volume"] / vma.replace(0, np.nan)).clip(0, 10)

    # Candle patterns
    fac["F157_Bullish_Engulf"] = (
        (g["close"].shift(1) < g["open"].shift(1)) &
        (df["close"] > df["open"]) &
        (df["open"] < g["close"].shift(1)) &
        (df["close"] > g["open"].shift(1))
    ).astype(float)

    fac["F158_Bearish_Engulf"] = (
        (g["close"].shift(1) > g["open"].shift(1)) &
        (df["close"] < df["open"]) &
        (df["open"] > g["close"].shift(1)) &
        (df["close"] < g["open"].shift(1))
    ).astype(float)

    body = (df["close"] - df["open"]).abs()
    hl_range = df["high"] - df["low"]
    fac["F159_Doji"] = ((body / hl_range.replace(0, np.nan)) < 0.1).astype(float)

    return fac


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    print(f"Start: {t0}")

    df = load_ohlcv()
    print(f"  Raw OHLCV: {len(df):,} rows, "
          f"{df.index.get_level_values('stock').nunique()} stocks")

    df = add_returns(df)
    df = df.reset_index()

    result = pd.DataFrame(index=pd.MultiIndex.from_frame(df[["stock", "date"]]))

    groups = [
        ("Value", group_value),
        ("Momentum", group_momentum),
        ("Risk", group_risk),
        ("Liquidity", group_liquidity),
        ("Technical", group_technical),
    ]

    for name, fn in groups:
        sub = 60
        try:
            fac = fn(df)
            result = result.join(fac)
            n = len([c for c in fac.columns])
            cov = fac.notna().mean().mean()
            print(f"  {name}: {n} factors, cov {cov:.1%}")
            sub = 60
        except Exception as e:
            print(f"  [ERR] {name}: {e}")

    # Normalize
    print("Normalizing...")
    cols = list(result.columns)
    for col in cols:
        s = result[col]
        s_clean = winsorize(s)
        result[f"{col}_zscore"] = cs_zscore(s_clean)
        result[f"{col}_rank"] = cs_rank_pct(s_clean)

    result = result.sort_index()
    result.to_parquet(f"{DATA_DIR}/all_factors.parquet")

    # Summary
    n_denom = len(cols)
    coverage = result[cols].notna().mean()
    print(f"\n{'='*60}")
    print(f"COMPLETE: {n_denom} raw factors")
    print(f"  Shape: {result.shape[0]:,} x {result.shape[1]}")
    print(f"  Stocks: {result.index.get_level_values('stock').nunique()}")
    print(f"  Cov mean: {coverage.mean():.1%}, range: {coverage.min():.1%}~{coverage.max():.1%}")

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Saved: {DATA_DIR}/all_factors.parquet")


if __name__ == "__main__":
    main()
