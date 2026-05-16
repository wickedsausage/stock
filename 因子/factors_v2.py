"""Robust factor computation — 50+ factors from OHLCV only."""
import pandas as pd
import numpy as np
import time

DATA = "C:/factor_data"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Loading OHLCV...")
df = pd.read_parquet(f"{DATA}/ohlcv.parquet")
log(f"  {df.index.get_level_values('stock').nunique()} stocks, {len(df):,} rows")

# Unstack to (date, stock) panel for vectorized ops
log("Unstacking...")
close = df["close"].unstack()   # index=date, columns=stock
high  = df["high"].unstack()
low   = df["low"].unstack()
open_ = df["open"].unstack()
vol   = df["volume"].unstack()
amt   = df["amount"].unstack()

log(f"  Panel: {len(close)} dates x {len(close.columns)} stocks")

# Forward fill for suspension gaps (max 5 days)
close = close.ffill(limit=5)
high = high.ffill(limit=5)
low = low.ffill(limit=5)
open_ = open_.ffill(limit=5)
vol = vol.fillna(0)
amt = amt.fillna(0)

ret = close.pct_change()
ret20 = close.pct_change(20)
ret60 = close.pct_change(60)
ret120 = close.pct_change(120)

# ====== Factor Computation ======
log("Computing factors...")
F = {}  # factor_name -> DataFrame (date x stock)
t0 = time.time()

# --- MOMENTUM (F081-F096) ---
F["F081_Mom_1M"] = ret20
F["F082_Mom_3M"] = ret60
F["F083_Mom_6M"] = ret120
F["F084_Mom_12m1m"] = close.pct_change(252) - ret20  # 12-1 momentum proxy
F["F090_LongRev"] = -ret120
F["F091_ShortRev"] = -close.pct_change(5)

# 52-week high/low
h52 = high.rolling(252).max()
F["F088_52WH"] = close / h52
F["F089_52WH_gap"] = close / h52 - 1

# Intraday/overnight
F["F092_Overnight"] = (open_ - close.shift(1)) / close.shift(1)
F["F093_Intraday"] = (close - open_) / open_

# Path-adjusted momentum
daily_std = ret.rolling(20).std()
F["F085_PathAdjMom"] = ret60 / daily_std.replace(0, np.nan)

# Price delay (simplified: R2 ratio)
def price_delay(x, n=60):
    r = x.pct_change()
    r1 = r.shift(1)
    r2 = r.shift(2)
    r3 = r.shift(3)
    # R2 from lagged returns vs current
    beta1 = r1.rolling(n).cov(r) / r1.rolling(n).var()
    beta12 = r12 = None
    # Simplified: correlation decay
    ac1 = r.rolling(n).apply(lambda y: y.autocorr(lag=1) if len(y)>2 else np.nan, raw=False)
    ac5 = r.rolling(n).apply(lambda y: y.autocorr(lag=5) if len(y)>5 else np.nan, raw=False)
    return 1 - (ac5.abs() / ac1.abs().replace(0, np.nan))
# Skip price delay for now (slow)

# --- REVERSAL ---
F["F091_Rev_5D"] = -ret.pct_change(5)
rev_1m_mean = -ret20.rolling(20).mean()
F["Rev_20D_smooth"] = rev_1m_mean

# --- VOLATILITY / RISK (F097-F116) ---
F["F100_Vol_20D"] = ret.rolling(20).std()
F["F101_Vol_60D"] = ret.rolling(60).std()
F["F106_DownsideVol"] = ret.clip(upper=0).rolling(20).std()
F["F107_UpsideVol"] = ret.clip(lower=0).rolling(20).std()
F["F108_Skew"] = ret.rolling(20).skew()
F["F109_Kurt"] = ret.rolling(20).kurt()
F["F112_MAX"] = ret.rolling(20).max()

# Parkinson vol
park = np.sqrt((np.log(high/low)**2 / (4*np.log(2))).rolling(20).mean())
F["F102_Parkinson"] = park

# ATR
tr1 = high - low
tr2 = (high - close.shift(1)).abs()
tr3 = (low - close.shift(1)).abs()
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
F["F104_ATR"] = tr.rolling(14).mean() / close

# Idio vol (market-hedged)
mkt_ret = ret.mean(axis=1)
idio_std = ret.sub(mkt_ret, axis=0).rolling(60).std()
F["F105_IdioVol"] = idio_std

# Beta (60-day rolling)
cov_rm = ret.rolling(60).cov(mkt_ret)
var_m = mkt_ret.rolling(60).var()
F["F097_Beta"] = cov_rm.div(var_m, axis=0)

# --- LIQUIDITY (F117-F136) ---
F["F117_Turnover"] = vol.rolling(20).mean()
F["F118_Turnover_60"] = vol.rolling(60).mean()
F["F119_AbnTurnover"] = vol.rolling(20).mean() / vol.rolling(120).mean() - 1
F["F123_AvgAmt"] = amt.rolling(20).mean()
F["F127_VolShock"] = vol / vol.rolling(20).mean() - 1
F["F128_VolSurprise"] = (vol - vol.rolling(120).mean()) / vol.rolling(120).std()

# Amihud illiquidity
illiq = (ret.abs() / amt.replace(0, np.nan)).rolling(20).mean()
F["F120_Amihud"] = illiq

# --- TECHNICAL (selected from TEC group) ---
F["F171_MA5"] = close / close.rolling(5).mean()
F["F172_MA20"] = close / close.rolling(20).mean()
F["F173_MA60"] = close / close.rolling(60).mean()
F["F173_STD20"] = close.rolling(20).std() / close

# RSV (Stochastic)
l20 = low.rolling(20).min()
h20 = high.rolling(20).max()
F["F180_RSV"] = (close - l20) / (h20 - l20 + 1e-10)

# RSI
delta = close.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
F["RSI_14"] = 100 - 100/(1+rs)

# Candle patterns
F["F157_KMID"] = (close - open_) / open_
F["F158_KLEN"] = (high - low) / open_
F["F160_KUP"] = (high - close.clip(lower=open_)) / open_
F["F162_KLOW"] = (close.clip(upper=open_) - low) / open_
F["F164_KSFT"] = (2*close - high - low) / open_

# Volume MA
F["F187_VMA20"] = vol.rolling(20).mean() / (vol + 1)

# --- CROSS-SECTIONAL versions ---
log("  Cross-sectional ranking...")
cs_factors = {}
for name, fac in F.items():
    cs_factors[f"{name}_CS"] = fac.rank(axis=1, pct=True)
F.update(cs_factors)

# --- COMPOSITE / INTERACTION factors ---
log("  Composite factors...")
F["Comp_Mom_LowVol"] = F["F081_Mom_1M_CS"] * (1 - F["F100_Vol_20D_CS"])
F["Comp_Rev_Liq"] = F["F091_ShortRev_CS"] * F["F117_Turnover_CS"]
F["Comp_Mom_52WH"] = F["F081_Mom_1M_CS"] * F["F088_52WH_CS"]
F["Comp_Vol_ATR"] = F["F100_Vol_20D_CS"] * F["F104_ATR_CS"]
F["Comp_MA_RSI"] = (F["F171_MA5_CS"] - 0.5) * (F["RSI_14_CS"] - 0.5)
F["Comp_Amihud_Skew"] = F["F120_Amihud_CS"] * F["F108_Skew_CS"]
F["Comp_MomAccel"] = (F["F081_Mom_1M_CS"] - F["F082_Mom_3M_CS"])  # momentum acceleration
F["Comp_VolRegime"] = F["F101_Vol_60D_CS"] - F["F100_Vol_20D_CS"]  # vol contraction

log(f"  Total factors computed: {len(F)}")
log(f"  Time: {(time.time()-t0)/60:.1f}min")

# ====== Stack back to MultiIndex ======
log("Stacking to MultiIndex...")
rows = []
for name, fac in F.items():
    s = fac.stack(dropna=False)
    s.name = name
    rows.append(s)
result = pd.concat(rows, axis=1)
result.index.names = ["date", "stock"]
result = result.swaplevel().sort_index()  # (stock, date) MultiIndex

# Remove rows where ALL factors are NaN
result = result.dropna(how="all")
log(f"  Final: {result.index.get_level_values('stock').nunique()} stocks, {len(result):,} rows, {len(result.columns)} factors")

# Save
result.to_parquet(f"{DATA}/all_factors.parquet")
log(f"Saved all_factors.parquet ({len(result.columns)} factors)")
log("=== FACTORS DONE ===")
