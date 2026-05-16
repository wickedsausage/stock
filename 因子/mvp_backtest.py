"""Factor computation + backtest for MVP."""
import pandas as pd, numpy as np, time
from scipy.stats import rankdata

DATA = 'C:/factor_data'

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Loading...")
adj = pd.read_parquet(f"{DATA}/adj_ohlcv.parquet")
labels = pd.read_parquet(f"{DATA}/labels.parquet")
uni = pd.read_parquet(f"{DATA}/universe.parquet")

close = adj['close'].unstack(level=0)
high = adj['high'].unstack(level=0)
low = adj['low'].unstack(level=0)
open_ = adj['open'].unstack(level=0)
vol = adj['volume'].unstack(level=0)
amt = adj['amount'].unstack(level=0)

for p in [close, high, low, open_]:
    p.ffill(limit=3, inplace=True)
vol = vol.fillna(0)
amt = amt.fillna(0)

log("Factors...")
ret = close.pct_change()
F = {}

# Momentum (6)
F['F081_Mom_1M'] = close.pct_change(20)
F['F082_Mom_3M'] = close.pct_change(60)
F['F083_Mom_6M'] = close.pct_change(120)
F['F084_12m1m'] = close.pct_change(252) - close.pct_change(20)
F['F092_Overnight'] = (open_ - close.shift(1)) / close.shift(1)
F['F093_Intraday'] = (close - open_) / open_

# Reversal (3)
F['F091_ShortRev'] = -close.pct_change(5)
F['F090_LongRev'] = -close.pct_change(120)
F['Rev20_smooth'] = -close.pct_change(20).rolling(20).mean()

# Volatility (6)
F['F100_Vol20'] = ret.rolling(20).std()
F['F101_Vol60'] = ret.rolling(60).std()
F['F106_DownVol'] = ret.clip(upper=0).rolling(20).std()
F['F102_Parkinson'] = np.sqrt((np.log(high/low)**2/(4*np.log(2))).rolling(20).mean())
mkt_ret = ret.mean(axis=1)
F['F105_IdioVol'] = ret.sub(mkt_ret, axis=0).rolling(60).std()
F['F097_Beta'] = ret.rolling(60).cov(mkt_ret).div(mkt_ret.rolling(60).var(), axis=0)

# Liquidity (6)
F['F117_Turn20'] = vol.rolling(20).mean()
F['F118_Turn60'] = vol.rolling(60).mean()
F['F119_AbnTurn'] = vol.rolling(20).mean() / vol.rolling(120).mean().replace(0,np.nan) - 1
F['F120_Amihud'] = (ret.abs() / amt.replace(0,np.nan)).rolling(20).mean()
F['F127_VolShock'] = vol / vol.rolling(20).mean().replace(0,np.nan) - 1
F['F128_VolSurp'] = (vol - vol.rolling(120).mean()) / vol.rolling(120).std()

# Technical (7)
F['MA5'] = close / close.rolling(5).mean()
F['MA20'] = close / close.rolling(20).mean()
F['MA60'] = close / close.rolling(60).mean()
F['STD20'] = close.rolling(20).std() / close
F['RSV'] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min() + 1e-10)
F['RSI14'] = 100 - 100/(1 + ret.clip(lower=0).rolling(14).mean()/(-ret.clip(upper=0)).rolling(14).mean().replace(0,1))
F['Skew20'] = ret.rolling(20).skew()

# Candle (5)
F['KMID'] = (close - open_) / open_
F['KLEN'] = (high - low) / open_
F['KUP'] = (high - close.clip(lower=open_)) / open_
F['KLOW'] = (close.clip(upper=open_) - low) / open_
F['KSFT'] = (2*close - high - low) / open_

# CS rank versions
log(f"  {len(F)} base factors, computing CS ranks...")
for name in list(F.keys()):
    F[f'{name}_cs'] = F[name].rank(axis=1, pct=True)

log(f"  {len(F)} total factors")

# Preprocessing (per-date cross-sectional)
log("Preprocessing...")
for name in list(F.keys()):
    fac = F[name]
    # winsorize per row (date): clip at 0.5% / 99.5%
    q_lo = fac.quantile(0.005, axis=1)
    q_hi = fac.quantile(0.995, axis=1)
    fac = fac.clip(q_lo, axis=0).clip(upper=q_hi, axis=0)
    # fillna with cross-sectional median per date
    med = fac.median(axis=1)
    fac = fac.T.fillna(med).T
    # zscore per date (cross-sectional)
    mu = fac.mean(axis=1)
    sd = fac.std(axis=1).replace(0, np.nan)
    fac = fac.sub(mu, axis=0).div(sd, axis=0)
    F[name] = fac

# Stack to MultiIndex
log("Stacking and saving...")
stacked = {}
for name, fac in F.items():
    s = fac.stack(dropna=False)
    s.name = name
    stacked[name] = s
result = pd.DataFrame(stacked)
result.index.names = ['date', 'stock']
result = result.swaplevel().sort_index()
result = result.dropna(how='all')
result.to_parquet(f"{DATA}/all_factors.parquet")
log(f"Saved: {len(result):,} rows, {len(result.columns)} factors")
del result, stacked, F, close, high, low, open_, vol, amt, ret, mkt_ret
import gc; gc.collect()

# === BACKTEST ===
log("Backtest...")
result = pd.read_parquet(f"{DATA}/all_factors.parquet")
# Align factors with labels
common_idx = result.index.intersection(labels.index)
result = result.loc[common_idx]
labels = labels.loc[common_idx]
fnames = list(result.columns)

label20 = labels['H20_excess']
label20_tr = labels['H20_tradable']
date_idx = result.index.get_level_values('date')
all_dates = sorted(date_idx.unique())
n = len(all_dates)
train_dates = set(all_dates[:int(n*0.6)])
valid_dates = set(all_dates[int(n*0.6):int(n*0.8)])
test_dates = set(all_dates[int(n*0.8):])
log(f"  Train={len(train_dates)}d Valid={len(valid_dates)}d Test={len(test_dates)}d")

reports = []
t0 = time.time()

for fi, fn in enumerate(fnames):
    fac = result[fn].values
    lab = label20.values
    trd = label20_tr.values
    dts = date_idx.values

    vmask = trd & ~np.isnan(fac) & ~np.isnan(lab)

    rankic_vals = []
    period_tags = []
    for d in all_dates:
        dm = (dts == d) & vmask
        if dm.sum() < 30: continue
        x = rankdata(fac[dm])
        y = rankdata(lab[dm])
        if np.std(x) < 1e-12 or np.std(y) < 1e-12: continue
        rankic_vals.append(np.corrcoef(x, y)[0,1])
        period_tags.append('train' if d in train_dates else ('valid' if d in valid_dates else 'test'))

    rankic = np.array(rankic_vals)
    tags = np.array(period_tags)

    train_mask = tags == 'train'
    if train_mask.sum() < 10: continue
    train_ric = rankic[train_mask].mean()
    direction = 1 if train_ric > 0 else -1

    row = {'factor': fn, 'direction': direction, 'coverage': vmask.mean()}
    for period in ['train', 'valid', 'test']:
        pm = tags == period
        if pm.sum() < 10: continue
        rp = rankic[pm] * direction
        row[f'{period}_RankIC'] = rp.mean()
        row[f'{period}_ICIR'] = rp.mean()/rp.std() if rp.std()>0 else 0
        row[f'{period}_win'] = (rp>0).mean()
    reports.append(row)

    if (fi+1) % 20 == 0:
        elapsed = time.time() - t0
        log(f"  {fi+1}/{len(fnames)} ({elapsed:.0f}s, ~{elapsed/(fi+1)*len(fnames):.0f}s total)")

rep = pd.DataFrame(reports)
rep = rep.sort_values('valid_RankIC', ascending=False)
rep.to_csv(f"{DATA}/single_factor_report.csv", index=False)
rep[rep['train_ICIR'].abs() < 0.1].to_csv(f"{DATA}/rejected_factor_report.csv", index=False)
rep[rep['coverage'] < 0.5].to_csv(f"{DATA}/low_coverage_factor_list.csv", index=False)

print(f"\n=== TOP 20 ({len(rep)} factors) ===")
for _, r in rep.head(20).iterrows():
    print(f"  {r['factor']:25s} train={r.get('train_RankIC',0):+.4f} valid={r.get('valid_RankIC',0):+.4f} test={r.get('test_RankIC',0):+.4f}  IR={r.get('valid_ICIR',0):+.2f}")

log(f"Done. {len(rep)} factors reported")
