"""Single-factor backtest: IC, RankIC, group returns for all factors."""
import pandas as pd, numpy as np, time
from pathlib import Path

DATA = Path("C:/factor_data")
OUT = DATA / "group_returns"
OUT.mkdir(exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Loading factors...")
af = pd.read_parquet(DATA / "all_factors.parquet")
log(f"  {len(af):,} rows, {len(af.columns)} factors")

# Fix: index names are swapped — level 0 is actually date, level 1 is stock
af.index = af.index.set_names("date", level=0)
af.index = af.index.set_names("stock", level=1)
# Now swap to (stock, date) for groupby operations
af = af.swaplevel().sort_index()  # → (stock, date)

# Load close prices for forward returns
log("Loading close prices...")
ohlcv = pd.read_parquet(DATA / "ohlcv.parquet")
close = ohlcv["close"]

# Compute forward 20-day return
fwd_ret = close.groupby("stock").pct_change(20).shift(-20)
fwd_ret.name = "fwd_ret"

# Align with factors
log("Aligning data...")
common_idx = af.index.intersection(fwd_ret.index)
af = af.loc[common_idx]
fwd_ret = fwd_ret.loc[common_idx]

# Train/valid split
all_dates = sorted(af.index.get_level_values("date").unique())
# Use simple 80/20 split (dates already in order)
split_idx = int(len(all_dates) * 0.8)
train_dates = all_dates[:split_idx]
valid_dates = all_dates[split_idx:]
log(f"Train: {len(train_dates)} days, Valid: {len(valid_dates)} days")

results = []
t0 = time.time()

for fi, factor_name in enumerate(af.columns):
    fac = af[factor_name]

    # Align factor with forward return
    common = fac.dropna().index.intersection(fwd_ret.dropna().index)
    if len(common) < 1000:
        continue

    fac_aligned = fac.loc[common]
    fwd_aligned = fwd_ret.loc[common]

    # Split
    train_mask = fac_aligned.index.get_level_values("date").isin(train_dates)
    valid_mask = fac_aligned.index.get_level_values("date").isin(valid_dates)

    metrics = {"factor": factor_name, "n_obs": len(common)}

    for period, mask in [("train", train_mask), ("valid", valid_mask)]:
        if not mask.any():
            continue
        f = fac_aligned[mask]
        fw = fwd_aligned[mask]

        # Cross-sectional IC — fast per-date loop
        dates = f.index.get_level_values("date")
        unique_dates = dates.unique()
        ic_vals, rankic_vals = [], []
        for d in unique_dates:
            mask = dates == d
            f_d = f[mask].dropna()
            fw_d = fw[mask].dropna()
            common = f_d.index.intersection(fw_d.index)
            if len(common) < 30:
                continue
            x = f_d.loc[common].values
            y = fw_d.loc[common].values
            if x.std() > 1e-12 and y.std() > 1e-12:
                ic_vals.append(np.corrcoef(x, y)[0, 1])
                # Spearman via rank
                from scipy.stats import rankdata
                rankic_vals.append(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
        ic_series = pd.Series(ic_vals, dtype=float)
        rankic_series = pd.Series(rankic_vals, dtype=float)

        if len(ic_series) > 0:
            metrics[f"{period}_IC_mean"] = ic_series.mean()
            metrics[f"{period}_IC_std"] = ic_series.std()
            metrics[f"{period}_ICIR"] = ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0
            metrics[f"{period}_IC_win"] = (ic_series > 0).mean()
            metrics[f"{period}_IC_N"] = len(ic_series)

        if len(rankic_series) > 0:
            metrics[f"{period}_RankIC_mean"] = rankic_series.mean()
            metrics[f"{period}_RankIC_std"] = rankic_series.std()
            metrics[f"{period}_RankICIR"] = rankic_series.mean() / rankic_series.std() if rankic_series.std() > 0 else 0
            metrics[f"{period}_RankIC_win"] = (rankic_series > 0).mean()

    results.append(metrics)

    if (fi + 1) % 10 == 0:
        elapsed = time.time() - t0
        rate = (fi + 1) / elapsed
        log(f"  {fi+1}/{len(af.columns)} ({rate:.0f} factors/s)")

# Save report
report = pd.DataFrame(results)
if len(report) > 0 and "valid_RankIC_mean" in report.columns:
    report = report.sort_values("valid_RankIC_mean", ascending=False)
report.to_csv(DATA / "single_factor_report.csv", index=False)
log(f"Report saved: {len(report)} factors")

# Top 10
print("\n=== TOP 10 FACTORS (by valid RankIC) ===")
for _, row in report.head(10).iterrows():
    print(f"  {row['factor']:25s}  RankIC={row.get('valid_RankIC_mean',0):+.4f}  "
          f"ICIR={row.get('valid_RankICIR',0):+.2f}  "
          f"Win={row.get('valid_RankIC_win',0):.1%}")

log(f"Done in {(time.time()-t0)/60:.1f}min")
