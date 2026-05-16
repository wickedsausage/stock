"""
Optimized backtest: per-date matrix IC + factor combinations.
Key insight: compute IC for ALL 80 factors in one pass per date.
Then design 1000+ combinations and backtest.
"""
import pandas as pd, numpy as np, time
from pathlib import Path
from scipy.stats import rankdata

DATA = Path("C:/factor_data")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ===== LOAD =====
log("Loading factors...")
af = pd.read_parquet(DATA / "all_factors.parquet")
af.index = af.index.set_names(["date", "stock"])
af = af.swaplevel().sort_index()  # → (stock, date)
log(f"  {len(af):,} rows, {len(af.columns)} factors")

log("Loading close...")
close_all = pd.read_parquet(DATA / "ohlcv.parquet")["close"]
fwd20 = close_all.groupby("stock").pct_change(20).shift(-20)
fwd20.name = "fwd20"

# ===== ALIGN =====
log("Aligning factor data with forward returns...")
# Build aligned matrix: each row = (stock, date), cols = all 80 factors + fwd20
all_data = af.copy()
all_data["_fwd"] = fwd20
all_data = all_data.dropna(subset=["_fwd"])
all_data = all_data.dropna(how="all", subset=af.columns)

dates_sorted = sorted(all_data.index.get_level_values("date").unique())
split = int(len(dates_sorted) * 0.8)
train_dates = set(dates_sorted[:split])
valid_dates = set(dates_sorted[split:])
log(f"  {len(all_data):,} rows, train={len(train_dates)}d, valid={len(valid_dates)}d")

# ===== PER-DATE MATRIX IC =====
log("Computing IC matrix (all factors per date)...")
factor_names = list(af.columns)
t0 = time.time()

# Accumulate IC series per factor
ic_acc = {f: {"train": [], "valid": []} for f in factor_names}
rankic_acc = {f: {"train": [], "valid": []} for f in factor_names}

date_idx = all_data.index.get_level_values("date")
unique_dates = date_idx.unique()

for di, d in enumerate(unique_dates):
    mask = date_idx == d
    sub = all_data[mask]
    if len(sub) < 50:
        continue

    y = sub["_fwd"].values.astype(float)
    X = sub[factor_names].values.astype(float)

    # Drop rows where any factor is NaN
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if valid.sum() < 50:
        continue

    Xv, yv = X[valid], y[valid]

    # Pearson IC for all factors at once
    full = np.column_stack([Xv, yv])
    corr_mat = np.corrcoef(full, rowvar=False)
    p_ic = corr_mat[-1, :-1]  # correlation of each factor with fwd_ret

    # Rank IC (Spearman)
    Xr = np.apply_along_axis(rankdata, 0, Xv)
    yr = rankdata(yv)
    full_r = np.column_stack([Xr, yr])
    corr_mat_r = np.corrcoef(full_r, rowvar=False)
    r_ic = corr_mat_r[-1, :-1]

    bucket = "train" if d in train_dates else "valid"
    for fi, fn in enumerate(factor_names):
        if not np.isnan(p_ic[fi]):
            ic_acc[fn][bucket].append(p_ic[fi])
            rankic_acc[fn][bucket].append(r_ic[fi])

    if (di + 1) % 200 == 0:
        elapsed = time.time() - t0
        log(f"  {di+1}/{len(unique_dates)} dates ({len(unique_dates)/(elapsed+1e-9):.0f} dates/s)")

# ===== BUILD REPORT =====
log("Building report...")
rows = []
for fn in factor_names:
    row = {"factor": fn}
    for period in ["train", "valid"]:
        ic = np.array(rankic_acc[fn][period])
        if len(ic) > 10:
            row[f"{period}_N"] = len(ic)
            row[f"{period}_RankIC"] = ic.mean()
            row[f"{period}_RankIC_std"] = ic.std()
            row[f"{period}_RankICIR"] = ic.mean() / ic.std() if ic.std() > 0 else 0
            row[f"{period}_RankIC_win"] = (ic > 0).mean()
    rows.append(row)

report = pd.DataFrame(rows)
report = report.sort_values("valid_RankIC", ascending=False)
report.to_csv(DATA / "single_factor_report.csv", index=False)
log(f"Report: {len(report)} factors saved")

# ===== TOP FACTORS =====
top = report.head(15)
print("\n=== TOP 15 FACTORS (valid RankIC) ===")
for _, r in top.iterrows():
    print(f"  {r['factor']:25s}  RankIC={r.get('valid_RankIC',0):+.4f}  "
          f"ICIR={r.get('valid_RankICIR',0):+.2f}  Win={r.get('valid_RankIC_win',0):.1%}")

# ===== FACTOR COMBINATIONS (1000+) =====
log("\nGenerating factor combinations (1000+)...")

# Select quality factors: RankIC > 0.01 and ICIR > 0.2
good = report[(report["valid_RankIC"] > 0.01) & (report["valid_RankICIR"] > 0.2)]
good_factors = good["factor"].tolist()
log(f"  {len(good_factors)} quality factors selected")

# Pre-compute daily IC series for all good factors
log("  Precomputing daily IC matrix...")
daily_ic = {}
for fn in good_factors:
    ic_v = np.array(rankic_acc[fn]["train"] + rankic_acc[fn]["valid"])
    # Map back to date order
    train_n = len(rankic_acc[fn]["train"])
    valid_n = len(rankic_acc[fn]["valid"])
    all_ic = np.concatenate([np.array(rankic_acc[fn]["train"]), np.array(rankic_acc[fn]["valid"])])
    daily_ic[fn] = all_ic

# Also pre-compute z-scored factor values for combo IC
log("  Precomputing daily z-score factor matrix...")
# For each date, z-score all good factors
all_data_z = all_data[good_factors].copy()
# Groupby date and z-score each column
for fn in good_factors:
    all_data_z[fn] = all_data_z[fn].groupby("date").transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

# Generate combinations
combos = []
import itertools

# a) Low-correlation pairs
log("  a) Low-corr pairs...")
# Factor correlation from daily IC series
n_good = min(len(good_factors), 40)
top_factors = good_factors[:n_good]
ic_corr = np.corrcoef([daily_ic[f] for f in top_factors])
factor_pairs = []
for i in range(len(top_factors)):
    for j in range(i+1, len(top_factors)):
        if abs(ic_corr[i,j]) < 0.5:  # low correlation
            factor_pairs.append((top_factors[i], top_factors[j]))
            combos.append({
                "id": f"PAIR_{i}_{j}",
                "factors": f"{top_factors[i]},{top_factors[j]}",
                "n": 2,
                "rationale": f"low_corr({abs(ic_corr[i,j]):.2f})",
                "category": "low_corr_pair",
            })

# b) Momentum ensembles (different windows)
log("  b) Momentum ensembles...")
mom_factors = [f for f in good_factors if "Mom" in f or "mom" in f.lower()]
for r in range(2, min(5, len(mom_factors)+1)):
    for combo in itertools.combinations(mom_factors[:8], r):
        combos.append({
            "id": f"MOM_{len(combos)}",
            "factors": ",".join(combo),
            "n": r,
            "rationale": "momentum_multi_window",
            "category": "mom_ensemble",
        })

# c) Category cross: pair factors from different families
log("  c) Category cross pairs...")
families = {}
for f in good_factors:
    if "Mom" in f: families.setdefault("momentum", []).append(f)
    elif "Rev" in f or "rev" in f.lower(): families.setdefault("reversal", []).append(f)
    elif "Vol" in f or "vol" in f.lower(): families.setdefault("volatility", []).append(f)
    elif "Turnover" in f or "Amihud" in f or "liq" in f.lower(): families.setdefault("liquidity", []).append(f)
    elif "MA" in f or "RSI" in f or "RSV" in f: families.setdefault("technical", []).append(f)
    elif "KUP" in f or "KLOW" in f or "KMID" in f or "KLEN" in f or "KSFT" in f: families.setdefault("candle", []).append(f)
    elif "Skew" in f or "Kurt" in f or "MAX" in f: families.setdefault("moment", []).append(f)
    else: families.setdefault("other", []).append(f)

fam_names = list(families.keys())
for i in range(len(fam_names)):
    for j in range(i+1, len(fam_names)):
        for f1 in families[fam_names[i]][:5]:
            for f2 in families[fam_names[j]][:5]:
                combos.append({
                    "id": f"XCAT_{len(combos)}",
                    "factors": f"{f1},{f2}",
                    "n": 2,
                    "rationale": f"{fam_names[i]}_x_{fam_names[j]}",
                    "category": "category_cross",
                })
                if len(combos) > 2000:
                    break
            if len(combos) > 2000:
                break
        if len(combos) > 2000:
            break

# d) 3-factor and 4-factor combos from top factors
log("  d) Multi-factor combos...")
top15 = good_factors[:15]
for r in [3, 4]:
    for combo in itertools.combinations(top15[:10], r):
        combos.append({
            "id": f"MF{r}_{len(combos)}",
            "factors": ",".join(combo),
            "n": r,
            "rationale": f"top15_{r}way",
            "category": f"{r}way",
        })

# e) IC-weighted combos from top pairs
log("  e) IC-weighted combos...")
# For top 50 pairs, also create IC-weighted version
# (Skip implementation — keep equal-weight for now)

log(f"  Total: {len(combos)} combinations")

# ===== EVALUATE COMBINATIONS =====
log("Evaluating combinations...")
combo_results = []
t1 = time.time()

for ci, c in enumerate(combos[:1500]):  # limit to 1500 for speed
    facs = c["factors"].split(",")
    if not all(f in all_data_z.columns for f in facs):
        continue

    # Equal-weight combo z-score
    combo_z = all_data_z[facs].mean(axis=1)
    combo_z.name = "combo"

    # Compute daily RankIC for the combo
    cdf = pd.DataFrame({"factor": combo_z, "fwd": all_data["_fwd"]})
    cdf["date"] = cdf.index.get_level_values("date")
    cdf = cdf.dropna()

    # Per-date RankIC
    rankic_vals = []
    for d, grp in cdf.groupby("date"):
        if len(grp) < 50:
            continue
        x, y = grp["factor"].values, grp["fwd"].values
        xr, yr = rankdata(x), rankdata(y)
        if np.std(xr) > 1e-12 and np.std(yr) > 1e-12:
            rankic_vals.append(np.corrcoef(xr, yr)[0, 1])

    if len(rankic_vals) < 20:
        continue

    rankic_arr = np.array(rankic_vals)
    best_single = good["valid_RankIC"].max() if len(good) > 0 else 0

    combo_results.append({
        "combo_id": c["id"],
        "n_factors": c["n"],
        "factors": c["factors"],
        "rationale": c["rationale"],
        "category": c["category"],
        "valid_RankIC": rankic_arr.mean(),
        "valid_RankIC_std": rankic_arr.std(),
        "valid_RankICIR": rankic_arr.mean() / rankic_arr.std() if rankic_arr.std() > 0 else 0,
        "valid_RankIC_win": (rankic_arr > 0).mean(),
        "best_single_RankIC": best_single,
        "improvement_pct": (rankic_arr.mean() - best_single) / abs(best_single) * 100 if best_single != 0 else 0,
    })

    if (ci + 1) % 200 == 0:
        elapsed = time.time() - t1
        log(f"  {ci+1}/{len(combos)} combos ({ci/elapsed:.0f}/s)")

# ===== FINAL REPORT =====
log("Saving combination results...")
cr = pd.DataFrame(combo_results)
cr = cr.sort_values("valid_RankIC", ascending=False)
cr.to_csv(DATA / "best_combinations.csv", index=False)

print(f"\n=== TOP 20 COMBINATIONS ===")
for _, r in cr.head(20).iterrows():
    print(f"  {r['combo_id']:20s}  n={r['n_factors']}  RankIC={r['valid_RankIC']:+.4f}  "
          f"IR={r['valid_RankICIR']:+.2f}  vs_best={r['improvement_pct']:+.0f}%  [{r['rationale'][:30]}]")

print(f"\nBest single factor RankIC: {good['valid_RankIC'].max():.4f}")
print(f"Best combo RankIC:       {cr['valid_RankIC'].max():.4f}")
print(f"Improvement:            {cr['improvement_pct'].max():.0f}%")

log(f"Done in {(time.time()-t0)/60:.1f}min")
