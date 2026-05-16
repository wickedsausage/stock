"""Factor combination generation + backtest with quality controls."""
import pandas as pd, numpy as np, time, itertools
from scipy.stats import rankdata

DATA = "C:/factor_data"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Loading...")
rep = pd.read_csv(f"{DATA}/single_factor_report.csv")
result = pd.read_parquet(f"{DATA}/all_factors.parquet")
labels = pd.read_parquet(f"{DATA}/labels.parquet")

# Align
ci = result.index.intersection(labels.index)
result = result.loc[ci]
labels = labels.loc[ci]

# === STEP 1: Factor pre-filtering (TRAIN ONLY) ===
log("Pre-filtering factors (train only)...")
good = rep[(rep["train_RIC"].abs() > 0.01) & (rep["train_IR"].abs() > 0.2)]
good_factors = good["factor"].tolist()
log(f"  {len(good_factors)} quality factors from {len(rep)}")

# === STEP 2: Correlation filter (rolling median) ===
log("Correlation filtering...")
n_sel = min(len(good_factors), 35)
sel = good_factors[:n_sel]
sel_data = result[sel].dropna()

# Compute rolling 12M corr for each pair
dates = sorted(sel_data.index.get_level_values("date").unique())
window = 252  # ~1 year
corr_pairs = {}
for i in range(len(sel)):
    for j in range(i+1, len(sel)):
        f1, f2 = sel[i], sel[j]
        # Sample every 10th date for speed
        corrs = []
        for d in dates[::10][window//10:]:
            mask = sel_data.index.get_level_values("date") == d
            sub = sel_data[mask]
            if len(sub) > 50:
                c = sub[f1].corr(sub[f2])
                if not np.isnan(c):
                    corrs.append(c)
        if corrs:
            median_abs = np.median(np.abs(corrs))
            if median_abs < 0.7:
                corr_pairs[(f1, f2)] = median_abs

log(f"  {len(corr_pairs)} low-corr pairs found")

# === STEP 3: Generate combos ===
log("Generating combos...")
combos = []

# a) Low-corr pairs
for (f1, f2), c in corr_pairs.items():
    combos.append({"factors": [f1, f2], "rationale": f"low_corr_{c:.2f}", "cat": "pair"})

# b) Cross-category pairs
cats = {"mom": [], "rev": [], "vol": [], "liq": [], "tech": [], "candle": []}
for f in sel:
    fn = f.lower()
    if "mom" in fn: cats["mom"].append(f)
    elif "rev" in fn: cats["rev"].append(f)
    elif "vol" in fn or "down" in fn or "parkinson" in fn or "idio" in fn or "beta" in fn: cats["vol"].append(f)
    elif "turn" in fn or "amihud" in fn or "shock" in fn or "surp" in fn: cats["liq"].append(f)
    elif "ma" in fn or "rsi" in fn or "rsv" in fn or "std" in fn or "skew" in fn: cats["tech"].append(f)
    elif "kmid" in fn or "klen" in fn or "kup" in fn or "klow" in fn or "ksft" in fn: cats["candle"].append(f)
cat_names = [c for c in cats if len(cats[c]) > 0]
for i in range(len(cat_names)):
    for j in range(i+1, len(cat_names)):
        for f1 in cats[cat_names[i]][:3]:
            for f2 in cats[cat_names[j]][:3]:
                combos.append({"factors": [f1, f2], "rationale": f"{cat_names[i]}_x_{cat_names[j]}", "cat": "cross_cat"})

# c) Triples from top-10
top10 = sel[:10]
for combo in itertools.combinations(top10, 3):
    combos.append({"factors": list(combo), "rationale": "top10_triple", "cat": "triple"})

# d) Top-5 quads
top5 = sel[:5]
for combo in itertools.combinations(top5, 4):
    combos.append({"factors": list(combo), "rationale": "top5_quad", "cat": "quad"})

# e) Equal-weight baselines
combos.append({"factors": sel[:3], "rationale": "top3_ew", "cat": "baseline"})
combos.append({"factors": sel[:5], "rationale": "top5_ew", "cat": "baseline"})
combos.append({"factors": sel[:8], "rationale": "top8_ew", "cat": "baseline"})

# Deduplicate (same factor set)
seen = set()
unique = []
for c in combos:
    key = tuple(sorted(c["factors"]))
    if key not in seen:
        seen.add(key)
        unique.append(c)
combos = unique
log(f"  {len(combos)} unique combos")

# === STEP 4: Evaluate combos ===
log("Evaluating combos...")
label20 = labels["H20_excess"].values
label20_tr = labels["H20_tradable"].values
date_idx = result.index.get_level_values("date").values
all_dates = sorted(result.index.get_level_values("date").unique())
n = len(all_dates)
train_d = set(all_dates[:int(n*0.6)])
valid_d = set(all_dates[int(n*0.6):int(n*0.8)])
test_d = set(all_dates[int(n*0.8):])

# Pre-compute z-scored factor values (already z-scored from preprocessing)
# Pre-group dates for speed
date_groups = {}
for d in all_dates:
    mask = (date_idx == d) & label20_tr
    if mask.sum() >= 50:
        date_groups[d] = mask

combo_results = []
t0 = time.time()
best_single_ric = good["valid_RIC"].max()

for ci, c in enumerate(combos):
    facs = [f for f in c["factors"] if f in result.columns]
    if len(facs) < 2:
        continue

    # Compute equal-weight combo z-score per date
    fac_data = result[facs].values
    combo_z = np.nanmean(fac_data, axis=1)  # equal weight

    # Per-date RankIC
    ric_period = {"train": [], "valid": [], "test": []}
    for d in all_dates:
        mask = date_groups.get(d)
        if mask is None: continue
        z = combo_z[mask]
        y = label20[mask]
        v = ~np.isnan(z) & ~np.isnan(y)
        if v.sum() < 30: continue
        xv, yv = z[v], y[v]
        if np.std(xv) < 1e-12 or np.std(yv) < 1e-12: continue
        ric = np.corrcoef(rankdata(xv), rankdata(yv))[0, 1]
        bucket = "train" if d in train_d else ("valid" if d in valid_d else "test")
        ric_period[bucket].append(ric)

    row = {
        "combo_id": f"C{ci:04d}",
        "n": len(facs),
        "factors": ",".join(facs),
        "rationale": c["rationale"],
        "cat": c["cat"],
    }

    for p in ["train", "valid", "test"]:
        vals = np.array(ric_period[p])
        if len(vals) < 10: continue
        row[f"{p}_RIC"] = vals.mean()
        row[f"{p}_IR"] = vals.mean() / vals.std() if vals.std() > 0 else 0
        row[f"{p}_win"] = (vals > 0).mean()

    if "valid_RIC" not in row: continue

    # Penalty
    penalty = 0.01 * np.log(row["n"]) + 0.005 * (1 - np.clip(row.get("valid_IR", 0), 0, 5) / 5)
    row["penalty"] = penalty
    row["score"] = row["valid_RIC"] - penalty

    # Incremental value check
    row["improve_pct"] = (row["valid_RIC"] - best_single_ric) / abs(best_single_ric) * 100

    combo_results.append(row)

    if (ci + 1) % 200 == 0:
        log(f"  {ci+1}/{len(combos)} ({time.time()-t0:.0f}s)")

# === STEP 5: Output ===
cr = pd.DataFrame(combo_results)
cr = cr.sort_values("score", ascending=False)
cr.to_csv(f"{DATA}/best_combinations.csv", index=False)

# Failure reports
bad_combo = cr[cr["improve_pct"] <= 0]
bad_combo.to_csv(f"{DATA}/rejected_combo_report.csv", index=False)

# Test-gone-bad: good in valid, bad in test
tgb = cr[(cr["valid_RIC"] > 0.05) & (cr.get("test_RIC", 0) < 0.01)]
tgb.to_csv(f"{DATA}/valid_good_test_bad.csv", index=False)

print(f"\n=== TOP 20 COMBOS ({len(cr)} tested) ===")
for _, r in cr.head(20).iterrows():
    print(f"  {r['combo_id']:6s} n={r['n']}  valid={r['valid_RIC']:+.4f}  test={r.get('test_RIC',0):+.4f}  "
          f"score={r['score']:+.4f}  +{r['improve_pct']:+.0f}%  [{r['rationale'][:25]}]")

log(f"Done. Best combo: {cr['valid_RIC'].max():.4f} (single best: {best_single_ric:.4f})")
log(f"Max improvement: {cr['improve_pct'].max():.0f}%")
log(f"N combos: {len(cr)}, rejected: {len(bad_combo)}")
