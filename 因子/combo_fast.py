"""Fast factor combination backtest."""
import pandas as pd, numpy as np, time, itertools
from scipy.stats import rankdata

DATA = "C:/factor_data"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Loading...")
rep = pd.read_csv(f"{DATA}/single_factor_report.csv")
result = pd.read_parquet(f"{DATA}/all_factors.parquet")
labels = pd.read_parquet(f"{DATA}/labels.parquet")
ci = result.index.intersection(labels.index)
result = result.loc[ci]
labels = labels.loc[ci]

# Pre-filter: train |RIC| > 0.01 (top 25)
good = rep[(rep["train_RIC"].abs() > 0.01)].head(25)
sel = good["factor"].tolist()
log(f"  {len(sel)} factors selected")

# Full-sample correlation filter (fast)
sel_data = result[sel].dropna()
corr_mat = sel_data.corr().abs()
pairs = []
for i in range(len(sel)):
    for j in range(i+1, len(sel)):
        if corr_mat.iloc[i, j] < 0.7:
            pairs.append((sel[i], sel[j]))
log(f"  {len(pairs)} low-corr pairs")

# Generate combos
combos = []
for f1, f2 in pairs:
    combos.append({"factors": [f1, f2], "r": "pair", "n": 2})
for combo in itertools.combinations(sel[:10], 3):
    combos.append({"factors": list(combo), "r": "triple", "n": 3})
for combo in itertools.combinations(sel[:6], 4):
    combos.append({"factors": list(combo), "r": "quad", "n": 4})
for k in [3, 5, 8]:
    combos.append({"factors": sel[:k], "r": f"top{k}_ew", "n": k})
# Cross-cat pairs
cats = {"mom": [], "rev": [], "vol": [], "liq": [], "tech": []}
for f in sel:
    fn = f.lower()
    if "mom" in fn: cats["mom"].append(f)
    elif "rev" in fn: cats["rev"].append(f)
    elif "vol" in fn or "down" in fn or "parkinson" in fn: cats["vol"].append(f)
    elif "turn" in fn or "amihud" in fn or "shock" in fn: cats["liq"].append(f)
    elif "ma" in fn or "std" in fn or "rsi" in fn or "rsv" in fn: cats["tech"].append(f)
cn = [c for c in cats if cats[c]]
for i in range(len(cn)):
    for j in range(i+1, len(cn)):
        for f1 in cats[cn[i]][:2]:
            for f2 in cats[cn[j]][:2]:
                combos.append({"factors": [f1, f2], "r": f"{cn[i]}_{cn[j]}", "n": 2})

# Deduplicate
seen = set()
unique = []
for c in combos:
    key = tuple(sorted(c["factors"]))
    if key not in seen:
        seen.add(key)
        unique.append(c)
combos = unique[:1500]
log(f"  {len(combos)} unique combos")

# Evaluate
log("Evaluating...")
label20 = labels["H20_excess"].values
label20_tr = labels["H20_tradable"].values
date_idx = result.index.get_level_values("date").values
all_dates = sorted(result.index.get_level_values("date").unique())
n = len(all_dates)
train_d = set(all_dates[:int(n*0.6)])
valid_d = set(all_dates[int(n*0.6):int(n*0.8)])
test_d = set(all_dates[int(n*0.8):])

# Pre-group dates
date_groups = {}
for d in all_dates:
    m = (date_idx == d) & label20_tr
    if m.sum() >= 50:
        date_groups[d] = m

fac_data_all = result[sel].values
combo_results = []
t0 = time.time()

for ci, c in enumerate(combos):
    facs = [f for f in c["factors"] if f in sel]
    if len(facs) < 2: continue
    indices = [sel.index(f) for f in facs]
    combo_z = np.nanmean(fac_data_all[:, indices], axis=1)

    ric_p = {"train": [], "valid": [], "test": []}
    for d in all_dates:
        mask = date_groups.get(d)
        if mask is None: continue
        z = combo_z[mask]
        y = label20[mask]
        v = ~np.isnan(z) & ~np.isnan(y)
        if v.sum() < 30: continue
        xv, yv = z[v], y[v]
        if np.std(xv) < 1e-12 or np.std(yv) < 1e-12: continue
        ric = np.corrcoef(rankdata(xv), rankdata(yv))[0,1]
        b = "train" if d in train_d else ("valid" if d in valid_d else "test")
        ric_p[b].append(ric)

    row = {"id": f"C{ci:04d}", "n": len(facs), "factors": ",".join(facs), "rationale": c["r"]}
    for p in ["train", "valid", "test"]:
        vv = np.array(ric_p[p])
        if len(vv) >= 10:
            row[f"{p}_RIC"] = vv.mean()
            row[f"{p}_IR"] = vv.mean() / vv.std() if vv.std() > 0 else 0
    if "valid_RIC" in row:
        penalty = 0.01 * np.log(row["n"])
        row["score"] = row["valid_RIC"] - penalty
        best_s = rep["valid_RIC"].max()
        row["vs_best%"] = (row["valid_RIC"] - best_s) / abs(best_s) * 100
        combo_results.append(row)

    if (ci+1) % 200 == 0:
        log(f"  {ci+1}/{len(combos)} ({time.time()-t0:.0f}s)")

cr = pd.DataFrame(combo_results)
cr = cr.sort_values("score", ascending=False)
cr.to_csv(f"{DATA}/best_combinations.csv", index=False)
cr[cr["vs_best%"] <= 0].to_csv(f"{DATA}/rejected_combo_report.csv", index=False)

print(f"\n=== TOP 20 COMBOS ({len(cr)} total) ===")
for _, r in cr.head(20).iterrows():
    print(f"  {r['id']:6s} n={r['n']} V={r['valid_RIC']:+.4f} T={r.get('test_RIC',0):+.4f} "
          f"score={r['score']:+.4f} +{r['vs_best%']:+.0f}% [{r['rationale'][:20]}]")

log(f"Done in {(time.time()-t0)/60:.1f}min")
