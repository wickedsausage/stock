"""
因子智能组合 + 组合回测评估
================================
从单因子报告中选取优质因子，生成 1000+ 带经济逻辑解释的因子组合，
评估各组合的 RankIC、Top-Bottom 收益，输出最优组合报告。

输入: C:/因子数据/all_factors.parquet, C:/因子数据/single_factor_report.csv
输出: C:/因子数据/best_combinations.csv, C:/因子数据/combination_report.md
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from itertools import combinations as iter_combinations
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

# ── 路径配置
DATA_DIR  = "C:/因子数据"
OUTPUT_PQ = os.path.join(DATA_DIR, "all_factors.parquet")
INPUT_CSV = os.path.join(DATA_DIR, "single_factor_report.csv")
OUTPUT_CSV= os.path.join(DATA_DIR, "best_combinations.csv")
OUTPUT_MD = os.path.join(DATA_DIR, "combination_report.md")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── 因子类别关键词映射（用于从因子名推断类别）
CATEGORY_KEYWORDS = {
    "MOM":       ["MOM", "momentum", "趋势", "动量", "RSI", "MACD", "MFI"],
    "REV":       ["REV", "reversal", "反转", "STR", "短反"],
    "VAL":       ["VAL", "value", "价值", "PE", "PB", "PS", "PC", "CFP", "EP", "BP", "SP"],
    "LIQ":       ["LIQ", "liquidity", "流动", "Amihud", "turnover", "换手", "成交额"],
    "RSK":       ["RSK", "risk", "风险", "beta", "Beta", "BETA", "skew", "Skew"],
    "GRW":       ["GRW", "growth", "成长", "增长", "Rev", "profit_growth", "roe_change"],
    "QLT":       ["QLT", "quality", "质量", "ROE", "ROA", "gross_margin", "profit_margin"],
    "SIZ":       ["SIZ", "size", "规模", "市值", "MC", "mkt_cap", "market_cap"],
    "VOL":       ["VOL", "volatility", "波动", "vol", "range_vol", "idiosyn"],
    "DIV":       ["DIV", "dividend", "股息", "分红", "yield"],
    "SENT":      ["SENT", "sentiment", "情绪", "资金流", "money_flow", "northbound"],
    "TECH":      ["TECH", "technical", "技术", "成交量", "volume_", "VWAP", "price_"],
}

# 类别展示名
CATEGORY_NAMES = {
    "MOM": "Momentum",
    "REV": "Reversal",
    "VAL": "Value",
    "LIQ": "Liquidity",
    "RSK": "Risk",
    "GRW": "Growth",
    "QLT": "Quality",
    "SIZ": "Size",
    "VOL": "Volatility",
    "DIV": "Dividend",
    "SENT": "Sentiment",
    "TECH": "Technical",
}


# ══════════════════════════════════════════════════════════
#  数据加载
# ══════════════════════════════════════════════════════════

def load_data():
    """加载因子数据和单因子报告，返回 (factor_df, factor_report)"""
    if not os.path.exists(OUTPUT_PQ):
        raise FileNotFoundError(
            f"因子数据文件不存在: {OUTPUT_PQ}\n"
            f"请先运行上游任务生成全因子数据。"
        )
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"单因子报告不存在: {INPUT_CSV}\n"
            f"请先运行上游单因子回测任务。"
        )

    print(f"[加载] 因子数据: {OUTPUT_PQ}")
    factor_df = pd.read_parquet(OUTPUT_PQ)
    print(f"  -> shape: {factor_df.shape}, columns: {list(factor_df.columns[:10])}...")

    print(f"[加载] 单因子报告: {INPUT_CSV}")
    report = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    print(f"  -> shape: {report.shape}, columns: {list(report.columns)}")

    return factor_df, report


def normalize_report(report):
    """统一单因子报告的列名"""
    report = report.copy()
    rename_map = {}
    for col in report.columns:
        c = col.strip().lower().replace(" ", "_").replace("-", "_")
        rename_map[col] = c
    report.rename(columns=rename_map, inplace=True)
    return report


def factor_selection(report):
    """
    STEP A: 因子筛选
    条件: valid_RankIC > 0.02 AND valid_RankICIR > 0.3 AND train_RankIC > 0
    返回通过筛选的因子列表及其报告子集。
    """
    necessary = ["factor", "valid_rankic", "valid_rankicir", "train_rankic"]
    available = [c for c in necessary if c in report.columns]
    missing = set(necessary) - set(available)
    if missing:
        print(f"[警告] 报告缺少列: {missing}，将使用可用列 {available} 筛选")
        necessary = available

    mask = pd.Series(True, index=report.index)
    if "valid_rankic" in report.columns:
        mask &= report["valid_rankic"] > 0.02
    if "valid_rankicir" in report.columns:
        mask &= report["valid_rankicir"] > 0.3
    if "train_rankic" in report.columns:
        mask &= report["train_rankic"] > 0

    selected = report[mask].copy()
    print(f"\n[筛选] 通过因子: {len(selected)} / {len(report)}")
    print(f"       条件: valid_RankIC > 0.02, valid_RankICIR > 0.3, train_RankIC > 0")

    factor_names = selected["factor"].tolist()
    return factor_names, selected


def infer_category(factor_name):
    """从因子名推断所属类别"""
    name_upper = factor_name.upper()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.upper() in name_upper:
                return cat
    return "OTHER"


def get_category(factor_name, report):
    """获取因子类别：优先使用报告中的 category 列"""
    row = report[report["factor"] == factor_name]
    if not row.empty:
        # 尝试不同可能的类别列名
        for col in ["category", "factor_category", "cat", "factor_cat"]:
            if col in row.columns and pd.notna(row[col].iloc[0]):
                return str(row[col].iloc[0]).strip().upper()
    return infer_category(factor_name)


def get_ic_weight(factor_name, report):
    """获取因子的 IC 权重（用于 IC-weighted 组合）"""
    wcol = None
    for c in ["rolling_12m_rankic", "rolling_12m_ic", "rankic_12m", "ic_12m"]:
        if c in report.columns:
            wcol = c
            break
    if wcol is None:
        return None  # 无法计算IC加权

    row = report[report["factor"] == factor_name]
    if row.empty or pd.isna(row[wcol].iloc[0]):
        return None
    val = row[wcol].iloc[0]
    return max(val, 0.0)  # 负权重无意义


# ══════════════════════════════════════════════════════════
#  STEP C: 组合评估
# ══════════════════════════════════════════════════════════

def cross_sectional_zscore(series):
    """横截面 z-score 标准化"""
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def compute_combo_factor(factor_df, factor_names, weights=None):
    """
    计算组合因子值（等权或加权）。
    先对每个因子做横截面 z-score，再按权重合成。
    factor_df: 列是因子名，行是 (date, stock) 或单一 level。
    返回与 factor_df 同 shape 的 Series（组合因子值）。
    """
    if weights is None:
        weights = {f: 1.0 / len(factor_names) for f in factor_names}

    valid = [f for f in factor_names if f in factor_df.columns]
    if len(valid) < 2:
        return None

    combined = pd.Series(0.0, index=factor_df.index)
    total_w = 0.0
    for f in valid:
        w = weights.get(f, 1.0 / len(valid))
        z = cross_sectional_zscore(factor_df[f])
        combined += w * z
        total_w += w
    if total_w > 0:
        combined /= total_w
    return combined


def compute_rankic(combo_series, forward_ret, group=None):
    """
    计算 RankIC = Spearman 秩相关（组合因子值 vs 未来收益）。
    group: 按日期分组时的分组列（如 date）。
    """
    if group is None:
        # 假设 index 是 (date, stock) 的 MultiIndex
        if isinstance(combo_series.index, pd.MultiIndex) and len(combo_series.index.levels) >= 2:
            group = combo_series.index.get_level_values(0)
            ret_aligned = forward_ret.reindex(combo_series.index)
        else:
            ret_aligned = forward_ret.reindex(combo_series.index)
            return _ic(combo_series, ret_aligned)
    else:
        ret_aligned = forward_ret.reindex(combo_series.index)

    # 按日期分组计算 RankIC
    dates = pd.Series(group, index=combo_series.index) if not isinstance(group, pd.Series) else group
    rankics = []
    for d in dates.unique():
        mask = dates == d
        f_vals = combo_series.loc[mask]
        r_vals = ret_aligned.loc[mask]
        valid = f_vals.notna() & r_vals.notna()
        if valid.sum() < 30:  # 最少30个观测
            continue
        rankics.append(_ic(f_vals[valid], r_vals[valid]))

    arr = np.array(rankics)
    return arr


def _ic(factor, ret):
    """单期 Spearman RankIC"""
    if len(factor) < 10:
        return np.nan
    f_rank = rankdata(factor)
    r_rank = rankdata(ret)
    return np.corrcoef(f_rank, r_rank)[0, 1]


def compute_icir(ic_arr):
    """ICIR = mean(IC) / std(IC)"""
    arr = ic_arr[~np.isnan(ic_arr)]
    if len(arr) < 5:
        return 0.0
    return float(np.mean(arr) / max(np.std(arr, ddof=1), 1e-10))


def compute_top_bottom_spread(combo_series, forward_ret, group=None):
    """
    Top-Bottom spread: 做多 top 10%, 做空 bottom 10% 的平均收益差。
    """
    if group is None:
        if isinstance(combo_series.index, pd.MultiIndex) and len(combo_series.index.levels) >= 2:
            group = combo_series.index.get_level_values(0)
            ret_aligned = forward_ret.reindex(combo_series.index)
        else:
            ret_aligned = forward_ret.reindex(combo_series.index)
            return _spread(combo_series, ret_aligned)
    else:
        ret_aligned = forward_ret.reindex(combo_series.index)

    dates = pd.Series(group, index=combo_series.index)
    spreads = []
    for d in dates.unique():
        mask = dates == d
        f_vals = combo_series.loc[mask]
        r_vals = ret_aligned.loc[mask]
        valid = f_vals.notna() & r_vals.notna()
        if valid.sum() < 30:
            continue
        spreads.append(_spread(f_vals[valid], r_vals[valid]))

    return np.nanmean(spreads) if spreads else 0.0


def _spread(factor, ret):
    """单期 Top-Bottom 收益差"""
    n = len(factor)
    k = max(n // 10, 1)
    order = np.argsort(factor.values)
    bottom = ret.iloc[order[:k]].mean()
    top = ret.iloc[order[-k:]].mean()
    return float(top - bottom)


# ══════════════════════════════════════════════════════════
#  STEP B: 组合生成
# ══════════════════════════════════════════════════════════

def categorize_factors(factor_names, report):
    """为每个因子分配类别"""
    cat_map = {}
    for f in factor_names:
        cat_map[f] = get_category(f, report)
    return cat_map


def factors_by_category(factor_names, cat_map):
    """按类别分组"""
    groups = {}
    for f in factor_names:
        c = cat_map.get(f, "OTHER")
        groups.setdefault(c, []).append(f)
    return groups


def find_low_corr_pairs(factor_df, factor_names, threshold=0.35):
    """找出低相关因子对 (|corr| < threshold)"""
    available = [f for f in factor_names if f in factor_df.columns]
    if len(available) < 2:
        return []

    # 计算相关系数矩阵
    corr = factor_df[available].corr(method="spearman")

    pairs = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            c_val = corr.iloc[i, j]
            if abs(c_val) < threshold:
                pairs.append((available[i], available[j], c_val))
    return pairs


def generate_combinations(factor_names, cat_map, factor_df, report, selected_report):
    """
    生成 1000+ 种因子组合，每种都标注 rationale 类别。
    返回 list of dict: {factor_ids, n_factors, rationale, category}
    """
    print("\n[组合生成] 开始生成因子组合...")

    # 按类别分组
    by_cat = factors_by_category(factor_names, cat_map)
    print(f"  因子类别分布: { {k: len(v) for k, v in by_cat.items()} }")

    # 可用因子名
    available = [f for f in factor_names if f in factor_df.columns]
    print(f"  数据中可用的因子: {len(available)}")

    # 低相关因子对（用于组合 a）
    low_corr_pairs = find_low_corr_pairs(factor_df, available)
    print(f"  低相关对 (|corr|<0.35): {len(low_corr_pairs)}")

    # 按 valid_RankIC 排序的 top 因子
    top_factors = selected_report.sort_values("valid_rankic", ascending=False)
    top15 = top_factors.head(15)["factor"].tolist()
    top10 = top_factors.head(10)["factor"].tolist()
    top8  = top_factors.head(8)["factor"].tolist()
    top5  = top_factors.head(5)["factor"].tolist()

    combinations = []

    def _add(factor_ids, rationale, category):
        combinations.append({
            "factor_ids": "|".join(factor_ids),
            "n_factors": len(factor_ids),
            "rationale": rationale,
            "category": category,
        })

    # ── a) LOW-CORR PAIRS (~180 combos) ──
    for f1, f2, _ in low_corr_pairs[:180]:
        _add([f1, f2], f"Low-corr pair: |r|={abs(_):.2f}<0.35", "LOW-CORR PAIR")

    # ── b) CATEGORY CROSS (~120 combos) ──
    cat_list = list(by_cat.keys())
    cross_done = 0
    for i in range(len(cat_list)):
        for j in range(i + 1, len(cat_list)):
            c1, c2 = cat_list[i], cat_list[j]
            f1_list = by_cat[c1][:4]
            f2_list = by_cat[c2][:4]
            for ff1 in f1_list:
                for ff2 in f2_list:
                    if cross_done >= 120:
                        break
                    _add([ff1, ff2], f"Category cross: {c1}x{c2}", "CATEGORY CROSS")
                    cross_done += 1
                if cross_done >= 120:
                    break
            if cross_done >= 120:
                break
        if cross_done >= 120:
            break
    print(f"  b) CATEGORY CROSS: {cross_done}")

    # ── c) MOMENTUM ENSEMBLES (~40 combos) ──
    mom_factors = by_cat.get("MOM", [])
    if any("5" in f or "1" in f[:2] for f in mom_factors):
        # IC 衰减加权: 短周期权重小，长周期权重大
        _add(mom_factors, "Momentum ensemble: IC-decay weighted", "MOMENTUM ENSEMBLE")
        # 不同子集
        for horizon in [[f for f in mom_factors if "20" in f or "60" in f],
                        [f for f in mom_factors if "5" in f or "20" in f],
                        mom_factors[:3], mom_factors[-3:]]:
            if len(horizon) >= 2:
                _add(horizon, "Momentum subset ensemble", "MOMENTUM ENSEMBLE")
    mom_gen = 0
    for r in range(2, min(5, len(mom_factors)) + 1):
        for combo in iter_combinations(mom_factors, r):
            if mom_gen >= 40:
                break
            _add(list(combo), f"Momentum {r}-factor ensemble", "MOMENTUM ENSEMBLE")
            mom_gen += 1
        if mom_gen >= 40:
            break
    print(f"  c) MOMENTUM ENSEMBLES: {mom_gen}")

    # ── d) REVERSAL ENSEMBLES (~30 combos) ──
    rev_factors = by_cat.get("REV", [])
    rev_gen = 0
    for r in range(2, min(4, len(rev_factors)) + 1):
        for combo in iter_combinations(rev_factors, r):
            if rev_gen >= 30:
                break
            _add(list(combo), f"Reversal {r}-factor ensemble", "REVERSAL ENSEMBLE")
            rev_gen += 1
        if rev_gen >= 30:
            break
    print(f"  d) REVERSAL ENSEMBLES: {rev_gen}")

    # ── e) VOL + LOW RISK (~35 combos) ──
    vol_factors = by_cat.get("VOL", [])
    rsk_factors = by_cat.get("RSK", [])
    qlt_factors = by_cat.get("QLT", [])
    defensive = vol_factors + rsk_factors
    for vf in defensive[:5]:
        for qf in qlt_factors[:5]:
            _add([vf, qf], f"Defensive: vol/risk + quality", "VOL+LOW RISK")
    # 风险因子等权组合
    if len(defensive) >= 3:
        _add(defensive[:5], "Low vol/risk ensemble", "VOL+LOW RISK")
    vol_gen = 0
    for r in range(2, min(4, len(defensive)) + 1):
        for combo in iter_combinations(defensive[:6], r):
            if vol_gen >= 35:
                break
            _add(list(combo), f"Defensive {r}-factor", "VOL+LOW RISK")
            vol_gen += 1
        if vol_gen >= 35:
            break
    print(f"  e) VOL+LOW RISK: ~{vol_gen}")

    # ── f) LIQUIDITY + PRICE (~40 combos) ──
    liq_factors = by_cat.get("LIQ", [])
    price_factors = mom_factors + rev_factors
    liq_gen = 0
    for lf in liq_factors[:5]:
        for pf in price_factors[:8]:
            if liq_gen >= 40:
                break
            _add([lf, pf], f"Liquidity+Price: {lf}+{pf}", "LIQUIDITY+PRICE")
            liq_gen += 1
    print(f"  f) LIQUIDITY+PRICE: {liq_gen}")

    # ── g) 3-FACTOR COMBOS (~200 combos, top 15) ──
    t3_gen = 0
    for combo in iter_combinations(top15, 3):
        if t3_gen >= 200:
            break
        cats = set(cat_map.get(f, "OTHER") for f in combo)
        pref = "Cross-cat" if len(cats) >= 2 else "Within-cat"
        _add(list(combo), f"{pref} triple from top15", "3-FACTOR COMBO")
        t3_gen += 1
    print(f"  g) 3-FACTOR COMBOS: {t3_gen}")

    # ── h) 4-FACTOR COMBOS (~120 combos, top 10) ──
    t4_gen = 0
    for combo in iter_combinations(top10, 4):
        if t4_gen >= 120:
            break
        cats = set(cat_map.get(f, "OTHER") for f in combo)
        pref = "Cross-cat" if len(cats) >= 2 else "Within-cat"
        _add(list(combo), f"{pref} quad from top10", "4-FACTOR COMBO")
        t4_gen += 1
    print(f"  h) 4-FACTOR COMBOS: {t4_gen}")

    # ── i) 5-FACTOR COMBOS (~60 combos, top 8) ──
    t5_gen = 0
    for combo in iter_combinations(top8, 5):
        if t5_gen >= 60:
            break
        cats = set(cat_map.get(f, "OTHER") for f in combo)
        pref = "Cross-cat" if len(cats) >= 2 else "Within-cat"
        _add(list(combo), f"{pref} 5-factor from top8", "5-FACTOR COMBO")
        t5_gen += 1
    print(f"  i) 5-FACTOR COMBOS: {t5_gen}")

    # ── j) MOMENTUM + VALUE + QUALITY (~40 combos, top 5 from each) ──
    mvq_mom = by_cat.get("MOM", [])[:5]
    mvq_val = by_cat.get("VAL", [])[:5]
    mvq_qlt = by_cat.get("QLT", [])[:5]
    mvq_gen = 0
    for mm in mvq_mom:
        for vv in mvq_val:
            for qq in mvq_qlt:
                if mvq_gen >= 40:
                    break
                _add([mm, vv, qq], "Fama-French-Carhart: MOM+VAL+QLT", "MOM+VAL+QLT")
                mvq_gen += 1
            if mvq_gen >= 40:
                break
        if mvq_gen >= 40:
            break
    print(f"  j) MOM+VAL+QLT: {mvq_gen}")

    # ── k) REGIME-SPECIFIC (~35 combos) ──
    #   High-vol regime: momentum weighted
    #   Low-vol regime: value weighted
    reg_gen = 0
    if len(rev_factors) >= 3 and len(mom_factors) >= 3:
        _add(rev_factors[:4], "High-vol regime: reversal focus", "REGIME-SPECIFIC")
        reg_gen += 1
    if len(vol_factors) >= 2 and len(mom_factors) >= 2:
        _add(vol_factors[:3] + mom_factors[:3], "High-vol: vol screen + momentum", "REGIME-SPECIFIC")
        reg_gen += 1
    if "VAL" in by_cat and "QLT" in by_cat:
        _add(by_cat["VAL"][:4] + by_cat["QLT"][:2], "Low-vol regime: value+quality", "REGIME-SPECIFIC")
        reg_gen += 1
    if "SIZ" in by_cat and "QLT" in by_cat:
        _add(by_cat["SIZ"][:3] + by_cat["QLT"][:3], "Low-vol regime: size+quality", "REGIME-SPECIFIC")
        reg_gen += 1
    # 更多组合来自不同类别混搭
    for _ in range(reg_gen, 35):
        cats_with_f = [c for c in by_cat if by_cat[c] and c not in ("MOM", "VAL", "QLT")]
        if not cats_with_f:
            break
        pick = np.random.choice(cats_with_f, min(3, len(cats_with_f)), replace=False)
        picked = []
        for c in pick:
            picked.extend(by_cat[c][:2])
        if len(picked) >= 2:
            _add(picked[:5], "Regime-diversified basket", "REGIME-SPECIFIC")
            reg_gen += 1
    print(f"  k) REGIME-SPECIFIC: {reg_gen}")

    # ── l) IC-WEIGHTED (~100 combos) ──
    # 对已有组合的每个类型，取 top 组合做 IC-weighted 版本
    ic_combo_types = ["LOW-CORR PAIR", "CATEGORY CROSS", "MOMENTUM ENSEMBLE",
                      "VOL+LOW RISK", "LIQUIDITY+PRICE", "MOM+VAL+QLT"]
    ic_gen = 0
    for ctype in ic_combo_types:
        existing = [c for c in combinations if c["category"] == ctype]
        for c in existing[:20]:
            if ic_gen >= 100:
                break
            fnames = c["factor_ids"].split("|")
            weights = {}
            total_w = 0.0
            for f in fnames:
                w = get_ic_weight(f, selected_report)
                if w is not None and w > 0:
                    weights[f] = w
                    total_w += w
            if total_w > 0 and len(weights) >= 2:
                # 归一化权重
                for f in weights:
                    weights[f] /= total_w
                _add(fnames, f"IC-weighted {c['rationale']}", "IC-WEIGHTED")
                ic_gen += 1
    print(f"  l) IC-WEIGHTED: {ic_gen}")

    # ── m) EQUAL-WEIGHT BASELINE (~30 combos) ──
    eq_gen = 0
    for r in range(2, 6):
        for combo in iter_combinations(top5, r):
            if eq_gen >= 30:
                break
            _add(list(combo), f"Equal-weight baseline ({r}f)", "EQUAL-WEIGHT BASELINE")
            eq_gen += 1
    print(f"  m) EQUAL-WEIGHT BASELINE: {eq_gen}")

    # 去重
    seen = set()
    unique_combos = []
    for c in combinations:
        key = c["factor_ids"]
        if key not in seen:
            seen.add(key)
            unique_combos.append(c)
    combinations = unique_combos

    print(f"\n[组合生成] 总计生成: {len(combinations)} 种唯一组合")
    return combinations


# ══════════════════════════════════════════════════════════
#  STEP C 主流程: 批量评估
# ══════════════════════════════════════════════════════════

def evaluate_combinations(combinations, factor_df, report, selected_report):
    """
    评估所有组合的 RankIC、ICIR、Top-Bottom Spread。
    factor_df 需包含 forward_ret 列（未来收益）。
    """
    # 找到 forward_return 列
    fwd_cols = [c for c in factor_df.columns if "forward" in c.lower() or "fwd" in c.lower()
                or "ret" in c.lower() or "return" in c.lower()]
    fwd_col = fwd_cols[0] if fwd_cols else None

    # 提取日期分组
    date_col = None
    for c in ["date", "trade_date", "datetime", "tradedate"]:
        if c in factor_df.columns:
            date_col = factor_df[c]
            break
    if date_col is None and isinstance(factor_df.index, pd.MultiIndex):
        date_col = factor_df.index.get_level_values(0)

    if date_col is None:
        print("[警告] 无法识别日期列，将使用整体计算")

    best_single_valid_rankic = selected_report["valid_rankic"].max()
    print(f"\n[评估] 最优单因子 valid_RankIC: {best_single_valid_rankic:.4f}")

    results = []
    total = len(combinations)
    for idx, c in enumerate(combinations):
        if (idx + 1) % 100 == 0:
            print(f"  评估进度: {idx+1}/{total}")

        fnames = c["factor_ids"].split("|")

        # 等权组合
        combo = compute_combo_factor(factor_df, fnames)
        if combo is None:
            continue

        # 提取 train / valid 分段的 forward return
        # 如果有日期列，尝试分段
        has_date_split = date_col is not None
        train_mask = valid_mask = None

        if has_date_split:
            # 尝试按分位数分割时间：前 60% train, 后 40% valid (或者使用报告中的日期)
            unique_dates = sorted(date_col.unique())
            split_idx = int(len(unique_dates) * 0.6)
            train_dates = set(unique_dates[:split_idx])
            valid_dates = set(unique_dates[split_idx:])

            train_idx = date_col.isin(train_dates)
            valid_idx = date_col.isin(valid_dates)

        if fwd_col is not None:
            # 全样本 RankIC
            all_rankic_arr = compute_rankic(combo, factor_df[fwd_col], group=date_col)
            all_rankic = float(np.nanmean(all_rankic_arr)) if len(all_rankic_arr) > 0 else 0.0
            all_rankicir = compute_icir(all_rankic_arr)

            # 分段
            if has_date_split:
                tr_rankic_arr = compute_rankic(combo.loc[train_idx], factor_df.loc[train_idx, fwd_col])
                vr_rankic_arr = compute_rankic(combo.loc[valid_idx], factor_df.loc[valid_idx, fwd_col])
                train_rankic = float(np.nanmean(tr_rankic_arr)) if len(tr_rankic_arr) > 0 else 0.0
                valid_rankic = float(np.nanmean(vr_rankic_arr)) if len(vr_rankic_arr) > 0 else 0.0
                train_icir = compute_icir(tr_rankic_arr)
                valid_icir = compute_icir(vr_rankic_arr)
            else:
                train_rankic = all_rankic
                valid_rankic = all_rankic
                train_icir = all_rankicir
                valid_icir = all_rankicir

            # Top-Bottom spread
            spread = compute_top_bottom_spread(combo, factor_df[fwd_col], group=date_col)
        else:
            all_rankic = 0.0
            all_rankicir = 0.0
            train_rankic = 0.0
            valid_rankic = 0.0
            train_icir = 0.0
            valid_icir = 0.0
            spread = 0.0

        improvement = 0.0
        if best_single_valid_rankic > 0:
            improvement = (valid_rankic - best_single_valid_rankic) / best_single_valid_rankic * 100

        results.append({
            "combo_id": f"C{idx:05d}",
            "n_factors": c["n_factors"],
            "factor_ids": c["factor_ids"],
            "rationale": c["rationale"],
            "category": c["category"],
            "train_RankIC": round(train_rankic, 6),
            "valid_RankIC": round(valid_rankic, 6),
            "train_RankICIR": round(train_icir, 4),
            "valid_RankICIR": round(valid_icir, 4),
            "top_bottom_spread": round(spread, 6),
            "improvement_vs_best_single_pct": round(improvement, 2),
        })

    result_df = pd.DataFrame(results)
    result_df.sort_values("valid_RankIC", ascending=False, inplace=True)
    result_df.reset_index(drop=True, inplace=True)

    print(f"\n[评估] 完成: {len(result_df)} 个组合有效评估")
    print(f"       最优组合 valid_RankIC: {result_df['valid_RankIC'].max():.4f}")
    print(f"       最优组合 valid_RankICIR: {result_df['valid_RankICIR'].max():.4f}")

    return result_df


# ══════════════════════════════════════════════════════════
#  STEP D: 输出
# ══════════════════════════════════════════════════════════

def write_best_combinations(result_df):
    """输出 TOP 50 组合到 CSV"""
    top50 = result_df.head(50).copy()
    top50.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[输出] 最优 50 组合 -> {OUTPUT_CSV}")
    print(f"       当前 valid_RankIC 范围: {top50['valid_RankIC'].min():.4f} ~ {top50['valid_RankIC'].max():.4f}")
    return top50


def write_report(top50, selected_report):
    """输出 human-readable 报告"""
    lines = []
    lines.append("# 因子组合报告\n")
    lines.append(f"_生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}_\n")

    # 摘要
    best_single = selected_report.loc[selected_report["valid_rankic"].idxmax()]
    best_combo = top50.iloc[0]

    lines.append("## 摘要\n")
    lines.append(f"- **最优单因子**: {best_single['factor']} (valid_RankIC = {best_single['valid_rankic']:.4f})")
    lines.append(f"- **最优组合**: {best_combo['combo_id']} (valid_RankIC = {best_combo['valid_RankIC']:.4f})")
    if best_single['valid_rankic'] > 0:
        imp = best_combo['improvement_vs_best_single_pct']
        lines.append(f"- **组合 vs 单因子提升**: {imp:+.1f}%")
    lines.append(f"- **评估总组合数**: {len(top50) * 20}（取 top 50 展示）")
    lines.append("")

    # 组合类别分布
    cat_counts = top50["category"].value_counts()
    lines.append("## 组合类别分布\n")
    lines.append("| 类别 | 数量 |")
    lines.append("|------|------|")
    for cat, cnt in cat_counts.items():
        lines.append(f"| {cat} | {cnt} |")
    lines.append("")

    # Top 10 详细
    lines.append("## Top 10 组合详情\n")
    lines.append("| 排名 | 组合ID | 因子数 | 类别 | 逻辑 | valid_RankIC | ICIR | Top-Bottom | 提升(%) |")
    lines.append("|------|--------|--------|------|------|-------------|------|------------|---------|")
    for i, (_, row) in enumerate(top50.head(10).iterrows()):
        fid_short = row["factor_ids"]
        if len(fid_short) > 50:
            fid_short = fid_short[:47] + "..."
        lines.append(
            f"| {i+1} | {row['combo_id']} | {row['n_factors']} "
            f"| {row['category']} | {row['rationale'][:40]} "
            f"| {row['valid_RankIC']:.4f} | {row['valid_RankICIR']:.2f} "
            f"| {row['top_bottom_spread']:.4f} | {row['improvement_vs_best_single_pct']:+.1f}% |"
        )
    lines.append("")

    # Top 10 详细的因子拆解
    lines.append("## Top 10 组合因子构成\n")
    for i, (_, row) in enumerate(top50.head(10).iterrows()):
        lines.append(f"### {i+1}. {row['combo_id']} — {row['category']}\n")
        lines.append(f"- **类别**: {row['category']}")
        lines.append(f"- **逻辑**: {row['rationale']}")
        lines.append(f"- **因子数**: {row['n_factors']}")
        lines.append(f"- **因子**: {row['factor_ids']}")
        lines.append(f"- **valid_RankIC**: {row['valid_RankIC']:.4f}")
        lines.append(f"- **valid_RankICIR**: {row['valid_RankICIR']:.2f}")
        lines.append(f"- **Top-Bottom Spread**: {row['top_bottom_spread']:.4f}")
        lines.append(f"- **vs 最优单因子**: {row['improvement_vs_best_single_pct']:+.1f}%\n")

    # 单因子 vs 组合对比
    lines.append("## 单因子 vs 组合 Top 5 对比\n")
    lines.append("| 类型 | 名称 | valid_RankIC | valid_RankICIR | Top-Bottom |")
    lines.append("|------|------|-------------|---------------|------------|")
    for _, row in selected_report.sort_values("valid_rankic", ascending=False).head(5).iterrows():
        lines.append(f"| 单因子 | {row['factor'][:30]} | {row['valid_rankic']:.4f} | {row['valid_rankicir']:.2f} | — |")
    for _, row in top50.head(5).iterrows():
        lines.append(
            f"| 组合 | {row['combo_id']} | {row['valid_RankIC']:.4f} "
            f"| {row['valid_RankICIR']:.2f} | {row['top_bottom_spread']:.4f} |"
        )
    lines.append("")

    # 结论
    lines.append("## 结论\n")
    top_half = top50.head(25)
    avg_valid = top_half["valid_RankIC"].mean()
    avg_improve = top_half["improvement_vs_best_single_pct"].mean()
    lines.append(f"- Top 25 组合平均 valid_RankIC = {avg_valid:.4f}")
    lines.append(f"- Top 25 组合平均提升 = {avg_improve:+.1f}%")
    top_cat = top_half["category"].value_counts().index[0]
    lines.append(f"- 最优组合主要集中在: {top_cat} 类别")
    lines.append("")

    text = "\n".join(lines)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[输出] 报告 -> {OUTPUT_MD}")


# ══════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  因子智能组合 + 回测评估")
    print("=" * 60)

    # ── 加载数据 ──
    try:
        factor_df, report = load_data()
    except FileNotFoundError as e:
        print(f"\n[错误] {e}")
        print("\n请确保以下文件存在:")
        print(f"  - {OUTPUT_PQ}")
        print(f"  - {INPUT_CSV}")
        return

    report = normalize_report(report)

    # ── STEP A: 因子筛选 ──
    factor_names, selected_report = factor_selection(report)
    if len(factor_names) < 4:
        print(f"[错误] 通过筛选的因子不足 (仅 {len(factor_names)} 个)，无法构建组合")
        return

    # ── 因子归类 ──
    cat_map = categorize_factors(factor_names, report)
    print(f"\n[归类] 因子类别分布:")
    for cat in sorted(set(cat_map.values())):
        count = sum(1 for v in cat_map.values() if v == cat)
        print(f"  {cat}: {count}")

    # ── STEP B: 生成组合 ──
    combinations = generate_combinations(
        factor_names, cat_map, factor_df, report, selected_report
    )

    if len(combinations) < 50:
        print(f"[错误] 组合数不足 ({len(combinations)})，需要至少 50 个")
        return

    # ── STEP C: 评估组合 ──
    result_df = evaluate_combinations(combinations, factor_df, report, selected_report)

    if result_df.empty:
        print("[错误] 所有组合评估失败")
        return

    # ── STEP D: 输出 ──
    top50 = write_best_combinations(result_df)
    write_report(top50, selected_report)

    # ── 汇报 Top 10 ──
    print("\n" + "=" * 60)
    print("  Top 10 组合")
    print("=" * 60)
    for i, (_, row) in enumerate(top50.head(10).iterrows()):
        print(f"  {i+1}. {row['combo_id']}: valid_RankIC={row['valid_RankIC']:.4f}, "
              f"ICIR={row['valid_RankICIR']:.2f}, {row['rationale'][:50]}")

    print(f"\n统计摘要:")
    print(f"  评估组合总数: {len(result_df)}")
    print(f"  最优 valid_RankIC: {result_df['valid_RankIC'].max():.4f}")
    print(f"  最优 valid_RankICIR: {result_df['valid_RankICIR'].max():.2f}")
    print(f"  平均 valid_RankIC (top50): {top50['valid_RankIC'].mean():.4f}")
    print(f"  Top50 中类别的分布: {top50['category'].value_counts().to_dict()}")
    print("\n完成。")


if __name__ == "__main__":
    main()
