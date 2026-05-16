"""Feature fetcher — downloads OHLCV via mootdx + valuation via 腾讯财经, writes qlib-format .bin files."""
import logging
import struct
import time
from pathlib import Path
import numpy as np
import pandas as pd
import urllib.request

logger = logging.getLogger(__name__)

try:
    from mootdx.quotes import Quotes
except ImportError:
    Quotes = None
    logger.warning("mootdx not installed; OHLCV fetch will fail")

# qlib field name → source mapping
# 腾讯财经 field index → qlib field
TENCENT_FIELDS = {
    "pe_ttm": 39,        # PE(TTM)
    "pb": 46,            # PB(市净率)
    "mcap_yi": 44,       # 总市值(亿)
    "float_mcap_yi": 45,  # 流通市值(亿)
    "turnover_pct": 38,   # 换手率%
    "amplitude_pct": 43,  # 振幅%
    "pe_static": 52,     # PE(静)
}


def get_prefix(code: str) -> str:
    """6-digit code → market prefix for mootdx/tencent."""
    if code.startswith(("6", "9")):
        return "sh"
    elif code.startswith("8"):
        return "bj"
    return "sz"


def _calendar_index_map(cal: pd.DatetimeIndex) -> dict:
    """Build a mapping from date string (YYYY-MM-DD) → integer index in calendar."""
    return {d.strftime("%Y-%m-%d"): i for i, d in enumerate(cal)}


def fetch_mootdx_klines(code: str, market: int = 0, offset: int = 500) -> pd.DataFrame:
    """Fetch K-line data for a single stock via mootdx.

    Args:
        code: 6-digit stock code
        market: 0 for SZ, 1 for SH
        offset: number of bars to fetch (max ~5000)

    Returns DataFrame with columns: datetime, open, close, high, low, vol, amount
    """
    if Quotes is None:
        raise ImportError("mootdx is required for K-line fetching")

    if code.startswith("6"):
        market = 1
    elif code.startswith(("0", "3")):
        market = 0
    # code 8 → BJ, market=0 (no separate BJ market in mootdx)

    client = Quotes.factory(market="std")
    bars = client.bars(symbol=code, category=4, offset=offset)  # category=4 → day
    if bars is None or len(bars) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(bars)
    # mootdx returns datetime as both index and column — resolve ambiguity
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index(drop=True)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"])
    # Drop any remaining duplicate datetime labels
    if "datetime" in df.columns:
        dup_cols = [c for c in df.columns if c == "datetime"]
        if len(dup_cols) > 1:
            df = df.loc[:, ~df.columns.duplicated()]
    return df.sort_values("datetime")


def fetch_tencent_valuation(codes: list[str]) -> dict[str, dict]:
    """Batch fetch valuation data (PE/PB/市值 etc.) from 腾讯财经.

    Returns {code: {pe_ttm, pb, mcap_yi, ...}}
    """
    prefixed = []
    for c in codes:
        pfx = get_prefix(c)
        prefixed.append(f"{pfx}{c}")

    url = "https://qt.gtimg.cn/q=" + ",".join(prefixed)
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0")
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        data = resp.read().decode("gbk")
    except Exception:
        return {}

    result = {}
    for line in data.strip().split(";"):
        if not line.strip() or "=" not in line or '"' not in line:
            continue
        key = line.split("=")[0].split("_")[-1]
        vals = line.split('"')[1].split("~")
        if len(vals) < 53:
            continue
        code = key[2:]
        result[code] = {
            "name": vals[1],
            "price": float(vals[3]) if vals[3] else 0,
            "last_close": float(vals[4]) if vals[4] else 0,
            "open": float(vals[5]) if vals[5] else 0,
            "high": float(vals[33]) if vals[33] else 0,
            "low": float(vals[34]) if vals[34] else 0,
            "amount_wan": float(vals[37]) if vals[37] else 0,
            "turnover_pct": float(vals[38]) if vals[38] else 0,
            "pe_ttm": float(vals[39]) if vals[39] else 0,
            "amplitude_pct": float(vals[43]) if vals[43] else 0,
            "mcap_yi": float(vals[44]) if vals[44] else 0,
            "float_mcap_yi": float(vals[45]) if vals[45] else 0,
            "pb": float(vals[46]) if vals[46] else 0,
            "limit_up": float(vals[47]) if vals[47] else 0,
            "limit_down": float(vals[48]) if vals[48] else 0,
            "pe_static": float(vals[52]) if vals[52] else 0,
        }
    return result


def _write_bin_file(filepath: Path, values: np.ndarray, start_index: int) -> None:
    """Write a single qlib-format binary feature file.

    Format: [start_index:int32][value1:float32][value2:float32]...
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(struct.pack("<i", start_index))  # little-endian int32
        f.write(values.astype("<f4").tobytes())   # little-endian float32


def _build_series(klines: pd.DataFrame, cal_map: dict, field: str) -> tuple:
    """Align K-line data with calendar and build a numpy array + start_index.

    Args:
        klines: DataFrame with datetime column
        cal_map: {date_str: calendar_index}
        field: OHLCV field name

    Returns (values_ndarray, start_index) or (None, None) if no data
    """
    if klines.empty or field not in klines.columns:
        return None, None

    dates = klines["datetime"].dt.strftime("%Y-%m-%d")
    values = klines[field].values.astype(float)

    # Filter to dates in calendar
    indices = [cal_map.get(d) for d in dates]
    valid_mask = [idx is not None for idx in indices]

    if not any(valid_mask):
        return None, None

    filtered_dates = [d for d, v in zip(dates, valid_mask) if v]
    filtered_values = values[valid_mask]
    filtered_indices = [cal_map[d] for d in filtered_dates]

    # Build dense array: value at each calendar position
    min_idx = min(filtered_indices)
    max_idx = max(filtered_indices)
    arr = np.full(max_idx - min_idx + 1, np.nan, dtype=np.float32)

    for i, idx in enumerate(filtered_indices):
        arr[idx - min_idx] = filtered_values[i]

    # Forward-fill NaNs (for non-trading gaps like suspension)
    mask = np.isnan(arr)
    if mask.any():
        arr = pd.Series(arr).ffill().bfill().values

    return arr, min_idx


def download_stock_features(
    code: str,
    cal_map: dict,
    provider_uri: str,
    freq: str = "day",
    ohlcv_only: bool = False,
) -> dict:
    """Download all features for a single stock and write to qlib .bin files.

    Args:
        code: 6-digit stock code
        cal_map: {date_str: calendar_index}
        provider_uri: path to qlib data root
        freq: frequency (day, 1min, etc.)
        ohlcv_only: if True, skip tencent valuation fields

    Returns {field: status}
    """
    result = {}
    feat_dir = Path(provider_uri) / "features" / code
    feat_dir.mkdir(parents=True, exist_ok=True)

    # 1. OHLCV via mootdx
    try:
        klines = fetch_mootdx_klines(code, offset=800)
    except Exception as e:
        logger.warning(f"mootdx fetch failed for {code}: {e}")
        klines = pd.DataFrame()

    ohlcv_fields = ["open", "close", "high", "low", "vol", "amount"]
    for field in ohlcv_fields:
        values, start_idx = _build_series(klines, cal_map, field)
        if values is not None and len(values) > 0:
            filepath = feat_dir / f"{field}.{freq}.bin"
            _write_bin_file(filepath, values, start_idx)
            result[field] = f"OK({len(values)})"
        else:
            result[field] = "skip(no_data)"

    # VWAP: amount / vol (volume in shares, amount in yuan)
    if "vol" in klines.columns and "amount" in klines.columns:
        vwap = (klines["amount"] / klines["vol"]).replace([np.inf, -np.inf], np.nan)
        klines["vwap"] = vwap
        values, start_idx = _build_series(klines, cal_map, "vwap")
        if values is not None and len(values) > 0:
            filepath = feat_dir / f"vwap.{freq}.bin"
            _write_bin_file(filepath, values, start_idx)
            result["vwap"] = f"OK({len(values)})"

    if ohlcv_only:
        return result

    # 2. Valuation via 腾讯财经 (only current snapshot, write as latest value)
    # Note: 腾讯财经 only provides current data, not historical.
    # We write the current snapshot as the latest value.
    # For historical PE/PB, use mootdx finance data instead.
    try:
        tencent = fetch_tencent_valuation([code])
        if code in tencent:
            tq = tencent[code]
            # Write as single-value features (current snapshot at last known position)
            for tfield, tidx in TENCENT_FIELDS.items():
                val = tq.get(tfield, 0)
                if val and val > 0:
                    # Use latest calendar index
                    latest_idx = max(cal_map.values()) if cal_map else 0
                    filepath = feat_dir / f"{tfield}.{freq}.bin"
                    _write_bin_file(filepath, np.array([val], dtype=np.float32), latest_idx)
                    result[tfield] = f"OK(snapshot @ idx {latest_idx})"
    except Exception as e:
        logger.warning(f"Tencent fetch failed for {code}: {e}")

    return result


def download_all_features(
    provider_uri: str,
    codes: list[str],
    cal: pd.DatetimeIndex,
    freq: str = "day",
    ohlcv_only: bool = False,
    batch_delay: float = 0.3,
) -> dict:
    """Download features for all stocks in bulk.

    Args:
        provider_uri: qlib data root
        codes: list of 6-digit stock codes
        cal: trading calendar DatetimeIndex
        freq: frequency string
        ohlcv_only: skip valuation fields
        batch_delay: seconds between stocks to avoid rate limiting

    Returns summary dict
    """
    cal_map = _calendar_index_map(cal)
    summary = {"success": 0, "failed": 0, "skipped": 0}
    total = len(codes)

    for i, code in enumerate(codes):
        feat_dir = Path(provider_uri) / "features" / code
        # Check if already downloaded (has close.day.bin)
        if (feat_dir / f"close.{freq}.bin").exists():
            summary["skipped"] += 1
            if i % 50 == 0:
                logger.info(f"[{i+1}/{total}] {code}: already cached, skipped")
            continue

        try:
            result = download_stock_features(code, cal_map, provider_uri, freq, ohlcv_only)
            summary["success"] += 1
            if i % 10 == 0:
                logger.info(f"[{i+1}/{total}] {code}: ok ({', '.join(f'{k}={v}' for k, v in sorted(result.items())[:5])})")
        except Exception as e:
            summary["failed"] += 1
            logger.warning(f"[{i+1}/{total}] {code}: FAILED — {e}")
            if i % 50 == 0:
                logger.warning(f"  Failed: {e}")

        time.sleep(batch_delay)

    return summary
