"""
daa_qlib — 数据源_大A → qlib 数据桥接层

让 qlib 全部通过"数据源_大A"的方案来调取数据。
一行初始化：

    import daa_qlib
    daa_qlib.init()

数据流向：
    数据源_大A (mootdx/腾讯) → qlib 本地格式 (.bin/.txt) → qlib LocalProvider

Usage:
    import daa_qlib
    daa_qlib.init()           # 默认 C:/qlib_data/
    daa_qlib.init(stock_count=200)  # 下载前200只
    daa_qlib.update()         # 增量更新（只拉缺失的）
"""
import logging
import os
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ASCII-safe paths to avoid Windows + joblib Unicode issues
DEFAULT_URI = "C:/qlib_data/"
JOBLIB_TEMP = "C:/temp_joblib/"
_CURRENT_URI = None
_QLIB_IMPORTED = False


def _fix_windows_encoding():
    """Fix joblib UnicodeEncodeError on Windows with Chinese usernames/paths.

    joblib's memmapping pool tries to encode temp paths as ASCII, which breaks
    when the user home directory has CJK characters. Redirect to ASCII path.
    Also fix for any subprocess-based parallelism.
    """
    if os.name == "nt" and "JOBLIB_TEMP_FOLDER" not in os.environ:
        os.makedirs(JOBLIB_TEMP, exist_ok=True)
        os.environ["JOBLIB_TEMP_FOLDER"] = JOBLIB_TEMP
        os.environ["JOBLIB_MULTIPROCESSING"] = "0"  # prefer threading


def _ensure_qlib_path():
    """Ensure qlib is importable — submodule has nested qlib/qlib/ structure."""
    outer_qlib = Path(__file__).parent.parent / "qlib"
    if outer_qlib.is_dir() and str(outer_qlib) not in sys.path:
        sys.path.insert(0, str(outer_qlib))


def _import_qlib():
    """Import qlib with proper path setup and encoding fix."""
    global _QLIB_IMPORTED
    if not _QLIB_IMPORTED:
        _fix_windows_encoding()
        _ensure_qlib_path()
        for m in list(sys.modules):
            if m.startswith("qlib"):
                del sys.modules[m]
        _QLIB_IMPORTED = True
    import qlib
    return qlib


def _ensure_deps():
    """Verify required packages are installed."""
    missing = []
    for pkg in ["mootdx", "pandas", "numpy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"缺少依赖: {', '.join(missing)}。请运行:\n"
            f"  pip install {' '.join(missing)}"
        )


def _write_instruments_for_downloaded(provider_uri: str) -> int:
    """Rewrite instruments/all.txt to include ONLY stocks with data on disk.

    This shrinks the instrument list from ~5800 potential codes to only the
    stocks that actually have downloaded features, dramatically speeding up
    qlib's D.features() queries (which iterate the full instrument list).

    Returns count of stocks written.
    """
    feat_root = Path(provider_uri) / "features"
    if not feat_root.is_dir():
        return 0
    codes = sorted(
        d.name for d in feat_root.iterdir()
        if d.is_dir() and (d / "close.day.bin").exists()
    )
    if not codes:
        return 0
    inst_dir = Path(provider_uri) / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    lines = [f"{c}\t1990-01-01\t{today}" for c in codes]
    (inst_dir / "all.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(codes)


def init(
    provider_uri: str = DEFAULT_URI,
    freq: str = "day",
    stock_count: int = None,
    ohlcv_only: bool = False,
    batch_delay: float = 0.3,
    auto_register: bool = True,
) -> dict:
    """初始化 daa_qlib — 下载数据 + 注册 qlib provider。

    Args:
        provider_uri: 数据缓存目录 (默认 C:/qlib_data/，ASCII 路径避免编码问题)
        freq: K线频率 ("day", "1min", "5min" 等)
        stock_count: 下载股票数量 (None=全市场 ~5800只, 指定数字=前N只)
        ohlcv_only: 仅行情 OHLCV (True), 含估值 PE/PB (False)
        batch_delay: 股票间延迟秒数（mootdx TCP 防限流）
        auto_register: 是否自动调用 qlib.init()

    Returns:
        dict: provider_uri, instrument_count, feature_status, qlib_registered
    """
    _fix_windows_encoding()
    _ensure_deps()

    from .calendar_fetcher import fetch_trading_calendar, write_calendar_files
    from .instrument_fetcher import generate_a_stock_list
    from .feature_fetcher import download_all_features

    global _CURRENT_URI
    _CURRENT_URI = provider_uri

    logger.info(f"=== daa_qlib init: provider_uri={provider_uri} ===")

    # Step 1: Calendar from mootdx (TCP, no HTTP dependency)
    logger.info("Step 1/3: Trading calendar from mootdx K-line dates...")
    cal = fetch_trading_calendar(provider_uri)
    write_calendar_files(provider_uri, cal)

    # Step 2: Generate instrument list
    logger.info("Step 2/3: Stock list...")
    stock_df = generate_a_stock_list()
    if stock_count is not None and stock_count > 0:
        stock_df = stock_df.head(stock_count)
    codes = stock_df["code"].tolist()

    # Step 3: Download features → mootdx OHLCV + optional tencent PE/PB
    logger.info(f"Step 3/3: Downloading features for {len(codes)} stocks...")
    feat_summary = download_all_features(
        provider_uri=provider_uri,
        codes=codes,
        cal=cal,
        freq=freq,
        ohlcv_only=ohlcv_only,
        batch_delay=batch_delay,
    )

    # Step 3.5: Write instrument file with only downloaded stocks
    written = _write_instruments_for_downloaded(provider_uri)
    logger.info(f"Instrument file: {written} stocks with data on disk")

    result = {
        "provider_uri": provider_uri,
        "instrument_count": written,
        "calendar_start": cal[0].strftime("%Y-%m-%d"),
        "calendar_end": cal[-1].strftime("%Y-%m-%d"),
        "feature_status": feat_summary,
        "qlib_registered": False,
    }

    # Step 4: Register with qlib
    if auto_register:
        try:
            qlib_mod = _import_qlib()
            qlib_mod.init(
                provider_uri=provider_uri,
                region="cn",
                freq=freq,
            )
            result["qlib_registered"] = True
            from qlib.data import D
            logger.info(f"qlib ready — {len(D.instruments(market='all'))} stocks, "
                        f"calendar {D.calendar()[0]} ~ {D.calendar()[-1]}")
        except Exception as e:
            logger.warning(f"qlib.init() failed: {e}")

    logger.info(f"=== done: {written} stocks, {len(cal)} trading days ===")
    return result


def update(
    provider_uri: str = None,
    freq: str = "day",
    ohlcv_only: bool = False,
    batch_delay: float = 0.3,
) -> dict:
    """增量更新 — 刷新日历 + 下载新增/缺失股票的数据。

    已有数据的股票直接跳过，全缓存命中时秒级完成。

    Returns:
        dict: updated_count, total_stocks
    """
    if provider_uri is None:
        provider_uri = _CURRENT_URI or DEFAULT_URI

    _fix_windows_encoding()
    _ensure_deps()

    from .calendar_fetcher import fetch_trading_calendar, write_calendar_files
    from .instrument_fetcher import generate_a_stock_list
    from .feature_fetcher import download_stock_features, _calendar_index_map

    # Refresh calendar
    cal = fetch_trading_calendar(provider_uri)
    write_calendar_files(provider_uri, cal)
    cal_map = _calendar_index_map(cal)

    # Check all codes for missing features
    stock_df = generate_a_stock_list()
    codes = stock_df["code"].tolist()

    import time
    updated = 0
    for code in codes:
        feat_dir = Path(provider_uri) / "features" / code
        if (feat_dir / f"close.{freq}.bin").exists():
            continue
        try:
            download_stock_features(code, cal_map, provider_uri, freq, ohlcv_only)
            updated += 1
            if updated % 20 == 0:
                logger.info(f"  {updated} new stocks downloaded...")
        except Exception as e:
            logger.warning(f"  {code}: {e}")
        time.sleep(batch_delay)

    if updated > 0:
        _write_instruments_for_downloaded(provider_uri)

    logger.info(f"Update: {updated} new, {len(codes) - updated} unchanged")
    return {"updated_count": updated, "total_stocks": len(codes)}


def daily_sync(
    provider_uri: str = None,
    ohlcv_only: bool = True,
) -> dict:
    """每日盘后快速同步 — 重新拉取日历 + 最近交易日数据。

    只刷新日历和最近几个交易日的增量数据。比 update() 更快。

    Returns:
        dict: synced_count, calendar_updated
    """
    if provider_uri is None:
        provider_uri = _CURRENT_URI or DEFAULT_URI

    _fix_windows_encoding()
    _ensure_deps()

    from .calendar_fetcher import fetch_trading_calendar, write_calendar_files
    from .feature_fetcher import download_stock_features, _calendar_index_map

    # Refresh calendar
    cal = fetch_trading_calendar(provider_uri)
    write_calendar_files(provider_uri, cal)
    cal_map = _calendar_index_map(cal)

    # Only refresh stocks that already have data (skip new/unlisted codes)
    feat_root = Path(provider_uri) / "features"
    codes = sorted(
        d.name for d in feat_root.iterdir()
        if d.is_dir() and (d / "close.day.bin").exists()
    )

    import time
    synced = 0
    for code in codes:
        try:
            download_stock_features(code, cal_map, provider_uri, "day", ohlcv_only)
            synced += 1
        except Exception:
            pass
        time.sleep(0.05)  # light delay; same-day data is cached by mootdx

    logger.info(f"Daily sync: {synced}/{len(codes)} stocks refreshed")
    return {"synced_count": synced, "calendar_updated": True}


def is_initialized() -> bool:
    """Check if daa_qlib has been initialized."""
    return _CURRENT_URI is not None
