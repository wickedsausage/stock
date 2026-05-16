"""
daa_qlib — 数据源_大A → qlib 数据桥接层

让 qlib 全部通过"数据源_大A"的方案来调取数据。
一行初始化：

    import daa_qlib
    daa_qlib.init("./qlib_data/")

数据流向：
    数据源_大A (mootdx/腾讯/akshare) → qlib 本地格式 (.bin/.txt) → qlib LocalProvider

Usage:
    import daa_qlib
    daa_qlib.init()                    # 默认 ./daa_qlib_data/
    daa_qlib.init("./my_data/")        # 指定目录
    daa_qlib.update("./qlib_data/")    # 增量更新
"""
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Default to ASCII path to avoid Windows + joblib Unicode issues with Chinese paths
DEFAULT_URI = "C:/qlib_data/"
_CURRENT_URI = None
_QLIB_IMPORTED = False


def _ensure_qlib_path():
    """Ensure qlib is importable — qlib submodule has nested qlib/qlib/ structure.

    The qlib git submodule root (outer dir) has no __init__.py.
    When added to sys.path, `import qlib` finds the inner qlib/qlib/ package.
    """
    import sys
    from pathlib import Path
    outer_qlib = Path(__file__).parent.parent / "qlib"
    if outer_qlib.is_dir() and str(outer_qlib) not in sys.path:
        sys.path.insert(0, str(outer_qlib))


def _import_qlib():
    """Import qlib module with proper path configuration."""
    global _QLIB_IMPORTED
    if not _QLIB_IMPORTED:
        _ensure_qlib_path()
        # Clear any cached namespace imports
        import sys
        for m in list(sys.modules):
            if m.startswith("qlib"):
                del sys.modules[m]
        _QLIB_IMPORTED = True
    import qlib
    return qlib


def _ensure_deps():
    """Verify all required packages are installed."""
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
        provider_uri: 数据缓存目录（qlib 数据根目录）
        freq: K线频率 ("day", "1min", "5min" 等)
        stock_count: 下载股票数量（None=全市场, 指定数字=前N只）
        ohlcv_only: True=仅行情, False=含估值(PE/PB等)
        batch_delay: 股票间延迟秒数，防被封
        auto_register: 是否自动调用 qlib.init() 注册

    Returns:
        dict with keys: provider_uri, instrument_count, feature_status, qlib_registered
    """
    _ensure_deps()

    from .calendar_fetcher import fetch_trading_calendar, write_calendar_files
    from .instrument_fetcher import generate_a_stock_list, write_instrument_files
    from .feature_fetcher import download_all_features

    global _CURRENT_URI
    _CURRENT_URI = provider_uri

    logger.info(f"=== daa_qlib init: provider_uri={provider_uri} ===")

    # Step 1: Calendar (via mootdx K-line dates, no HTTP needed)
    logger.info("Step 1/3: Deriving trading calendar from mootdx...")
    cal = fetch_trading_calendar(provider_uri)
    write_calendar_files(provider_uri, cal)

    # Step 2: Instruments (from static A-share ranges)
    logger.info("Step 2/3: Generating stock list...")
    stock_df = generate_a_stock_list()
    if stock_count is not None and stock_count > 0:
        stock_df = stock_df.head(stock_count)
    write_instrument_files(provider_uri, stock_df)

    # Step 3: Features
    codes = stock_df["code"].tolist()
    logger.info(f"Step 3/3: Downloading features for {len(codes)} stocks...")
    feat_summary = download_all_features(
        provider_uri=provider_uri,
        codes=codes,
        cal=cal,
        freq=freq,
        ohlcv_only=ohlcv_only,
        batch_delay=batch_delay,
    )

    result = {
        "provider_uri": provider_uri,
        "instrument_count": len(codes),
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
            logger.info("qlib.init() succeeded")
            from qlib.data import D
            logger.info(f"  D instruments: {len(D.instruments(market='all'))} stocks")
            logger.info(f"  D calendar: {D.calendar()[0]} ~ {D.calendar()[-1]}")
        except Exception as e:
            logger.warning(f"qlib.init() failed: {e}")
            logger.warning(f"You can manually run: qlib.init(provider_uri='{provider_uri}')")

    logger.info("=== daa_qlib init complete ===")
    return result


def update(
    provider_uri: str = None,
    freq: str = "day",
    ohlcv_only: bool = False,
    batch_delay: float = 0.3,
) -> dict:
    """增量更新 — 只下载新增的股票和缺失的数据。

    Args:
        provider_uri: 数据目录（默认用上次 init 的目录）
        freq: K线频率
        ohlcv_only: 仅行情
        batch_delay: 请求间隔

    Returns:
        dict: updated_count, new_stocks, etc.
    """
    if provider_uri is None:
        provider_uri = _CURRENT_URI or DEFAULT_URI

    _ensure_deps()

    from .calendar_fetcher import fetch_trading_calendar, write_calendar_files
    from .instrument_fetcher import generate_a_stock_list, write_instrument_files
    from .feature_fetcher import download_stock_features, _calendar_index_map

    # Refresh calendar
    cal = fetch_trading_calendar(provider_uri)
    write_calendar_files(provider_uri, cal)
    cal_map = _calendar_index_map(cal)

    # Refresh instruments
    stock_df = generate_a_stock_list()
    write_instrument_files(provider_uri, stock_df)
    codes = stock_df["code"].tolist()

    # Only download missing features
    import time
    updated = 0
    for i, code in enumerate(codes):
        feat_dir = Path(provider_uri) / "features" / code
        if (feat_dir / f"close.{freq}.bin").exists():
            continue
        try:
            download_stock_features(code, cal_map, provider_uri, freq, ohlcv_only)
            updated += 1
            if updated % 20 == 0:
                logger.info(f"Updated {updated} new stocks...")
        except Exception as e:
            logger.warning(f"Update failed for {code}: {e}")
        time.sleep(batch_delay)

    logger.info(f"Update complete: {updated} new stocks downloaded, {len(codes) - updated} unchanged")
    return {"updated_count": updated, "total_stocks": len(codes)}


def is_initialized() -> bool:
    """Check if daa_qlib has been initialized."""
    return _CURRENT_URI is not None
