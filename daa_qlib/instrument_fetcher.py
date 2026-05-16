"""Instrument (stock list) fetcher — generates A-share codes from known ranges, writes qlib-format instrument files.

Uses static A-share code ranges — no HTTP requests needed, works completely offline.
For a more precise list, call validate_stocks() to verify each code against mootdx.
"""
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

STOCK_LIST = "all.txt"

# A-share stock code ranges (沪深京)
# These are the main active ranges as of 2026.
A_SHARE_RANGES = [
    # Shanghai main board: 600xxx, 601xxx, 603xxx, 605xxx
    (600000, 606000),
    # Shanghai STAR (科创板): 688xxx
    (688000, 690000),
    # Shenzhen main board: 000xxx-003xxx
    (0, 4000),
    # Shenzhen GEM (创业板): 300xxx-302xxx
    (300000, 302000),
    # Beijing (北交所): 83xxxx-87xxxx
    (830000, 877000),
]

# Approximate count of valid codes in each range (for progress reporting)
_TOTAL_ESTIMATE = 5800


def generate_a_stock_list() -> pd.DataFrame:
    """Generate all potential A-share stock codes from known ranges.

    Returns DataFrame with columns: code (6-digit zero-padded), name

    Note: This generates ~6000 codes, including some that may be inactive/delisted.
    qlib gracefully handles missing data for non-existent stocks.
    To validate, call validate_stocks() which checks each code against mootdx.
    """
    codes = []
    for lo, hi in A_SHARE_RANGES:
        codes.extend(str(c).zfill(6) for c in range(lo, hi + 1))

    logger.info(f"Generated {len(codes)} potential A-share codes from static ranges")
    df = pd.DataFrame({"code": codes, "name": ["A" + c for c in codes]})
    return df


def validate_stocks(df: pd.DataFrame, max_check: int = None) -> pd.DataFrame:
    """Validate stock codes by checking if mootdx returns K-line data for them.

    A stock is considered valid if mootdx returns at least 1 day of K-line data.
    This is slow (0.3-0.5s per stock via TCP), so it's optional.

    Args:
        df: DataFrame from generate_a_stock_list()
        max_check: Limit how many to validate (None = all)

    Returns filtered DataFrame with only valid/listed stocks.
    """
    try:
        from mootdx.quotes import Quotes
    except ImportError:
        logger.warning("mootdx not available; skipping validation")
        return df

    client = Quotes.factory(market="std")
    codes = df["code"].tolist()
    if max_check:
        codes = codes[:max_check]

    valid = []
    total = len(codes)
    for i, code in enumerate(codes):
        try:
            bars = client.bars(symbol=code, category=4, offset=1)
            if bars is not None and len(bars) > 0:
                valid.append(code)
        except Exception:
            pass
        if (i + 1) % 200 == 0:
            logger.info(f"  validated {i+1}/{total}: {len(valid)} listed so far")

    logger.info(f"Validation complete: {len(valid)}/{total} stocks are listed")
    return df[df["code"].isin(valid)].copy()


def write_instrument_files(provider_uri: str, df: pd.DataFrame = None) -> pd.DataFrame:
    """Write A-share instrument list to qlib-format tab-separated files.

    Writes instruments/all.txt with columns: instrument, start_datetime, end_datetime

    Returns the instrument DataFrame used.
    """
    if df is None:
        df = generate_a_stock_list()

    inst_dir = Path(provider_uri) / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    for _, row in df.iterrows():
        code = str(row["code"]).zfill(6)[:6]
        lines.append(f"{code}\t1990-01-01\t{today}")

    all_file = inst_dir / STOCK_LIST
    all_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Wrote {len(lines)} instruments to {all_file}")
    return df
