"""Trading calendar fetcher — derives trading days from mootdx K-line dates, writes qlib-format calendars."""
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def _derive_calendar_from_mootdx(start_date: str = "2005-01-01") -> pd.DatetimeIndex:
    """Derive trading calendar from mootdx K-line dates.

    Fetches K-lines for a long-listed benchmark stock (000001 平安银行, listed since 1991)
    and extracts all trading dates from the datetime column. This is the most reliable
    method since mootdx uses TCP protocol that bypasses HTTP proxies.

    Falls back to weekday generation if mootdx is unavailable.
    """
    try:
        from mootdx.quotes import Quotes
        client = Quotes.factory(market="std")
        # 000001 平安银行 — listed since 1991-04-03, one of the oldest A-share stocks
        bars = client.bars(symbol="000001", category=4, offset=5000)
        if bars is not None and len(bars) > 0:
            df = pd.DataFrame(bars)
            date_col = "datetime" if "datetime" in df.columns else "date"
            dates = pd.DatetimeIndex(pd.to_datetime(df[date_col]).sort_values().unique())
            dates = dates[dates >= pd.Timestamp(start_date)]
            logger.info(f"Derived {len(dates)} trading days from mootdx (000001 K-lines)")
            if len(dates) > 100:
                return dates
    except Exception as e:
        logger.warning(f"mootdx calendar derivation failed: {e}")

    # Fallback: generate all weekdays
    logger.warning("Using weekday-based calendar (no holiday filtering)")
    all_days = pd.date_range(start=start_date, end=pd.Timestamp.now(), freq="B")
    logger.info(f"Generated {len(all_days)} weekdays as calendar")
    return all_days


def fetch_trading_calendar(provider_uri: str, start_date: str = "2005-01-01") -> pd.DatetimeIndex:
    """Fetch A-share trading calendar.

    Primary: derives from mootdx K-line dates (TCP, no proxy issues).
    Fallback: generates all weekdays.

    Returns a DatetimeIndex of trading days, sorted.
    """
    logger.info("Deriving trading calendar...")
    cal = _derive_calendar_from_mootdx(start_date=start_date)
    logger.info(f"Calendar: {len(cal)} days from {cal[0].strftime('%Y-%m-%d')} to {cal[-1].strftime('%Y-%m-%d')}")
    return cal


def write_calendar_files(provider_uri: str, cal: pd.DatetimeIndex) -> None:
    """Write trading calendar to qlib-format text files.

    Writes calendars/day.txt under provider_uri (one date per line, YYYY-MM-DD).
    Also writes calendars/day_future.txt for the next 30 expected trading days.
    """
    cal_dir = Path(provider_uri) / "calendars"
    cal_dir.mkdir(parents=True, exist_ok=True)

    day_file = cal_dir / "day.txt"
    dates = sorted(d.strftime("%Y-%m-%d") for d in cal)
    day_file.write_text("\n".join(dates) + "\n", encoding="utf-8")
    logger.info(f"Wrote {len(dates)} dates to {day_file}")

    # Future: estimate next 30 expected trading days (skip weekends)
    future_dates = []
    last = cal[-1]
    while len(future_dates) < 30:
        last += pd.Timedelta(days=1)
        if last.dayofweek < 5:
            future_dates.append(last.strftime("%Y-%m-%d"))
    future_file = cal_dir / "day_future.txt"
    future_file.write_text("\n".join(future_dates) + "\n", encoding="utf-8")
    logger.info(f"Wrote {len(future_dates)} future dates to {future_file}")
