"""NSE cash session calendar.

exchange_calendars has no `XNSE` name; India cash holidays live on `XBOM`
(BSE/NSE share the same holiday list in that package).
"""

from __future__ import annotations

import pandas as pd
from exchange_calendars import get_calendar

from ml_scan.storage.buckets import NSE_TZ, ensure_ist

_SESSION_START = (9, 15)
_BARS_5M = 75
_BARS_15M = 25
_BARS_60M = 7


def _naive_day(ts: pd.Timestamp) -> pd.Timestamp:
    local = ensure_ist(ts)
    return pd.Timestamp(year=local.year, month=local.month, day=local.day)


class NSECalendar:
    """XNSE sessions in Asia/Kolkata. Does not invent OHLCV."""

    def __init__(self) -> None:
        self._cal = get_calendar("XBOM")

    def session_days(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        sessions = self._cal.sessions_in_range(_naive_day(start), _naive_day(end))
        days = []
        for session in sessions:
            stamp = pd.Timestamp(session)
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize(NSE_TZ)
            else:
                stamp = stamp.tz_convert(NSE_TZ)
            days.append(stamp.normalize())
        return pd.DatetimeIndex(days)

    def is_trading_day(self, ts: pd.Timestamp) -> bool:
        return bool(self._cal.is_session(_naive_day(ts)))

    def minute5_index(self, day: pd.Timestamp) -> pd.DatetimeIndex:
        return self._intraday_index(day, _BARS_5M, "5min", _SESSION_START)

    def minute15_index(self, day: pd.Timestamp) -> pd.DatetimeIndex:
        return self._intraday_index(day, _BARS_15M, "15min", _SESSION_START)

    def hourly_index(self, day: pd.Timestamp) -> pd.DatetimeIndex:
        return self._intraday_index(day, _BARS_60M, "60min", _SESSION_START)

    def _intraday_index(
        self,
        day: pd.Timestamp,
        periods: int,
        freq: str,
        start_hm: tuple[int, int],
    ) -> pd.DatetimeIndex:
        if not self.is_trading_day(day):
            return pd.DatetimeIndex([], tz=NSE_TZ)
        local = ensure_ist(day).replace(
            hour=start_hm[0],
            minute=start_hm[1],
            second=0,
            microsecond=0,
            nanosecond=0,
        )
        return pd.date_range(start=local, periods=periods, freq=freq, tz=NSE_TZ)
