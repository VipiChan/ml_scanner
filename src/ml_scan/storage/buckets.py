"""NSE 09:15 IST time buckets, completed-bar cutoff, partial last hour.

Timescale `time_bucket(width, ts, origin)` is:

    origin + width * floor((ts - origin) / width)

Default Unix/UTC 60-minute buckets put 09:15 IST into 08:30–09:30 IST.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

NSE_TZ = "Asia/Kolkata"
CAGG_ORIGIN = pd.Timestamp("2000-01-03 09:15:00+05:30")
PARTIAL_HOUR = (15, 15)
SESSION_CLOSE = (15, 30)

IntervalName = Literal["5minute", "15minute", "60minute", "day"]

INTERVAL_WIDTH: dict[str, pd.Timedelta] = {
    "5minute": pd.Timedelta("5min"),
    "15minute": pd.Timedelta("15min"),
    "60minute": pd.Timedelta("60min"),
}


def ensure_ist(ts: pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        return stamp.tz_localize(NSE_TZ)
    return stamp.tz_convert(NSE_TZ)


def time_bucket(
    ts: pd.Timestamp,
    width: pd.Timedelta,
    origin: pd.Timestamp = CAGG_ORIGIN,
) -> pd.Timestamp:
    """Origin-aligned floor bucket, matching Timescale `time_bucket`."""
    ts_utc = ensure_ist(ts).tz_convert("UTC")
    origin_utc = ensure_ist(origin).tz_convert("UTC")
    width_ns = int(width.value)
    if width_ns <= 0:
        raise ValueError("bucket width must be positive")
    n = (ts_utc.value - origin_utc.value) // width_ns
    bucket = pd.Timestamp(origin_utc.value + n * width_ns, tz="UTC")
    return bucket.tz_convert(NSE_TZ)


def bucket_5m(ts: pd.Timestamp) -> pd.Timestamp:
    return time_bucket(ts, INTERVAL_WIDTH["5minute"])


def bucket_15m(ts: pd.Timestamp) -> pd.Timestamp:
    return time_bucket(ts, INTERVAL_WIDTH["15minute"])


def bucket_60m(ts: pd.Timestamp) -> pd.Timestamp:
    return time_bucket(ts, INTERVAL_WIDTH["60minute"])


def is_partial_hour_bucket(ts: pd.Timestamp) -> bool:
    """True for the terminal NSE hour 15:15–15:30 IST (15 minutes, not 60)."""
    local = ensure_ist(ts)
    return (local.hour, local.minute) == PARTIAL_HOUR


def session_close_on(ts: pd.Timestamp) -> pd.Timestamp:
    local = ensure_ist(ts)
    hour, minute = SESSION_CLOSE
    return local.replace(hour=hour, minute=minute, second=0, microsecond=0, nanosecond=0)


def _add_ns(ts: pd.Timestamp, nanoseconds: int) -> pd.Timestamp:
    utc = ensure_ist(ts).tz_convert("UTC")
    return pd.Timestamp(utc.value + nanoseconds, tz="UTC").tz_convert(NSE_TZ)


def bucket_end(ts: pd.Timestamp, interval: IntervalName) -> pd.Timestamp:
    """First instant at which this bar is closed (exclusive end of the bucket)."""
    start = ensure_ist(ts)
    if interval == "day":
        return session_close_on(start)
    if interval not in INTERVAL_WIDTH:
        raise ValueError(f"unsupported interval: {interval}")
    end = _add_ns(start, int(INTERVAL_WIDTH[interval].value))
    if interval == "60minute":
        return min(end, session_close_on(start))
    return end


def is_completed(
    ts: pd.Timestamp,
    interval: IntervalName,
    completed_asof: pd.Timestamp,
) -> bool:
    return bucket_end(ts, interval) <= ensure_ist(completed_asof)
