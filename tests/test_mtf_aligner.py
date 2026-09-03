from __future__ import annotations

import pandas as pd

from ml_scan.data.mtf_aligner import MTFAligner
from ml_scan.data.schemas import MTFBundle
from ml_scan.storage.buckets import NSE_TZ


def _hourly(ts: str, close: float, symbol: str = "RELIANCE") -> dict:
    stamp = pd.Timestamp(ts, tz=NSE_TZ)
    return {
        "ts": stamp,
        "symbol": symbol,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1.0,
        "interval": "60minute",
        "source": "cagg",
        "instrument_token": 1,
    }


def _daily(day: str, close: float, symbol: str = "RELIANCE") -> dict:
    stamp = pd.Timestamp(day, tz=NSE_TZ).normalize()
    return {
        "ts": stamp,
        "symbol": symbol,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1.0,
        "interval": "day",
        "source": "kite_day",
        "instrument_token": 1,
    }


def test_same_session_daily_close_not_visible_at_open() -> None:
    hourly = pd.DataFrame(
        [
            _hourly("2024-01-02 09:15", 100),
            _hourly("2024-01-03 09:15", 101),
        ]
    )
    daily = pd.DataFrame(
        [
            _daily("2024-01-01", 90),
            _daily("2024-01-02", 999),
        ]
    )
    aligned = MTFAligner().align_to_hourly(MTFBundle(hourly=hourly, daily=daily, minute15=pd.DataFrame()))
    row_open = aligned.loc[aligned["ts"] == pd.Timestamp("2024-01-02 09:15", tz=NSE_TZ)].iloc[0]
    assert float(row_open["d_close"]) == 90.0
    row_next = aligned.loc[aligned["ts"] == pd.Timestamp("2024-01-03 09:15", tz=NSE_TZ)].iloc[0]
    assert float(row_next["d_close"]) == 999.0


def test_m15_asof_uses_completed_bars_only() -> None:
    hourly = pd.DataFrame([_hourly("2024-01-02 10:15", 100)])
    m15 = pd.DataFrame(
        [
            {
                "ts": pd.Timestamp("2024-01-02 10:15", tz=NSE_TZ),
                "symbol": "RELIANCE",
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 55,
                "volume": 1,
                "interval": "15minute",
                "source": "cagg",
                "instrument_token": 1,
            },
            {
                "ts": pd.Timestamp("2024-01-02 11:15", tz=NSE_TZ),
                "symbol": "RELIANCE",
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 77,
                "volume": 1,
                "interval": "15minute",
                "source": "cagg",
                "instrument_token": 1,
            },
        ]
    )
    aligned = MTFAligner().align_to_hourly(MTFBundle(hourly=hourly, daily=pd.DataFrame(), minute15=m15))
    # Hour 10:15 ends 11:15; the 10:15 15m bar ends 10:30 and is visible; 11:15 is not.
    assert float(aligned.iloc[0]["m15_close"]) == 55.0
