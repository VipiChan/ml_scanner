from __future__ import annotations

import pandas as pd
import pytest

from ml_scan.storage.buckets import NSE_TZ


@pytest.fixture
def hourly_fixture() -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 09:15", periods=40, freq="60min", tz=NSE_TZ)
    # Keep only NSE-like hours 09:15-14:15
    idx = idx[(idx.hour > 9) | ((idx.hour == 9) & (idx.minute >= 15))]
    idx = idx[idx.hour < 16]
    close = 100 + pd.Series(range(len(idx)), dtype=float)
    return pd.DataFrame(
        {
            "ts": idx,
            "symbol": "TEST",
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 10_000.0,
        }
    )
