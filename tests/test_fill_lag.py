from __future__ import annotations

import pandas as pd

from ml_scan.backtest.signals import align_next_open
from ml_scan.storage.buckets import NSE_TZ


def test_signal_bar_does_not_fill() -> None:
    hourly = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "ts": [
                pd.Timestamp("2024-01-02 09:15", tz=NSE_TZ),
                pd.Timestamp("2024-01-02 10:15", tz=NSE_TZ),
            ],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
        }
    )
    signals = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "asof_ts": [pd.Timestamp("2024-01-02 09:15", tz=NSE_TZ)],
            "score": [0.8],
            "sl_px": [95.0],
            "tp_px": [110.0],
        }
    )
    aligned = align_next_open(signals, hourly)
    assert aligned.iloc[0]["fill_ts"] == pd.Timestamp("2024-01-02 10:15", tz=NSE_TZ)
    assert float(aligned.iloc[0]["fill_open"]) == 101.0
    assert aligned.iloc[0]["fill_ts"] != aligned.iloc[0]["asof_ts"]
