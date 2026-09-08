from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_scan.data.parquet_source import drop_partial_hours, hourly_to_daily, load_hourly_parquet
from ml_scan.storage.buckets import NSE_TZ


def _hourly_frame() -> pd.DataFrame:
    day = pd.Timestamp("2024-01-02", tz=NSE_TZ)
    stamps = [
        day.replace(hour=9, minute=15),
        day.replace(hour=10, minute=15),
        day.replace(hour=14, minute=15),
        day.replace(hour=15, minute=15),
    ]
    return pd.DataFrame(
        {
            "ts": stamps,
            "symbol": ["AAA"] * 4,
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [101.0, 102.0, 103.0, 104.0],
            "volume": [10.0, 20.0, 30.0, 40.0],
        }
    )


def test_drop_partial_hours_removes_1515() -> None:
    kept = drop_partial_hours(_hourly_frame())
    times = kept["ts"].dt.strftime("%H:%M").tolist()
    assert "15:15" not in times
    assert len(kept) == 3


def test_hourly_to_daily_uses_session_close_print(tmp_path: Path) -> None:
    daily = hourly_to_daily(_hourly_frame())
    assert len(daily) == 1
    row = daily.iloc[0]
    assert float(row["open"]) == 100.0
    assert float(row["high"]) == 105.0
    assert float(row["low"]) == 99.0
    assert float(row["close"]) == 104.0
    assert float(row["volume"]) == 100.0
    assert row["interval"] == "day"


def test_load_hourly_parquet_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "train_ohlcv_60minute.parquet"
    _hourly_frame().to_parquet(path, index=False)
    loaded = load_hourly_parquet(path, drop_partial_hour=True)
    assert set(loaded["symbol"]) == {"AAA"}
    assert (loaded["interval"] == "60minute").all()
    assert not loaded["ts"].dt.strftime("%H:%M").eq("15:15").any()
    assert {"ts", "symbol", "open", "high", "low", "close", "volume"} <= set(loaded.columns)
