"""Read facade over the existing scan_trade Timescale warehouse."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from ml_scan.config import Settings
from ml_scan.data.schemas import MTFBundle
from ml_scan.storage.buckets import NSE_TZ, ensure_ist
from ml_scan.storage.db import Database
from ml_scan.storage.repository import OhlcvRepository

ReadInterval = Literal["15minute", "60minute", "day"]


class TimescaleAdapter:
    def __init__(self, settings: Settings, *, database: Database | None = None) -> None:
        self._settings = settings
        self._owns = database is None
        self._db = database or Database.from_settings(settings)
        self._repo = OhlcvRepository(self._db)

    def close(self) -> None:
        if self._owns:
            self._db.close()

    def read(
        self,
        symbols: list[str],
        interval: ReadInterval,
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
        *,
        completed_asof: pd.Timestamp | None = None,
        drop_partial_hour: bool = True,
    ) -> pd.DataFrame:
        return self._repo.read(
            list(symbols),
            interval,
            ensure_ist(pd.Timestamp(start)),
            ensure_ist(pd.Timestamp(end)),
            completed_asof=completed_asof,
            drop_partial_hour=drop_partial_hour,
        )

    def read_mtf(
        self,
        symbols: list[str],
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
        *,
        completed_asof: pd.Timestamp | None = None,
        minute15_lookback_days: int | None = None,
    ) -> MTFBundle:
        start_ts = ensure_ist(pd.Timestamp(start))
        end_ts = ensure_ist(pd.Timestamp(end))
        m15_days = minute15_lookback_days or self._settings.data.minute15_days
        m15_start = end_ts - pd.Timedelta(days=int(m15_days))
        if m15_start < start_ts:
            m15_start = start_ts
        daily_start = start_ts - pd.Timedelta(days=40)
        return MTFBundle(
            hourly=self.read(symbols, "60minute", start_ts, end_ts, completed_asof=completed_asof),
            daily=self.read(symbols, "day", daily_start, end_ts, completed_asof=completed_asof),
            minute15=self.read(symbols, "15minute", m15_start, end_ts, completed_asof=completed_asof),
        )

    def coverage(
        self,
        symbols: list[str],
        interval: ReadInterval,
    ) -> pd.DataFrame:
        rows: list[dict] = []
        start = pd.Timestamp("2015-01-01", tz=NSE_TZ)
        end = pd.Timestamp.now(tz=NSE_TZ)
        frame = self.read(symbols, interval, start, end)
        if frame.empty:
            return pd.DataFrame(columns=["symbol", "interval", "n_bars", "first_ts", "last_ts"])
        grouped = frame.groupby("symbol", sort=True)
        for symbol, part in grouped:
            rows.append(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "n_bars": int(len(part)),
                    "first_ts": part["ts"].min(),
                    "last_ts": part["ts"].max(),
                }
            )
        return pd.DataFrame(rows)
