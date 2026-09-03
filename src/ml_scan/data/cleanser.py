"""Sort, tz-convert, drop duplicates. Never synthesize missing bars."""

from __future__ import annotations

import pandas as pd

from ml_scan.data.schemas import OHLCVRow
from ml_scan.storage.buckets import NSE_TZ, ensure_ist

_OHLC = ("open", "high", "low", "close")


class Cleanser:
    def to_frame(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Sort, tz IST, drop duplicates. Do not synthesize missing 5m/day bars."""
        if df is None or df.empty:
            return self._empty(interval)

        frame = df.copy()
        if "ts" not in frame.columns and "date" in frame.columns:
            frame = frame.rename(columns={"date": "ts"})
        if "datetime" in frame.columns and "ts" not in frame.columns:
            frame = frame.rename(columns={"datetime": "ts"})
        if "ts" not in frame.columns:
            raise ValueError("OHLCV frame needs a ts/date column")

        parsed = pd.to_datetime(frame["ts"])
        if getattr(parsed.dt, "tz", None) is None:
            frame["ts"] = parsed.dt.tz_localize(NSE_TZ)
        else:
            frame["ts"] = parsed.dt.tz_convert(NSE_TZ)
        if interval == "day":
            frame["ts"] = frame["ts"].dt.tz_convert(NSE_TZ).dt.normalize()

        if "volume" not in frame.columns:
            frame["volume"] = 0.0
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)

        for col in _OHLC:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=list(_OHLC))

        frame = frame.sort_values("ts")
        frame = frame.drop_duplicates(subset=["ts"], keep="last")
        if "symbol" in frame.columns:
            frame = frame.drop_duplicates(subset=["symbol", "ts"], keep="last")

        frame["interval"] = interval
        if "source" not in frame.columns:
            frame["source"] = "historical"
        return frame.reset_index(drop=True)

    def to_rows(self, df: pd.DataFrame, interval: str) -> list[OHLCVRow]:
        frame = self.to_frame(df, interval)
        rows: list[OHLCVRow] = []
        for rec in frame.itertuples(index=False):
            rows.append(
                OHLCVRow(
                    ts=ensure_ist(pd.Timestamp(rec.ts)),
                    symbol=str(getattr(rec, "symbol", "")),
                    instrument_token=int(getattr(rec, "instrument_token", 0) or 0),
                    open=float(rec.open),
                    high=float(rec.high),
                    low=float(rec.low),
                    close=float(rec.close),
                    volume=float(rec.volume),
                    interval=interval,
                    source=str(getattr(rec, "source", "historical")),
                )
            )
        return rows

    @staticmethod
    def _empty(interval: str) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "ts",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "symbol",
                "interval",
                "instrument_token",
                "source",
            ]
        )
