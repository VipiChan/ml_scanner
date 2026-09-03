"""Single SQL door for OHLCV. Scan/backtest must not query Timescale directly."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Literal

import pandas as pd

from ml_scan.data.schemas import OHLCVRow
from ml_scan.exceptions import StorageError
from ml_scan.storage.buckets import (
    IntervalName,
    bucket_end,
    ensure_ist,
    is_partial_hour_bucket,
)
from ml_scan.storage.db import Database

_ALLOWED_READ: frozenset[str] = frozenset({"5minute", "15minute", "60minute", "day"})
_WATERMARK_INTERVALS: frozenset[str] = frozenset({"5minute", "day"})
_TABLE = {
    "5minute": "ohlcv_5m",
    "15minute": "ohlcv_15m",
    "60minute": "ohlcv_60m",
    "day": "ohlcv_1d",
}
_UPSERT_SQL = """
INSERT INTO {table} (
  ts, symbol, instrument_token, open, high, low, close, volume, source
) VALUES (
  %(ts)s, %(symbol)s, %(instrument_token)s, %(open)s, %(high)s, %(low)s,
  %(close)s, %(volume)s, %(source)s
)
ON CONFLICT (symbol, ts) DO UPDATE SET
  instrument_token = EXCLUDED.instrument_token,
  open = EXCLUDED.open,
  high = EXCLUDED.high,
  low = EXCLUDED.low,
  close = EXCLUDED.close,
  volume = EXCLUDED.volume,
  source = EXCLUDED.source
"""

_OHLCV_COLUMNS = [
    "ts",
    "symbol",
    "instrument_token",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "interval",
    "source",
]


class OhlcvRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def upsert_5m(self, rows: Sequence[OHLCVRow]) -> int:
        return self._upsert("ohlcv_5m", rows, expected_interval="5minute")

    def upsert_1d(self, rows: Sequence[OHLCVRow]) -> int:
        return self._upsert("ohlcv_1d", rows, expected_interval="day")

    def read(
        self,
        symbols: list[str],
        interval: Literal["5minute", "15minute", "60minute", "day"],
        start: pd.Timestamp,
        end: pd.Timestamp,
        completed_asof: pd.Timestamp | None = None,
        *,
        drop_partial_hour: bool = True,
    ) -> pd.DataFrame:
        if interval not in _ALLOWED_READ:
            raise StorageError(f"unsupported read interval: {interval}")
        if not symbols:
            return self._empty_frame(interval)

        table = _TABLE[interval]
        start_ist = ensure_ist(start)
        end_ist = ensure_ist(end)
        extra = ", n_5m" if interval in {"15minute", "60minute"} else ""
        source_sql = (
            "'cagg' AS source"
            if interval in {"15minute", "60minute"}
            else "source"
        )
        sql = f"""
            SELECT ts, symbol, instrument_token, open, high, low, close, volume,
                   {source_sql}{extra}
            FROM {table}
            WHERE symbol = ANY(%s)
              AND ts >= %s
              AND ts <= %s
            ORDER BY symbol, ts
        """
        with self._db.connection() as conn:
            rows = conn.execute(sql, (symbols, start_ist, end_ist)).fetchall()

        frame = pd.DataFrame(rows)
        if frame.empty:
            return self._empty_frame(interval)

        frame["ts"] = pd.to_datetime(frame["ts"], utc=True).dt.tz_convert("Asia/Kolkata")
        frame["interval"] = interval
        if "source" not in frame.columns:
            frame["source"] = "cagg"
        if completed_asof is not None:
            asof = ensure_ist(completed_asof)
            mask = frame["ts"].map(lambda ts: bucket_end(ts, interval) <= asof)
            frame = frame.loc[mask].reset_index(drop=True)
        if interval == "60minute":
            frame["is_partial_hour"] = frame["ts"].map(is_partial_hour_bucket)
            if drop_partial_hour and not frame.empty:
                frame = frame.loc[~frame["is_partial_hour"]].reset_index(drop=True)
        elif "is_partial_hour" not in frame.columns:
            frame["is_partial_hour"] = False
        return frame.reset_index(drop=True)

    def watermark(self, symbol: str, interval: str) -> pd.Timestamp | None:
        self._check_watermark_interval(interval)
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT last_ts FROM ingest_watermark WHERE symbol = %s AND interval = %s",
                (symbol, interval),
            ).fetchone()
        if not row:
            return None
        return ensure_ist(pd.Timestamp(row["last_ts"]))

    def set_watermark(self, symbol: str, interval: str, ts: pd.Timestamp) -> None:
        self._check_watermark_interval(interval)
        stamp = ensure_ist(ts)
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO ingest_watermark (symbol, interval, last_ts, updated_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (symbol, interval) DO UPDATE SET
                  last_ts = EXCLUDED.last_ts,
                  updated_at = now()
                """,
                (symbol, interval, stamp),
            )
            conn.commit()

    def ca_flagged(self, symbol: str, session: date) -> bool:
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT flagged FROM ca_flags WHERE symbol = %s AND session_date = %s",
                (symbol, session),
            ).fetchone()
        if not row:
            return False
        return bool(row["flagged"])

    def list_flagged(self, start: date, end: date) -> set[tuple[str, date]]:
        with self._db.connection() as conn:
            rows = conn.execute(
                """
                SELECT symbol, session_date FROM ca_flags
                WHERE flagged AND session_date >= %s AND session_date <= %s
                """,
                (start, end),
            ).fetchall()
        out: set[tuple[str, date]] = set()
        for row in rows:
            session = row["session_date"]
            if hasattr(session, "date") and not isinstance(session, date):
                session = session.date()
            out.add((str(row["symbol"]), session))
        return out

    def upsert_ca_flag(
        self,
        symbol: str,
        session_date: date,
        kite_close: float,
        rollup_close: float | None,
        abs_rel_diff: float | None,
        flagged: bool,
    ) -> None:
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO ca_flags (
                  symbol, session_date, kite_close, rollup_close, abs_rel_diff, flagged
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, session_date) DO UPDATE SET
                  kite_close = EXCLUDED.kite_close,
                  rollup_close = EXCLUDED.rollup_close,
                  abs_rel_diff = EXCLUDED.abs_rel_diff,
                  flagged = EXCLUDED.flagged
                """,
                (symbol, session_date, kite_close, rollup_close, abs_rel_diff, flagged),
            )
            conn.commit()

    def list_timestamps(
        self,
        symbol: str,
        interval: Literal["5minute", "day"],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[pd.Timestamp]:
        if interval not in {"5minute", "day"}:
            raise StorageError("list_timestamps only supports 5minute and day")
        table = _TABLE[interval]
        with self._db.connection() as conn:
            rows = conn.execute(
                f"SELECT ts FROM {table} WHERE symbol = %s AND ts >= %s AND ts <= %s",
                (symbol, ensure_ist(start), ensure_ist(end)),
            ).fetchall()
        return [ensure_ist(pd.Timestamp(row["ts"])) for row in rows]

    def replace_gaps(
        self,
        symbol: str,
        interval: str,
        missing: Sequence[pd.Timestamp],
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> int:
        start_ist = ensure_ist(start)
        end_ist = ensure_ist(end)
        with self._db.connection() as conn:
            conn.execute(
                """
                DELETE FROM ingest_gaps
                WHERE symbol = %s AND interval = %s
                  AND ts_missing >= %s AND ts_missing <= %s
                """,
                (symbol, interval, start_ist, end_ist),
            )
            if missing:
                conn.cursor().executemany(
                    """
                    INSERT INTO ingest_gaps (symbol, interval, ts_missing)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (symbol, interval, ts_missing) DO NOTHING
                    """,
                    [(symbol, interval, ensure_ist(ts)) for ts in missing],
                )
            conn.commit()
        return len(missing)

    def refresh_caggs(
        self,
        window_start: pd.Timestamp | None = None,
        window_end: pd.Timestamp | None = None,
    ) -> None:
        start = None if window_start is None else ensure_ist(window_start).to_pydatetime()
        end = None if window_end is None else ensure_ist(window_end).to_pydatetime()
        with self._db.connection() as conn:
            previous = conn.autocommit
            conn.autocommit = True
            try:
                for view in ("ohlcv_15m", "ohlcv_60m"):
                    conn.execute(
                        f"CALL refresh_continuous_aggregate('{view}', "
                        "%s::timestamptz, %s::timestamptz)",
                        (start, end),
                    )
            finally:
                conn.autocommit = previous

    def _upsert(
        self,
        table: str,
        rows: Sequence[OHLCVRow],
        *,
        expected_interval: str,
    ) -> int:
        if not rows:
            return 0
        payloads = []
        for row in rows:
            if row.interval != expected_interval:
                raise StorageError(
                    f"expected interval {expected_interval!r}, got {row.interval!r} "
                    f"for {row.symbol}"
                )
            payloads.append(
                {
                    "ts": ensure_ist(row.ts),
                    "symbol": row.symbol,
                    "instrument_token": int(row.instrument_token),
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                    "volume": float(row.volume),
                    "source": row.source,
                }
            )
        sql = _UPSERT_SQL.format(table=table)
        with self._db.connection() as conn:
            conn.cursor().executemany(sql, payloads)
            conn.commit()
        return len(payloads)

    @staticmethod
    def _check_watermark_interval(interval: str) -> None:
        if interval not in _WATERMARK_INTERVALS:
            raise StorageError(
                f"watermark interval must be 5minute or day, not {interval!r} "
                "(15m/60m are continuous aggregates)"
            )

    @staticmethod
    def _empty_frame(interval: IntervalName) -> pd.DataFrame:
        cols = list(_OHLCV_COLUMNS)
        frame = pd.DataFrame(columns=cols)
        frame["is_partial_hour"] = pd.Series(dtype=bool)
        if interval in {"15minute", "60minute"}:
            frame["n_5m"] = pd.Series(dtype="int64")
        frame["interval"] = interval
        return frame
