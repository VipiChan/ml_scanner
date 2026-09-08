"""Load scan_trade training OHLCV from parquet. No Timescale connection.

The freeze-list dump (`train_ohlcv_60minute.parquet`) is hourly only. Daily bars
are resampled from those hours (including the 15:15 IST close print so the daily
close matches the session). 15-minute bars cannot be reconstructed from hourly
data; optional `train_ohlcv_15minute.parquet` / `train_ohlcv_day.parquet` in the
same folder are used when present.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_scan.data.schemas import MTFBundle
from ml_scan.exceptions import IngestError
from ml_scan.storage.buckets import ensure_ist, is_partial_hour_bucket

_OHLC = ("open", "high", "low", "close")
_ALIASES = {
    "date": "ts",
    "datetime": "ts",
    "timestamp": "ts",
    "ticker": "symbol",
    "tradingsymbol": "symbol",
    "vol": "volume",
}


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    rename = {c: _ALIASES[c.lower()] for c in work.columns if c.lower() in _ALIASES and c not in _ALIASES.values()}
    if rename:
        work = work.rename(columns=rename)
    missing = {"ts", "symbol", *_OHLC} - set(work.columns)
    if missing:
        raise IngestError(f"OHLCV parquet is missing columns {sorted(missing)}")
    work["ts"] = pd.to_datetime(work["ts"], utc=True, errors="coerce")
    work["ts"] = work["ts"].map(lambda ts: ensure_ist(ts) if pd.notna(ts) else pd.NaT)
    work["symbol"] = work["symbol"].astype(str).str.strip()
    for col in _OHLC:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if "volume" in work.columns:
        work["volume"] = pd.to_numeric(work["volume"], errors="coerce").fillna(0.0)
    else:
        work["volume"] = 0.0
    work = work.dropna(subset=["ts", "symbol", *_OHLC])
    work = work.sort_values(["symbol", "ts"]).drop_duplicates(["symbol", "ts"], keep="last")
    return work.reset_index(drop=True)


def drop_partial_hours(hourly: pd.DataFrame) -> pd.DataFrame:
    """Drop the 15:15–15:30 IST close print so the hourly panel matches TimescaleAdapter."""
    if hourly.empty:
        return hourly
    work = hourly.copy()
    if "is_partial_hour" in work.columns:
        flag = work["is_partial_hour"].astype(bool)
    else:
        flag = work["ts"].map(is_partial_hour_bucket)
    return work.loc[~flag].reset_index(drop=True)


def load_hourly_parquet(path: Path | str, *, drop_partial_hour: bool = False) -> pd.DataFrame:
    dest = Path(path)
    if not dest.is_file():
        raise IngestError(f"OHLCV parquet not found: {dest}")
    work = _normalize_columns(pd.read_parquet(dest))
    work["interval"] = "60minute"
    if "source" not in work.columns:
        work["source"] = "parquet"
    if drop_partial_hour:
        work = drop_partial_hours(work)
    return work


def hourly_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Session daily bars from hourly OHLCV. Uses every hour, including 15:15."""
    if hourly is None or hourly.empty:
        return pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume", "interval"])
    work = hourly.copy()
    work["ts"] = work["ts"].map(ensure_ist)
    work["session"] = work["ts"].map(lambda ts: ts.normalize())
    daily = (
        work.groupby(["symbol", "session"], sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
        .rename(columns={"session": "ts"})
    )
    daily["interval"] = "day"
    daily["source"] = "resampled_hourly"
    return daily


def load_optional_interval(path: Path, *, interval: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    work = _normalize_columns(pd.read_parquet(path))
    work["interval"] = interval
    if "source" not in work.columns:
        work["source"] = "parquet"
    return work


def load_training_bundle(
    hourly_path: Path | str,
    *,
    daily_path: Path | str | None = None,
    minute15_path: Path | str | None = None,
    benchmark_symbol: str = "NIFTY 50",
) -> tuple[pd.DataFrame, pd.DataFrame, MTFBundle]:
    """Return (hourly_for_features, benchmark_daily, mtf_bundle).

    `hourly_for_features` drops the 15:15 partial hour. The bundle's daily frame
    keeps that print so session close is correct. 15m is empty unless a parquet
    is supplied — do not invent 15-minute bars from hourly data.
    """
    raw = load_hourly_parquet(hourly_path, drop_partial_hour=False)
    hourly = drop_partial_hours(raw)
    hourly["interval"] = "60minute"

    daily_file = Path(daily_path) if daily_path else Path(hourly_path).with_name("train_ohlcv_day.parquet")
    daily = load_optional_interval(daily_file, interval="day")
    if daily.empty:
        daily = hourly_to_daily(raw)

    m15_file = Path(minute15_path) if minute15_path else Path(hourly_path).with_name("train_ohlcv_15minute.parquet")
    minute15 = load_optional_interval(m15_file, interval="15minute")

    bench = daily.loc[daily["symbol"] == benchmark_symbol].copy() if "symbol" in daily.columns else pd.DataFrame()
    bundle = MTFBundle(hourly=hourly, daily=daily, minute15=minute15)
    return hourly, bench, bundle
