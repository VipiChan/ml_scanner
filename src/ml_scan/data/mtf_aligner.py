"""Backward asof joins of daily and 15m onto the hourly panel. No look-ahead."""

from __future__ import annotations

import pandas as pd

from ml_scan.data.schemas import MTFBundle
from ml_scan.storage.buckets import bucket_end, ensure_ist, session_close_on

_DAILY_KEEP = ["open", "high", "low", "close", "volume"]
_M15_KEEP = ["open", "high", "low", "close", "volume"]


class MTFAligner:
    def align_to_hourly(self, bundle: MTFBundle) -> pd.DataFrame:
        hourly = _require_symbol_ts(bundle.hourly, "hourly").copy()
        if hourly.empty:
            return hourly
        hourly["ts"] = hourly["ts"].map(ensure_ist)
        hourly = hourly.sort_values(["symbol", "ts"]).reset_index(drop=True)

        daily = self._prepare_daily(bundle.daily)
        minute15 = self._prepare_m15(bundle.minute15)

        parts: list[pd.DataFrame] = []
        for symbol, h_part in hourly.groupby("symbol", sort=False):
            left = h_part.sort_values("ts")
            d_part = daily.loc[daily["symbol"] == symbol].sort_values("available_ts")
            if not d_part.empty:
                left = pd.merge_asof(
                    left,
                    d_part.drop(columns=["symbol"]),
                    left_on="ts",
                    right_on="available_ts",
                    direction="backward",
                    suffixes=("", "_djoin"),
                )
                left = left.drop(columns=["available_ts"], errors="ignore")
            else:
                for col in [f"d_{c}" for c in _DAILY_KEEP] + ["d_ts"]:
                    left[col] = pd.NA

            m_part = minute15.loc[minute15["symbol"] == symbol].sort_values("available_ts")
            if not m_part.empty:
                left = left.assign(_asof=left["ts"].map(lambda ts: bucket_end(ts, "60minute")))
                left = left.sort_values("_asof")
                left = pd.merge_asof(
                    left,
                    m_part.drop(columns=["symbol"]),
                    left_on="_asof",
                    right_on="available_ts",
                    direction="backward",
                    suffixes=("", "_mjoin"),
                )
                left = left.drop(columns=["available_ts", "_asof"], errors="ignore")
            else:
                for col in [f"m15_{c}" for c in _M15_KEEP] + ["m15_ts"]:
                    left[col] = pd.NA
            parts.append(left)
        return pd.concat(parts, ignore_index=True).sort_values(["symbol", "ts"]).reset_index(drop=True)

    @staticmethod
    def _prepare_daily(daily: pd.DataFrame) -> pd.DataFrame:
        if daily is None or daily.empty:
            return pd.DataFrame(columns=["symbol", "available_ts", "d_ts"] + [f"d_{c}" for c in _DAILY_KEEP])
        frame = _require_symbol_ts(daily, "daily").copy()
        frame["ts"] = frame["ts"].map(ensure_ist)
        out = pd.DataFrame({"symbol": frame["symbol"], "d_ts": frame["ts"]})
        for col in _DAILY_KEEP:
            if col in frame.columns:
                out[f"d_{col}"] = frame[col]
        for col in frame.columns:
            if col.startswith("d_") and col not in out.columns:
                out[col] = frame[col]
        # Daily bar is only visible after that session's 15:30 IST close.
        out["available_ts"] = out["d_ts"].map(session_close_on)
        return out

    @staticmethod
    def _prepare_m15(minute15: pd.DataFrame) -> pd.DataFrame:
        if minute15 is None or minute15.empty:
            return pd.DataFrame(columns=["symbol", "available_ts", "m15_ts"] + [f"m15_{c}" for c in _M15_KEEP])
        frame = _require_symbol_ts(minute15, "15m").copy()
        frame["ts"] = frame["ts"].map(ensure_ist)
        out = pd.DataFrame({"symbol": frame["symbol"], "m15_ts": frame["ts"]})
        for col in _M15_KEEP:
            if col in frame.columns:
                out[f"m15_{col}"] = frame[col]
        for col in frame.columns:
            if col.startswith("m15_") and col not in out.columns:
                out[col] = frame[col]
        # 15m bar usable once its bucket ends.
        out["available_ts"] = out["m15_ts"].map(lambda ts: bucket_end(ts, "15minute"))
        return out


def _require_symbol_ts(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["symbol", "ts"])
    missing = {"symbol", "ts"} - set(frame.columns)
    if missing:
        raise ValueError(f"{name} frame missing columns {sorted(missing)}")
    return frame
