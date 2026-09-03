"""Vectorized next-open alignment. No fill on the signal bar."""

from __future__ import annotations

import pandas as pd

from ml_scan.storage.buckets import ensure_ist


def align_next_open(signals: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """Map each signal bar to the next completed hourly open of the same symbol.

    The fill bar is strictly after `asof_ts` / `ts`. Signal-bar OHLC is never used.
    """
    if signals.empty:
        out = signals.copy()
        out["fill_ts"] = pd.NaT
        out["fill_open"] = pd.NA
        return out
    sig = signals.copy()
    ts_col = "asof_ts" if "asof_ts" in sig.columns else "ts"
    sig[ts_col] = sig[ts_col].map(ensure_ist)
    bars = hourly[["symbol", "ts", "open"]].copy()
    bars["ts"] = bars["ts"].map(ensure_ist)
    bars = bars.sort_values(["symbol", "ts"])
    parts: list[pd.DataFrame] = []
    for symbol, part in sig.groupby("symbol", sort=False):
        cand = bars.loc[bars["symbol"] == symbol].sort_values("ts")
        left = part.sort_values(ts_col)
        if cand.empty:
            left = left.assign(fill_ts=pd.NaT, fill_open=pd.NA)
            parts.append(left)
            continue
        right = cand.rename(columns={"ts": "fill_ts", "open": "fill_open"}).drop(
            columns=["symbol"], errors="ignore"
        )
        merged = pd.merge_asof(
            left,
            right,
            left_on=ts_col,
            right_on="fill_ts",
            direction="forward",
            allow_exact_matches=False,
        )
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)
