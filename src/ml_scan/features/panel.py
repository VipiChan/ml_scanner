"""Per-symbol panel feature engineer. Never mix tickers in one TA pass."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from ml_scan.features.ta_engine import (
    ensure_atr,
    generate_all_ta_features,
    generate_lite_ta_features,
)
from ml_scan.storage.buckets import ensure_ist


class PanelFeatureEngineer:
    def __init__(self, *, mode: Literal["full", "lite"] = "lite", atr_period: int = 14) -> None:
        self.mode = mode
        self.atr_period = atr_period

    def transform(self, hourly: pd.DataFrame, *, group_col: str = "symbol") -> pd.DataFrame:
        frame = hourly.copy()
        if frame.empty:
            return frame
        if group_col not in frame.columns:
            raise ValueError(f"panel is missing {group_col!r}")
        if "ts" not in frame.columns:
            raise ValueError("panel is missing ts")
        frame["ts"] = frame["ts"].map(ensure_ist)
        parts: list[pd.DataFrame] = []
        for symbol, part in frame.groupby(group_col, sort=False):
            parts.append(self._one_symbol(str(symbol), part))
        out = pd.concat(parts, ignore_index=False)
        if not isinstance(out.index, pd.MultiIndex):
            out = out.set_index([group_col, "ts"], drop=False)
        return out.sort_index()

    def _one_symbol(self, symbol: str, part: pd.DataFrame) -> pd.DataFrame:
        work = part.sort_values("ts").copy()
        indexed = work.set_index("ts")
        ohlcv = indexed[["open", "high", "low", "close"]].copy()
        if "volume" in indexed.columns:
            ohlcv["volume"] = indexed["volume"]
        else:
            ohlcv["volume"] = 0.0
        engine = generate_all_ta_features if self.mode == "full" else generate_lite_ta_features
        featured = engine(ohlcv)
        featured = ensure_atr(featured, period=self.atr_period, col="ATR")
        extra = indexed.drop(columns=[c for c in ohlcv.columns if c in indexed.columns], errors="ignore")
        merged = featured.join(extra, how="left")
        if "symbol" not in merged.columns:
            merged["symbol"] = symbol
        merged = merged.reset_index().rename(columns={"index": "ts"})
        if "ts" not in merged.columns:
            merged["ts"] = work["ts"].to_numpy()
        merged["ts"] = merged["ts"].map(ensure_ist)
        merged["symbol"] = symbol
        return merged.set_index(["symbol", "ts"], drop=False)
