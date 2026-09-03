"""Price-action swing labels. Time expiry is not a winning class."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_scan.features.ta_engine import ensure_atr
from ml_scan.storage.buckets import ensure_ist

BARS_PER_SESSION = 7


class SwingLabeler:
    def __init__(
        self,
        *,
        atr_col: str = "ATR",
        sl_mult: float = 2.0,
        tp_r: float = 2.0,
        max_sessions: int = 10,
        stop_wins_same_bar: bool = True,
    ) -> None:
        self.atr_col = atr_col
        self.sl_mult = sl_mult
        self.tp_r = tp_r
        self.max_sessions = max_sessions
        self.stop_wins_same_bar = stop_wins_same_bar
        self.max_bars = int(max_sessions * BARS_PER_SESSION)

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        frame = panel.reset_index(drop=True).copy() if not isinstance(panel.index, pd.RangeIndex) else panel.copy()
        if "symbol" not in frame.columns or "ts" not in frame.columns:
            raise ValueError("panel needs symbol and ts columns")
        frame["ts"] = frame["ts"].map(ensure_ist)
        if frame.empty:
            frame = frame.copy()
            frame["y"] = pd.Series(dtype="int64")
            frame["y_reason"] = pd.Series(dtype="object")
            return frame
        parts: list[pd.DataFrame] = []
        for symbol, part in frame.groupby("symbol", sort=False):
            parts.append(self._label_symbol(str(symbol), part))
        if not parts:
            frame = frame.copy()
            frame["y"] = pd.Series(dtype="int64")
            frame["y_reason"] = pd.Series(dtype="object")
            return frame
        return pd.concat(parts, ignore_index=True).sort_values(["symbol", "ts"]).reset_index(drop=True)

    def _label_symbol(self, symbol: str, part: pd.DataFrame) -> pd.DataFrame:
        work = part.sort_values("ts").reset_index(drop=True)
        work = ensure_atr(work, col=self.atr_col)
        n = len(work)
        y = np.full(n, -1, dtype=int)
        reason = np.array(["unresolved"] * n, dtype=object)
        mfe = np.full(n, np.nan)
        mae = np.full(n, np.nan)
        bars_to = np.full(n, np.nan)
        highs = work["high"].to_numpy(dtype=float)
        lows = work["low"].to_numpy(dtype=float)
        closes = work["close"].to_numpy(dtype=float)
        atrs = work[self.atr_col].to_numpy(dtype=float)
        horizon = self.max_bars
        for i in range(n):
            atr = atrs[i]
            if not np.isfinite(atr) or atr <= 0:
                continue
            entry = closes[i]
            sl = entry - self.sl_mult * atr
            tp = entry + self.sl_mult * self.tp_r * atr
            end = min(n, i + 1 + horizon)
            best = 0.0
            worst = 0.0
            hit = None
            hit_j = None
            for j in range(i + 1, end):
                best = max(best, highs[j] - entry)
                worst = min(worst, lows[j] - entry)
                hit_sl = lows[j] <= sl
                hit_tp = highs[j] >= tp
                if hit_sl and hit_tp:
                    hit = "sl_hit" if self.stop_wins_same_bar else "tp_hit"
                    hit_j = j
                    break
                if hit_sl:
                    hit = "sl_hit"
                    hit_j = j
                    break
                if hit_tp:
                    hit = "tp_hit"
                    hit_j = j
                    break
            mfe[i] = best / entry if entry else np.nan
            mae[i] = abs(worst) / entry if entry else np.nan
            if hit is None:
                if end >= n:
                    reason[i] = "unresolved"
                    y[i] = -1
                else:
                    reason[i] = "unresolved"
                    y[i] = -1
                bars_to[i] = float(end - i - 1)
            else:
                reason[i] = hit
                y[i] = 1 if hit == "tp_hit" else 0
                bars_to[i] = float(hit_j - i)
        work["y"] = y
        work["y_reason"] = reason
        work["fwd_mfe"] = mfe
        work["fwd_mae"] = mae
        work["bars_to_outcome"] = bars_to
        work.loc[work["y"] < 0, "y"] = np.nan
        return work
