"""Next-open entries; stop-wins same-bar exits; gap-through fills at the open."""

from __future__ import annotations

import pandas as pd

from ml_scan.config import Settings
from ml_scan.data.schemas import Signal


class FillModel:
    def __init__(self, settings: Settings) -> None:
        self.slip = float(settings.costs.slippage_pct_per_side)
        self.stop_wins = bool(settings.backtest.stop_wins_same_bar)
        self.exit_on_stop = bool(settings.risk.exit_on_stop)
        self.exit_on_target = bool(settings.risk.exit_on_target)

    def entry_fill(self, signal: Signal | None, next_bar: pd.Series) -> float:
        """Buy at next hourly open, slipped against the trader."""
        _ = signal
        return float(next_bar["open"]) * (1.0 + self.slip)

    def next_open_fill(self, next_bar: pd.Series) -> float:
        return float(next_bar["open"]) * (1.0 + self.slip)

    def exit_fill(
        self,
        bar: pd.Series,
        stop: float,
        target: float,
    ) -> tuple[float, str] | None:
        """Sell fill. Returns (slipped price, reason) or None if neither level hits."""
        open_px = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])
        hit_stop = self.exit_on_stop and low <= stop
        hit_target = self.exit_on_target and high >= target
        raw: float | None = None
        reason: str | None = None
        if hit_stop and (self.stop_wins or not hit_target):
            if open_px < stop:
                raw, reason = open_px, "gap_stop"
            else:
                raw, reason = float(stop), "stop"
        elif hit_target:
            if open_px > target:
                raw, reason = open_px, "gap_target"
            else:
                raw, reason = float(target), "target"
        if raw is None or reason is None:
            return None
        return raw * (1.0 - self.slip), reason
