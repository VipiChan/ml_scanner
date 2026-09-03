"""Entry / ATR stop / take-profit. Scan preview uses bar close; backtest fills next open."""

from __future__ import annotations

from typing import Any

import pandas as pd


def compute_levels(
    row: pd.Series | dict[str, Any],
    *,
    sl_mult: float = 2.0,
    tp_r: float = 2.0,
    atr_col: str = "ATR",
    entry_col: str = "close",
) -> tuple[float, float, float]:
    entry = float(row[entry_col])
    atr = float(row[atr_col])
    if atr <= 0:
        atr = abs(entry) * 0.01
    sl = entry - sl_mult * atr
    tp = entry + sl_mult * tp_r * atr
    return entry, sl, tp
