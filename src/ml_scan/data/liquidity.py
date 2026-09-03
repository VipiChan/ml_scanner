"""Daily ADTV liquidity filter. No network I/O."""

from __future__ import annotations

import math

import pandas as pd

from ml_scan.config import LiquidityConfig
from ml_scan.data.schemas import LiquidityRow


def adtv_20(daily: pd.DataFrame, window: int = 20) -> float:
    """Mean(close * volume) over the last `window` daily bars. NaN if too short."""
    if daily is None or daily.empty or window < 1:
        return math.nan
    frame = daily.sort_values("ts") if "ts" in daily.columns else daily
    if len(frame) < window:
        return math.nan
    tail = frame.tail(window)
    notional = tail["close"].astype(float) * tail["volume"].astype(float)
    return float(notional.mean())


def evaluate_liquidity(
    symbol: str,
    daily: pd.DataFrame,
    config: LiquidityConfig | None = None,
) -> LiquidityRow:
    cfg = config or LiquidityConfig()
    n = 0 if daily is None or daily.empty else int(len(daily))
    if n < cfg.adtv_window:
        return LiquidityRow(
            symbol=symbol,
            adtv_20=math.nan,
            n_daily_bars=n,
            passed=False,
            reason="insufficient_daily_bars",
        )
    value = adtv_20(daily, cfg.adtv_window)
    if math.isnan(value) or value < cfg.adtv_min_inr:
        return LiquidityRow(
            symbol=symbol,
            adtv_20=value,
            n_daily_bars=n,
            passed=False,
            reason="illiquid",
        )
    return LiquidityRow(
        symbol=symbol,
        adtv_20=value,
        n_daily_bars=n,
        passed=True,
        reason=None,
    )


def filter_universe(
    universe: pd.DataFrame,
    daily: pd.DataFrame,
    config: LiquidityConfig | None = None,
) -> pd.DataFrame:
    """Return universe rows that pass the ADTV gate, with `adtv_20` attached."""
    cfg = config or LiquidityConfig()
    if universe.empty:
        return universe.assign(adtv_20=pd.Series(dtype=float), passed=pd.Series(dtype=bool))
    if daily is None or daily.empty:
        out = universe.copy()
        out["adtv_20"] = math.nan
        out["passed"] = False
        out["liq_reason"] = "no_daily"
        return out

    rows: list[dict] = []
    for rec in universe.itertuples(index=False):
        symbol = str(rec.symbol)
        sym_daily = daily.loc[daily["symbol"] == symbol]
        eval_row = evaluate_liquidity(symbol, sym_daily, cfg)
        payload = rec._asdict() if hasattr(rec, "_asdict") else {c: getattr(rec, c) for c in universe.columns}
        payload["adtv_20"] = eval_row.adtv_20
        payload["passed"] = eval_row.passed
        payload["liq_reason"] = eval_row.reason
        rows.append(payload)
    frame = pd.DataFrame(rows)
    return frame.loc[frame["passed"]].reset_index(drop=True)
