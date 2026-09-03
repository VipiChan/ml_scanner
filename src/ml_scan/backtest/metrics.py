"""Backtest performance metrics. Daily equity path; trade PnL is net of costs."""

from __future__ import annotations

import math

import pandas as pd

from ml_scan.data.schemas import Trade


def _nan() -> float:
    return math.nan


def compute_metrics(
    equity: pd.Series,
    trades: tuple[Trade, ...] | list[Trade],
    rf_annual: float = 0.0,
    periods: int = 252,
    *,
    starting_equity: float | None = None,
    exposed: pd.Series | None = None,
) -> dict[str, float]:
    n_trades = float(len(trades))
    empty = {
        "cagr": _nan(),
        "sharpe": _nan(),
        "sortino": _nan(),
        "max_dd": _nan(),
        "win_rate": 0.0 if n_trades == 0 else _nan(),
        "profit_factor": _nan(),
        "n_trades": n_trades,
        "avg_r": _nan(),
        "exposure": 0.0,
    }
    if equity is None or len(equity) == 0:
        return empty

    series = pd.Series(equity).astype(float).dropna()
    if series.empty:
        return empty
    # Collapse intraday equity to daily last for annualized stats.
    if getattr(series.index, "tz", None) is not None or hasattr(series.index, "normalize"):
        daily = series.groupby(series.index.map(lambda ts: pd.Timestamp(ts).tz_convert("Asia/Kolkata").normalize() if getattr(pd.Timestamp(ts), "tzinfo", None) else pd.Timestamp(ts).normalize())).last()
    else:
        daily = series
    e0 = float(starting_equity) if starting_equity is not None else float(daily.iloc[0])
    e_end = float(daily.iloc[-1])
    n_days = len(daily)
    t_years = n_days / float(periods) if periods else 0.0
    if t_years <= 0:
        cagr = _nan()
    elif e_end <= 0 or e0 <= 0:
        cagr = -1.0 if e_end <= 0 else _nan()
    else:
        cagr = (e_end / e0) ** (1.0 / t_years) - 1.0

    rets = daily.pct_change().dropna()
    daily_rf = float(rf_annual) / float(periods) if periods else 0.0
    excess = rets - daily_rf
    std = float(excess.std(ddof=1)) if len(excess) >= 2 else math.nan
    mean_ex = float(excess.mean()) if len(excess) else math.nan
    sharpe = mean_ex / std * math.sqrt(periods) if std and std > 0 and math.isfinite(std) else _nan()

    neg = excess[excess < 0]
    if len(neg) == 0:
        sortino = _nan()
    else:
        down = float(neg.std(ddof=1)) if len(neg) >= 2 else float(neg.std(ddof=0))
        sortino = (
            mean_ex / down * math.sqrt(periods) if down and down > 0 and math.isfinite(down) else _nan()
        )

    peak = daily.cummax()
    dd = daily / peak - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0

    wins = [t for t in trades if t.pnl_net > 0]
    losses = [t for t in trades if t.pnl_net < 0]
    win_rate = (len(wins) / len(trades)) if trades else 0.0
    gross_win = sum(t.pnl_net for t in wins)
    gross_loss = abs(sum(t.pnl_net for t in losses))
    if not trades:
        profit_factor = _nan()
    elif gross_loss == 0:
        profit_factor = math.inf if gross_win > 0 else _nan()
    else:
        profit_factor = gross_win / gross_loss
    avg_r = (
        float(sum(t.r_multiple_realized for t in trades) / len(trades)) if trades else _nan()
    )
    if exposed is None or len(exposed) == 0:
        exposure = 0.0
    else:
        exposure = float(pd.Series(exposed).astype(bool).mean())

    return {
        "cagr": float(cagr) if math.isfinite(cagr) or math.isnan(cagr) else cagr,
        "sharpe": float(sharpe) if not math.isnan(sharpe) else _nan(),
        "sortino": float(sortino) if not math.isnan(sortino) else _nan(),
        "max_dd": max_dd,
        "win_rate": float(win_rate),
        "profit_factor": profit_factor,
        "n_trades": n_trades,
        "avg_r": avg_r,
        "exposure": exposure,
    }


def equity_to_frame(equity: pd.Series) -> pd.DataFrame:
    series = pd.Series(equity).astype(float).sort_index()
    peak = series.cummax()
    return pd.DataFrame(
        {
            "ts": series.index,
            "equity": series.to_numpy(),
            "drawdown": (series / peak - 1.0).to_numpy(),
        }
    )
