"""Shared dataclasses. Do not pass ad-hoc dicts across module boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class OHLCVRow:
    ts: pd.Timestamp
    symbol: str
    instrument_token: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str
    source: str


@dataclass(frozen=True)
class UniverseRow:
    symbol: str
    isin: str
    industry: str
    company: str
    series: str
    instrument_token: int
    exchange: str


@dataclass(frozen=True)
class LiquidityRow:
    symbol: str
    adtv_20: float
    n_daily_bars: int
    passed: bool
    reason: str | None


@dataclass
class MTFBundle:
    hourly: pd.DataFrame
    daily: pd.DataFrame
    minute15: pd.DataFrame


@dataclass(frozen=True)
class ScanRow:
    symbol: str
    asof_ts: pd.Timestamp
    score: float
    p_win: float
    entry_px: float
    sl_px: float
    tp_px: float
    atr: float
    adtv_20: float
    rank: int


@dataclass(frozen=True)
class Signal:
    symbol: str
    setup: str
    bar_ts: pd.Timestamp
    entry: float
    stop: float
    target: float
    atr: float
    r_multiple: float
    score: float
    invalid_reason: str | None


@dataclass(frozen=True)
class Trade:
    symbol: str
    setup: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_px: float
    exit_px: float
    qty: int
    stop: float
    target: float
    exit_reason: str
    pnl_gross: float
    costs: float
    slippage: float
    pnl_net: float
    r_multiple_realized: float


@dataclass
class BacktestResult:
    run_ts: pd.Timestamp
    config_hash: str
    trades: tuple[Trade, ...] = ()
    equity: pd.Series | None = None
    metrics: dict[str, float] | None = None
    blotter: pd.DataFrame | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
