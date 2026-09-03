"""Small daily and 15m feature sets joined onto the hourly panel."""

from __future__ import annotations

import pandas as pd

from ml_scan.config import FeatureConfig
from ml_scan.data.mtf_aligner import MTFAligner
from ml_scan.data.schemas import MTFBundle
from ml_scan.storage.buckets import ensure_ist


def add_daily_features(
    daily: pd.DataFrame,
    *,
    benchmark: pd.DataFrame | None = None,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    cfg = config or FeatureConfig()
    if daily is None or daily.empty:
        return daily
    parts: list[pd.DataFrame] = []
    bench_ret = None
    if benchmark is not None and not benchmark.empty:
        b = benchmark.sort_values("ts").copy()
        b["ts"] = b["ts"].map(ensure_ist)
        bench_ret = b.set_index("ts")["close"].pct_change(cfg.daily_rs_lookback)
    for symbol, part in daily.groupby("symbol", sort=False):
        work = part.sort_values("ts").copy()
        close = work["close"].astype(float)
        work["d_ema50"] = close.ewm(span=cfg.daily_ema_fast, adjust=False).mean()
        work["d_ema200"] = close.ewm(span=cfg.daily_ema_slow, adjust=False).mean()
        work["d_trend"] = (work["d_ema50"] > work["d_ema200"]).astype(float)
        prev = close.shift(1)
        tr = pd.concat(
            [
                (work["high"] - work["low"]).abs(),
                (work["high"] - prev).abs(),
                (work["low"] - prev).abs(),
            ],
            axis=1,
        ).max(axis=1)
        work["d_atr"] = tr.rolling(cfg.atr_period, min_periods=cfg.atr_period).mean()
        stock_ret = close.pct_change(cfg.daily_rs_lookback)
        if bench_ret is not None:
            aligned = bench_ret.reindex(work["ts"].map(ensure_ist)).to_numpy()
            work["d_rs"] = stock_ret.to_numpy() - aligned
        else:
            work["d_rs"] = stock_ret
        if not work.empty:
            parts.append(work)
    if not parts:
        return daily
    return pd.concat(parts, ignore_index=True)


def add_m15_features(minute15: pd.DataFrame, *, config: FeatureConfig | None = None) -> pd.DataFrame:
    cfg = config or FeatureConfig()
    if minute15 is None or minute15.empty:
        return minute15
    parts: list[pd.DataFrame] = []
    for symbol, part in minute15.groupby("symbol", sort=False):
        work = part.sort_values("ts").copy()
        close = work["close"].astype(float)
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / cfg.m15_rsi_length, min_periods=cfg.m15_rsi_length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / cfg.m15_rsi_length, min_periods=cfg.m15_rsi_length, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        work["m15_rsi"] = 100 - (100 / (1 + rs))
        typical = (work["high"] + work["low"] + work["close"]) / 3.0
        vol = work["volume"].astype(float).clip(lower=0.0)
        cum_vol = vol.groupby(work["ts"].map(lambda ts: ensure_ist(ts).normalize())).cumsum()
        cum_tp = (typical * vol).groupby(work["ts"].map(lambda ts: ensure_ist(ts).normalize())).cumsum()
        vwap = cum_tp / cum_vol.replace(0, pd.NA)
        work["m15_vwap_dist"] = (close / vwap) - 1.0
        prev = close.shift(1)
        tr = pd.concat(
            [
                (work["high"] - work["low"]).abs(),
                (work["high"] - prev).abs(),
                (work["low"] - prev).abs(),
            ],
            axis=1,
        ).max(axis=1)
        work["m15_atr"] = tr.rolling(cfg.atr_period, min_periods=cfg.atr_period).mean()
        if not work.empty:
            parts.append(work)
    if not parts:
        return minute15
    return pd.concat(parts, ignore_index=True)


def join_htf_features(
    hourly_featured: pd.DataFrame,
    bundle: MTFBundle,
    *,
    benchmark_daily: pd.DataFrame | None = None,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Compute daily/15m extras then backward-asof them onto the hourly feature panel."""
    daily = add_daily_features(bundle.daily, benchmark=benchmark_daily, config=config)
    minute15 = add_m15_features(bundle.minute15, config=config)
    aligned = MTFAligner().align_to_hourly(MTFBundle(hourly=hourly_featured.reset_index(drop=True), daily=daily, minute15=minute15))
    extra_daily = [c for c in aligned.columns if c.startswith("d_") and c not in hourly_featured.columns]
    extra_m15 = [c for c in aligned.columns if c.startswith("m15_") and c not in hourly_featured.columns]
    keep = ["symbol", "ts"] + extra_daily + extra_m15
    keep = [c for c in keep if c in aligned.columns]
    right = aligned[keep].drop_duplicates(["symbol", "ts"])
    left = hourly_featured.reset_index(drop=True)
    left["ts"] = left["ts"].map(ensure_ist)
    right["ts"] = right["ts"].map(ensure_ist)
    merged = left.merge(right, on=["symbol", "ts"], how="left")
    return merged
