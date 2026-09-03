"""Offline smoke of features → labels → model → next-open backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_scan.backtest.engine import Backtester
from ml_scan.backtest.signals import align_next_open
from ml_scan.config import Settings
from ml_scan.data.schemas import MTFBundle
from ml_scan.execution.levels import compute_levels
from ml_scan.execution.scanner import InferenceScanner, history_signals
from ml_scan.features.panel import PanelFeatureEngineer
from ml_scan.features.qc import training_xy
from ml_scan.features.target import SwingLabeler
from ml_scan.ml_engine.estimator import MLEstimator
from ml_scan.ml_engine.selector import BorutaSelector, VIFPruner
from ml_scan.storage.buckets import NSE_TZ


def _synth_hourly(symbol: str, start: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq="h", tz=NSE_TZ)
    close = 100 + np.cumsum(rng.normal(0, 0.4, n))
    return pd.DataFrame(
        {
            "ts": ts,
            "symbol": symbol,
            "open": close - 0.1,
            "high": close + 0.6,
            "low": close - 0.6,
            "close": close,
            "volume": rng.integers(1_000, 5_000, n).astype(float),
            "interval": "60minute",
            "source": "synth",
            "instrument_token": 1,
        }
    )


def test_offline_feature_label_train_and_fill() -> None:
    hourly = pd.concat(
        [
            _synth_hourly("AAA", "2024-01-02 09:15", 220, 1),
            _synth_hourly("BBB", "2024-01-02 09:15", 220, 2),
        ],
        ignore_index=True,
    )
    featured = PanelFeatureEngineer(mode="lite").transform(hourly).reset_index(drop=True)
    assert "ATR" in featured.columns
    assert set(featured["symbol"]) == {"AAA", "BBB"}
    labeled = SwingLabeler(max_sessions=5).transform(featured)
    assert labeled["y_reason"].isin(["tp_hit", "sl_hit", "unresolved"]).all()
    X, y = training_xy(labeled)
    assert len(X) > 50
    keep = BorutaSelector(max_iter=5, random_state=42).fit(X, y).features_
    X2 = X[keep]
    pruned = VIFPruner(max_vif=20.0).fit(X2)
    X3 = pruned.transform(X2)
    est = MLEstimator(model="lightgbm", random_state=42).fit(X3, y.loc[X3.index])
    scanner = InferenceScanner(Settings(), estimator=est, features=list(X3.columns))
    scored = scanner.score_panel(labeled)
    assert {"score", "entry_px", "sl_px", "tp_px"} <= set(scored.columns)
    latest = scanner.latest_scan(scored)
    assert {"symbol", "asof_ts", "score", "entry_px", "sl_px", "tp_px"} <= set(latest.columns)
    hist = history_signals(scored, threshold=0.0).head(15)
    aligned = align_next_open(hist, hourly)
    assert aligned["fill_ts"].notna().any()
    settings = Settings()
    result = Backtester(settings).run(hist, MTFBundle(hourly=hourly, daily=pd.DataFrame(), minute15=pd.DataFrame()))
    assert result.equity is not None
    assert result.blotter is not None


def test_compute_levels_r_multiple() -> None:
    entry, sl, tp = compute_levels({"close": 100.0, "ATR": 2.0}, sl_mult=2.0, tp_r=2.0)
    assert entry == 100.0
    assert sl == 96.0
    assert tp == 108.0
