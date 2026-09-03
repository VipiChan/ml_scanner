"""Score the latest completed hour and emit entry / SL / TP."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_scan.config import Settings, load_settings
from ml_scan.data.liquidity import adtv_20
from ml_scan.execution.levels import compute_levels
from ml_scan.features.qc import feature_columns
from ml_scan.ml_engine.artifacts import load_model
from ml_scan.ml_engine.estimator import MLEstimator
from ml_scan.ml_engine.selector import load_feature_list
from ml_scan.storage.buckets import ensure_ist


class InferenceScanner:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        estimator: MLEstimator | None = None,
        features: list[str] | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.estimator = estimator
        self.features = features or []

    def load_artifacts(
        self,
        model_path: Path | str,
        features_path: Path | str | None = None,
    ) -> None:
        self.estimator = load_model(model_path)
        if features_path:
            self.features = load_feature_list(features_path)
        else:
            self.features = list(self.estimator.features_)

    def score_panel(self, panel: pd.DataFrame, daily: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.estimator is None:
            raise RuntimeError("InferenceScanner has no model; call load_artifacts() or pass estimator=")
        work = panel.reset_index(drop=True).copy()
        work["ts"] = work["ts"].map(ensure_ist)
        cols = self.features or feature_columns(work)
        X = work.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        proba = self.estimator.predict_proba(X)
        work["p_win"] = proba
        work["score"] = proba
        tgt = self.settings.target
        entries: list[float] = []
        stops: list[float] = []
        tps: list[float] = []
        for _, row in work.iterrows():
            entry, sl, tp = compute_levels(row, sl_mult=tgt.sl_mult, tp_r=tgt.tp_r, atr_col=tgt.atr_col)
            entries.append(entry)
            stops.append(sl)
            tps.append(tp)
        work["entry_px"] = entries
        work["sl_px"] = stops
        work["tp_px"] = tps
        work["atr"] = pd.to_numeric(work.get(tgt.atr_col), errors="coerce")
        if daily is not None and not daily.empty and "adtv_20" not in work.columns:
            adtv_map = {
                str(sym): adtv_20(part, self.settings.liquidity.adtv_window)
                for sym, part in daily.groupby("symbol")
            }
            work["adtv_20"] = work["symbol"].map(adtv_map)
        elif "adtv_20" not in work.columns:
            work["adtv_20"] = pd.NA
        return work

    def latest_scan(self, scored: pd.DataFrame) -> pd.DataFrame:
        if scored.empty:
            return scored
        latest = scored.sort_values("ts").groupby("symbol", as_index=False).tail(1)
        latest = latest.sort_values("score", ascending=False).reset_index(drop=True)
        latest["rank"] = latest.index + 1
        latest["asof_ts"] = latest["ts"]
        keep = [
            "symbol",
            "asof_ts",
            "score",
            "p_win",
            "entry_px",
            "sl_px",
            "tp_px",
            "atr",
            "adtv_20",
            "rank",
        ]
        return latest[[c for c in keep if c in latest.columns]]

    def run(
        self,
        asof: pd.Timestamp | str,
        panel: pd.DataFrame,
        daily: pd.DataFrame | None = None,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        work = panel.copy()
        if symbols:
            work = work.loc[work["symbol"].isin(symbols)]
        if str(asof) != "latest":
            cut = ensure_ist(pd.Timestamp(asof))
            work = work.loc[work["ts"].map(ensure_ist) <= cut]
        scored = self.score_panel(work, daily=daily)
        return self.latest_scan(scored)


def history_signals(scored: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    """Every hourly row that clears the score gate — used by the backtester."""
    work = scored.loc[scored["score"] >= threshold].copy()
    work["asof_ts"] = work["ts"].map(ensure_ist)
    work["signal"] = 1
    return work.sort_values(["ts", "score"], ascending=[True, False]).reset_index(drop=True)
