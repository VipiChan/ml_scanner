"""Boruta then VIF dimensionality reduction."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ml_scan.exceptions import ModelError
from ml_scan.ml_engine.metrics import class_weight_dict, scale_pos_weight


BORUTA_PARAMS: dict[str, dict[str, object]] = {
    "xgboost": dict(n_estimators=80, max_depth=4, learning_rate=0.08, n_jobs=1, eval_metric="logloss"),
    "lightgbm": dict(n_estimators=80, max_depth=4, learning_rate=0.08, n_jobs=1, verbose=-1),
    "rf": dict(n_estimators=200, max_depth=6, n_jobs=1, class_weight="balanced"),
}


def _boruta_estimator(name: str, random_state: int, y=None):
    """Shadow-feature estimator for Boruta. Falls back to XGBoost for unknown names.

    BorutaPy.fit(X, y) does not accept sample_weight, so imbalance is handled on
    the estimator itself: class_weight for RF/LightGBM, scale_pos_weight for XGBoost.
    """
    key = name if name in BORUTA_PARAMS else "xgboost"
    params = dict(BORUTA_PARAMS[key], random_state=random_state)
    if y is not None:
        if key == "xgboost":
            params["scale_pos_weight"] = scale_pos_weight(y)
        else:
            params["class_weight"] = class_weight_dict(y)
    if key == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(**params)
    if key == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**params)
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(**params)


class BorutaSelector:
    def __init__(self, *, max_iter: int = 50, random_state: int = 42, estimator_name: str = "xgboost") -> None:
        self.max_iter = max_iter
        self.random_state = random_state
        self.estimator_name = estimator_name
        self.features_: list[str] = []
        self.ranks_: dict[str, int] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BorutaSelector:
        if X.empty or y.empty:
            raise ModelError("Boruta needs a non-empty X/y")
        matrix = X.to_numpy(dtype=float)
        labels = y.to_numpy()
        try:
            from boruta import BorutaPy

            model = _boruta_estimator(self.estimator_name, self.random_state, y=labels)
            selector = BorutaPy(
                model,
                n_estimators="auto",
                verbose=0,
                random_state=self.random_state,
                max_iter=self.max_iter,
            )
            selector.fit(matrix, labels)
            keep = [c for c, k in zip(X.columns, selector.support_) if k]
            self.ranks_ = {c: int(r) for c, r in zip(X.columns, selector.ranking_)}
            if not keep:
                ranked = sorted(self.ranks_.items(), key=lambda kv: kv[1])
                keep = [c for c, _ in ranked[: min(20, len(ranked))]]
            self.features_ = keep
        except Exception:
            forest = RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=self.random_state,
                n_jobs=1,
                class_weight=class_weight_dict(labels),
            )
            forest.fit(matrix, labels)
            importances = pd.Series(forest.feature_importances_, index=X.columns)
            keep = importances.sort_values(ascending=False).head(min(20, len(importances)))
            self.features_ = list(keep.index)
            self.ranks_ = {c: i + 1 for i, c in enumerate(self.features_)}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.features_ if c in X.columns]
        if not cols:
            raise ModelError("BorutaSelector has no fitted features")
        return X[cols]

    def save(self, path: Path | str) -> Path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps(
                {"features": self.features_, "ranks": self.ranks_, "estimator_name": self.estimator_name},
                indent=2,
            ),
            encoding="utf-8",
        )
        return dest


class VIFPruner:
    def __init__(self, *, max_vif: float = 10.0, max_rows: int = 8000) -> None:
        self.max_vif = max_vif
        self.max_rows = max_rows
        self.features_: list[str] = []
        self.history_: list[dict] = []

    def fit(self, X: pd.DataFrame, *, max_vif: float | None = None) -> VIFPruner:
        cap = float(self.max_vif if max_vif is None else max_vif)
        work = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        nunique = work.nunique()
        cols = [c for c in work.columns if nunique.get(c, 0) > 1]
        work = work[cols]
        if self.max_rows and len(work) > self.max_rows:
            work = work.sample(n=self.max_rows, random_state=42)
        cols = _drop_corr_pairs(work, cols, threshold=0.95)
        work = work[cols]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            while len(cols) > 2:
                matrix = work[cols].to_numpy(dtype=float)
                vifs: list[tuple[str, float]] = []
                for i, col in enumerate(cols):
                    try:
                        value = float(variance_inflation_factor(matrix, i))
                    except Exception:
                        value = float("inf")
                    if not np.isfinite(value):
                        value = float("inf")
                    vifs.append((col, value))
                vifs.sort(key=lambda kv: kv[1], reverse=True)
                worst_col, worst_vif = vifs[0]
                self.history_.append({"dropped": None, "worst": worst_col, "vif": worst_vif, "n": len(cols)})
                if not np.isfinite(worst_vif) or worst_vif > cap:
                    cols.remove(worst_col)
                    work = work[cols]
                    self.history_[-1]["dropped"] = worst_col
                    continue
                break
        self.features_ = cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.features_ if c in X.columns]
        if not cols:
            raise ModelError("VIFPruner has no fitted features")
        return X[cols]

    def save(self, path: Path | str) -> Path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"features": self.features_, "history": self.history_}, indent=2), encoding="utf-8")
        return dest


def _drop_corr_pairs(work: pd.DataFrame, cols: list[str], *, threshold: float) -> list[str]:
    """Drop one of each highly correlated pair so VIF is not rank-deficient."""
    if len(cols) < 2:
        return cols
    corr = work[cols].corr().abs()
    keep = list(cols)
    dropped: set[str] = set()
    for i, a in enumerate(cols):
        if a in dropped:
            continue
        for b in cols[i + 1 :]:
            if b in dropped:
                continue
            value = corr.loc[a, b]
            if pd.notna(value) and float(value) >= threshold:
                dropped.add(b)
    return [c for c in keep if c not in dropped]


def load_feature_list(path: Path | str) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(x) for x in payload]
    return [str(x) for x in payload.get("features", [])]
