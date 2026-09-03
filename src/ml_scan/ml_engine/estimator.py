"""Tree-model estimator with optional walk-forward fold metrics."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from ml_scan.exceptions import ModelError
from ml_scan.ml_engine.metrics import (
    aggregate_fold_metrics,
    class_weight_dict,
    classification_metrics,
    sample_weight_vector,
)
from ml_scan.ml_engine.splitter import PurgedWalkForward

MODEL_CHOICES: tuple[str, ...] = ("lightgbm", "xgboost", "rf")
MODEL_LABELS: dict[str, str] = {
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "rf": "RandomForest",
}


def _default_params(name: Literal["lightgbm", "xgboost", "rf"], random_state: int) -> dict[str, Any]:
    if name == "lightgbm":
        return dict(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=1,
            verbose=-1,
        )
    if name == "xgboost":
        return dict(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=1,
            eval_metric="logloss",
        )
    return dict(
        n_estimators=300,
        max_depth=6,
        random_state=random_state,
        n_jobs=1,
    )


def _build_model(
    name: Literal["lightgbm", "xgboost", "rf"],
    random_state: int,
    params: dict[str, Any] | None = None,
):
    """Build a fresh, unfitted estimator. `params` overrides only the keys it sets."""
    merged = _default_params(name, random_state)
    if params:
        merged.update(params)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**merged)
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(**merged)
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(**merged)


class MLEstimator:
    def __init__(
        self,
        *,
        model: Literal["lightgbm", "xgboost", "rf"] = "lightgbm",
        random_state: int = 42,
        params: dict[str, Any] | None = None,
        use_class_weight: bool = True,
    ) -> None:
        self.model_name = model
        self.random_state = random_state
        self.params = dict(params) if params else {}
        self.use_class_weight = use_class_weight
        self._model = _build_model(model, random_state, self.params)
        self.features_: list[str] = []
        self.fold_metrics_: list[dict[str, Any]] = []
        self.class_weight_: dict[int, float] = {}

    def _weights_for(self, y: pd.Series | np.ndarray) -> tuple[dict[int, float], np.ndarray | None]:
        if not self.use_class_weight:
            return {}, None
        mapping = class_weight_dict(y)
        return mapping, sample_weight_vector(y, mapping)

    def fit(self, X: pd.DataFrame, y: pd.Series, *, sample_weight=None) -> MLEstimator:
        if X.empty:
            raise ModelError("MLEstimator.fit received an empty X")
        self.features_ = list(X.columns)
        mapping, auto_sw = self._weights_for(y)
        self.class_weight_ = mapping
        sw = sample_weight if sample_weight is not None else auto_sw
        self._model.fit(X.to_numpy(dtype=float), y.to_numpy(), sample_weight=sw)
        return self

    def fit_walk_forward(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        panel: pd.DataFrame,
        splitter: PurgedWalkForward | None = None,
    ) -> MLEstimator:
        splitter = splitter or PurgedWalkForward()
        self.features_ = list(X.columns)
        self.fold_metrics_ = []
        last_train_idx: np.ndarray | None = None
        aligned_panel = panel.loc[X.index] if not X.index.equals(panel.index) else panel
        for fold, (train_idx, test_idx) in enumerate(splitter.split(aligned_panel), start=1):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            if X_train.empty or X_test.empty:
                continue
            mapping, sw = self._weights_for(y_train)
            self.class_weight_ = mapping
            model = _build_model(self.model_name, self.random_state, self.params)
            model.fit(X_train.to_numpy(dtype=float), y_train.to_numpy(), sample_weight=sw)
            proba = model.predict_proba(X_test.to_numpy(dtype=float))[:, 1]
            row = {"fold": fold, "n_train": int(len(X_train)), "n_test": int(len(X_test))}
            if mapping:
                row["class_weight_0"] = float(mapping[0])
                row["class_weight_1"] = float(mapping[1])
            row.update(classification_metrics(y_test, proba))
            self.fold_metrics_.append(row)
            last_train_idx = train_idx
            self._model = model
        if last_train_idx is None:
            return self.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.features_:
            raise ModelError("MLEstimator is not fitted")
        matrix = X.reindex(columns=self.features_, fill_value=0.0).to_numpy(dtype=float)
        return self._model.predict_proba(matrix)[:, 1]

    def feature_importances(self) -> pd.Series:
        if hasattr(self._model, "feature_importances_"):
            return pd.Series(self._model.feature_importances_, index=self.features_).sort_values(ascending=False)
        return pd.Series(dtype=float)

    def aggregate_metrics(self) -> dict[str, float]:
        return aggregate_fold_metrics(self.fold_metrics_)


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    panel: pd.DataFrame,
    *,
    models: tuple[str, ...] = MODEL_CHOICES,
    splitter: PurgedWalkForward | None = None,
    random_state: int = 42,
    use_class_weight: bool = True,
) -> tuple[pd.DataFrame, dict[str, MLEstimator]]:
    """Fit every candidate model on the same purged walk-forward folds.

    Returns (summary_df, fitted_estimators). `summary_df` has one row per
    model with the fold-weighted average of every metric plus n_folds, so
    the notebook can print it directly and pick the winner.
    """
    splitter = splitter or PurgedWalkForward()
    rows: list[dict[str, Any]] = []
    fitted: dict[str, MLEstimator] = {}
    for name in models:
        est = MLEstimator(model=name, random_state=random_state, use_class_weight=use_class_weight)
        est.fit_walk_forward(X, y, panel, splitter)
        summary = est.aggregate_metrics()
        rows.append({"model": MODEL_LABELS.get(name, name), "model_key": name, **summary})
        fitted[name] = est
    summary_df = pd.DataFrame(rows).set_index("model")
    return summary_df, fitted


def select_best_model(summary_df: pd.DataFrame, *, metric: str = "roc_auc") -> str:
    """Return the `model_key` with the highest metric, falling back to accuracy if all NaN."""
    if summary_df.empty:
        raise ModelError("compare_models produced no rows")
    ranked = summary_df[metric]
    if ranked.notna().any():
        return str(summary_df.loc[ranked.idxmax(), "model_key"])
    fallback = summary_df["accuracy"]
    return str(summary_df.loc[fallback.idxmax(), "model_key"])
