"""Hyperparameter search that reuses the purged walk-forward folds as CV.

`RandomizedSearchCV` accepts `cv` as a plain iterable of (train_idx, test_idx)
positional-index tuples, so we hand it exactly the same purged, embargoed
folds used everywhere else in this project instead of sklearn's default
random/contiguous K-fold (which would leak future rows into training and
inflate the score used to pick hyperparameters).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from ml_scan.exceptions import ModelError
from ml_scan.ml_engine.estimator import _build_model
from ml_scan.ml_engine.metrics import sample_weight_vector
from ml_scan.ml_engine.splitter import PurgedWalkForward

PARAM_DISTRIBUTIONS: dict[str, dict[str, list[Any]]] = {
    "lightgbm": {
        "n_estimators": [100, 150, 200, 300, 400],
        "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
        "max_depth": [3, 4, 5, 6, -1],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [5, 10, 20, 40],
        "subsample": [0.6, 0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "reg_lambda": [0.0, 0.1, 1.0, 5.0],
    },
    "xgboost": {
        "n_estimators": [100, 150, 200, 300, 400],
        "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
        "max_depth": [3, 4, 5, 6],
        "subsample": [0.6, 0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 10],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    },
    "rf": {
        "n_estimators": [200, 300, 400, 600],
        "max_depth": [4, 6, 8, 10, None],
        "min_samples_leaf": [1, 2, 4, 8],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", 0.5, 0.8],
    },
}


def _grid_size(grid: dict[str, list[Any]]) -> int:
    size = 1
    for values in grid.values():
        size *= max(1, len(values))
    return size


def random_search(
    X: pd.DataFrame,
    y: pd.Series,
    panel: pd.DataFrame,
    *,
    model_name: str,
    splitter: PurgedWalkForward | None = None,
    n_iter: int = 20,
    random_state: int = 42,
    scoring: str = "roc_auc",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Randomized hyperparameter search scored on purged walk-forward folds.

    `X`, `y`, `panel` must share the same 0..n-1 row order (see
    `ml_scan.features.qc.select_xy`). Returns (best_params, cv_results) where
    `cv_results` is the sklearn cv_results_ table sorted by rank, trimmed to
    the searched params plus mean/std/rank test score.
    """
    splitter = splitter or PurgedWalkForward()
    folds = list(splitter.split(panel))
    if len(folds) < 2:
        raise ModelError("Need at least 2 purged folds to run a hyperparameter search")
    base = _build_model(model_name, random_state)
    grid = PARAM_DISTRIBUTIONS.get(model_name, {})
    if not grid:
        raise ModelError(f"No hyperparameter grid defined for model '{model_name}'")
    search = RandomizedSearchCV(
        base,
        param_distributions=grid,
        n_iter=min(n_iter, _grid_size(grid)),
        scoring=scoring,
        cv=folds,
        random_state=random_state,
        n_jobs=1,
        refit=False,
        error_score="raise",
    )
    search.fit(X.to_numpy(dtype=float), y.to_numpy(), sample_weight=sample_weight_vector(y))
    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").reset_index(drop=True)
    keep_cols = [c for c in results.columns if c.startswith("param_")]
    keep_cols += [c for c in ("mean_test_score", "std_test_score", "rank_test_score") if c in results.columns]
    return dict(search.best_params_), results[keep_cols]
