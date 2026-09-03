"""Shared classification metrics for model comparison and reporting.

Every stage that trains or evaluates a model (baseline comparison, Boruta,
VIF, tuned retrain) reports through this module so the numbers in
`user_command.ipynb` are computed the same way everywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_COLUMNS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc")


def class_weight_dict(y) -> dict[int, float]:
    """Inverse-frequency weights so both classes contribute equally (reference `cwts`).

    ``w_i = n / (2 * n_i)``. With these weights, ``w0 * n0 == w1 * n1 == n / 2``.
    """
    y = np.asarray(y).astype(int)
    counts = np.bincount(y)
    if len(counts) < 2 or counts[0] == 0 or counts[1] == 0:
        return {0: 1.0, 1: 1.0}
    n = float(len(y))
    return {0: float((1.0 / counts[0]) * (n / 2.0)), 1: float((1.0 / counts[1]) * (n / 2.0))}


def sample_weight_vector(y, weights: dict[int, float] | None = None) -> np.ndarray:
    """Per-row sample weights from a class-weight dict (XGBoost / shared fit path)."""
    mapping = weights or class_weight_dict(y)
    y = np.asarray(y).astype(int)
    return np.where(y == 1, mapping.get(1, 1.0), mapping.get(0, 1.0)).astype(float)


def scale_pos_weight(y) -> float:
    """XGBoost constructor form of the same imbalance correction: n_neg / n_pos."""
    y = np.asarray(y).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


def classification_metrics(y_true, y_proba, *, threshold: float = 0.5) -> dict[str, float]:
    """Accuracy, balanced accuracy, precision, recall, F1, and ROC-AUC for one fold."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    n = int(len(y_true))
    out: dict[str, float] = {"n": n, "positive_rate": float(np.mean(y_true)) if n else float("nan")}
    if n == 0:
        return {**out, **{m: float("nan") for m in METRIC_COLUMNS}}
    y_pred = (y_proba >= threshold).astype(int)
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    if len(np.unique(y_true)) > 1:
        out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            out["roc_auc"] = float("nan")
    else:
        # A single-class fold makes balanced accuracy / AUC undefined, not zero.
        out["balanced_accuracy"] = float("nan")
        out["roc_auc"] = float("nan")
    return out


def aggregate_fold_metrics(fold_rows: list[dict]) -> dict[str, float]:
    """Test-size-weighted mean of each metric across folds, skipping NaN folds."""
    if not fold_rows:
        return {m: float("nan") for m in METRIC_COLUMNS} | {"n_folds": 0, "n_test_total": 0}
    frame = pd.DataFrame(fold_rows)
    weights = frame["n_test"] if "n_test" in frame.columns else frame.get("n", pd.Series(1.0, index=frame.index))
    out: dict[str, float] = {}
    for col in METRIC_COLUMNS:
        if col not in frame.columns:
            out[col] = float("nan")
            continue
        vals = frame[col].astype(float)
        mask = vals.notna()
        if not mask.any():
            out[col] = float("nan")
            continue
        w = weights[mask].astype(float)
        out[col] = float((vals[mask] * w).sum() / w.sum()) if w.sum() > 0 else float(vals[mask].mean())
    out["n_folds"] = int(len(frame))
    out["n_test_total"] = int(weights.sum())
    return out


def fold_metrics_table(fold_rows: list[dict]) -> pd.DataFrame:
    """Tidy per-fold DataFrame ordered fold, n_train, n_test, then the metric columns."""
    if not fold_rows:
        return pd.DataFrame(columns=["fold", "n_train", "n_test", *METRIC_COLUMNS])
    frame = pd.DataFrame(fold_rows)
    lead = [c for c in ("fold", "n_train", "n_test", "positive_rate") if c in frame.columns]
    rest = [c for c in METRIC_COLUMNS if c in frame.columns]
    other = [c for c in frame.columns if c not in lead and c not in rest]
    return frame[lead + rest + other]
