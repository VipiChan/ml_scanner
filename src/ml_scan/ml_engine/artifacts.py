"""Model and metadata persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from ml_scan.ml_engine.estimator import MLEstimator


def save_model(estimator: MLEstimator, path: Path | str) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": estimator._model,
            "model_name": estimator.model_name,
            "features": estimator.features_,
            "fold_metrics": estimator.fold_metrics_,
            "random_state": estimator.random_state,
            "params": estimator.params,
            "class_weight": {str(k): v for k, v in (estimator.class_weight_ or {}).items()},
            "use_class_weight": estimator.use_class_weight,
        },
        dest,
    )
    meta = dest.with_suffix(".json")
    meta.write_text(
        json.dumps(
            {
                "model_name": estimator.model_name,
                "features": estimator.features_,
                "fold_metrics": estimator.fold_metrics_,
                "aggregate_metrics": estimator.aggregate_metrics(),
                "params": estimator.params,
                "class_weight": {str(k): v for k, v in (estimator.class_weight_ or {}).items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return dest


def load_model(path: Path | str) -> MLEstimator:
    payload: dict[str, Any] = joblib.load(Path(path))
    est = MLEstimator(
        model=payload.get("model_name", "lightgbm"),
        random_state=payload.get("random_state", 42),
        params=payload.get("params") or {},
        use_class_weight=payload.get("use_class_weight", True),
    )
    est._model = payload["model"]
    est.features_ = list(payload.get("features") or [])
    est.fold_metrics_ = list(payload.get("fold_metrics") or [])
    raw_cw = payload.get("class_weight") or {}
    est.class_weight_ = {int(k): float(v) for k, v in raw_cw.items()}
    return est
