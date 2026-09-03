"""Inverse-frequency class weights match the reference `cwts` helper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_scan.ml_engine.estimator import MLEstimator
from ml_scan.ml_engine.metrics import class_weight_dict, sample_weight_vector, scale_pos_weight


def test_class_weight_dict_balances_classes() -> None:
    y = np.array([0] * 70 + [1] * 30)
    weights = class_weight_dict(y)
    assert abs(weights[0] * 70 - weights[1] * 30) < 1e-9
    assert abs(weights[0] * 70 - 50.0) < 1e-9


def test_sample_weight_vector_maps_rows() -> None:
    y = pd.Series([0, 1, 0, 1])
    weights = {0: 0.5, 1: 2.0}
    sw = sample_weight_vector(y, weights)
    assert sw.tolist() == [0.5, 2.0, 0.5, 2.0]


def test_scale_pos_weight_is_neg_over_pos() -> None:
    y = np.array([0, 0, 0, 1])
    assert scale_pos_weight(y) == 3.0


def test_estimator_fit_stores_class_weights() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=40), "b": rng.normal(size=40)})
    y = pd.Series([0] * 30 + [1] * 10)
    est = MLEstimator(model="rf", random_state=42, use_class_weight=True).fit(X, y)
    assert est.class_weight_
    assert abs(est.class_weight_[0] * 30 - est.class_weight_[1] * 10) < 1e-9
