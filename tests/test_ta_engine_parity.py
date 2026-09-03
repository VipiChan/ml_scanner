from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_scan.features.ta_engine import generate_all_ta_features
from tests.fixtures.legacy_ta import generate_all_ta_features as legacy_generate


def test_ta_engine_matches_legacy_helper(tmp_path: Path) -> None:
    idx = pd.date_range("2024-01-02 09:15", periods=80, freq="h")
    close = 2500 + pd.Series(range(len(idx)), dtype=float).rolling(3).mean().bfill()
    raw = pd.DataFrame(
        {
            "open": close - 1,
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "volume": 1000 + close,
        },
        index=idx,
    )
    csv = tmp_path / "hourly.csv"
    raw.reset_index().rename(columns={"index": "datetime"}).to_csv(csv, index=False)
    ours = generate_all_ta_features(raw.copy())
    theirs = legacy_generate(raw.copy())
    assert set(ours.columns) == set(theirs.columns)
    shared = [c for c in ours.columns if pd.api.types.is_numeric_dtype(ours[c])]
    left = ours[shared].apply(pd.to_numeric, errors="coerce")
    right = theirs[shared].apply(pd.to_numeric, errors="coerce")
    pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-6, atol=1e-6)
