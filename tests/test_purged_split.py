from __future__ import annotations

import pandas as pd

from ml_scan.ml_engine.splitter import PurgedWalkForward
from ml_scan.storage.buckets import NSE_TZ


def test_purged_split_no_timestamp_overlap() -> None:
    ts = pd.date_range("2024-01-02 09:15", periods=400, freq="h", tz=NSE_TZ)
    panel = pd.DataFrame(
        {
            "ts": list(ts) * 2,
            "symbol": ["AAA"] * 400 + ["BBB"] * 400,
        }
    )
    splitter = PurgedWalkForward(n_splits=3, embargo_sessions=10)
    seen = False
    for train_idx, test_idx in splitter.split(panel):
        seen = True
        train_ts = set(panel.iloc[train_idx]["ts"])
        test_ts = set(panel.iloc[test_idx]["ts"])
        assert train_ts.isdisjoint(test_ts)
        assert train_idx.max() < test_idx.min() or max(train_ts) < min(test_ts)
        assert len(test_ts) >= 50
    assert seen
