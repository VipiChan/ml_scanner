"""Purged walk-forward splits on a date-aligned panel."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from ml_scan.storage.buckets import ensure_ist

BARS_PER_SESSION = 7


class PurgedWalkForward:
    def __init__(self, *, n_splits: int = 4, embargo_sessions: int = 10) -> None:
        self.n_splits = n_splits
        self.embargo_sessions = embargo_sessions
        self.embargo_bars = int(embargo_sessions * BARS_PER_SESSION)

    def split(self, panel: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) positional indices into `panel`.

        Expanding train, then an embargo gap, then the test block.
        Embargo never consumes more than 25% of a test block so short
        smoke windows still get usable holdouts.
        """
        if "ts" not in panel.columns:
            raise ValueError("panel needs a ts column")
        ts = pd.Series(panel["ts"].map(ensure_ist)).reset_index(drop=True)
        unique = pd.Index(sorted(ts.unique()))
        if len(unique) < self.n_splits + 2:
            raise ValueError(
                f"Need more timestamps than n_splits ({self.n_splits}); got {len(unique)}"
            )
        blocks = np.array_split(np.arange(len(unique)), self.n_splits + 1)
        train_pos = list(blocks[0])
        for test_block in blocks[1:]:
            if len(test_block) == 0 or not train_pos:
                continue
            embargo_n = min(self.embargo_bars, max(0, len(test_block) // 4))
            test_pos = test_block[embargo_n:]
            if len(test_pos) == 0:
                continue
            train_times = set(unique[train_pos])
            test_times = set(unique[test_pos])
            if train_times & test_times:
                raise RuntimeError("purged split leaked timestamps")
            train_idx = np.flatnonzero(ts.isin(train_times).to_numpy())
            test_idx = np.flatnonzero(ts.isin(test_times).to_numpy())
            if len(train_idx) and len(test_idx):
                yield train_idx, test_idx
            train_pos = train_pos + list(test_block)


def fold_symbol_table(panel: pd.DataFrame, splitter: "PurgedWalkForward | None" = None) -> pd.DataFrame:
    """Per fold, which symbols contributed rows to the train block vs. the test block.

    `panel` must have a 0..n-1 RangeIndex matching the X/y used for training
    (see `ml_scan.features.qc.select_xy`), since positions from `splitter.split`
    are looked up with `.iloc`.
    """
    if "symbol" not in panel.columns:
        raise ValueError("panel needs a symbol column")
    splitter = splitter or PurgedWalkForward()
    work = panel.reset_index(drop=True)
    rows: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(work), start=1):
        train_syms = set(work.iloc[train_idx]["symbol"])
        test_syms = set(work.iloc[test_idx]["symbol"])
        for sym in sorted(train_syms | test_syms):
            rows.append(
                {
                    "fold": fold,
                    "symbol": sym,
                    "in_train": sym in train_syms,
                    "in_test": sym in test_syms,
                    "n_train_rows": int((work.iloc[train_idx]["symbol"] == sym).sum()),
                    "n_test_rows": int((work.iloc[test_idx]["symbol"] == sym).sum()),
                }
            )
    return pd.DataFrame(rows)
