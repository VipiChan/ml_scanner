"""Plain DataFrame views for scan and blotter tables."""

from __future__ import annotations

import pandas as pd


def scan_table(scan: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "rank",
        "symbol",
        "asof_ts",
        "score",
        "p_win",
        "entry_px",
        "sl_px",
        "tp_px",
        "atr",
        "adtv_20",
    ]
    cols = [c for c in keep if c in scan.columns]
    return scan[cols].copy()


def blotter_table(blotter: pd.DataFrame) -> pd.DataFrame:
    return blotter.copy()


def metrics_table(metrics: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([metrics])
