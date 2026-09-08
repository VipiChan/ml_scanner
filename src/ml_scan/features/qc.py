"""Feature QC and leakage guards."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_COLS = ("open", "high", "low", "close", "volume")
META_COLS = (
    "symbol",
    "ts",
    "interval",
    "source",
    "instrument_token",
    "is_partial_hour",
    "n_5m",
    "y",
    "y_reason",
    "fwd_mfe",
    "fwd_mae",
    "bars_to_outcome",
    "d_ts",
    "m15_ts",
    "d_open",
    "d_high",
    "d_low",
    "d_close",
    "d_volume",
    "m15_open",
    "m15_high",
    "m15_low",
    "m15_close",
    "m15_volume",
)


def drop_low_quality_columns(
    df: pd.DataFrame,
    missing_threshold: float = 0.10,
    drop_zero_std: bool = True,
) -> pd.DataFrame:
    frame = df.copy()
    drop: list[str] = []
    for col in frame.columns:
        if col in META_COLS or col in PRICE_COLS:
            continue
        series = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if series.isna().mean() > missing_threshold:
            drop.append(col)
            continue
        if drop_zero_std and series.nunique(dropna=True) <= 1:
            drop.append(col)
    return frame.drop(columns=drop, errors="ignore")


def feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = set(META_COLS) | set(PRICE_COLS)
    cols: list[str] = []
    for col in df.columns:
        if col in blocked:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def leakage_audit(df: pd.DataFrame, *, close_col: str = "close") -> dict:
    """Assert no feature is a shifted future close (correlation ~1 after shift)."""
    if close_col not in df.columns:
        return {"ok": True, "suspects": []}
    feats = feature_columns(df)
    suspects: list[dict] = []
    if "symbol" in df.columns:
        groups = df.groupby("symbol", sort=False)
    else:
        groups = [(None, df)]
    for _symbol, part in groups:
        if close_col not in part.columns:
            continue
        future = part[close_col].shift(-1)
        for col in feats:
            a = pd.to_numeric(part[col], errors="coerce")
            if a.notna().sum() < 20 or future.notna().sum() < 20:
                continue
            if float(a.std(skipna=True) or 0) == 0 or float(future.std(skipna=True) or 0) == 0:
                continue
            corr = a.corr(future)
            if corr is not None and abs(float(corr)) > 0.999:
                suspects.append({"column": col, "corr_future_close": float(corr)})
    unique = {s["column"]: s for s in suspects}
    return {"ok": len(unique) == 0, "suspects": list(unique.values())}


def run_qc(
    df: pd.DataFrame,
    *,
    missing_threshold: float = 0.10,
    out_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """QC gate before expensive model training.

    Beyond missingness/zero-variance pruning, `features` excludes any column the
    leakage audit flags (>0.999 correlation with next-bar close) -- these are raw
    price-level trackers (moving averages, bands, VWAP, price channels) rather than
    look-ahead bugs, but they encode absolute price level, which varies by orders of
    magnitude across a cross-sectional universe and gives a tree model an easy,
    non-generalizing shortcut. `leakage.suspects` still lists them for transparency.
    """
    cleaned = drop_low_quality_columns(df, missing_threshold=missing_threshold)
    audit = leakage_audit(cleaned)
    suspect_cols = {s["column"] for s in audit["suspects"]}
    safe_features = [c for c in feature_columns(cleaned) if c not in suspect_cols]
    report = {
        "n_rows": int(len(cleaned)),
        "n_feature_cols": len(safe_features),
        "features": safe_features,
        "leakage": audit,
        "dropped_vs_input": sorted(set(df.columns) - set(cleaned.columns)),
        "dropped_leakage_suspects": sorted(suspect_cols),
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return cleaned, report


def training_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop unresolved labels and non-feature columns."""
    work = df.loc[df["y"].isin([0, 1])].copy() if "y" in df.columns else df.copy()
    cols = feature_columns(work)
    X = work[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    y = work["y"].astype(int) if "y" in work.columns else pd.Series(dtype=int)
    mask = y.notna()
    return X.loc[mask], y.loc[mask]


def select_xy(
    df: pd.DataFrame, feature_cols: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Resolve labels, reset to a shared 0..n-1 index, and build X for a feature list.

    Unlike `training_xy`, this also returns the row-aligned `panel` (same order,
    same length as X/y). A `PurgedWalkForward` split on that panel then yields
    positions that map directly onto `X.iloc` / `y.iloc` -- no separate
    re-alignment step needed by callers such as `compare_models` or
    `random_search`.
    """
    work = df.loc[df["y"].isin([0, 1])].reset_index(drop=True) if "y" in df.columns else df.reset_index(drop=True)
    cols = feature_cols if feature_cols is not None else feature_columns(work)
    X = work.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    y = work["y"].astype(int)
    return X, y, work


def symbol_row_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol row counts and label balance -- the audit trail for "which stocks feed the model"."""
    if "symbol" not in df.columns:
        raise ValueError("panel needs a symbol column")
    has_y = "y" in df.columns
    rows: list[dict] = []
    for sym, part in df.groupby("symbol", sort=True):
        row: dict[str, object] = {"symbol": sym, "n_rows": int(len(part))}
        if has_y:
            n_win = int((part["y"] == 1).sum())
            n_loss = int((part["y"] == 0).sum())
            n_unresolved = int(part["y"].isna().sum())
            resolved = n_win + n_loss
            row.update(
                n_win=n_win,
                n_loss=n_loss,
                n_unresolved=n_unresolved,
                win_rate=float(n_win / resolved) if resolved else float("nan"),
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
