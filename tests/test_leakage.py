from __future__ import annotations

import pandas as pd

from ml_scan.features.qc import leakage_audit, run_qc


def test_future_close_is_flagged() -> None:
    close = pd.Series(range(40), dtype=float)
    df = pd.DataFrame(
        {
            "symbol": ["A"] * 40,
            "ts": pd.date_range("2024-01-01", periods=40, freq="h"),
            "close": close,
            "open": close,
            "high": close,
            "low": close,
            "volume": 1.0,
            "LEAK": close.shift(-1),
            "RSI": close.rolling(3).mean(),
            "y": [0, 1] * 20,
        }
    )
    report = leakage_audit(df)
    assert report["ok"] is False
    assert "LEAK" in {s["column"] for s in report["suspects"]}


def test_honest_features_pass() -> None:
    close = pd.Series(range(40), dtype=float)
    df = pd.DataFrame(
        {
            "symbol": ["A"] * 40,
            "ts": pd.date_range("2024-01-01", periods=40, freq="h"),
            "close": close,
            "open": close,
            "high": close,
            "low": close,
            "volume": 1.0,
            "RSI": (close % 7) + 30,
            "y": [0, 1] * 20,
        }
    )
    cleaned, report = run_qc(df)
    assert report["leakage"]["ok"] is True
    assert "close" not in cleaned.columns or "close" in cleaned.columns
    assert "RSI" in cleaned.columns
