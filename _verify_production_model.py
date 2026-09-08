"""One-off verification: score the full labeled panel with the production model,
build history signals, run a full-history backtest, and write a report. Not part
of the package -- delete after the production model has been sanity-checked."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml_scan.config import load_settings
from ml_scan.execution.scanner import InferenceScanner, history_signals
from ml_scan.ops import backtest_run, write_report

ROOT = Path(__file__).resolve().parent
PROD = ROOT / "data" / "artifacts" / "production"

settings = load_settings()
scanner = InferenceScanner(settings)
scanner.load_artifacts(PROD / "model.joblib", PROD / "selected_features_full.json")

labeled = pd.read_parquet(PROD / "labeled_full.parquet")
print("labeled rows:", len(labeled))

scored = scanner.score_panel(labeled)
history = history_signals(scored, threshold=settings.ml.score_threshold)
hist_path = PROD / "scan_history_full.parquet"
history.to_parquet(hist_path, index=False)
print("history signals:", len(history), "->", hist_path)

latest = scanner.run("latest", labeled)
latest_path = PROD / "scan_latest_full.csv"
latest.to_csv(latest_path, index=False)
print("latest scan rows:", len(latest), "->", latest_path)

start = str(labeled["ts"].min().date())
end = str(labeled["ts"].max().date())
bt_dir = backtest_run(settings, signals=hist_path, start=start, end=end, out=PROD / "bt", model_path=PROD / "model.joblib", features_path=PROD / "selected_features_full.json")
print("backtest dir:", bt_dir)

metrics = json.loads((bt_dir / "metrics.json").read_text(encoding="utf-8"))
print(json.dumps(metrics, indent=2))

report_path = write_report(bt_dir, PROD / "report.html", model_path=PROD / "model.joblib")
print("report:", report_path)
