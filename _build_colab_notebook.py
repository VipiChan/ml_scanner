"""Generate colab_commands.ipynb. Run from repo root: python _build_colab_notebook.py"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "colab_commands.ipynb"


def md(source: str) -> dict:
    lines = source.strip("\n") + "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines.split("\n")[:-1]] + [lines.split("\n")[-1] + "\n"],
    }


def code(source: str) -> dict:
    text = source.strip("\n") + "\n"
    parts = text.split("\n")
    src = [p + "\n" for p in parts[:-1]]
    if parts[-1] or not src:
        src.append(parts[-1] + "\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


cells = [
    md(
        """# ml_scan — Colab training notebook

Colab-GPU equivalent of `user_command.ipynb`. **No TimescaleDB.** Training bars come from
`train_ohlcv_60minute.parquet` on Google Drive (`MyDrive/ml_train`).

**Colab setup (once)**
1. Runtime → Change runtime type → **T4 GPU** (or A100).
2. Upload `train_ohlcv_60minute.parquet` (from `scan-trade export-train-parquet`) to Drive folder `ml_train`.
3. File → Open notebook from GitHub → `VipiChan/ml_scanner` → `colab_commands.ipynb`, **or** run the clone cell below.
4. Run cells in order. After `pip install`, if imports fail: Runtime → Restart session, then re-run from the **paths** cell.

**What this notebook does not do**
- It does not query the warehouse. CLI commands `features hourly`, `features mtf`, `scan`, `backtest run`, and `e2e` still expect Timescale — do not use them here.
- 15-minute features are included only if you also upload `train_ohlcv_15minute.parquet`. Hourly bars cannot reconstruct 15m honestly; missing `m15_*` columns are dropped at QC.
- Daily context is resampled from hourly (including the 15:15 IST close print) unless you upload `train_ohlcv_day.parquet`.
"""
    ),
    md("## 0 — Clone repo, install package, mount Drive"),
    code(
        r"""
import os
import sys
from pathlib import Path

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

REPO_URL = "https://github.com/VipiChan/ml_scanner.git"
REPO_DIR = Path(".").resolve()
if IN_COLAB:
    for candidate in (Path.cwd(), Path("/content/ml_scan"), Path("/content/ml_scanner")):
        if (candidate / "src" / "ml_scan").is_dir():
            REPO_DIR = candidate
            break
    else:
        REPO_DIR = Path("/content/ml_scan")
        !git clone --depth 1 {REPO_URL} {REPO_DIR}
    os.chdir(REPO_DIR)
    # Colab already has pandas/numpy/sklearn. Install the package + remaining deps.
    %pip install -q -e ".[dev]"
else:
    os.chdir(REPO_DIR)

sys.path.insert(0, str(REPO_DIR / "src"))
print("IN_COLAB", IN_COLAB)
print("cwd", Path.cwd())
"""
    ),
    md(
        """## 0b — Paths, Drive parquet, settings

Edit `DRIVE_DIR` if your folder is not `MyDrive/ml_train`.
Set `SYMBOL_LIMIT` to an int (e.g. `20`) for a cheap smoke pass. `None` = every symbol in the parquet.
"""
    ),
    code(
        r"""
import os
import sys
from pathlib import Path

import pandas as pd

REPO_DIR = Path(".").resolve()
for candidate in (Path.cwd(), Path("/content/ml_scan"), Path("/content/ml_scanner")):
    if (candidate / "src" / "ml_scan").is_dir():
        REPO_DIR = candidate
        break
os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR / "src"))

from ml_scan.config import load_settings
from ml_scan.data.parquet_source import load_training_bundle
from ml_scan.ml_engine.estimator import describe_accelerator, gpu_available

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", 40)
pd.set_option("display.expand_frame_repr", False)

IN_COLAB = Path("/content").is_dir()
try:
    import google.colab
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# --- edit these ---
DRIVE_DIR = Path("/content/drive/MyDrive/ml_train") if IN_COLAB else Path("data/artifacts")
OHLCV_PATH = DRIVE_DIR / "train_ohlcv_60minute.parquet"
ARTIFACTS = DRIVE_DIR / "artifacts"
SYMBOL_LIMIT = None  # e.g. 20 for a smoke run
SAMPLE_ROWS = 300_000  # compare / Boruta / optimize only; final train uses all rows
# ------------------

ARTIFACTS.mkdir(parents=True, exist_ok=True)
assert OHLCV_PATH.is_file(), (
    f"Upload train_ohlcv_60minute.parquet to {DRIVE_DIR} "
    "(scan-trade export-train-parquet, then copy that file to Drive)."
)

settings = load_settings()
hourly, bench_daily, bundle = load_training_bundle(
    OHLCV_PATH,
    benchmark_symbol=settings.universe.benchmark_symbol,
)
if SYMBOL_LIMIT:
    keep = sorted(hourly["symbol"].unique())[: int(SYMBOL_LIMIT)]
    hourly = hourly.loc[hourly["symbol"].isin(keep)].reset_index(drop=True)
    bundle.hourly = hourly
    bundle.daily = bundle.daily.loc[bundle.daily["symbol"].isin(keep)].reset_index(drop=True)
    if bundle.minute15 is not None and not bundle.minute15.empty:
        bundle.minute15 = bundle.minute15.loc[bundle.minute15["symbol"].isin(keep)].reset_index(drop=True)
    if bench_daily is not None and not bench_daily.empty:
        pass  # keep full benchmark series

TRAIN_START = pd.Timestamp(hourly["ts"].min()).tz_convert("Asia/Kolkata").strftime("%Y-%m-%d")
TRAIN_END = pd.Timestamp(hourly["ts"].max()).tz_convert("Asia/Kolkata").strftime("%Y-%m-%d")
print("accelerator:", describe_accelerator())
print("gpu_available:", gpu_available())
print("parquet", OHLCV_PATH)
print("hourly rows", len(hourly), "symbols", hourly["symbol"].nunique(), TRAIN_START, "→", TRAIN_END)
print("daily rows", len(bundle.daily), "15m rows", 0 if bundle.minute15 is None else len(bundle.minute15))
print("benchmark daily rows", 0 if bench_daily is None else len(bench_daily))
print("artifacts", ARTIFACTS)
"""
    ),
    md("## M03 — NSE calendar (7 hourly bars per session)"),
    code(
        r"""
import pandas as pd
from ml_scan.data.calendar import NSECalendar

cal = NSECalendar()
n = len(cal.hourly_index(pd.Timestamp("2024-01-02", tz="Asia/Kolkata")))
print("hourly bars on 2024-01-02:", n)
assert n == 7
"""
    ),
    md(
        """## M04 — Parquet hourly reader (replaces TimescaleAdapter)

Expect standard OHLCV columns, `interval == 60minute`, and **no** 15:15 partial hours on the feature panel.
The 15:15 print is kept only for daily resampling.
"""
    ),
    code(
        r"""
assert len(hourly) > 0
assert {"ts", "symbol", "open", "high", "low", "close", "volume", "interval"} <= set(hourly.columns)
assert (hourly["interval"] == "60minute").all()
ts = pd.to_datetime(hourly["ts"], utc=True).dt.tz_convert("Asia/Kolkata")
assert not ts.dt.strftime("%H:%M").eq("15:15").any(), "15:15 bars must be dropped from the hourly feature panel"
probe = hourly.loc[hourly["symbol"] == hourly["symbol"].iloc[0]]
print(probe.head())
print("-" * 70)
print(probe.tail())
"""
    ),
    md("## M07 — Honest daily as-of join (daily resampled from hourly; no same-session leak)"),
    code(
        r"""
from ml_scan.data.mtf_aligner import MTFAligner
from ml_scan.data.schemas import MTFBundle

syms = [s for s in ("RELIANCE", "TCS") if s in set(hourly["symbol"])]
if len(syms) < 2:
    syms = sorted(hourly["symbol"].unique())[:2]
h = hourly.loc[hourly["symbol"].isin(syms)].copy()
d = bundle.daily.loc[bundle.daily["symbol"].isin(syms)].copy()
aligned = MTFAligner().align_to_hourly(MTFBundle(hourly=h, daily=d, minute15=pd.DataFrame()))
print(aligned.groupby("symbol", sort=False).head(7)[["symbol", "ts", "d_ts", "d_close"]])
print("aligned", aligned.shape)
print("M07 ok — morning rows of the first session should have empty d_close (no same-session leak)")
"""
    ),
    md("## M08 — TA engine parity (offline fixture, no warehouse)"),
    code(
        r"""
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pytest", "tests/test_ta_engine_parity.py", "tests/test_parquet_source.py", "-q"])
"""
    ),
    md("## M09 — Hourly pandas_ta panel from the Drive parquet"),
    code(
        r"""
from ml_scan.features.panel import PanelFeatureEngineer

feat_hourly_path = ARTIFACTS / "feat_hourly.parquet"
engineer = PanelFeatureEngineer(mode=settings.features.hourly_mode, atr_period=settings.features.atr_period)
print(f"building hourly features for {hourly['symbol'].nunique()} symbols, mode={settings.features.hourly_mode!r} …")
feat = engineer.transform(hourly).reset_index(drop=True)
feat.to_parquet(feat_hourly_path, index=False)
assert "ATR" in feat.columns
meta = {"symbol", "ts", "interval", "source", "instrument_token", "open", "high", "low", "close", "volume"}
n_ta = len([c for c in feat.columns if c not in meta])
print(f"hourly panel: {len(feat)} rows × {feat.shape[1]} columns ({n_ta} TA/derived)")
print("wrote", feat_hourly_path)
print(feat[["symbol", "ts"]].head())
"""
    ),
    md(
        """### M09b — Which stocks feed the model

The split later is **time-based**, not stock-based. Every symbol in the parquet is used on both train and test sides of each fold.
"""
    ),
    code(
        r"""
from IPython.display import display
from ml_scan.features.qc import symbol_row_counts
from ml_scan.features.ta_engine import generate_all_ta_features, generate_lite_ta_features

feat = pd.read_parquet(ARTIFACTS / "feat_hourly.parquet")
print(f"Q1 — {feat['symbol'].nunique()} symbols in the hourly panel")
display(symbol_row_counts(feat)[["symbol", "n_rows"]])

meta = {"symbol", "ts", "interval", "source", "instrument_token", "open", "high", "low", "close", "volume"}
print(f"Q2 — hourly_mode={settings.features.hourly_mode!r}: {len([c for c in feat.columns if c not in meta])} non-OHLCV columns")
sample = (
    feat.loc[feat["symbol"] == feat["symbol"].iloc[0], ["ts", "open", "high", "low", "close", "volume"]]
    .sort_values("ts")
    .set_index("ts")
)
ohlcv_cols = {"open", "high", "low", "close", "volume"}
print("lite", len(set(generate_lite_ta_features(sample).columns) - ohlcv_cols))
print("full", len(set(generate_all_ta_features(sample).columns) - ohlcv_cols))
"""
    ),
    md(
        """## M10 — Daily (+ optional 15m) join

Daily EMA/ATR/RS come from resampled hourly (or `train_ohlcv_day.parquet` if you uploaded it).
`d_rs` uses `NIFTY 50` only when that symbol is in the parquet; otherwise it is the stock's own return.
"""
    ),
    code(
        r"""
from ml_scan.features.htf import join_htf_features

feat = pd.read_parquet(ARTIFACTS / "feat_hourly.parquet")
mtf = join_htf_features(feat, bundle, benchmark_daily=bench_daily, config=settings.features)
mtf_path = ARTIFACTS / "feat_mtf.parquet"
mtf.to_parquet(mtf_path, index=False)
d_cols = [c for c in mtf.columns if c.startswith("d_")]
m15_cols = [c for c in mtf.columns if c.startswith("m15_")]
assert d_cols, "daily columns missing"
print("daily", d_cols[:10])
print("m15", m15_cols[:10] if m15_cols else "(none — upload train_ohlcv_15minute.parquet to add them)")
print("wrote", mtf_path, mtf.shape)
"""
    ),
    md("## M11 — Swing labels"),
    code(
        r"""
from ml_scan.features.target import SwingLabeler

mtf = pd.read_parquet(ARTIFACTS / "feat_mtf.parquet")
tgt = settings.target
labeled = SwingLabeler(
    atr_col=tgt.atr_col,
    sl_mult=tgt.sl_mult,
    tp_r=tgt.tp_r,
    max_sessions=tgt.max_sessions,
    stop_wins_same_bar=tgt.stop_wins_same_bar,
).transform(mtf)
labeled_path = ARTIFACTS / "labeled.parquet"
labeled.to_parquet(labeled_path, index=False)
print(labeled["y"].value_counts(dropna=False))
print(labeled["y_reason"].value_counts(dropna=False))
print("wrote", labeled_path)
"""
    ),
    md("### M11b — Class weights (`cwts`)"),
    code(
        r"""
from ml_scan.ml_engine.metrics import class_weight_dict

lab = pd.read_parquet(ARTIFACTS / "labeled.parquet")
resolved = lab.loc[lab["y"].isin([0, 1]), "y"].astype(int)
counts = resolved.value_counts().sort_index()
print(counts.to_string())
print(f"positive rate: {resolved.mean():.4f}")
cw = class_weight_dict(resolved)
print("class weights", {k: round(v, 4) for k, v in cw.items()})
print("weighted 0", cw[0] * int(counts.get(0, 0)))
print("weighted 1", cw[1] * int(counts.get(1, 0)))
"""
    ),
    md("## M12 — Feature QC (drops price-level leakage suspects)"),
    code(
        r"""
import json
from ml_scan.features.qc import run_qc

lab = pd.read_parquet(ARTIFACTS / "labeled.parquet")
_, qc_report = run_qc(
    lab,
    missing_threshold=settings.features.missing_threshold,
    out_path=ARTIFACTS / "feature_qc_report.json",
)
print(json.dumps({
    "n_rows": qc_report["n_rows"],
    "n_feature_cols": qc_report["n_feature_cols"],
    "n_leakage_suspects_excluded": len(qc_report["dropped_leakage_suspects"]),
    "leakage_ok": qc_report["leakage"]["ok"],
}, indent=2))
"""
    ),
    md(
        """### M12b — Baseline model bake-off (RF / XGBoost / LightGBM)

Same purged walk-forward folds. On a Colab GPU, **XGBoost uses CUDA**; pip LightGBM is usually CPU-only.
`SAMPLE_ROWS` keeps this stage tractable; the final train cell uses the full labeled panel.
"""
    ),
    code(
        r"""
import json
from IPython.display import display

from ml_scan.features.qc import select_xy
from ml_scan.ml_engine.estimator import compare_models, select_best_model
from ml_scan.ml_engine.metrics import fold_metrics_table
from ml_scan.ml_engine.splitter import PurgedWalkForward
from ml_scan.ops import stratified_sample

labeled = pd.read_parquet(ARTIFACTS / "labeled.parquet")
qc_report = json.loads((ARTIFACTS / "feature_qc_report.json").read_text(encoding="utf-8"))
initial_features = qc_report["features"]
print(f"{len(initial_features)} QC-passed features")

work = stratified_sample(labeled, SAMPLE_ROWS, random_state=settings.ml.random_state)
X, y, panel = select_xy(work, initial_features)
splitter = PurgedWalkForward(n_splits=settings.ml.n_splits, embargo_sessions=settings.ml.embargo_sessions)
baseline_summary, baseline_models = compare_models(
    X, y, panel, splitter=splitter, random_state=settings.ml.random_state
)
best_model = select_best_model(baseline_summary, metric="roc_auc")
display(baseline_summary.round(4))
print("best_model", best_model)
display(fold_metrics_table(baseline_models[best_model].fold_metrics_).round(4))
(ARTIFACTS / "model_comparison_initial.json").write_text(
    json.dumps(
        {
            "best_model": best_model,
            "n_initial_features": len(initial_features),
            "initial_features": initial_features,
            "summary": baseline_summary.reset_index().to_dict(orient="records"),
        },
        indent=2,
    ),
    encoding="utf-8",
)
"""
    ),
    md("## M13b — Boruta with the winning estimator"),
    code(
        r"""
import json
from IPython.display import display

from ml_scan.features.qc import select_xy
from ml_scan.ml_engine.estimator import compare_models
from ml_scan.ml_engine.metrics import fold_metrics_table
from ml_scan.ml_engine.selector import BorutaSelector
from ml_scan.ml_engine.splitter import PurgedWalkForward
from ml_scan.ops import stratified_sample

labeled = pd.read_parquet(ARTIFACTS / "labeled.parquet")
init = json.loads((ARTIFACTS / "model_comparison_initial.json").read_text(encoding="utf-8"))
best_model, initial_features = init["best_model"], init["initial_features"]
work = stratified_sample(labeled, SAMPLE_ROWS, random_state=settings.ml.random_state)
X, y, _ = select_xy(work, initial_features)
boruta = BorutaSelector(
    max_iter=settings.ml.boruta_max_iter,
    random_state=settings.ml.random_state,
    estimator_name=best_model,
).fit(X, y)
boruta.save(ARTIFACTS / "boruta_features.json")
print(f"Boruta ({best_model}) kept {len(boruta.features_)} of {len(initial_features)}")
print(boruta.features_)

Xb, yb, panelb = select_xy(work, boruta.features_)
splitter = PurgedWalkForward(n_splits=settings.ml.n_splits, embargo_sessions=settings.ml.embargo_sessions)
boruta_summary, boruta_models = compare_models(
    Xb, yb, panelb, models=(best_model,), splitter=splitter, random_state=settings.ml.random_state
)
display(boruta_summary.round(4))
display(fold_metrics_table(boruta_models[best_model].fold_metrics_).round(4))
(ARTIFACTS / "model_comparison_boruta.json").write_text(
    json.dumps(
        {"n_features": len(boruta.features_), "summary": boruta_summary.reset_index().to_dict(orient="records")},
        indent=2,
    ),
    encoding="utf-8",
)
"""
    ),
    md("## M14b — VIF prune + same-model comparison"),
    code(
        r"""
import json
from IPython.display import display

from ml_scan.features.qc import select_xy
from ml_scan.ml_engine.estimator import compare_models
from ml_scan.ml_engine.metrics import fold_metrics_table
from ml_scan.ml_engine.selector import VIFPruner, load_feature_list
from ml_scan.ml_engine.splitter import PurgedWalkForward
from ml_scan.ops import stratified_sample

labeled = pd.read_parquet(ARTIFACTS / "labeled.parquet")
init = json.loads((ARTIFACTS / "model_comparison_initial.json").read_text(encoding="utf-8"))
best_model = init["best_model"]
boruta_feats = load_feature_list(ARTIFACTS / "boruta_features.json")
work = stratified_sample(labeled, SAMPLE_ROWS, random_state=settings.ml.random_state)
X, _y, _ = select_xy(work, boruta_feats)
pruner = VIFPruner(max_vif=settings.ml.max_vif).fit(X)
pruner.save(ARTIFACTS / "selected_features.json")
print(f"VIF kept {len(pruner.features_)} of {len(boruta_feats)}")

Xs, ys, panels = select_xy(work, pruner.features_)
splitter = PurgedWalkForward(n_splits=settings.ml.n_splits, embargo_sessions=settings.ml.embargo_sessions)
vif_summary, vif_models = compare_models(
    Xs, ys, panels, models=(best_model,), splitter=splitter, random_state=settings.ml.random_state
)
display(vif_summary.round(4))
display(fold_metrics_table(vif_models[best_model].fold_metrics_).round(4))
(ARTIFACTS / "model_comparison_vif.json").write_text(
    json.dumps({"n_features": len(pruner.features_), "summary": vif_summary.reset_index().to_dict(orient="records")}, indent=2),
    encoding="utf-8",
)
"""
    ),
    md("## M15 — Purged walk-forward (no timestamp overlap)"),
    code(
        r"""
import subprocess
import sys
from IPython.display import display

from ml_scan.features.qc import select_xy, symbol_row_counts
from ml_scan.ml_engine.selector import load_feature_list
from ml_scan.ml_engine.splitter import PurgedWalkForward, fold_symbol_table

subprocess.check_call([sys.executable, "-m", "pytest", "tests/test_purged_split.py", "-q"])

labeled = pd.read_parquet(ARTIFACTS / "labeled.parquet")
selected = load_feature_list(ARTIFACTS / "selected_features.json")
_, _, panel = select_xy(labeled, selected)
splitter = PurgedWalkForward(n_splits=settings.ml.n_splits, embargo_sessions=settings.ml.embargo_sessions)
print("resolved-label rows per symbol:")
display(symbol_row_counts(labeled).head(20))
fold_table = fold_symbol_table(panel, splitter)
print(f"{fold_table['symbol'].nunique()} symbols across {fold_table['fold'].nunique()} folds")
display(
    fold_table.pivot_table(index="symbol", columns="fold", values=["n_train_rows", "n_test_rows"], aggfunc="sum")
    .fillna(0)
    .astype(int)
    .head(15)
)
"""
    ),
    md("## M16a — RandomizedSearch on purged folds (sampled)"),
    code(
        r"""
import json
from IPython.display import display

from ml_scan.features.qc import select_xy
from ml_scan.ml_engine.optimize import random_search
from ml_scan.ml_engine.selector import load_feature_list
from ml_scan.ml_engine.splitter import PurgedWalkForward
from ml_scan.ops import stratified_sample

labeled = pd.read_parquet(ARTIFACTS / "labeled.parquet")
best_model = json.loads((ARTIFACTS / "model_comparison_initial.json").read_text(encoding="utf-8"))["best_model"]
selected = load_feature_list(ARTIFACTS / "selected_features.json")
work = stratified_sample(labeled, SAMPLE_ROWS, random_state=settings.ml.random_state)
Xs, ys, panels = select_xy(work, selected)
splitter = PurgedWalkForward(n_splits=settings.ml.n_splits, embargo_sessions=settings.ml.embargo_sessions)
best_params, search_results = random_search(
    Xs, ys, panels, model_name=best_model, splitter=splitter, n_iter=20, random_state=settings.ml.random_state
)
print("best_model", best_model)
print(json.dumps(best_params, indent=2, default=str))
(ARTIFACTS / "best_params.json").write_text(
    json.dumps({"model_name": best_model, "best_params": best_params}, indent=2, default=str),
    encoding="utf-8",
)
display(search_results.head(10))
"""
    ),
    md(
        """## M16b — Final train on the **full** labeled panel (GPU if available)

This is the model you copy to the other project. Walk-forward fold metrics are reported; the last fold's fitted trees are saved.
"""
    ),
    code(
        r"""
import json
from IPython.display import display

from ml_scan.features.qc import select_xy
from ml_scan.ml_engine.artifacts import save_model
from ml_scan.ml_engine.estimator import MLEstimator
from ml_scan.ml_engine.metrics import fold_metrics_table
from ml_scan.ml_engine.selector import load_feature_list
from ml_scan.ml_engine.splitter import PurgedWalkForward

labeled = pd.read_parquet(ARTIFACTS / "labeled.parquet")
best_model = json.loads((ARTIFACTS / "model_comparison_initial.json").read_text(encoding="utf-8"))["best_model"]
best_params = json.loads((ARTIFACTS / "best_params.json").read_text(encoding="utf-8"))["best_params"]
selected = load_feature_list(ARTIFACTS / "selected_features.json")
Xs, ys, panels = select_xy(labeled, selected)
splitter = PurgedWalkForward(n_splits=settings.ml.n_splits, embargo_sessions=settings.ml.embargo_sessions)
tuned = MLEstimator(
    model=best_model,
    random_state=settings.ml.random_state,
    params=best_params,
    use_class_weight=settings.ml.use_class_weight,
)
tuned.fit_walk_forward(Xs, ys, panels, splitter)
display(fold_metrics_table(tuned.fold_metrics_).round(4))
print(json.dumps(tuned.aggregate_metrics(), indent=2))
model_path = save_model(tuned, ARTIFACTS / "model.joblib")
print("wrote", model_path)
print("sidecar", model_path.with_suffix(".json"))
"""
    ),
    md("## M17 — Scan latest bar from the labeled panel (no live warehouse)"),
    code(
        r"""
from IPython.display import display
from ml_scan.execution.scanner import InferenceScanner

lab = pd.read_parquet(ARTIFACTS / "labeled.parquet")
scanner = InferenceScanner(settings)
scanner.load_artifacts(ARTIFACTS / "model.joblib", ARTIFACTS / "selected_features.json")
scan = scanner.run("latest", lab, daily=bundle.daily)
scan_path = ARTIFACTS / "scan_latest.csv"
scan.to_csv(scan_path, index=False)
print("wrote", scan_path)
display(scan)
"""
    ),
    md("## M18 / M19 — Cost model and next-open fill (offline tests)"),
    code(
        r"""
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pytest", "tests/test_costs.py", "tests/test_fill_lag.py", "-q"])
"""
    ),
    md("## M20–M22 — History signals, backtest on parquet hourly bars, report"),
    code(
        r"""
import json
from IPython.display import display

from ml_scan.backtest.engine import Backtester
from ml_scan.backtest.metrics import compute_metrics, equity_to_frame
from ml_scan.execution.scanner import InferenceScanner, history_signals
from ml_scan.reporting.charts import equity_figure, importance_figure, write_html
from ml_scan.ml_engine.artifacts import load_model

lab = pd.read_parquet(ARTIFACTS / "labeled.parquet")
scanner = InferenceScanner(settings)
scanner.load_artifacts(ARTIFACTS / "model.joblib", ARTIFACTS / "selected_features.json")
hist = history_signals(scanner.score_panel(lab), threshold=settings.ml.score_threshold)
hist_path = ARTIFACTS / "scan_history.parquet"
hist.to_parquet(hist_path, index=False)
print("history signals", len(hist))

bt_dir = ARTIFACTS / "bt"
bt_dir.mkdir(parents=True, exist_ok=True)
result = Backtester(settings).run(hist, bundle)
result.metrics = compute_metrics(
    result.equity if result.equity is not None else pd.Series(dtype=float),
    result.trades,
    rf_annual=settings.backtest.rf_annual,
    periods=settings.backtest.trading_days_per_year,
    starting_equity=settings.risk.starting_equity,
)
if result.blotter is not None:
    result.blotter.to_parquet(bt_dir / "trades.parquet", index=False)
if result.equity is not None:
    equity_to_frame(result.equity).to_parquet(bt_dir / "equity.parquet", index=False)
(bt_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
print(json.dumps(result.metrics, indent=2))

eq = pd.read_parquet(bt_dir / "equity.parquet")
fig = equity_figure(eq)
fig.show()
write_html(fig, ARTIFACTS / "report.html")
imps = load_model(ARTIFACTS / "model.joblib").feature_importances()
if not imps.empty:
    importance_figure(imps).show()
print("wrote", ARTIFACTS / "report.html")
"""
    ),
    md(
        """## Package — `ml_signal_model_ddMMMyyyy`

Writes the joblib + JSON sidecar into `MyDrive/ml_train` (and `artifacts/`) for the other project.
The other project needs: the `.joblib` file, `selected_features.json`, and the same hourly feature recipe (`PanelFeatureEngineer` + daily join). No scaler.
"""
    ),
    code(
        r"""
import json
import shutil
from datetime import date
from pathlib import Path

stamp = date.today().strftime("%d%b%Y")  # e.g. 08Sep2026
stem = f"ml_signal_model_{stamp}"
src_joblib = ARTIFACTS / "model.joblib"
src_json = ARTIFACTS / "model.json"
src_feats = ARTIFACTS / "selected_features.json"
assert src_joblib.is_file(), "run M16b first"
for dest_dir in (DRIVE_DIR, ARTIFACTS):
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_joblib, dest_dir / f"{stem}.joblib")
    shutil.copy2(src_json, dest_dir / f"{stem}.json")
    shutil.copy2(src_feats, dest_dir / f"{stem}_features.json")

meta = json.loads(src_json.read_text(encoding="utf-8"))
print("packaged", stem)
print("model", meta.get("model_name"), "features", len(meta.get("features") or []))
print("aggregate", json.dumps(meta.get("aggregate_metrics"), indent=2))
print("Drive copies:")
for p in sorted(DRIVE_DIR.glob(f"{stem}*")):
    print(" ", p, p.stat().st_size)
"""
    ),
]


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "cells": cells,
}

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", NB_PATH, "cells", len(cells))
