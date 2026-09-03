"""Orchestrator functions shared by the CLI and notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml_scan.backtest.engine import Backtester
from ml_scan.backtest.metrics import compute_metrics, equity_to_frame
from ml_scan.config import Settings, load_settings, project_root
from ml_scan.data.liquidity import filter_universe
from ml_scan.data.schemas import MTFBundle
from ml_scan.data.timescale_adapter import TimescaleAdapter
from ml_scan.data.universe import (
    copy_snapshot,
    liquid_path,
    load_snapshot,
    load_symbol_list,
    smoke_path,
    snapshot_path,
    write_symbol_frame,
)
from ml_scan.execution.scanner import InferenceScanner, history_signals
from ml_scan.features.htf import join_htf_features
from ml_scan.features.panel import PanelFeatureEngineer
from ml_scan.features.qc import run_qc, select_xy, training_xy
from ml_scan.features.target import SwingLabeler
from ml_scan.ml_engine.artifacts import load_model, save_model
from ml_scan.ml_engine.estimator import MLEstimator, compare_models
from ml_scan.ml_engine.optimize import random_search
from ml_scan.ml_engine.selector import BorutaSelector, VIFPruner, load_feature_list
from ml_scan.ml_engine.splitter import PurgedWalkForward
from ml_scan.reporting.charts import equity_figure, importance_figure, write_html
from ml_scan.storage.buckets import NSE_TZ, ensure_ist


def _resolve(path: Path | str) -> Path:
    dest = Path(path)
    return dest if dest.is_absolute() else project_root() / dest


def _parse_symbols(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [s.strip() for s in str(raw).split(",") if s.strip()]


def _read_table(path: Path | str) -> pd.DataFrame:
    dest = _resolve(path)
    if dest.suffix == ".parquet":
        return pd.read_parquet(dest)
    return pd.read_csv(dest)


def _write_table(frame: pd.DataFrame, path: Path | str) -> Path:
    dest = _resolve(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.suffix == ".parquet":
        frame.to_parquet(dest, index=False)
    else:
        frame.to_csv(dest, index=False)
    return dest


def coverage(settings: Settings | None = None, symbols: str = "RELIANCE") -> pd.DataFrame:
    settings = settings or load_settings()
    adapter = TimescaleAdapter(settings)
    try:
        return adapter.coverage(_parse_symbols(symbols) or ["RELIANCE"], "60minute")
    finally:
        adapter.close()


def universe_snapshot(source: str, dest: Path | str | None = None) -> Path:
    return copy_snapshot(source, _resolve(dest) if dest else snapshot_path())


def universe_liquid(
    settings: Settings | None = None,
    *,
    adtv_min: float | None = None,
    smoke_n: int | None = None,
    out: Path | str | None = None,
) -> tuple[Path, Path]:
    settings = settings or load_settings()
    cfg = settings.liquidity.model_copy()
    if adtv_min is not None:
        cfg.adtv_min_inr = float(adtv_min)
    n_smoke = int(smoke_n if smoke_n is not None else cfg.smoke_n)
    universe = load_snapshot()
    names = [s for s in universe["symbol"].tolist() if s != settings.universe.benchmark_symbol]
    adapter = TimescaleAdapter(settings)
    try:
        end = pd.Timestamp.now(tz=NSE_TZ)
        start = end - pd.Timedelta(days=80)
        daily = adapter.read(names + [settings.universe.benchmark_symbol], "day", start, end)
    finally:
        adapter.close()
    liquid = filter_universe(universe, daily, cfg)
    liquid = liquid.sort_values("adtv_20", ascending=False).reset_index(drop=True)
    out_path = write_symbol_frame(liquid, out or liquid_path())
    smoke = liquid.head(n_smoke)
    smoke_out = write_symbol_frame(smoke, smoke_path())
    return out_path, smoke_out


def align_symbols(
    settings: Settings | None = None,
    *,
    symbols: str,
    start: str,
    end: str,
    out: Path | str,
) -> Path:
    from ml_scan.data.mtf_aligner import MTFAligner

    settings = settings or load_settings()
    adapter = TimescaleAdapter(settings)
    try:
        bundle = adapter.read_mtf(_parse_symbols(symbols), start, end)
    finally:
        adapter.close()
    aligned = MTFAligner().align_to_hourly(bundle)
    return _write_table(aligned, out)


def features_hourly(
    settings: Settings | None = None,
    *,
    universe: Path | str,
    start: str,
    end: str,
    out: Path | str,
) -> Path:
    settings = settings or load_settings()
    symbols = load_symbol_list(universe)
    adapter = TimescaleAdapter(settings)
    try:
        hourly = adapter.read(symbols, "60minute", start, end)
    finally:
        adapter.close()
    featured = PanelFeatureEngineer(
        mode=settings.features.hourly_mode,
        atr_period=settings.features.atr_period,
    ).transform(hourly)
    return _write_table(featured.reset_index(drop=True), out)


def features_mtf(
    settings: Settings | None = None,
    *,
    inbound: Path | str,
    out: Path | str,
) -> Path:
    settings = settings or load_settings()
    hourly = _read_table(inbound)
    if hourly.empty:
        return _write_table(hourly, out)
    symbols = sorted(hourly["symbol"].dropna().astype(str).unique().tolist())
    start = hourly["ts"].min()
    end = hourly["ts"].max()
    if not symbols or pd.isna(start) or pd.isna(end):
        return _write_table(hourly, out)
    adapter = TimescaleAdapter(settings)
    try:
        bundle = adapter.read_mtf(symbols + [settings.universe.benchmark_symbol], start, end)
        bench = adapter.read([settings.universe.benchmark_symbol], "day", start, end)
    finally:
        adapter.close()
    joined = join_htf_features(hourly, bundle, benchmark_daily=bench, config=settings.features)
    return _write_table(joined, out)


def features_label(
    settings: Settings | None = None,
    *,
    inbound: Path | str,
    out: Path | str,
) -> Path:
    settings = settings or load_settings()
    panel = _read_table(inbound)
    tgt = settings.target
    labeled = SwingLabeler(
        atr_col=tgt.atr_col,
        sl_mult=tgt.sl_mult,
        tp_r=tgt.tp_r,
        max_sessions=tgt.max_sessions,
        stop_wins_same_bar=tgt.stop_wins_same_bar,
    ).transform(panel)
    path = _write_table(labeled, out)
    print(labeled["y"].value_counts(dropna=False).to_string())
    print(labeled["y_reason"].value_counts(dropna=False).to_string())
    return path


def features_qc(
    settings: Settings | None = None,
    *,
    inbound: Path | str,
    out: Path | str,
) -> Path:
    settings = settings or load_settings()
    panel = _read_table(inbound)
    _, report = run_qc(
        panel,
        missing_threshold=settings.features.missing_threshold,
        out_path=_resolve(out),
    )
    print(json.dumps({"n_rows": report["n_rows"], "n_feature_cols": report["n_feature_cols"], "leakage_ok": report["leakage"]["ok"]}, indent=2))
    return _resolve(out)


def ml_boruta(
    settings: Settings | None = None,
    *,
    inbound: Path | str,
    out: Path | str,
    max_iter: int | None = None,
    estimator_name: str | None = None,
) -> Path:
    settings = settings or load_settings()
    panel = _read_table(inbound)
    X, y = training_xy(panel)
    selector = BorutaSelector(
        max_iter=int(max_iter or settings.ml.boruta_max_iter),
        random_state=settings.ml.random_state,
        estimator_name=estimator_name or settings.ml.model,
    ).fit(X, y)
    dest = selector.save(_resolve(out))
    print(f"boruta ({selector.estimator_name}) kept {len(selector.features_)} features")
    return dest


def ml_vif(
    settings: Settings | None = None,
    *,
    inbound: Path | str,
    features: Path | str,
    out: Path | str,
    max_vif: float | None = None,
) -> Path:
    settings = settings or load_settings()
    panel = _read_table(inbound)
    X, _y = training_xy(panel)
    keep = load_feature_list(features)
    X = X[[c for c in keep if c in X.columns]]
    pruner = VIFPruner(max_vif=float(max_vif if max_vif is not None else settings.ml.max_vif)).fit(X)
    dest = pruner.save(_resolve(out))
    print(f"vif kept {len(pruner.features_)} features")
    return dest


def ml_train(
    settings: Settings | None = None,
    *,
    inbound: Path | str,
    features: Path | str,
    out: Path | str,
    model_name: str | None = None,
    params: dict | None = None,
) -> Path:
    settings = settings or load_settings()
    panel = _read_table(inbound)
    keep = load_feature_list(features)
    X, y, labeled = select_xy(panel, keep)
    est = MLEstimator(
        model=model_name or settings.ml.model,
        random_state=settings.ml.random_state,
        params=params,
        use_class_weight=settings.ml.use_class_weight,
    )
    splitter = PurgedWalkForward(n_splits=settings.ml.n_splits, embargo_sessions=settings.ml.embargo_sessions)
    try:
        est.fit_walk_forward(X, y, labeled, splitter)
    except ValueError:
        est.fit(X, y)
    dest = save_model(est, _resolve(out))
    print(json.dumps(est.fold_metrics_, indent=2))
    return dest


def ml_compare_models(
    settings: Settings | None = None,
    *,
    inbound: Path | str,
    features: Path | str | list[str] | None = None,
    out: Path | str | None = None,
) -> pd.DataFrame:
    """Baseline check: fit every candidate model on the same feature set and folds.

    `features=None` uses every surviving QC'd numeric column (the "initial
    feature set" before Boruta/VIF ever run). Prints and optionally saves the
    fold-weighted accuracy / balanced accuracy / precision / recall / ROC-AUC
    for each model so the notebook can pick a winner before feature selection.
    """
    settings = settings or load_settings()
    panel = _read_table(inbound)
    keep = features if isinstance(features, list) else (load_feature_list(features) if features else None)
    X, y, work = select_xy(panel, keep)
    splitter = PurgedWalkForward(n_splits=settings.ml.n_splits, embargo_sessions=settings.ml.embargo_sessions)
    summary, _fitted = compare_models(
        X, y, work, splitter=splitter, random_state=settings.ml.random_state, use_class_weight=settings.ml.use_class_weight
    )
    print(summary.round(4).to_string())
    if out is not None:
        dest = _resolve(out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(summary.reset_index().to_json(orient="records", indent=2), encoding="utf-8")
    return summary


def ml_optimize(
    settings: Settings | None = None,
    *,
    inbound: Path | str,
    features: Path | str,
    model_name: str,
    out: Path | str,
    n_iter: int = 20,
) -> Path:
    """Randomized hyperparameter search scored on the same purged walk-forward folds."""
    settings = settings or load_settings()
    panel = _read_table(inbound)
    keep = load_feature_list(features)
    X, y, work = select_xy(panel, keep)
    splitter = PurgedWalkForward(n_splits=settings.ml.n_splits, embargo_sessions=settings.ml.embargo_sessions)
    best_params, results = random_search(
        X, y, work, model_name=model_name, splitter=splitter, n_iter=n_iter, random_state=settings.ml.random_state
    )
    dest = _resolve(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps({"model_name": model_name, "best_params": best_params}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"optimize ({model_name}) best_params = {best_params}")
    results_path = dest.with_name(dest.stem + "_cv_results.csv")
    results.to_csv(results_path, index=False)
    return dest


def scan_latest(
    settings: Settings | None = None,
    *,
    asof: str = "latest",
    universe: Path | str,
    out: Path | str,
    model_path: Path | str | None = None,
    features_path: Path | str | None = None,
    labeled_path: Path | str | None = None,
) -> Path:
    settings = settings or load_settings()
    model_path = model_path or (project_root() / "data" / "artifacts" / "model.joblib")
    features_path = features_path or (project_root() / "data" / "artifacts" / "selected_features.json")
    scanner = InferenceScanner(settings)
    scanner.load_artifacts(model_path, features_path)
    if labeled_path:
        panel = _read_table(labeled_path)
    else:
        symbols = load_symbol_list(universe)
        end = pd.Timestamp.now(tz=NSE_TZ) if asof == "latest" else ensure_ist(pd.Timestamp(asof))
        start = end - pd.Timedelta(days=max(settings.data.smoke_hourly_days, 80))
        adapter = TimescaleAdapter(settings)
        try:
            hourly = adapter.read(symbols, "60minute", start, end, completed_asof=end)
            bundle = adapter.read_mtf(symbols + [settings.universe.benchmark_symbol], start, end, completed_asof=end)
            bench = adapter.read([settings.universe.benchmark_symbol], "day", start, end, completed_asof=end)
        finally:
            adapter.close()
        featured = PanelFeatureEngineer(mode=settings.features.hourly_mode, atr_period=settings.features.atr_period).transform(hourly)
        panel = join_htf_features(featured.reset_index(drop=True), bundle, benchmark_daily=bench, config=settings.features)
    result = scanner.run(asof, panel)
    return _write_table(result, out)


def backtest_run(
    settings: Settings | None = None,
    *,
    signals: Path | str,
    start: str,
    end: str,
    out: Path | str,
    labeled_path: Path | str | None = None,
) -> Path:
    settings = settings or load_settings()
    dest = _resolve(out)
    dest.mkdir(parents=True, exist_ok=True)
    sig = _read_table(signals)
    symbols = sorted(sig["symbol"].unique().tolist())
    adapter = TimescaleAdapter(settings)
    try:
        bundle = adapter.read_mtf(symbols, start, end)
    finally:
        adapter.close()
    if "score" not in sig.columns and labeled_path:
        labeled = _read_table(labeled_path)
        scanner = InferenceScanner(settings, estimator=load_model(project_root() / "data" / "artifacts" / "model.joblib"))
        scanner.features = load_feature_list(project_root() / "data" / "artifacts" / "selected_features.json")
        scored = scanner.score_panel(labeled)
        sig = history_signals(scored, threshold=settings.ml.score_threshold)
        _write_table(sig, dest / "scan_history.parquet")
    result = Backtester(settings).run(sig, bundle)
    result.metrics = compute_metrics(
        result.equity if result.equity is not None else pd.Series(dtype=float),
        result.trades,
        rf_annual=settings.backtest.rf_annual,
        periods=settings.backtest.trading_days_per_year,
        starting_equity=settings.risk.starting_equity,
    )
    if result.blotter is not None:
        result.blotter.to_parquet(dest / "trades.parquet", index=False)
    if result.equity is not None:
        equity_to_frame(result.equity).to_parquet(dest / "equity.parquet", index=False)
    (dest / "metrics.json").write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    return dest


def backtest_metrics(run_dir: Path | str) -> dict:
    dest = _resolve(run_dir)
    payload = json.loads((dest / "metrics.json").read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2))
    return payload


def write_report(run_dir: Path | str, out: Path | str, model_path: Path | str | None = None) -> Path:
    dest = _resolve(run_dir)
    equity = pd.read_parquet(dest / "equity.parquet")
    fig = equity_figure(equity)
    extras = []
    model_file = Path(model_path) if model_path else project_root() / "data" / "artifacts" / "model.joblib"
    if model_file.is_file():
        est = load_model(model_file)
        imps = est.feature_importances()
        if not imps.empty:
            extras.append(importance_figure(imps, title="Model feature importance"))
    html = write_html(fig, out)
    if extras:
        write_html(extras[0], Path(html).with_name("importance.html"))
    return html


def e2e_smoke(settings: Settings | None = None, *, dest_name: str = "e2e_smoke") -> Path:
    settings = settings or load_settings()
    root = project_root() / "data" / "artifacts" / dest_name
    root.mkdir(parents=True, exist_ok=True)
    if not smoke_path().is_file():
        universe_liquid(settings)
    end = pd.Timestamp.now(tz=NSE_TZ)
    start = end - pd.Timedelta(days=settings.data.smoke_hourly_days)
    hourly_path = features_hourly(settings, universe=smoke_path(), start=str(start.date()), end=str(end.date()), out=root / "feat_hourly.parquet")
    mtf_path = features_mtf(settings, inbound=hourly_path, out=root / "feat_mtf.parquet")
    labeled_path = features_label(settings, inbound=mtf_path, out=root / "labeled.parquet")
    features_qc(settings, inbound=labeled_path, out=root / "feature_qc_report.json")
    boruta_path = ml_boruta(settings, inbound=labeled_path, out=root / "boruta_features.json")
    selected = ml_vif(settings, inbound=labeled_path, features=boruta_path, out=root / "selected_features.json")
    model_path = ml_train(settings, inbound=labeled_path, features=selected, out=root / "model.joblib")
    scan_path = scan_latest(
        settings,
        universe=smoke_path(),
        out=root / "scan_latest.csv",
        model_path=model_path,
        features_path=selected,
        labeled_path=labeled_path,
    )
    scanner = InferenceScanner(settings)
    scanner.load_artifacts(model_path, selected)
    scored = scanner.score_panel(_read_table(labeled_path))
    history = history_signals(scored, threshold=settings.ml.score_threshold)
    hist_path = _write_table(history, root / "scan_history.parquet")
    bt_dir = backtest_run(
        settings,
        signals=hist_path,
        start=str(start.date()),
        end=str(end.date()),
        out=root / "bt",
    )
    write_report(bt_dir, root / "report.html", model_path=model_path)
    print(f"e2e artifacts at {root} scan={scan_path}")
    return root


def e2e_liquid(settings: Settings | None = None, *, max_symbols: int = 80) -> Path:
    settings = settings or load_settings()
    if not liquid_path().is_file():
        universe_liquid(settings)
    liquid = pd.read_csv(liquid_path())
    subset = liquid.head(int(max_symbols))
    tmp = project_root() / "data" / "universe" / "liquid_cap.csv"
    write_symbol_frame(subset, tmp)
    # Temporarily point smoke path content by running the same pipeline on the cap file.
    original = settings
    root = project_root() / "data" / "artifacts" / "e2e_liquid"
    root.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp.now(tz=NSE_TZ)
    start = end - pd.Timedelta(days=min(original.data.hourly_days, 400))
    hourly_path = features_hourly(original, universe=tmp, start=str(start.date()), end=str(end.date()), out=root / "feat_hourly.parquet")
    mtf_path = features_mtf(original, inbound=hourly_path, out=root / "feat_mtf.parquet")
    labeled_path = features_label(original, inbound=mtf_path, out=root / "labeled.parquet")
    boruta_path = ml_boruta(original, inbound=labeled_path, out=root / "boruta_features.json")
    selected = ml_vif(original, inbound=labeled_path, features=boruta_path, out=root / "selected_features.json")
    model_path = ml_train(original, inbound=labeled_path, features=selected, out=root / "model.joblib")
    scan_latest(original, universe=tmp, out=root / "scan_latest.csv", model_path=model_path, features_path=selected, labeled_path=labeled_path)
    scanner = InferenceScanner(original)
    scanner.load_artifacts(model_path, selected)
    history = history_signals(scanner.score_panel(_read_table(labeled_path)), threshold=original.ml.score_threshold)
    hist_path = _write_table(history, root / "scan_history.parquet")
    backtest_run(original, signals=hist_path, start=str(start.date()), end=str(end.date()), out=root / "bt")
    return root
