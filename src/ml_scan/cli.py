"""Typer CLI: coverage, universe, data, features, ml (compare/boruta/vif/optimize/train), scan, backtest, report, e2e."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ml_scan.config import load_settings
from ml_scan.logging import setup_logging
from ml_scan import ops

app = typer.Typer(help="Nifty 500 MTF ML scanner")
universe_app = typer.Typer(help="Universe snapshot and liquidity")
data_app = typer.Typer(help="MTF reads and alignment")
features_app = typer.Typer(help="Feature engineering and labels")
ml_app = typer.Typer(help="Selection and training")
backtest_app = typer.Typer(help="Backtest run and metrics")
e2e_app = typer.Typer(help="End-to-end pipelines")
app.add_typer(universe_app, name="universe")
app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(ml_app, name="ml")
app.add_typer(backtest_app, name="backtest")
app.add_typer(e2e_app, name="e2e")


@app.callback()
def _root() -> None:
    setup_logging()


@app.command("coverage")
def coverage_cmd(symbols: str = typer.Option("RELIANCE", "--symbols")) -> None:
    frame = ops.coverage(load_settings(), symbols=symbols)
    typer.echo(frame.to_string(index=False))


@universe_app.command("snapshot")
def universe_snapshot(
    source: str = typer.Option(..., "--source"),
) -> None:
    dest = ops.universe_snapshot(source)
    typer.echo(f"wrote {dest}")


@universe_app.command("liquid")
def universe_liquid(
    adtv_min: float = typer.Option(50_000_000, "--adtv-min"),
    smoke_n: int = typer.Option(8, "--smoke-n"),
    out: Path = typer.Option(Path("data/universe/liquid_universe.csv"), "--out"),
) -> None:
    liquid, smoke = ops.universe_liquid(load_settings(), adtv_min=adtv_min, smoke_n=smoke_n, out=out)
    typer.echo(f"liquid={liquid} smoke={smoke}")


@data_app.command("align")
def data_align(
    symbols: str = typer.Option(..., "--symbols"),
    start: str = typer.Option(..., "--start"),
    end: str = typer.Option(..., "--end"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    dest = ops.align_symbols(load_settings(), symbols=symbols, start=start, end=end, out=out)
    typer.echo(f"wrote {dest}")


@features_app.command("hourly")
def features_hourly(
    universe: Path = typer.Option(..., "--universe"),
    start: str = typer.Option(..., "--start"),
    end: str = typer.Option(..., "--end"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    dest = ops.features_hourly(load_settings(), universe=universe, start=start, end=end, out=out)
    typer.echo(f"wrote {dest}")


@features_app.command("mtf")
def features_mtf(
    inbound: Path = typer.Option(..., "--in"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    dest = ops.features_mtf(load_settings(), inbound=inbound, out=out)
    typer.echo(f"wrote {dest}")


@features_app.command("label")
def features_label(
    inbound: Path = typer.Option(..., "--in"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    dest = ops.features_label(load_settings(), inbound=inbound, out=out)
    typer.echo(f"wrote {dest}")


@features_app.command("qc")
def features_qc(
    inbound: Path = typer.Option(..., "--in"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    dest = ops.features_qc(load_settings(), inbound=inbound, out=out)
    typer.echo(f"wrote {dest}")


@ml_app.command("boruta")
def ml_boruta(
    inbound: Path = typer.Option(..., "--in"),
    out: Path = typer.Option(..., "--out"),
    max_iter: int = typer.Option(50, "--max-iter"),
    estimator_name: str = typer.Option(None, "--estimator-name", help="rf | xgboost | lightgbm; default = configured model"),
    sample_rows: int = typer.Option(None, "--sample-rows", help="Subsample rows (spread across symbols) before fitting; large universes only"),
) -> None:
    dest = ops.ml_boruta(
        load_settings(), inbound=inbound, out=out, max_iter=max_iter, estimator_name=estimator_name, sample_rows=sample_rows
    )
    typer.echo(f"wrote {dest}")


@ml_app.command("compare")
def ml_compare(
    inbound: Path = typer.Option(..., "--in"),
    features: Path = typer.Option(None, "--features", help="Feature list JSON; omit to use every QC'd feature"),
    out: Path = typer.Option(None, "--out"),
    sample_rows: int = typer.Option(None, "--sample-rows", help="Subsample rows (spread across symbols) before fitting; large universes only"),
) -> None:
    ops.ml_compare_models(load_settings(), inbound=inbound, features=features, out=out, sample_rows=sample_rows)


@ml_app.command("optimize")
def ml_optimize(
    inbound: Path = typer.Option(..., "--in"),
    features: Path = typer.Option(..., "--features"),
    model_name: str = typer.Option(..., "--model", help="rf | xgboost | lightgbm"),
    out: Path = typer.Option(..., "--out"),
    n_iter: int = typer.Option(20, "--n-iter"),
    sample_rows: int = typer.Option(None, "--sample-rows", help="Subsample rows (spread across symbols) before searching; large universes only"),
) -> None:
    dest = ops.ml_optimize(
        load_settings(), inbound=inbound, features=features, model_name=model_name, out=out, n_iter=n_iter, sample_rows=sample_rows
    )
    typer.echo(f"wrote {dest}")


@ml_app.command("vif")
def ml_vif(
    inbound: Path = typer.Option(..., "--in"),
    features: Path = typer.Option(..., "--features"),
    out: Path = typer.Option(..., "--out"),
    max_vif: float = typer.Option(10.0, "--max-vif"),
) -> None:
    dest = ops.ml_vif(load_settings(), inbound=inbound, features=features, out=out, max_vif=max_vif)
    typer.echo(f"wrote {dest}")


@ml_app.command("train")
def ml_train(
    inbound: Path = typer.Option(..., "--in"),
    features: Path = typer.Option(..., "--features"),
    out: Path = typer.Option(..., "--out"),
    model_name: str = typer.Option(None, "--model", help="rf | xgboost | lightgbm; default = configured model"),
    params: Path = typer.Option(None, "--params", help="JSON file of hyperparameter overrides, e.g. from ml optimize"),
) -> None:
    overrides = json.loads(Path(params).read_text())["best_params"] if params else None
    dest = ops.ml_train(load_settings(), inbound=inbound, features=features, out=out, model_name=model_name, params=overrides)
    typer.echo(f"wrote {dest}")


@app.command("scan")
def scan_cmd(
    asof: str = typer.Option("latest", "--asof"),
    universe: Path = typer.Option(..., "--universe"),
    out: Path = typer.Option(..., "--out"),
    model_path: Path = typer.Option(None, "--model-path", help="Default: data/artifacts/model.joblib"),
    features_path: Path = typer.Option(None, "--features-path", help="Default: data/artifacts/selected_features.json"),
) -> None:
    dest = ops.scan_latest(
        load_settings(), asof=asof, universe=universe, out=out, model_path=model_path, features_path=features_path
    )
    typer.echo(f"wrote {dest}")


@backtest_app.command("run")
def backtest_run(
    signals: Path = typer.Option(..., "--signals"),
    start: str = typer.Option(..., "--start"),
    end: str = typer.Option(..., "--end"),
    out: Path = typer.Option(..., "--out"),
    model_path: Path = typer.Option(None, "--model-path", help="Default: data/artifacts/model.joblib"),
    features_path: Path = typer.Option(None, "--features-path", help="Default: data/artifacts/selected_features.json"),
) -> None:
    dest = ops.backtest_run(
        load_settings(), signals=signals, start=start, end=end, out=out, model_path=model_path, features_path=features_path
    )
    typer.echo(f"wrote {dest}")


@backtest_app.command("metrics")
def backtest_metrics(run: Path = typer.Option(..., "--run")) -> None:
    ops.backtest_metrics(run)


@app.command("report")
def report_cmd(
    run: Path = typer.Option(..., "--run"),
    out: Path = typer.Option(..., "--out"),
    model_path: Path = typer.Option(None, "--model-path", help="Default: data/artifacts/model.joblib"),
) -> None:
    dest = ops.write_report(run, out, model_path=model_path)
    typer.echo(f"wrote {dest}")


@e2e_app.command("smoke")
def e2e_smoke() -> None:
    dest = ops.e2e_smoke(load_settings())
    typer.echo(f"ok {dest}")


@e2e_app.command("liquid")
def e2e_liquid(max_symbols: int = typer.Option(80, "--max-symbols")) -> None:
    dest = ops.e2e_liquid(load_settings(), max_symbols=max_symbols)
    typer.echo(f"ok {dest}")


if __name__ == "__main__":
    app()
