# ml_scan

Standalone Nifty 500 multi-timeframe ML scanner, backtester, and execution pipeline.

It reads the existing TimescaleDB `scan_trade` warehouse (hourly / daily / 15m),
builds a cross-sectional feature panel, trains a tree model, and emits
entry / stop / target levels.

See [user_command.md](user_command.md) for milestone acceptance commands, and
[user_command.ipynb](user_command.ipynb) for the same checks as an interactive notebook
with extra diagnostics (model bake-off, Boruta, hyperparameter search).

Copy `.env.example` to `.env` and set `TIMESCALE_PASSWORD` to the same value
used by your existing `scan_trade` Timescale instance (`localhost:5433`).
The warehouse name stays `scan_trade`; this project only reads it.

## Training universe

The symbol list used for feature building, training, and scanning is **not** the local
`data/universe/` CSVs by default. It's owned by a separate project (`scan_trade`) and frozen
into `model_training_symbol.csv`, refreshed there whenever the model is retrained. Point
`TRAINING_UNIVERSE` (notebook) or `$UNI` (`user_command.md`) at that file; re-run feature
build through training after every refresh. The local `universe snapshot` / `universe liquid`
/ `e2e liquid` commands still work but are optional tooling, not the production training path.

## CLI surface

`ml-scan --help` (or `python -m ml_scan.cli --help`) lists all command groups:
`coverage`, `universe` (snapshot/liquid), `data` (align), `features` (hourly/mtf/label/qc),
`ml` (compare/boruta/vif/optimize/train), `scan`, `backtest` (run/metrics), `report`, `e2e` (smoke/liquid).

`ml compare`, `ml boruta`, and `ml optimize` accept `--sample-rows` to subsample (spread across
symbols) before the expensive selection/search step — useful on large universes. `ml train`
always fits on the full input file.
