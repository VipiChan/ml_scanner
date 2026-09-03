# ml_scan

Standalone Nifty 500 multi-timeframe ML scanner, backtester, and execution pipeline.

It reads the existing TimescaleDB `scan_trade` warehouse (hourly / daily / 15m),
builds a cross-sectional feature panel, trains a tree model, and emits
entry / stop / target levels.

See [user_command.md](user_command.md) for milestone acceptance commands.

Copy `.env.example` to `.env` and set `TIMESCALE_PASSWORD` to the same value
used by your existing `scan_trade` Timescale instance (`localhost:5433`).
The warehouse name stays `scan_trade`; this project only reads it.
