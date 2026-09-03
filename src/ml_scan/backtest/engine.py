"""Event-driven hourly portfolio backtester. Next-open fills; SL/TP path replay."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from ml_scan.backtest.costs import IndianCostModel
from ml_scan.backtest.fills import FillModel
from ml_scan.backtest.signals import align_next_open
from ml_scan.config import Settings
from ml_scan.data.schemas import BacktestResult, MTFBundle, Trade
from ml_scan.storage.buckets import NSE_TZ, ensure_ist


@dataclass
class _Open:
    symbol: str
    setup: str
    entry_ts: pd.Timestamp
    entry_px: float
    qty: int
    stop: float
    target: float
    score: float
    buy_slip: float
    buy_costs: float
    bars_held: int = 0


class Backtester:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.costs = IndianCostModel(settings)
        self.fills = FillModel(settings)

    def run(self, signals: pd.DataFrame, panels: MTFBundle) -> BacktestResult:
        hourly = panels.hourly.copy()
        hourly["ts"] = hourly["ts"].map(ensure_ist)
        hourly = hourly.sort_values(["ts", "symbol"]).reset_index(drop=True)
        aligned = align_next_open(signals, hourly)
        aligned = aligned.dropna(subset=["fill_ts"])
        pending: dict[tuple[str, pd.Timestamp], pd.Series] = {}
        for _, row in aligned.iterrows():
            pending[(str(row["symbol"]), ensure_ist(row["fill_ts"]))] = row

        equity0 = float(self.settings.risk.starting_equity)
        cash = equity0
        opens: dict[str, _Open] = {}
        trades: list[Trade] = []
        equity_rows: list[dict] = []
        max_hold = int(self.settings.risk.max_hold_sessions)
        max_conc = int(self.settings.risk.max_concurrent)
        risk_frac = float(self.settings.risk.risk_frac)

        by_ts = {ts: part for ts, part in hourly.groupby("ts", sort=True)}
        timestamps = sorted(by_ts)

        for ts in timestamps:
            bars = by_ts[ts]
            bar_map = {str(row["symbol"]): row for _, row in bars.iterrows()}

            # Exits first on this bar.
            for symbol in list(opens):
                pos = opens[symbol]
                bar = bar_map.get(symbol)
                if bar is None:
                    continue
                hit = self.fills.exit_fill(bar, pos.stop, pos.target)
                reason = None
                exit_px = None
                if hit:
                    exit_px, reason = hit
                else:
                    pos.bars_held += 1
                    if self.settings.risk.exit_on_time_stop and pos.bars_held >= max_hold * 7:
                        exit_px = float(bar["close"]) * (1.0 - self.fills.slip)
                        reason = "time_stop"
                if reason is None or exit_px is None:
                    continue
                trade = self._close(pos, ts, exit_px, reason)
                cash += trade.qty * exit_px - trade.costs
                trades.append(trade)
                del opens[symbol]

            # Entries at this bar's open (signal was on a prior bar).
            for symbol, bar in bar_map.items():
                key = (symbol, ts)
                if key not in pending or symbol in opens or len(opens) >= max_conc:
                    continue
                sig = pending[key]
                stop = float(sig["sl_px"])
                target = float(sig["tp_px"])
                fill_px = self.fills.next_open_fill(pd.Series({"open": float(bar["open"])}))
                risk_ps = max(fill_px - stop, fill_px * 0.003)
                qty = int((cash * risk_frac) / risk_ps) if risk_ps > 0 else 0
                if qty <= 0:
                    continue
                notional = qty * fill_px
                if notional > cash * 0.95:
                    qty = int((cash * 0.95) / fill_px)
                    notional = qty * fill_px
                if qty <= 0:
                    continue
                session = _session(ts)
                charge = self.costs.charge("buy", notional, symbol, session)
                cash -= notional + charge.statutory
                opens[symbol] = _Open(
                    symbol=symbol,
                    setup="ml_swing",
                    entry_ts=ts,
                    entry_px=fill_px,
                    qty=qty,
                    stop=stop,
                    target=target,
                    score=float(sig.get("score", 0.0)),
                    buy_slip=charge.slippage,
                    buy_costs=charge.statutory,
                )

            mtm = cash + sum(pos.qty * float(bar_map[pos.symbol]["close"]) for pos in opens.values() if pos.symbol in bar_map)
            equity_rows.append({"ts": ts, "equity": mtm, "cash": cash, "n_open": len(opens)})

        # Flatten leftovers at last close.
        if timestamps:
            last_ts = timestamps[-1]
            last_bars = {str(row["symbol"]): row for _, row in by_ts[last_ts].iterrows()}
            for symbol, pos in list(opens.items()):
                bar = last_bars.get(symbol)
                if bar is None:
                    continue
                exit_px = float(bar["close"]) * (1.0 - self.fills.slip)
                trade = self._close(pos, last_ts, exit_px, "eod_flatten")
                trades.append(trade)

        equity = pd.Series(
            {row["ts"]: row["equity"] for row in equity_rows},
            name="equity",
        ).sort_index()
        blotter = _blotter(trades)
        return BacktestResult(
            run_ts=pd.Timestamp.now(tz=NSE_TZ),
            config_hash=self.settings.config_hash(),
            trades=tuple(trades),
            equity=equity,
            blotter=blotter,
        )

    def _close(self, pos: _Open, exit_ts: pd.Timestamp, exit_px: float, reason: str) -> Trade:
        sell_notional = pos.qty * exit_px
        sell = self.costs.charge("sell", sell_notional, pos.symbol, _session(exit_ts))
        costs = pos.buy_costs + sell.statutory
        slip = pos.buy_slip + sell.slippage
        gross = pos.qty * (exit_px - pos.entry_px)
        net = gross - costs
        risk = max(pos.entry_px - pos.stop, pos.entry_px * 0.003)
        r_mult = ((exit_px - pos.entry_px) / risk) if risk else 0.0
        return Trade(
            symbol=pos.symbol,
            setup=pos.setup,
            entry_ts=pos.entry_ts,
            exit_ts=exit_ts,
            entry_px=pos.entry_px,
            exit_px=exit_px,
            qty=pos.qty,
            stop=pos.stop,
            target=pos.target,
            exit_reason=reason,
            pnl_gross=gross,
            costs=costs,
            slippage=slip,
            pnl_net=net,
            r_multiple_realized=r_mult,
        )


def _session(ts: pd.Timestamp) -> date:
    return ensure_ist(ts).date()


def _blotter(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "symbol",
                "entry_ts",
                "exit_ts",
                "entry_px",
                "exit_px",
                "qty",
                "side",
                "reason",
                "gross_pnl",
                "costs",
                "net_pnl",
            ]
        )
    rows = []
    for t in trades:
        rows.append(
            {
                "symbol": t.symbol,
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
                "entry_px": t.entry_px,
                "exit_px": t.exit_px,
                "qty": t.qty,
                "side": "long",
                "reason": t.exit_reason,
                "gross_pnl": t.pnl_gross,
                "costs": t.costs,
                "net_pnl": t.pnl_net,
            }
        )
    return pd.DataFrame(rows)
