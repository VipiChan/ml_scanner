from __future__ import annotations

from datetime import date

from ml_scan.backtest.costs import IndianCostModel
from ml_scan.config import CostsConfig


def test_cnc_round_trip_stt_stamp_and_slip() -> None:
    cfg = CostsConfig()
    model = IndianCostModel(cfg)
    buy_n = 100_000.0
    sell_n = 110_000.0
    trip = model.round_trip(buy_n, sell_n, "RELIANCE", date(2024, 1, 2), date(2024, 1, 5))
    buy = model.charge("buy", buy_n, "RELIANCE", date(2024, 1, 2))
    sell = model.charge("sell", sell_n, "RELIANCE", date(2024, 1, 5))
    assert buy.stt == buy_n * cfg.stt_buy_pct
    assert sell.stt == sell_n * cfg.stt_sell_pct
    assert buy.stamp == buy_n * cfg.stamp_buy_pct
    assert sell.stamp == 0.0
    assert abs(buy.slippage - buy_n * 0.0003) < 1e-9
    assert abs(sell.slippage - sell_n * 0.0003) < 1e-9
    assert trip.stt == buy.stt + sell.stt
    assert trip.total == buy.total + sell.total
