"""Zerodha equity cost model (CNC default; MIS config-only). Copied from scan_trade."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from ml_scan.config import CostsConfig, Settings

_CRORE = 10_000_000.0
_MIS_BROKERAGE_PCT = 0.0003
_MIS_STT_SELL_PCT = 0.00025
_MIS_STAMP_BUY_PCT = 0.00003


@dataclass(frozen=True)
class CostBreakdown:
    brokerage: float
    stt: float
    stamp: float
    txn: float
    sebi: float
    gst: float
    dp: float
    ipft: float
    slippage: float
    total: float

    @property
    def statutory(self) -> float:
        return self.total - self.slippage

    def __add__(self, other: CostBreakdown) -> CostBreakdown:
        return CostBreakdown(
            brokerage=self.brokerage + other.brokerage,
            stt=self.stt + other.stt,
            stamp=self.stamp + other.stamp,
            txn=self.txn + other.txn,
            sebi=self.sebi + other.sebi,
            gst=self.gst + other.gst,
            dp=self.dp + other.dp,
            ipft=self.ipft + other.ipft,
            slippage=self.slippage + other.slippage,
            total=self.total + other.total,
        )


def _breakdown(
    *,
    brokerage: float,
    stt: float,
    stamp: float,
    txn: float,
    sebi: float,
    gst: float,
    dp: float,
    ipft: float,
    slippage: float,
) -> CostBreakdown:
    total = brokerage + stt + stamp + txn + sebi + gst + dp + ipft + slippage
    return CostBreakdown(
        brokerage=brokerage,
        stt=stt,
        stamp=stamp,
        txn=txn,
        sebi=sebi,
        gst=gst,
        dp=dp,
        ipft=ipft,
        slippage=slippage,
        total=total,
    )


class IndianCostModel:
    """Statutory charges on fill notional. Slippage is reported, not double-applied."""

    def __init__(self, settings: Settings | CostsConfig) -> None:
        self.cfg = settings.costs if isinstance(settings, Settings) else settings

    def charge(
        self,
        side: Literal["buy", "sell"],
        notional: float,
        symbol: str,
        session: date,
        *,
        charge_dp: bool = True,
    ) -> CostBreakdown:
        _ = (symbol, session)
        n = float(notional)
        if n < 0:
            raise ValueError("notional must be >= 0")
        cfg = self.cfg
        buy = side == "buy"
        sebi = n * (cfg.sebi_per_crore / _CRORE)
        txn = n * cfg.nse_txn_pct
        ipft = n * (cfg.ipft_per_crore / _CRORE)
        slip = n * cfg.slippage_pct_per_side

        if cfg.product == "MIS":
            brokerage = min(_MIS_BROKERAGE_PCT * n, cfg.brokerage_flat_cap)
            stt = (_MIS_STT_SELL_PCT * n) if not buy else 0.0
            stamp = (_MIS_STAMP_BUY_PCT * n) if buy else 0.0
            dp = 0.0
        else:
            brokerage = 0.0 if cfg.brokerage_pct == 0.0 else min(cfg.brokerage_pct * n, cfg.brokerage_flat_cap)
            stt = (cfg.stt_buy_pct if buy else cfg.stt_sell_pct) * n
            stamp = cfg.stamp_buy_pct * n if buy else 0.0
            dp = cfg.dp_sell_flat if (not buy and charge_dp) else 0.0

        gst = cfg.gst_pct * (brokerage + sebi + txn)
        return _breakdown(
            brokerage=brokerage,
            stt=stt,
            stamp=stamp,
            txn=txn,
            sebi=sebi,
            gst=gst,
            dp=dp,
            ipft=ipft,
            slippage=slip,
        )

    def round_trip(
        self,
        buy_notional: float,
        sell_notional: float,
        symbol: str,
        buy_day: date,
        sell_day: date,
    ) -> CostBreakdown:
        buy = self.charge("buy", buy_notional, symbol, buy_day)
        sell = self.charge("sell", sell_notional, symbol, sell_day, charge_dp=True)
        return buy + sell
