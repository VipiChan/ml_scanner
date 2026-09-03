"""Load YAML config plus env secrets. Timescale DB stays `scan_trade`."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ml_scan.exceptions import ConfigError

ALLOWED_TIMESCALE_DB = "scan_trade"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_yaml_path() -> Path:
    return project_root() / "configs" / "default.yaml"


class Secrets(BaseSettings):
    """Timescale credentials from the environment only."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    timescale_host: str | None = None
    timescale_port: int | None = None
    timescale_db: str | None = None
    timescale_user: str | None = None
    timescale_password: str = ""


class UniverseConfig(BaseModel):
    index_name: str = "NIFTY500"
    constituents_url: str = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    series: str = "EQ"
    benchmark_symbol: str = "NIFTY 50"
    exchange: str = "NSE"


class DataConfig(BaseModel):
    daily_days: int = 730
    hourly_days: int = 730
    minute15_days: int = 90
    smoke_hourly_days: int = 90
    parquet_export_dir: str = "data/artifacts"


class TimescaleConfig(BaseModel):
    host: str = "localhost"
    port: int = 5433
    db: str = ALLOWED_TIMESCALE_DB
    chunk_5m: str = "7d"
    compress_after: str = "14d"
    cagg_refresh_minutes: int = 1
    ca_rel_diff: float = 0.02
    user: str = "scan_trade"
    password: str = ""

    @field_validator("db")
    @classmethod
    def db_must_be_scan_trade(cls, value: str) -> str:
        if value != ALLOWED_TIMESCALE_DB:
            raise ValueError(
                f"Timescale database must be {ALLOWED_TIMESCALE_DB!r}, not {value!r}."
            )
        return value


class LiquidityConfig(BaseModel):
    adtv_window: int = 20
    adtv_min_inr: float = 50_000_000
    smoke_n: int = 8


class FeatureConfig(BaseModel):
    hourly_mode: Literal["full", "lite"] = "full"
    atr_period: int = 14
    daily_ema_fast: int = 50
    daily_ema_slow: int = 200
    daily_rs_lookback: int = 63
    m15_rsi_length: int = 14
    missing_threshold: float = 0.10


class TargetConfig(BaseModel):
    atr_col: str = "ATR"
    sl_mult: float = 2.0
    tp_r: float = 2.0
    max_sessions: int = 10
    stop_wins_same_bar: bool = True


class MLConfig(BaseModel):
    model: Literal["lightgbm", "xgboost", "rf"] = "lightgbm"
    n_splits: int = 4
    embargo_sessions: int = 10
    max_vif: float = 10.0
    boruta_max_iter: int = 50
    random_state: int = 42
    score_threshold: float = 0.55
    use_class_weight: bool = True


class RiskConfig(BaseModel):
    risk_frac: float = 0.01
    max_concurrent: int = 5
    max_hold_sessions: int = 10
    starting_equity: float = 1_000_000.0
    exit_on_stop: bool = True
    exit_on_target: bool = True
    exit_on_time_stop: bool = True


class CostsConfig(BaseModel):
    product: Literal["CNC", "MIS"] = "CNC"
    slippage_pct_per_side: float = 0.0003
    brokerage_pct: float = 0.0
    brokerage_flat_cap: float = 20.0
    stt_buy_pct: float = 0.001
    stt_sell_pct: float = 0.001
    stamp_buy_pct: float = 0.00015
    nse_txn_pct: float = 0.0000307
    sebi_per_crore: float = 10.0
    gst_pct: float = 0.18
    dp_sell_flat: float = 15.34
    ipft_per_crore: float = 0.01


class BacktestConfig(BaseModel):
    fill: str = "next_open"
    stop_wins_same_bar: bool = True
    rf_annual: float = 0.0
    trading_days_per_year: int = 252
    max_dd_constraint: float = -0.25


class Settings(BaseModel):
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    timescale: TimescaleConfig = Field(default_factory=TimescaleConfig)
    liquidity: LiquidityConfig = Field(default_factory=LiquidityConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    yaml_path: Path | None = None

    def config_hash(self) -> str:
        payload = self.model_dump(
            exclude={"yaml_path": True, "timescale": {"password": True}},
        )
        canonical = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> Settings:
        load_dotenv(project_root() / ".env", override=False)
        yaml_path = Path(path) if path else default_yaml_path()
        if not yaml_path.is_file():
            raise ConfigError(f"Config file not found: {yaml_path}")

        raw: dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        secrets = Secrets()
        ts = dict(raw.get("timescale") or {})
        if secrets.timescale_host:
            ts["host"] = secrets.timescale_host
        if secrets.timescale_port is not None:
            ts["port"] = secrets.timescale_port
        if secrets.timescale_db:
            ts["db"] = secrets.timescale_db
        if secrets.timescale_user:
            ts["user"] = secrets.timescale_user
        if secrets.timescale_password:
            ts["password"] = secrets.timescale_password
        raw["timescale"] = ts
        raw["yaml_path"] = yaml_path
        return cls.model_validate(raw)


def load_settings(path: Path | str | None = None) -> Settings:
    load_dotenv(project_root() / ".env", override=False)
    return Settings.from_yaml(path)
