"""Domain exceptions for ml_scan."""


class MLScanError(Exception):
    """Base error for this package."""


class ConfigError(MLScanError):
    """Invalid or missing configuration."""


class StorageError(MLScanError):
    """Timescale connection, migration, or query error."""


class IngestError(MLScanError):
    """Universe snapshot or mapping error."""


class FeatureError(MLScanError):
    """Feature engineering or labeling error."""


class ModelError(MLScanError):
    """Training, selection, or inference error."""


class BacktestError(MLScanError):
    """Backtest engine or fill error."""
