"""Nifty 500 snapshot loader. Refresh via Kite lives in scan_trade; this repo reads a CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_scan.config import project_root
from ml_scan.exceptions import IngestError
from ml_scan.storage.buckets import NSE_TZ


def snapshot_path() -> Path:
    return project_root() / "data" / "universe" / "nifty500_mapped.csv"


def liquid_path() -> Path:
    return project_root() / "data" / "universe" / "liquid_universe.csv"


def smoke_path() -> Path:
    return project_root() / "data" / "universe" / "smoke_symbols.csv"


def resolve_csv_path(raw: str | Path) -> Path:
    text = str(raw).strip().strip('"').strip("'")
    if not text:
        raise IngestError("Universe CSV path is empty.")
    path = Path(text).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(project_root() / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise IngestError(f"Universe CSV not found: {path}.")


def load_snapshot(path: Path | str | None = None) -> pd.DataFrame:
    snap = resolve_csv_path(path) if path is not None else snapshot_path()
    if not snap.is_file():
        raise IngestError(
            f"No mapped snapshot yet at {snap}. "
            "Copy nifty500_mapped.csv from scan_trade or run `ml-scan universe snapshot`."
        )
    frame = pd.read_csv(snap)
    if "instrument_token" in frame.columns:
        frame["instrument_token"] = pd.array(frame["instrument_token"], dtype="Int64")
    if "symbol" not in frame.columns:
        raise IngestError(f"Snapshot {snap} is missing a symbol column.")
    frame["symbol"] = frame["symbol"].astype(str).str.strip()
    return frame


def copy_snapshot(source: Path | str, dest: Path | None = None) -> Path:
    src = resolve_csv_path(source)
    dest_path = dest or snapshot_path()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    frame = load_snapshot(src)
    frame.to_csv(dest_path, index=False)
    return dest_path


def load_symbol_list(path: Path | str) -> list[str]:
    frame = pd.read_csv(resolve_csv_path(path))
    if "symbol" not in frame.columns:
        raise IngestError(f"{path} needs a symbol column")
    return [str(s).strip() for s in frame["symbol"].tolist() if str(s).strip()]


def write_symbol_frame(frame: pd.DataFrame, path: Path | str) -> Path:
    dest = Path(path)
    if not dest.is_absolute():
        dest = project_root() / dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    if "asof_ist" not in out.columns:
        out["asof_ist"] = pd.Timestamp.now(tz=NSE_TZ).isoformat()
    out.to_csv(dest, index=False)
    return dest
