"""Stdlib logging with optional Rich handler."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(level: int = logging.INFO, *, force: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    handler = logging.StreamHandler(sys.stderr)
    try:
        from rich.logging import RichHandler

        handler = RichHandler(rich_tracebacks=True, show_path=False)
        fmt = "%(message)s"
    except ImportError:
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    logging.basicConfig(level=level, format=fmt, handlers=[handler], force=True)
    logging.getLogger("ml_scan").setLevel(level)
    _CONFIGURED = True
