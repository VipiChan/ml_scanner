"""Timescale connection pool. The only module besides schema/repository that opens SQL."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from ml_scan.exceptions import StorageError

if TYPE_CHECKING:
    from ml_scan.config import Settings


def dsn_from_settings(settings: Settings) -> str:
    ts = settings.timescale
    return (
        f"host={ts.host} port={ts.port} dbname={ts.db} "
        f"user={ts.user} password={ts.password}"
    )


class Database:
    """psycopg pool bound to the dedicated `scan_trade` database."""

    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 8) -> None:
        try:
            self._pool = ConnectionPool(
                conninfo=dsn,
                min_size=min_size,
                max_size=max_size,
                kwargs={"row_factory": dict_row, "autocommit": False, "connect_timeout": 5},
                open=True,
                timeout=10,
                reconnect_timeout=5,
            )
        except Exception as exc:
            raise StorageError(
                "Could not create a Timescale connection pool. "
                "Check TIMESCALE_* in .env (existing scan_trade instance on port 5433)."
            ) from exc
        self.dsn = dsn

    @classmethod
    def from_settings(
        cls, settings: Settings, *, min_size: int = 1, max_size: int = 8
    ) -> Database:
        if not settings.timescale.password:
            raise StorageError(
                "TIMESCALE_PASSWORD is empty. Copy .env.example to .env and set "
                "the password used by your existing scan_trade Timescale instance."
            )
        return cls(dsn_from_settings(settings), min_size=min_size, max_size=max_size)

    @contextmanager
    def connection(self) -> Iterator[Connection]:
        try:
            acquired = self._pool.connection()
        except Exception as exc:
            raise StorageError(f"Timescale query failed: {exc}") from exc
        with acquired as conn:
            yield conn

    def close(self) -> None:
        self._pool.close()
