"""Apply `sql/migrations/*.sql` (standalone ingest later)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import psycopg

from ml_scan.config import project_root
from ml_scan.exceptions import StorageError
from ml_scan.storage.db import Database, dsn_from_settings

if TYPE_CHECKING:
    from ml_scan.config import Settings

log = logging.getLogger(__name__)

_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
  filename   text        PRIMARY KEY,
  applied_at timestamptz NOT NULL DEFAULT now()
)
"""


def migrations_dir() -> Path:
    return project_root() / "sql" / "migrations"


def split_sql_statements(script: str) -> list[str]:
    """Split on ';' outside quotes and $tag$ dollar-quotes."""
    statements: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(script)
    in_single = False
    in_double = False
    dollar_tag: str | None = None

    while i < n:
        ch = script[i]
        nxt = script[i + 1] if i + 1 < n else ""

        if dollar_tag is not None:
            if script.startswith(dollar_tag, i):
                buf.append(dollar_tag)
                i += len(dollar_tag)
                dollar_tag = None
                continue
            buf.append(ch)
            i += 1
            continue

        if in_single:
            buf.append(ch)
            if ch == "'" and nxt == "'":
                buf.append(nxt)
                i += 2
                continue
            if ch == "'":
                in_single = False
            i += 1
            continue

        if in_double:
            buf.append(ch)
            if ch == '"':
                in_double = False
            i += 1
            continue

        if ch == "-" and nxt == "-":
            i += 2
            while i < n and script[i] not in "\n\r":
                i += 1
            continue

        if ch == "/" and nxt == "*":
            i += 2
            while i < n - 1 and not (script[i] == "*" and script[i + 1] == "/"):
                i += 1
            i += 2
            continue

        if ch == "'":
            in_single = True
            buf.append(ch)
            i += 1
            continue

        if ch == '"':
            in_double = True
            buf.append(ch)
            i += 1
            continue

        if ch == "$":
            match = re.match(r"\$[A-Za-z0-9_]*\$", script[i:])
            if match:
                dollar_tag = match.group(0)
                buf.append(dollar_tag)
                i += len(dollar_tag)
                continue

        if ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
            i += 1
            continue

        buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def apply_migrations(settings: Settings, *, database: Database | None = None) -> list[str]:
    """Run pending SQL files in name order. Idempotent (IF NOT EXISTS + ledger)."""
    folder = migrations_dir()
    if not folder.is_dir():
        raise StorageError(f"Migrations directory not found: {folder}")

    files = sorted(p for p in folder.glob("*.sql") if p.is_file())
    if not files:
        raise StorageError(f"No *.sql files in {folder}")

    owns_db = database is None
    if owns_db:
        _ensure_reachable(settings)
        db = Database.from_settings(settings, min_size=1, max_size=2)
    else:
        db = database
    applied: list[str] = []
    try:
        with db.connection() as conn:
            previous_autocommit = conn.autocommit
            conn.autocommit = True
            try:
                conn.execute(_MIGRATIONS_TABLE)
                done = {
                    row["filename"]
                    for row in conn.execute("SELECT filename FROM schema_migrations").fetchall()
                }
                for path in files:
                    name = path.name
                    if name in done:
                        log.info("migration already applied: %s", name)
                        continue
                    script = path.read_text(encoding="utf-8")
                    for statement in split_sql_statements(script):
                        conn.execute(statement)
                    conn.execute(
                        "INSERT INTO schema_migrations (filename) VALUES (%s)",
                        (name,),
                    )
                    applied.append(name)
                    log.info("applied migration %s", name)
            finally:
                conn.autocommit = previous_autocommit
    except StorageError:
        raise
    except Exception as exc:
        raise StorageError(f"db upgrade failed: {exc}") from exc
    finally:
        if owns_db:
            db.close()
    return applied


def _ensure_reachable(settings: Settings) -> None:
    dsn = dsn_from_settings(settings)
    try:
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            conn.execute("SELECT 1")
    except Exception as exc:
        raise StorageError(
            "Cannot connect to Timescale. "
            "Ensure the existing scan_trade Timescale instance is running on port 5433. "
            f"({exc})"
        ) from exc
