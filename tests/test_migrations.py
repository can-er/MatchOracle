"""Alembic migration smoke test (Sprint 01): upgrade/downgrade on a vierge DB."""

from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from sqlalchemy import create_engine, inspect

from alembic import command

EXPECTED_TABLES = {
    "predictions",
    "agent_results",
    "connectors",
    "outcomes",
    "feedback",
    "accuracy_snapshots",
    "tenants",
    "users",
    "audit_logs",
}


def _config(db_path: Path) -> Config:
    cfg = Config("alembic.ini")
    cfg.set_main_option("script_location", "alembic")
    cfg.set_main_option("sqlalchemy.url", f"sqlite+pysqlite:///{db_path.as_posix()}")
    return cfg


def _tables(db_path: Path) -> set[str]:
    engine = create_engine(f"sqlite+pysqlite:///{db_path.as_posix()}")
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_upgrade_head_creates_all_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "m.db"
    command.upgrade(_config(db_path), "head")

    tables = _tables(db_path)
    assert tables >= EXPECTED_TABLES
    assert "alembic_version" in tables


def test_downgrade_base_drops_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "m.db"
    cfg = _config(db_path)
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")

    assert not (EXPECTED_TABLES & _tables(db_path))
