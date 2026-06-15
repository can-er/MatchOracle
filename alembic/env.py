"""Alembic environment (Sprint 01).

The database URL comes from the application settings (``MO_DATABASE_URL``) unless
one is already provided on the Alembic config (used by tests). Custom column
types (``GUID``, ``JSONType``) are rendered with an explicit import so migrations
stay dialect-agnostic.
"""

from __future__ import annotations

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from app.config import settings
from app.db import models  # noqa: F401 — register models on the metadata
from app.db.base import GUID, Base, JSONType

config = context.config

# Let callers (e.g. tests) override the URL; otherwise use the app settings.
if not config.get_main_option("sqlalchemy.url"):
    config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def render_item(type_: str, obj: object, autogen_context: object) -> str | bool:
    """Render our custom TypeDecorators with a usable import in migrations."""
    if type_ == "type" and isinstance(obj, (GUID, JSONType)):
        autogen_context.imports.add(  # type: ignore[attr-defined]
            "from app.db.base import GUID, JSONType"
        )
        return f"{obj.__class__.__name__}()"
    return False


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        render_item=render_item,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=connection.dialect.name == "sqlite",
            render_item=render_item,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
