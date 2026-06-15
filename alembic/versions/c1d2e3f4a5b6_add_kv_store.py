"""add kv_store (durable cache for the serverless / Postgres backend)

Revision ID: c1d2e3f4a5b6
Revises: dd35aa8af719
Create Date: 2026-06-15 02:30:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from app.db.base import JSONType

# revision identifiers, used by Alembic.
revision: str = "c1d2e3f4a5b6"
down_revision: str | Sequence[str] | None = "dd35aa8af719"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "kv_store",
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("value", JSONType(), nullable=True),
        sa.Column("expires_at", sa.Float(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("key"),
    )


def downgrade() -> None:
    op.drop_table("kv_store")
