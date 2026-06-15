"""Connector schemas (Sprints 06 & 08)."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ConnectorType(str, Enum):
    rest = "rest"
    graphql = "graphql"
    sql = "sql"
    nosql = "nosql"
    mcp = "mcp"


class ConnectorStatus(str, Enum):
    active = "active"
    inactive = "inactive"
    error = "error"


class ConnectorCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    type: ConnectorType
    configuration: dict = Field(default_factory=dict)


class ConnectorUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=128)
    configuration: dict | None = None
    status: ConnectorStatus | None = None


class ConnectorRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    type: str
    status: str
    configuration: dict
    last_checked_at: datetime | None = None
    created_at: datetime


class ConnectorHealth(BaseModel):
    id: uuid.UUID
    status: ConnectorStatus
    detail: str | None = None
