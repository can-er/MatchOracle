"""Shared schema primitives."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    """Generic pagination envelope used by list endpoints."""

    items: list[T]
    total: int
    limit: int
    offset: int


class Message(BaseModel):
    message: str


class ErrorResponse(BaseModel):
    detail: str
    code: str = Field(default="error")
