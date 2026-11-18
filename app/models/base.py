"""ORM 基类与公共列。"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def new_id() -> str:
    """32 位十六进制 UUID（去掉横线，便于作为向量库 ID）。"""
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


class IdMixin:
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )
