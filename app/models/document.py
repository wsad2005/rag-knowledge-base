"""文档主表。"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, IdMixin, TimestampMixin

if TYPE_CHECKING:  # 仅用于类型检查，避免与 chunk.py 循环导入
    from app.models.chunk import Chunk


class DocumentStatus(enum.StrEnum):
    """文档处理状态机：pending -> processing -> ready | failed。"""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Document(IdMixin, TimestampMixin, Base):
    """一份已上传文档的元信息。

    ``content_hash`` 唯一索引实现重复文件检测：同一内容只入库一次。
    ``storage_path`` 存相对路径（相对 ``data/uploads``），避免把本机绝对路径写进数据库。
    """

    __tablename__ = "documents"

    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str | None] = mapped_column(String(255))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    file_ext: Mapped[str] = mapped_column(String(16), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(255), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=DocumentStatus.PENDING.value, index=True
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    page_count: Mapped[int | None] = mapped_column(Integer)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Chunk.chunk_index",
    )

    __table_args__ = (Index("ix_documents_status_created", "status", "created_at"),)

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<Document id={self.id} name={self.filename!r} status={self.status}>"
