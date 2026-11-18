"""文本块表：一行 = 一个可检索片段，与向量库中的记录一一对应。"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, IdMixin, utcnow

if TYPE_CHECKING:
    from app.models.document import Document


class Chunk(IdMixin, Base):
    __tablename__ = "chunks"

    document_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: PDF 页码（从 1 开始）；TXT/Markdown/DOCX 为 NULL（按段落/标题定位）
    page_number: Mapped[int | None] = mapped_column(Integer)
    #: Markdown 标题 / DOCX 标题样式文本，便于展示上下文来源
    section: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (UniqueConstraint("document_id", "chunk_index", name="uq_chunk_doc_index"),)

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<Chunk doc={self.document_id} idx={self.chunk_index} chars={self.char_count}>"
