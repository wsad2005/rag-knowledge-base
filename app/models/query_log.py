"""问答日志表：记录每次提问的检索参数、耗时与来源，便于事后复盘。"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, IdMixin, utcnow


class QueryLog(IdMixin, Base):
    __tablename__ = "query_logs"

    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text)
    #: llm / extractive / no_context
    answer_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: JSON 字符串：来源摘要，复盘时不必再查向量库
    sources_json: Mapped[str | None] = mapped_column(Text)
    error_code: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
