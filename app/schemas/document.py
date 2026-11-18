"""文档相关响应模型。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    chunk_index: int
    content: str
    char_count: int
    page_number: int | None = None
    section: str | None = None


class DocumentSummary(BaseModel):
    """列表项：不含正文，避免列表接口返回大 payload。"""

    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    title: str | None = None
    file_ext: str
    mime_type: str
    source_type: str
    file_size: int
    status: str
    error_message: str | None = None
    chunk_count: int
    char_count: int
    page_count: int | None = None
    created_at: datetime
    processed_at: datetime | None = None


class DocumentDetail(DocumentSummary):
    content_hash: str
    chunks: list[ChunkOut] = Field(default_factory=list, description="按 chunk_index 升序")


class DocumentListResponse(BaseModel):
    total: int = Field(description="满足筛选条件的文档总数")
    skip: int
    limit: int
    items: list[DocumentSummary]


class UploadResponse(BaseModel):
    document: DocumentSummary
    message: str = "上传并解析成功"
