"""向量库接口。

业务层只依赖这个协议，因此把 ChromaDB 换成 pgvector / Qdrant 时，
只需要再实现一份 ``VectorStore``，不用改检索与问答代码。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(slots=True)
class VectorRecord:
    """待写入的一条向量记录（元数据必须能回到原文）。"""

    id: str
    embedding: list[float]
    content: str
    document_id: str
    chunk_id: str
    chunk_index: int
    document_name: str
    page_number: int | None = None
    section: str | None = None
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(slots=True)
class VectorMatch:
    """检索命中的一条记录。"""

    id: str
    content: str
    score: float
    document_id: str
    chunk_id: str
    chunk_index: int
    document_name: str
    page_number: int | None = None
    section: str | None = None


class VectorStore(ABC):
    """向量库抽象。"""

    @abstractmethod
    def add(self, records: Sequence[VectorRecord]) -> int:
        """批量写入，返回写入条数。"""

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[VectorMatch]:
        """按余弦相似度召回 Top-K。"""

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """删除某文档的全部向量，返回删除条数。"""

    @abstractmethod
    def count(self) -> int:
        """当前集合内的向量总数。"""

    @abstractmethod
    def health(self) -> dict[str, object]:
        """健康检查信息。"""
