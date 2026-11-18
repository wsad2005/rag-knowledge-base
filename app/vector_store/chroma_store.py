"""ChromaDB 向量库实现（持久化到本地目录）。

设计取舍：

- **集合名带模型指纹**：``knowledge_chunks_hashing-...-512``。不同 Embedding 模型的向量
  不在同一语义空间，混在一个集合里会让距离失去意义；带指纹后切换模型是“新建集合”，
  不会污染旧数据（旧集合可用 ``scripts/reindex.py`` 清理）。
- **元数据完整回指原文**：每条向量都带 ``document_id/chunk_id/chunk_index/document_name/
  page_number``，检索层无需再回表就能给出引用来源。
- **维度校验**：建集合时把维度写进集合元数据，打开时比对；不一致直接报错，而不是让
  Chroma 在写入时抛出难以理解的底层异常。
"""

from __future__ import annotations

import contextlib
import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.core.errors import VectorStoreError
from app.core.logging import get_logger
from app.vector_store.base import VectorMatch, VectorRecord, VectorStore

logger = get_logger(__name__)

_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9._-]")
_UNKNOWN_PAGE = -1


def _collection_name(base: str, signature: str) -> str:
    """生成符合 Chroma 规则（3~63 字符、字母数字 ._-）且带上模型指纹的集合名。"""
    raw = _INVALID_NAME_CHARS.sub("-", f"{base}_{signature}")
    if len(raw) > 60:
        digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:8]
        raw = f"{raw[:48]}-{digest}"
    return raw[:63].strip("-.") or "knowledge_chunks"


class ChromaVectorStore(VectorStore):
    """基于 ChromaDB PersistentClient 的向量库。"""

    def __init__(
        self,
        path: Path,
        *,
        collection_base: str = "knowledge_chunks",
        embedding_signature: str = "default",
        dimension: int | None = None,
    ) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:  # pragma: no cover - 依赖缺失属于部署问题
            raise VectorStoreError(
                "未安装 chromadb，无法使用向量库",
                detail={"install": "pip install -r requirements.txt"},
            ) from exc

        path.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._dimension = dimension
        self._collection_name = _collection_name(collection_base, embedding_signature)
        self._signature = embedding_signature

        try:
            self._client = chromadb.PersistentClient(
                path=str(path),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            self._collection = self._open_collection()
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001 - chroma 底层异常类型繁多
            raise VectorStoreError(
                "向量库初始化失败", detail={"path": str(path), "reason": str(exc)}
            ) from exc

        logger.info("向量库就绪：collection=%s", self._collection_name)

    # ------------------------------------------------------------------ 初始化
    def _open_collection(self) -> Any:
        existing = {collection.name for collection in self._client.list_collections()}
        if self._collection_name in existing:
            collection = self._client.get_collection(
                name=self._collection_name, embedding_function=None
            )
            stored_dim = (collection.metadata or {}).get("embedding_dimension")
            if self._dimension and stored_dim and int(stored_dim) != self._dimension:
                raise VectorStoreError(
                    f"向量维度不一致：集合为 {stored_dim}，当前模型为 {self._dimension}",
                    detail={
                        "collection": self._collection_name,
                        "hint": "请重建集合或更换 CHROMA_COLLECTION",
                    },
                )
            return collection

        metadata: dict[str, Any] = {
            "hnsw:space": "cosine",
            "embedding_signature": self._signature,
        }
        if self._dimension:
            metadata["embedding_dimension"] = self._dimension
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata=metadata,
            embedding_function=None,
        )

    # ------------------------------------------------------------------ 写入
    def add(self, records: Sequence[VectorRecord]) -> int:
        if not records:
            return 0
        self._check_dimensions([record.embedding for record in records])

        ids = [record.id for record in records]
        documents = [record.content for record in records]
        embeddings = [record.embedding for record in records]
        metadatas = [self._build_metadata(record) for record in records]

        try:
            self._collection.upsert(
                ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError("向量写入失败", detail={"reason": str(exc)}) from exc
        return len(ids)

    @staticmethod
    def _build_metadata(record: VectorRecord) -> dict[str, str | int | float | bool]:
        """Chroma 的元数据只接受标量，None 需要编码成哨兵值。"""
        metadata: dict[str, str | int | float | bool] = {
            "document_id": record.document_id,
            "chunk_id": record.chunk_id,
            "chunk_index": record.chunk_index,
            "document_name": record.document_name,
            "page_number": record.page_number if record.page_number is not None else _UNKNOWN_PAGE,
            "section": record.section or "",
        }
        metadata.update(record.metadata)
        return metadata

    def _check_dimensions(self, embeddings: Sequence[Sequence[float]]) -> None:
        if not self._dimension:
            return
        for embedding in embeddings:
            if len(embedding) != self._dimension:
                raise VectorStoreError(
                    f"向量维度不一致：期望 {self._dimension}，实际 {len(embedding)}",
                    detail={"collection": self._collection_name},
                )

    # ------------------------------------------------------------------ 检索
    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[VectorMatch]:
        if top_k <= 0:
            return []
        self._check_dimensions([query_embedding])

        where: dict[str, Any] | None = None
        if document_ids:
            where = {"document_id": {"$in": list(document_ids)}}

        try:
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError("向量检索失败", detail={"reason": str(exc)}) from exc

        return self._to_matches(result)

    @staticmethod
    def _to_matches(result: dict[str, Any]) -> list[VectorMatch]:
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        matches: list[VectorMatch] = []
        for position, vector_id in enumerate(ids):
            metadata = metadatas[position] or {}
            distance = float(distances[position])
            # cosine 空间下 distance = 1 - similarity，裁剪到 [-1, 1] 防止浮点误差
            score = max(-1.0, min(1.0, 1.0 - distance))
            page = metadata.get("page_number", _UNKNOWN_PAGE)
            matches.append(
                VectorMatch(
                    id=str(vector_id),
                    content=documents[position] or "",
                    score=score,
                    document_id=str(metadata.get("document_id", "")),
                    chunk_id=str(metadata.get("chunk_id", vector_id)),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    document_name=str(metadata.get("document_name", "")),
                    page_number=None if int(page) == _UNKNOWN_PAGE else int(page),
                    section=str(metadata.get("section") or "") or None,
                )
            )
        return matches

    # ------------------------------------------------------------------ 删除与统计
    def delete_document(self, document_id: str) -> int:
        try:
            existing = self._collection.get(where={"document_id": document_id}, include=[])
            ids = existing.get("ids") or []
            if ids:
                self._collection.delete(ids=ids)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(
                "删除文档向量失败", detail={"document_id": document_id, "reason": str(exc)}
            ) from exc
        return len(ids)

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError("读取向量数量失败", detail={"reason": str(exc)}) from exc

    def health(self) -> dict[str, object]:
        return {
            "provider": "chromadb",
            "status": "ok",
            "collection": self._collection_name,
            "vectors": self.count(),
            "embedding_signature": self._signature,
        }

    # ------------------------------------------------------------------ 测试辅助
    def reset(self) -> None:
        """清空当前集合（仅测试与脚本使用，不暴露为 API）。"""
        with contextlib.suppress(Exception):  # 集合不存在时忽略
            self._client.delete_collection(self._collection_name)
        self._collection = self._open_collection()
