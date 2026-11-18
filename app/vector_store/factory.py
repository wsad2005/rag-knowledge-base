"""向量库工厂。"""

from __future__ import annotations

from app.core.config import Settings
from app.core.errors import VectorStoreError
from app.rag.embeddings.base import EmbeddingProvider
from app.vector_store.base import VectorStore
from app.vector_store.chroma_store import ChromaVectorStore


def build_vector_store(settings: Settings, embedder: EmbeddingProvider) -> VectorStore:
    if settings.vector_store_provider != "chroma":  # pragma: no cover - 配置枚举已限制
        raise VectorStoreError(f"未知的向量库提供方：{settings.vector_store_provider}")

    try:
        dimension: int | None = embedder.dimension
    except Exception:  # noqa: BLE001 - 远程 Embedding 首次调用前维度未知
        dimension = None

    return ChromaVectorStore(
        settings.chroma_path,
        collection_base=settings.chroma_collection,
        embedding_signature=embedder.signature,
        dimension=dimension,
    )
