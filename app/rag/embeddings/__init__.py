"""Embedding 子包：接口 + 三种实现 + 工厂。"""

from app.rag.embeddings.base import EmbeddingProvider
from app.rag.embeddings.factory import build_embedder
from app.rag.embeddings.hashing import HashingEmbedder

__all__ = ["EmbeddingProvider", "HashingEmbedder", "build_embedder"]
