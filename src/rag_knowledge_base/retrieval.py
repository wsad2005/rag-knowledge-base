"""检索：把「查询文本」变成「最相关的文档块」列表。"""

from .embeddings import EmbeddingClient
from .vector_store import SearchHit, VectorStore


class Retriever:
    """组合向量化客户端与向量存储的检索器。"""

    def __init__(self, embedder: EmbeddingClient, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, query: str, top_k: int = 3) -> list[SearchHit]:
        """把查询向量化并返回最相关的块；空白查询视为无效。"""
        text = query.strip()
        if not text:
            raise ValueError("查询为空")
        return self._store.search(self._embedder.embed(text), top_k)
