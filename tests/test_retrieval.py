"""retrieval 模块测试。"""

import pytest

from rag_knowledge_base.embeddings import MockEmbeddingClient
from rag_knowledge_base.retrieval import Retriever
from rag_knowledge_base.text_splitter import Chunk
from rag_knowledge_base.vector_store import InMemoryVectorStore, VectorRecord


def build_retriever() -> Retriever:
    embedder = MockEmbeddingClient()
    store = InMemoryVectorStore()
    texts = ["向量检索按余弦相似度排序", "今天天气很好"]
    for index, text in enumerate(texts):
        chunk = Chunk(text=text, source="s", index=index)
        store.add([VectorRecord(chunk=chunk, vector=embedder.embed(text))])
    return Retriever(embedder=embedder, store=store)


def test_retrieve_ranks_relevant_text_first() -> None:
    hits = build_retriever().retrieve("余弦相似度", top_k=2)
    assert "余弦相似度" in hits[0].chunk.text


def test_blank_query_raises() -> None:
    retriever = Retriever(embedder=MockEmbeddingClient(), store=InMemoryVectorStore())
    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_top_k_is_forwarded() -> None:
    assert len(build_retriever().retrieve("向量", top_k=1)) == 1
