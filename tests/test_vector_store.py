"""vector_store 模块测试。"""

import pytest

from rag_knowledge_base.text_splitter import Chunk
from rag_knowledge_base.vector_store import InMemoryVectorStore, VectorRecord, VectorStore


def record(text: str, vector: list[float]) -> VectorRecord:
    return VectorRecord(chunk=Chunk(text=text, source="s", index=0), vector=vector)


def test_search_orders_by_similarity_desc() -> None:
    store = InMemoryVectorStore()
    store.add(
        [
            record("a", [1.0, 0.0]),
            record("b", [0.0, 1.0]),
            record("c", [1.0, 1.0]),
        ]
    )
    hits = store.search([1.0, 0.0], top_k=2)
    assert [hit.chunk.text for hit in hits] == ["a", "c"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].score > hits[1].score


def test_top_k_larger_than_store_returns_all() -> None:
    store = InMemoryVectorStore()
    store.add([record("a", [1.0])])
    assert len(store.search([1.0], top_k=5)) == 1


def test_empty_store_returns_empty_list() -> None:
    assert InMemoryVectorStore().search([1.0], top_k=3) == []


def test_invalid_top_k_raises() -> None:
    with pytest.raises(ValueError):
        InMemoryVectorStore().search([1.0], top_k=0)


def test_zero_vector_scores_zero() -> None:
    store = InMemoryVectorStore()
    store.add([record("z", [0.0, 0.0])])
    assert store.search([1.0, 0.0], top_k=1)[0].score == 0.0


def test_store_satisfies_protocol() -> None:
    assert isinstance(InMemoryVectorStore(), VectorStore)
