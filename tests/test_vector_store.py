"""向量库测试：真实 ChromaDB（临时目录），覆盖写入、检索、过滤、删除、维度校验与持久化。"""

from __future__ import annotations

import pytest

from app.core.errors import VectorStoreError
from app.vector_store.base import VectorRecord
from app.vector_store.chroma_store import ChromaVectorStore

DIM = 4


def make_record(
    *,
    record_id: str,
    document_id: str = "doc-1",
    document_name: str = "制度.txt",
    chunk_index: int = 0,
    embedding: list[float] | None = None,
    content: str = "内容",
    page: int | None = None,
    section: str | None = None,
) -> VectorRecord:
    return VectorRecord(
        id=record_id,
        embedding=embedding or [1.0, 0.0, 0.0, 0.0],
        content=content,
        document_id=document_id,
        chunk_id=record_id,
        chunk_index=chunk_index,
        document_name=document_name,
        page_number=page,
        section=section,
    )


@pytest.fixture
def store(tmp_path) -> ChromaVectorStore:
    return ChromaVectorStore(
        tmp_path / "chroma",
        collection_base="test_chunks",
        embedding_signature="hashing-unit-test-4",
        dimension=DIM,
    )


class TestWriteAndSearch:
    def test_add_and_count(self, store: ChromaVectorStore) -> None:
        assert store.count() == 0
        written = store.add(
            [
                make_record(record_id="a", embedding=[1.0, 0.0, 0.0, 0.0]),
                make_record(record_id="b", embedding=[0.0, 1.0, 0.0, 0.0], chunk_index=1),
            ]
        )
        assert written == 2
        assert store.count() == 2

    def test_search_orders_by_similarity(self, store: ChromaVectorStore) -> None:
        store.add(
            [
                make_record(record_id="a", content="完全匹配", embedding=[1.0, 0.0, 0.0, 0.0]),
                make_record(record_id="b", content="正交", embedding=[0.0, 1.0, 0.0, 0.0]),
                make_record(record_id="c", content="部分匹配", embedding=[0.7, 0.7, 0.0, 0.0]),
            ]
        )
        matches = store.search([1.0, 0.0, 0.0, 0.0], top_k=3)
        assert matches[0].content == "完全匹配"
        assert matches[0].score == pytest.approx(1.0, abs=1e-5)
        assert matches[0].score > matches[1].score >= matches[2].score

    def test_metadata_roundtrip(self, store: ChromaVectorStore) -> None:
        store.add(
            [
                make_record(
                    record_id="a",
                    page=3,
                    section="报销标准",
                    content="一线城市 600 元",
                    document_name="员工手册.pdf",
                )
            ]
        )
        match = store.search([1.0, 0.0, 0.0, 0.0], top_k=1)[0]
        assert match.document_id == "doc-1"
        assert match.document_name == "员工手册.pdf"
        assert match.page_number == 3
        assert match.section == "报销标准"
        assert match.chunk_id == "a"

    def test_null_page_and_section(self, store: ChromaVectorStore) -> None:
        store.add([make_record(record_id="a")])
        match = store.search([1.0, 0.0, 0.0, 0.0], top_k=1)[0]
        assert match.page_number is None
        assert match.section is None

    def test_top_k_zero_returns_empty(self, store: ChromaVectorStore) -> None:
        store.add([make_record(record_id="a")])
        assert store.search([1.0, 0.0, 0.0, 0.0], top_k=0) == []

    def test_filter_by_document_ids(self, store: ChromaVectorStore) -> None:
        store.add(
            [
                make_record(record_id="a", document_id="doc-1"),
                make_record(record_id="b", document_id="doc-2", content="另一篇"),
            ]
        )
        matches = store.search([1.0, 0.0, 0.0, 0.0], top_k=5, document_ids=["doc-2"])
        assert len(matches) == 1
        assert matches[0].document_id == "doc-2"

    def test_empty_add_is_noop(self, store: ChromaVectorStore) -> None:
        assert store.add([]) == 0


class TestDelete:
    def test_delete_document_removes_only_its_vectors(self, store: ChromaVectorStore) -> None:
        store.add(
            [
                make_record(record_id="a", document_id="doc-1"),
                make_record(record_id="b", document_id="doc-1", chunk_index=1),
                make_record(record_id="c", document_id="doc-2"),
            ]
        )
        removed = store.delete_document("doc-1")
        assert removed == 2
        assert store.count() == 1
        assert store.search([1.0, 0.0, 0.0, 0.0], top_k=5)[0].document_id == "doc-2"

    def test_delete_unknown_document(self, store: ChromaVectorStore) -> None:
        assert store.delete_document("not-exist") == 0


class TestGuards:
    def test_dimension_mismatch_on_write(self, store: ChromaVectorStore) -> None:
        with pytest.raises(VectorStoreError) as excinfo:
            store.add([make_record(record_id="a", embedding=[1.0, 0.0])])
        assert "维度不一致" in excinfo.value.message

    def test_dimension_mismatch_on_query(self, store: ChromaVectorStore) -> None:
        with pytest.raises(VectorStoreError):
            store.search([1.0, 0.0], top_k=1)

    def test_reopen_with_different_dimension_fails(self, tmp_path) -> None:
        ChromaVectorStore(
            tmp_path / "chroma",
            collection_base="test_chunks",
            embedding_signature="same-signature",
            dimension=4,
        )
        with pytest.raises(VectorStoreError) as excinfo:
            ChromaVectorStore(
                tmp_path / "chroma",
                collection_base="test_chunks",
                embedding_signature="same-signature",
                dimension=8,
            )
        assert "向量维度不一致" in excinfo.value.message


class TestPersistence:
    def test_vectors_survive_reopen(self, tmp_path) -> None:
        first = ChromaVectorStore(
            tmp_path / "chroma",
            collection_base="persist",
            embedding_signature="hashing-persist-4",
            dimension=4,
        )
        first.add([make_record(record_id="a", content="持久化内容")])
        assert first.count() == 1

        second = ChromaVectorStore(
            tmp_path / "chroma",
            collection_base="persist",
            embedding_signature="hashing-persist-4",
            dimension=4,
        )
        assert second.count() == 1
        assert second.search([1.0, 0.0, 0.0, 0.0], top_k=1)[0].content == "持久化内容"

    def test_health_payload(self, store: ChromaVectorStore) -> None:
        health = store.health()
        assert health["provider"] == "chromadb"
        assert health["status"] == "ok"
        assert health["vectors"] == 0
        assert "collection" in health


class TestCollectionNaming:
    def test_long_signature_is_shortened(self) -> None:
        from app.vector_store.chroma_store import _collection_name

        name = _collection_name(
            "knowledge_chunks", "sentence-transformers-BAAI_bge-small-zh-v1.5-512-extra-long"
        )
        assert len(name) <= 63
        assert name.isascii()

    def test_illegal_chars_are_replaced(self) -> None:
        from app.vector_store.chroma_store import _collection_name

        name = _collection_name("kb", "model/with:illegal chars")
        assert "/" not in name and ":" not in name and " " not in name
