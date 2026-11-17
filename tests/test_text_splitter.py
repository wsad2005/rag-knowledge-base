"""text_splitter 模块测试。"""

import pytest

from rag_knowledge_base.document_loader import Document
from rag_knowledge_base.text_splitter import FixedSizeSplitter, TextSplitter


def test_empty_document_returns_no_chunks() -> None:
    assert FixedSizeSplitter().split(Document(content="   ", source="s")) == []


def test_short_document_is_single_chunk() -> None:
    chunks = FixedSizeSplitter(chunk_size=50).split(Document(content="短文本", source="s"))
    assert len(chunks) == 1
    assert chunks[0].text == "短文本"
    assert chunks[0].source == "s"
    assert chunks[0].index == 0


def test_long_document_chunks_cover_text_with_overlap() -> None:
    text = "abcdefghij" * 10  # 100 字符
    splitter = FixedSizeSplitter(chunk_size=30, overlap=5)
    chunks = splitter.split(Document(content=text, source="s"))
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 30 for chunk in chunks)
    assert chunks[1].text[:5] == chunks[0].text[-5:]  # 相邻块共享 overlap 字符
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].text.startswith(text[:30])
    assert text.endswith(chunks[-1].text)  # 最后一块覆盖到结尾


def test_invalid_params_raise_value_error() -> None:
    with pytest.raises(ValueError):
        FixedSizeSplitter(chunk_size=0)
    with pytest.raises(ValueError):
        FixedSizeSplitter(chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        FixedSizeSplitter(chunk_size=10, overlap=-1)


def test_splitter_satisfies_protocol() -> None:
    assert isinstance(FixedSizeSplitter(), TextSplitter)
