"""document_loader 模块测试。"""

from pathlib import Path

import pytest

from rag_knowledge_base.document_loader import DocumentLoader, TextLoader


def test_text_loader_reads_utf8(tmp_path: Path) -> None:
    path = tmp_path / "doc.txt"
    path.write_text("第一行\n第二行", encoding="utf-8")
    documents = TextLoader(path).load()
    assert len(documents) == 1
    assert documents[0].content == "第一行\n第二行"
    assert documents[0].source == str(path)


def test_text_loader_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TextLoader(tmp_path / "missing.txt").load()


def test_text_loader_satisfies_protocol() -> None:
    assert isinstance(TextLoader("x.txt"), DocumentLoader)
