"""四种格式的解析测试：PDF 页码、DOCX 标题与表格、TXT 编码、Markdown 章节。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.errors import DocumentParseError, UnsupportedFileTypeError
from app.rag.loaders.docx_loader import DocxLoader
from app.rag.loaders.markdown_loader import MarkdownLoader
from app.rag.loaders.pdf_loader import PdfLoader
from app.rag.loaders.registry import load_document
from app.rag.loaders.text_loader import TextLoader
from tests import samples


def _write(tmp_path: Path, name: str, content: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


class TestTextLoader:
    def test_utf8_paragraphs(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "制度.txt", samples.TXT_CONTENT.encode("utf-8"))
        parsed = TextLoader().load(path)
        assert len(parsed.blocks) == 5  # 标题 + 四条
        assert parsed.encoding in {"utf-8", "utf-8-sig"}
        assert "15 个工作日" in parsed.blocks[1].text

    def test_gbk_fallback(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "gbk.txt", samples.gbk_txt_bytes())
        parsed = TextLoader().load(path)
        assert parsed.encoding in {"gb18030", "latin-1"}
        assert "GBK 编码" in parsed.blocks[0].text

    def test_empty_file(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "empty.txt", b"   \n\n  ")
        with pytest.raises(DocumentParseError):
            TextLoader().load(path)


class TestMarkdownLoader:
    def test_sections_and_code_fence(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "deploy.md", samples.MD_CONTENT.encode("utf-8"))
        parsed = MarkdownLoader().load(path)

        assert parsed.title == "项目部署手册"
        sections = {block.section for block in parsed.blocks}
        assert "项目部署手册" in sections
        assert "环境准备" in sections
        assert "常见问题" in sections

        code_blocks = [block for block in parsed.blocks if "uvicorn" in block.text]
        assert code_blocks, "代码块内容应当被保留"
        assert "```" in code_blocks[0].text


class TestDocxLoader:
    def test_headings_and_tables(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "需求.docx", samples.docx_bytes())
        parsed = DocxLoader().load(path)

        assert parsed.title == samples.DOCX_HEADING
        text = "\n".join(block.text for block in parsed.blocks)
        assert samples.DOCX_PARAGRAPH in text
        assert "检索 | 高 | 支持 Top-K 与相似度阈值" in text
        assert any(block.section == "功能清单" for block in parsed.blocks)

    def test_invalid_docx(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "broken.docx", b"not a real docx")
        with pytest.raises(DocumentParseError):
            DocxLoader().load(path)


class TestPdfLoader:
    def test_page_numbers_are_kept(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "overview.pdf", samples.pdf_bytes())
        parsed = PdfLoader().load(path)

        assert parsed.page_count == 2
        page_numbers = [block.page_number for block in parsed.blocks]
        assert page_numbers == [1, 2]
        assert "Retrieval Augmented Generation" in parsed.blocks[0].text
        assert "Evaluation" in parsed.blocks[1].text

    def test_scanned_pdf_raises_clear_error(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "scan.pdf", samples.scanned_pdf_bytes())
        with pytest.raises(DocumentParseError) as excinfo:
            PdfLoader().load(path)
        assert "扫描" in excinfo.value.message

    def test_corrupted_pdf(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "bad.pdf", b"%PDF-1.4 broken")
        with pytest.raises(DocumentParseError):
            PdfLoader().load(path)


class TestRegistry:
    def test_dispatches_by_extension(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "note.md", b"# Title\n\nBody text here.")
        parsed = load_document(path)
        assert parsed.blocks[0].text == "Title"

    def test_unknown_extension(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "data.bin", b"\x00\x01")
        with pytest.raises(UnsupportedFileTypeError):
            load_document(path)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(DocumentParseError):
            load_document(tmp_path / "nope.txt")
