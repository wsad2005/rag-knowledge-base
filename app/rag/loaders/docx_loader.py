"""DOCX 解析（基于 python-docx）。

按正文 XML 顺序遍历段落与表格，保留“标题样式 -> 章节名”的映射，
表格转成 ``单元格 | 单元格`` 的文本行，避免表格内容整块丢失。
"""

from __future__ import annotations

from pathlib import Path

from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.core.errors import DocumentParseError
from app.rag.loaders.base import ParsedDocument, TextBlock

_HEADING_STYLE_PREFIXES = ("heading", "标题")
_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class DocxLoader:
    source_type = "docx"

    def load(self, path: Path) -> ParsedDocument:
        try:
            document = DocxDocument(str(path))
        except Exception as exc:  # noqa: BLE001 - 损坏文件可能抛各种异常
            raise DocumentParseError(
                "DOCX 文件损坏或不是有效的 Office 文档", detail={"reason": str(exc)}
            ) from exc

        blocks: list[TextBlock] = []
        current_section: str | None = None
        order = 0

        for child in document.element.body.iterchildren():
            if child.tag == f"{_W_NS}p":
                paragraph = Paragraph(child, document)
                text = paragraph.text.strip()
                if not text:
                    continue
                if _is_heading(paragraph):
                    current_section = text[:255]
                blocks.append(
                    TextBlock(text=text, section=current_section, order=order)
                )
                order += 1
            elif child.tag == f"{_W_NS}tbl":
                table_text = _table_to_text(Table(child, document))
                if table_text:
                    blocks.append(
                        TextBlock(text=table_text, section=current_section, order=order)
                    )
                    order += 1

        if not blocks:
            raise DocumentParseError("DOCX 中没有可提取的文本内容")

        title = _heading_title(blocks) or path.stem
        return ParsedDocument(blocks=blocks, title=title)


def _is_heading(paragraph: Paragraph) -> bool:
    try:
        style_name = (paragraph.style.name or "").lower()
    except AttributeError:  # pragma: no cover - 极端样式缺失
        return False
    return any(style_name.startswith(prefix) for prefix in _HEADING_STYLE_PREFIXES)


def _table_to_text(table: Table) -> str:
    rows: list[str] = []
    for row in table.rows:
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _heading_title(blocks: list[TextBlock]) -> str | None:
    """优先用第一个带章节名的短段落作为标题。"""
    for block in blocks:
        if block.section and 2 <= len(block.section) <= 60:
            return block.section
    return None
