"""PDF 解析（基于 pypdf），逐页提取以保留页码。

pypdf 只做文本层提取：扫描版 PDF（纯图片）没有文本层，这里会明确报错而不是返回空内容，
避免把“没解析出东西”伪装成“文档为空”。
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.errors import DocumentParseError
from app.core.logging import get_logger
from app.rag.loaders.base import ParsedDocument, TextBlock
from app.rag.text_utils import split_paragraphs

logger = get_logger(__name__)

#: 全文字符数低于该阈值视为“没有可用文本层”
_MIN_USEFUL_CHARS = 20


class PdfLoader:
    source_type = "pdf"

    def load(self, path: Path) -> ParsedDocument:
        try:
            reader = PdfReader(str(path))
        except (PdfReadError, OSError) as exc:
            raise DocumentParseError(
                "PDF 文件损坏或格式不受支持", detail={"reason": str(exc)}
            ) from exc

        if reader.is_encrypted:
            # 空密码可解的加密 PDF 很常见，先尝试一次
            try:
                if reader.decrypt("") == 0:
                    raise DocumentParseError("PDF 已加密，无法读取内容")
            except Exception as exc:  # noqa: BLE001 - pypdf 抛出的异常类型不稳定
                raise DocumentParseError("PDF 已加密，无法读取内容") from exc

        blocks: list[TextBlock] = []
        order = 0
        total_chars = 0

        for page_index, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001 - 单页失败不应中断整篇
                logger.warning("PDF 第 %s 页提取失败：%s", page_index + 1, exc)
                continue

            for paragraph in split_paragraphs(page_text):
                blocks.append(
                    TextBlock(text=paragraph, page_number=page_index + 1, order=order)
                )
                order += 1
                total_chars += len(paragraph)

        if total_chars < _MIN_USEFUL_CHARS:
            raise DocumentParseError(
                "PDF 中没有提取到可用的文本，可能是扫描版（需要 OCR）",
                detail={"pages": len(reader.pages), "extracted_chars": total_chars},
            )

        title = _guess_title(blocks, path)
        return ParsedDocument(blocks=blocks, page_count=len(reader.pages), title=title)


def _guess_title(blocks: list[TextBlock], path: Path) -> str:
    """取首个较短的段落作为标题，否则用文件名。"""
    for block in blocks[:3]:
        text = block.text.strip()
        if 4 <= len(text) <= 60:
            return text
    return path.stem
