"""解析器注册表：扩展名 -> 解析器实例。"""

from __future__ import annotations

from pathlib import Path

from app.core.errors import DocumentParseError, UnsupportedFileTypeError
from app.core.security import parser_kind_for
from app.rag.loaders.base import ParsedDocument
from app.rag.loaders.docx_loader import DocxLoader
from app.rag.loaders.markdown_loader import MarkdownLoader
from app.rag.loaders.pdf_loader import PdfLoader
from app.rag.loaders.text_loader import TextLoader

_LOADERS = {
    "pdf": PdfLoader(),
    "docx": DocxLoader(),
    "text": TextLoader(),
    "markdown": MarkdownLoader(),
}


def supported_source_types() -> list[str]:
    return sorted(_LOADERS)


def get_loader(source_type: str):
    try:
        return _LOADERS[source_type]
    except KeyError as exc:
        raise UnsupportedFileTypeError(f"没有可用的解析器：{source_type}") from exc


def load_document(path: Path, *, source_type: str | None = None) -> ParsedDocument:
    """按文件扩展名选择解析器并解析。

    显式传入 ``source_type`` 可跳过扩展名推断（测试与内部调用用）。
    """
    resolved_type = source_type or parser_kind_for(path.suffix.lower())
    if resolved_type == "unknown":
        raise UnsupportedFileTypeError(f"不支持的文件类型：{path.suffix or '无扩展名'}")
    if not path.exists():
        raise DocumentParseError("文件不存在或已被删除", detail={"path_suffix": path.suffix})

    loader = get_loader(resolved_type)
    return loader.load(path)
