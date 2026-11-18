"""纯文本解析：按空行分段，并做编码兼容。

Windows 上中文 txt 常见 GBK/GB18030，macOS/Linux 常见 UTF-8；
这里按 ``utf-8-sig -> utf-8 -> gb18030 -> utf-16`` 顺序尝试，并把最终编码记录下来，
方便排查“乱码”类问题。
"""

from __future__ import annotations

from pathlib import Path

from app.core.errors import DocumentParseError
from app.rag.loaders.base import ParsedDocument, TextBlock
from app.rag.text_utils import split_paragraphs

_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "utf-16", "latin-1")


def read_text_with_fallback(path: Path) -> tuple[str, str]:
    """返回 ``(text, encoding)``。"""
    raw = path.read_bytes()
    if not raw.strip():
        raise DocumentParseError("文件内容为空")
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding), encoding
        except (UnicodeDecodeError, LookupError):
            continue
    raise DocumentParseError("无法识别文件编码，请另存为 UTF-8 后重试")


class TextLoader:
    source_type = "text"

    def load(self, path: Path) -> ParsedDocument:
        text, encoding = read_text_with_fallback(path)
        paragraphs = split_paragraphs(text)
        if not paragraphs:
            raise DocumentParseError("文件中没有可提取的文本内容")

        blocks = [
            TextBlock(text=paragraph, order=index) for index, paragraph in enumerate(paragraphs)
        ]
        return ParsedDocument(
            blocks=blocks,
            title=_guess_title(paragraphs, path),
            encoding=encoding,
        )


def _guess_title(paragraphs: list[str], path: Path) -> str:
    first = paragraphs[0].strip()
    if 4 <= len(first) <= 60:
        return first
    return path.stem
