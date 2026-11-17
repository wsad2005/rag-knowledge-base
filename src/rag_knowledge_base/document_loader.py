"""文档读取：把本地文本文件转换为统一的 Document 结构。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Document:
    """一份待切分的原始文档。"""

    content: str
    source: str


@runtime_checkable
class DocumentLoader(Protocol):
    """文档加载器接口：返回 Document 列表。"""

    def load(self) -> list[Document]:
        """加载并返回文档列表。"""
        ...


class TextLoader:
    """读取单个 UTF-8 文本文件（离线、零依赖）。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def load(self) -> list[Document]:
        """读取文件内容，封装为单个 Document。"""
        content = self._path.read_text(encoding="utf-8")
        return [Document(content=content, source=str(self._path))]
