"""解析层的数据结构与接口。

解析器的职责只有一件事：**把文件变成有序的文本块，并保留可追溯的元数据**
（PDF 页码 / Markdown 标题 / DOCX 段落序号）。切分与向量化都是后续环节的事。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class TextBlock:
    """解析出的一个逻辑段落。"""

    text: str
    #: 1-based 页码，仅 PDF 有
    page_number: int | None = None
    #: 所在章节标题，Markdown/DOCX 有
    section: str | None = None
    #: 在文档中的顺序（从 0 开始）
    order: int = 0


@dataclass(slots=True)
class ParsedDocument:
    """一份文档的解析结果。"""

    blocks: list[TextBlock] = field(default_factory=list)
    page_count: int | None = None
    title: str | None = None
    encoding: str | None = None

    @property
    def char_count(self) -> int:
        return sum(len(block.text) for block in self.blocks)

    @property
    def is_empty(self) -> bool:
        return not any(block.text.strip() for block in self.blocks)


class DocumentLoader(Protocol):
    """解析器协议：新增格式只需实现 ``load`` 并在 registry 注册。"""

    source_type: str

    def load(self, path: Path) -> ParsedDocument:  # pragma: no cover - 协议声明
        ...
