"""文本切分：把长文档切成带重叠的小块，作为检索的最小单元。"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .document_loader import Document


@dataclass(frozen=True)
class Chunk:
    """切分后的最小检索单元。"""

    text: str
    source: str
    index: int


@runtime_checkable
class TextSplitter(Protocol):
    """切分器接口：Document 进，Chunk 列表出。"""

    def split(self, document: Document) -> list[Chunk]:
        """把文档切分为块列表。"""
        ...


class FixedSizeSplitter:
    """按固定字符长度滑动窗口切分，相邻块保留 overlap 字符重叠。"""

    def __init__(self, chunk_size: int = 200, overlap: int = 20) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须为正整数")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")
        self._chunk_size = chunk_size
        self._overlap = overlap

    def split(self, document: Document) -> list[Chunk]:
        """按固定窗口切分；空白文档返回空列表。"""
        text = document.content.strip()
        if not text:
            return []
        step = self._chunk_size - self._overlap
        chunks: list[Chunk] = []
        for index, start in enumerate(range(0, len(text), step)):
            piece = text[start : start + self._chunk_size]
            chunks.append(Chunk(text=piece, source=document.source, index=index))
            if start + self._chunk_size >= len(text):
                break
        return chunks
