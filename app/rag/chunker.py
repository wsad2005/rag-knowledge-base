"""文本切分：段落感知 + 句子边界 + 重叠窗口。

为什么不是简单的 ``text[i:i+size]``：

1. **不切断句子**：超长段落先按中英文句末标点切句，再按句子装箱；
2. **保留段落边界**：块内段落用换行分隔，不会把两段话粘成一句；
3. **重叠有语义**：重叠窗口从整句开始，而不是从半个词开始；
4. **元数据跟着走**：每个块记住来源段落的第一页 / 所属章节，答案才能溯源。

切分长度按**字符**计算（中文场景比 token 更直观，且不依赖 tokenizer）。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.core.errors import InvalidRequestError
from app.rag.loaders.base import TextBlock
from app.rag.text_utils import split_sentences

#: 用于定位重叠窗口的句子起点
_SENTENCE_BOUNDARY = re.compile(r"[。！？；!?;\n]")
_CJK_SENTENCE_END = "。！？；"


@dataclass(slots=True)
class Chunk:
    """切分产物。"""

    index: int
    content: str
    page_number: int | None = None
    section: str | None = None

    @property
    def char_count(self) -> int:
        return len(self.content)


@dataclass(slots=True)
class _Unit:
    """最小装箱单位：一个不超过 chunk_size 的文本片段。"""

    text: str
    page_number: int | None
    section: str | None


class TextChunker:
    """可配置的文本切分器。"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
        min_chunk_chars: int = 20,
    ) -> None:
        if chunk_size <= 0:
            raise InvalidRequestError("chunk_size 必须为正整数")
        if chunk_overlap < 0:
            raise InvalidRequestError("chunk_overlap 不能为负数")
        if chunk_overlap >= chunk_size:
            raise InvalidRequestError("chunk_overlap 必须小于 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars

    # ------------------------------------------------------------------ 对外
    def split(self, blocks: Sequence[TextBlock]) -> list[Chunk]:
        """把解析出的段落装箱成切分块（空块自动过滤，过短尾块并入前一块）。"""
        units = self._explode(blocks)
        if not units:
            return []

        chunks: list[Chunk] = []
        parts: list[str] = []
        length = 0
        page: int | None = None
        section: str | None = None
        overlap_only = False

        def emit() -> str:
            """把当前缓冲落成一块，返回该块文本（供计算重叠）。"""
            nonlocal parts, length, page, section, overlap_only
            content = "\n".join(parts).strip()
            if content:
                chunks.append(
                    Chunk(
                        index=len(chunks),
                        content=content,
                        page_number=page,
                        section=section,
                    )
                )
            parts = []
            length = 0
            page = None
            section = None
            overlap_only = False
            return content

        for unit in units:
            addition = len(unit.text) + (1 if parts else 0)
            if parts and length + addition > self.chunk_size:
                previous = emit()
                tail = self._overlap_tail(previous)
                if tail and len(tail) + len(unit.text) + 1 <= self.chunk_size:
                    # 重叠部分只是上下文，页码与章节仍以新内容为准
                    parts = [tail]
                    length = len(tail)
                    overlap_only = True
            if not parts or overlap_only:
                page = unit.page_number
                section = unit.section
                overlap_only = False
            parts.append(unit.text)
            length += len(unit.text) + (1 if length else 0)

        emit()
        return self._merge_short_tail(chunks)

    # ------------------------------------------------------------------ 内部
    def _explode(self, blocks: Sequence[TextBlock]) -> list[_Unit]:
        """把段落拆成不超过 chunk_size 的片段。"""
        units: list[_Unit] = []
        for block in blocks:
            text = block.text.strip()
            if not text:
                continue
            if len(text) <= self.chunk_size:
                units.append(_Unit(text, block.page_number, block.section))
                continue
            for piece in self._split_long_text(text):
                units.append(_Unit(piece, block.page_number, block.section))
        return units

    def _split_long_text(self, text: str) -> list[str]:
        """按句子装箱；单句超长时按字符硬切，保证算法一定收敛。"""
        pieces: list[str] = []
        buffer: list[str] = []

        def flush() -> None:
            if buffer:
                pieces.append(_join_sentences(buffer).strip())
                buffer.clear()

        buffer_len = 0
        for sentence in split_sentences(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > self.chunk_size:
                flush()
                buffer_len = 0
                for start in range(0, len(sentence), self.chunk_size):
                    pieces.append(sentence[start : start + self.chunk_size])
                continue
            if buffer and buffer_len + len(sentence) + 1 > self.chunk_size:
                flush()
                buffer_len = 0
            buffer_len += len(sentence) + (1 if buffer_len else 0)
            buffer.append(sentence)
        flush()
        return [piece for piece in pieces if piece.strip()]

    def _overlap_tail(self, text: str) -> str:
        """从上一块结尾取重叠内容，并尽量从整句起点开始。"""
        if self.chunk_overlap <= 0 or not text:
            return ""
        tail = text[-self.chunk_overlap :]
        match = _SENTENCE_BOUNDARY.search(tail)
        if match and len(tail) - match.end() >= max(self.chunk_overlap // 3, 1):
            tail = tail[match.end() :]
        return tail.strip()

    def _merge_short_tail(self, chunks: list[Chunk]) -> list[Chunk]:
        """末尾过短的碎片并入前一块，避免出现“半句话”块。"""
        if len(chunks) < 2 or self.min_chunk_chars <= 0:
            return chunks
        last = chunks[-1]
        if last.char_count >= self.min_chunk_chars:
            return chunks

        previous = chunks[-2]
        merged = Chunk(
            index=previous.index,
            content=f"{previous.content}\n{last.content}".strip(),
            page_number=previous.page_number,
            section=previous.section,
        )
        return [*chunks[:-2], merged]


def _join_sentences(sentences: Sequence[str]) -> str:
    """中文句子直接拼接，英文句子补空格。"""
    result = ""
    for sentence in sentences:
        if result and result[-1] not in _CJK_SENTENCE_END:
            result += " "
        result += sentence
    return result
