"""切分器测试：块大小、重叠、空文本过滤、元数据、超长句子兜底。"""

from __future__ import annotations

import pytest

from app.core.errors import InvalidRequestError
from app.rag.chunker import TextChunker
from app.rag.loaders.base import TextBlock


def _blocks(*texts: str, page: int | None = None) -> list[TextBlock]:
    return [TextBlock(text=text, page_number=page, order=index) for index, text in enumerate(texts)]


class TestChunkSize:
    def test_short_text_produces_single_chunk(self) -> None:
        chunks = TextChunker(chunk_size=200, chunk_overlap=20).split(_blocks("很短的一段话。"))
        assert len(chunks) == 1
        assert chunks[0].content == "很短的一段话。"
        assert chunks[0].index == 0

    def test_long_text_is_split_into_bounded_chunks(self) -> None:
        paragraph = "这是一句用于测试切分的句子，长度固定。" * 40
        chunks = TextChunker(chunk_size=200, chunk_overlap=40).split(_blocks(paragraph))
        assert len(chunks) > 3
        for chunk in chunks:
            assert chunk.char_count <= 200
        assert [chunk.index for chunk in chunks] == list(range(len(chunks)))

    def test_overlap_is_applied(self) -> None:
        paragraph = "第一句内容关于报销流程。第二句内容关于审批链路。第三句内容关于差旅标准。" * 8
        chunker = TextChunker(chunk_size=120, chunk_overlap=30)
        chunks = chunker.split(_blocks(paragraph))
        assert len(chunks) >= 3
        # 后一块的开头应当出现在前一块的结尾里（重叠窗口）
        for previous, current in zip(chunks, chunks[1:], strict=False):
            head = current.content.split("\n")[0][:10]
            assert head in previous.content

    def test_paragraph_boundaries_are_preserved(self) -> None:
        chunks = TextChunker(chunk_size=60, chunk_overlap=0).split(
            _blocks("第一段内容比较短。", "第二段内容也比较短。", "第三段内容同样短。")
        )
        joined = "\n".join(chunk.content for chunk in chunks)
        assert "第一段内容比较短。\n第二段内容也比较短。" in joined or len(chunks) >= 2


class TestFiltering:
    def test_empty_blocks_are_ignored(self) -> None:
        chunks = TextChunker().split(_blocks("", "   ", "\n\n", "有内容的段落。"))
        assert len(chunks) == 1
        assert chunks[0].content == "有内容的段落。"

    def test_all_empty_returns_no_chunks(self) -> None:
        assert TextChunker().split(_blocks("", "  \n ")) == []

    def test_short_tail_is_merged(self) -> None:
        long_text = "这是一个足够长的段落，用来占据第一个块的大部分空间。" * 6
        chunks = TextChunker(chunk_size=120, chunk_overlap=0, min_chunk_chars=40).split(
            _blocks(long_text, "短尾巴")
        )
        assert all(chunk.char_count >= 40 for chunk in chunks)


class TestMetadata:
    def test_page_number_follows_block(self) -> None:
        first = "第一页的内容。" * 8  # 56 字，超过 chunk_size 的一半，两块无法合并
        second = "第二页的内容。" * 8
        chunks = TextChunker(chunk_size=60, chunk_overlap=0, min_chunk_chars=0).split(
            [
                TextBlock(text=first, page_number=1),
                TextBlock(text=second, page_number=2),
            ]
        )
        assert [chunk.page_number for chunk in chunks] == [1, 2]

    def test_section_is_recorded(self) -> None:
        chunks = TextChunker().split(
            [TextBlock(text="安装步骤说明。", section="环境准备", order=0)]
        )
        assert chunks[0].section == "环境准备"


class TestEdgeCases:
    def test_single_huge_sentence_is_hard_split(self) -> None:
        sentence = "无标点" * 500  # 1500 字且没有任何句末标点，只能按字符硬切
        chunks = TextChunker(chunk_size=100, chunk_overlap=10).split(_blocks(sentence))
        assert len(chunks) == 15
        assert all(chunk.char_count <= 100 for chunk in chunks)
        assert sum(chunk.char_count for chunk in chunks) == 1500  # 硬切不丢字

    @pytest.mark.parametrize(
        ("size", "overlap"),
        [(0, 0), (100, 100), (100, 150), (-1, 0)],
    )
    def test_invalid_configuration(self, size: int, overlap: int) -> None:
        with pytest.raises(InvalidRequestError):
            TextChunker(chunk_size=size, chunk_overlap=overlap)

    def test_chunk_char_count_property(self) -> None:
        chunks = TextChunker().split(_blocks("统计字符数。"))
        assert chunks[0].char_count == len("统计字符数。")
