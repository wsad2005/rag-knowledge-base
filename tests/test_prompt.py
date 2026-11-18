"""上下文组装与 Prompt 构建测试。"""

from __future__ import annotations

from app.rag.prompt import (
    NO_CONTEXT_MESSAGE,
    SYSTEM_PROMPT,
    build_context,
    build_extractive_answer,
    build_messages,
    format_source_label,
)
from app.schemas.knowledge import Source


def make_source(
    index: int,
    *,
    name: str = "员工手册.pdf",
    page: int | None = 3,
    section: str | None = "报销",
    snippet: str = "一线城市住宿标准为每晚 600 元。",
    score: float = 0.82,
) -> Source:
    return Source(
        index=index,
        document_id=f"doc-{index}",
        document_name=name,
        chunk_id=f"chunk-{index}",
        chunk_index=index - 1,
        page_number=page,
        section=section,
        score=score,
        snippet=snippet,
    )


class TestLabels:
    def test_full_label(self) -> None:
        assert format_source_label(1, make_source(1)) == "[1] 《员工手册.pdf》 第 3 页 章节：报销"

    def test_without_page_and_section(self) -> None:
        label = format_source_label(2, make_source(2, page=None, section=None))
        assert label == "[2] 《员工手册.pdf》"


class TestBuildContext:
    def test_contains_all_sources(self) -> None:
        bundle = build_context([make_source(1), make_source(2)], max_chars=4000)
        assert len(bundle.used_sources) == 2
        assert "[1] 《员工手册.pdf》" in bundle.text
        assert "[2] 《员工手册.pdf》" in bundle.text
        assert bundle.total_chars > 0

    def test_truncates_when_budget_exceeded(self) -> None:
        sources = [make_source(i, snippet="内容" * 100) for i in range(1, 6)]
        bundle = build_context(sources, max_chars=300)
        assert 1 <= len(bundle.used_sources) < 5
        assert len(bundle.text) <= 320  # 允许第一块被裁剪后的少量余量

    def test_first_source_is_kept_even_if_too_long(self) -> None:
        bundle = build_context([make_source(1, snippet="长" * 1000)], max_chars=200)
        assert len(bundle.used_sources) == 1
        assert len(bundle.text) <= 220

    def test_empty_snippet_skipped(self) -> None:
        bundle = build_context([make_source(1, snippet="   ")], max_chars=1000)
        assert bundle.is_empty
        assert bundle.text == ""

    def test_empty_sources(self) -> None:
        bundle = build_context([], max_chars=1000)
        assert bundle.is_empty


class TestMessages:
    def test_system_prompt_forbids_fabrication(self) -> None:
        assert "只能依据" in SYSTEM_PROMPT
        assert "不要编造" in SYSTEM_PROMPT

    def test_messages_structure(self) -> None:
        bundle = build_context([make_source(1)], max_chars=1000)
        messages = build_messages("住宿标准是多少？", bundle)
        assert [message["role"] for message in messages] == ["system", "user"]
        assert "住宿标准是多少？" in messages[1]["content"]
        assert "参考资料" in messages[1]["content"]

    def test_no_context_message(self) -> None:
        messages = build_messages("随便问问", build_context([], max_chars=1000))
        assert messages[1]["content"].endswith("参考资料：无")


class TestExtractiveAnswer:
    def test_lists_sources(self) -> None:
        answer = build_extractive_answer([make_source(1), make_source(2)])
        assert "未配置大模型" in answer
        assert "《员工手册.pdf》 第 3 页" in answer
        assert "0.820" in answer

    def test_limits_items(self) -> None:
        sources = [make_source(i) for i in range(1, 6)]
        answer = build_extractive_answer(sources, max_items=2)
        assert "3." not in answer.split("\n")[-2:]

    def test_no_sources(self) -> None:
        assert build_extractive_answer([]) == NO_CONTEXT_MESSAGE
