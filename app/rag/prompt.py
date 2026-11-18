"""Prompt 构建与上下文组装。

上下文不是“把 Top-K 拼起来”就完事，这里做了三件事：

1. **格式统一**：每条资料带 ``[n] 《文档名》 第 X 页 · 章节``，模型引用有据可依；
2. **长度受控**：按 ``MAX_CONTEXT_CHARS`` 逐条累加，超预算的资料来源被丢弃并记录条数，
   避免把 20 万字的上下文塞给模型导致截断或超费；
3. **反幻觉约束**：系统提示明确“资料不足就说不足”，并要求引用编号，不允许编造来源。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.rag.text_utils import truncate
from app.schemas.knowledge import Source

SYSTEM_PROMPT = """你是一个严谨的企业知识库问答助手。请遵守以下规则：

1. 只能依据「参考资料」中的内容回答，不要使用资料以外的知识，也不要猜测。
2. 每个结论后面必须标注来源编号，格式为 [1]、[2]；多个来源写成 [1][3]。
3. 如果参考资料不足以回答问题，请直接说明"根据现有资料无法回答"，并指出缺少什么信息。
4. 不要编造文档名、页码或数据。资料中没有出现的数字、日期、人名一律不得写入答案。
5. 用简体中文回答，条理清晰；能分点时优先分点，不要复述整个资料段落。"""

NO_CONTEXT_MESSAGE = "根据现有资料无法回答这个问题。"


@dataclass(slots=True)
class ContextBundle:
    """组装后的上下文与统计信息。"""

    text: str
    used_sources: list[Source]
    total_chars: int

    @property
    def is_empty(self) -> bool:
        return not self.used_sources


def format_source_label(index: int, source: Source) -> str:
    """``[1] 《员工手册.pdf》 第 3 页 · 章节：报销``"""
    parts = [f"[{index}] 《{source.document_name}》"]
    if source.page_number is not None:
        parts.append(f"第 {source.page_number} 页")
    if source.section:
        parts.append(f"章节：{source.section}")
    return " ".join(parts)


def build_context(sources: Sequence[Source], max_chars: int) -> ContextBundle:
    """把检索结果拼成带编号的上下文，并做长度截断。"""
    blocks: list[str] = []
    used: list[Source] = []
    total = 0

    for source in sources:
        header = format_source_label(source.index, source)
        body = source.snippet.strip()
        if not body:
            continue
        block = f"{header}\n{body}"
        # 剩余预算不足一条时停止；至少保留一条，避免完全空上下文
        if used and total + len(block) + 2 > max_chars:
            break
        if not used and len(block) > max_chars:
            block = block[:max_chars]
        blocks.append(block)
        used.append(source)
        total += len(block) + 2

    return ContextBundle(text="\n\n".join(blocks), used_sources=used, total_chars=total)


def build_messages(question: str, context: ContextBundle) -> list[dict[str, str]]:
    """构造 OpenAI 兼容的 messages。"""
    if context.is_empty:
        user_content = f"问题：{question}\n\n参考资料：无"
    else:
        user_content = (
            f"参考资料：\n{context.text}\n\n"
            f"问题：{question}\n\n"
            "请依据上述资料作答，并在结论后标注来源编号。"
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_extractive_answer(sources: Sequence[Source], *, max_items: int = 3) -> str:
    """未配置大模型时的降级答案：直接给出最相关的原文片段。"""
    if not sources:
        return NO_CONTEXT_MESSAGE

    lines = [
        "当前未配置大模型，无法生成归纳性回答。以下是与你的问题最相关的原文片段，供参考：",
        "",
    ]
    for position, source in enumerate(sources[:max_items], start=1):
        label = f"《{source.document_name}》"
        if source.page_number is not None:
            label += f" 第 {source.page_number} 页"
        if source.section:
            label += f" · {source.section}"
        lines.append(f"{position}. {label}（相似度 {source.score:.3f}）")
        # 降级答案只做“片段预览”，完整原文在 sources 里，避免刷屏
        lines.append(f"   {truncate(source.snippet, 240)}")
        lines.append("")
    return "\n".join(lines).strip()
