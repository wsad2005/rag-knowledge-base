"""文本清洗与句子切分工具。

中文文档的真实脏数据主要是：CRLF、全角空格、PDF 逐行硬换行、页眉页脚残留空行。
这里统一在解析阶段清洗，切分器只处理“干净的段落”，职责更清晰。
"""

from __future__ import annotations

import re

#: 中日韩字符范围（用于判断换行两侧能否直接拼接）
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]")
#: 句末标点（中英文），用于段落内的句子切分
_SENTENCE_END = re.compile(r"(?<=[。！？；!?;])|(?<=\.)(?=\s)")
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def _is_cjk(char: str) -> bool:
    return bool(_CJK_PATTERN.match(char))


def join_wrapped_lines(block: str) -> str:
    """把段落内的硬换行拼回一行。

    - 英文断词 ``exam-\\nple`` -> ``example``
    - 两侧都是中日韩字符时不加空格（中文排版不需要）
    - 其余情况补一个空格，避免英文单词粘连
    """
    block = _HYPHEN_BREAK.sub(r"\1\2", block)
    lines = [line.strip() for line in block.split("\n")]
    lines = [line for line in lines if line]
    if not lines:
        return ""

    merged = lines[0]
    for line in lines[1:]:
        if merged and line:
            if _is_cjk(merged[-1]) or _is_cjk(line[0]):
                merged += line
            else:
                merged += " " + line
        else:
            merged += line
    return merged


def normalize_text(text: str) -> str:
    """基础清洗：统一换行/空白、去掉不可见字符、压缩连续空行。"""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = text.replace("\ufeff", "").replace("\u200b", "")
    # 去掉每行行尾空白，但保留行首缩进信息（对代码块有意义）
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _MULTI_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    """按空行切分段落，并过滤纯空白段落。"""
    normalized = normalize_text(text)
    if not normalized:
        return []
    raw_blocks = re.split(r"\n\s*\n", normalized)
    paragraphs: list[str] = []
    for block in raw_blocks:
        joined = join_wrapped_lines(block)
        if joined.strip():
            paragraphs.append(joined.strip())
    return paragraphs


def split_sentences(text: str) -> list[str]:
    """按中英文句末标点切句，保留标点；无法切分时返回整段。"""
    if not text:
        return []
    parts = [part.strip() for part in _SENTENCE_END.split(text) if part and part.strip()]
    return parts or [text.strip()]


def truncate(text: str, limit: int, *, suffix: str = "…") -> str:
    """按字符数截断，用于日志与 API 返回的片段预览。"""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(limit - len(suffix), 0)] + suffix
