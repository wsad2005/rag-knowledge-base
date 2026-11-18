"""Markdown 解析：按标题切章节，保留 fenced code block 完整。

Markdown 的标题层级天然是章节边界，把它记录到 ``section`` 后，
检索结果就能显示“来自《安装说明》> 常见问题”，比只给文件名更有用。
"""

from __future__ import annotations

import re
from pathlib import Path

from app.core.errors import DocumentParseError
from app.rag.loaders.base import ParsedDocument, TextBlock
from app.rag.loaders.text_loader import read_text_with_fallback

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE = re.compile(r"^(```|~~~)")


class MarkdownLoader:
    source_type = "markdown"

    def load(self, path: Path) -> ParsedDocument:
        text, encoding = read_text_with_fallback(path)
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        blocks: list[TextBlock] = []
        current_section: str | None = None
        buffer: list[str] = []
        in_fence = False
        order = 0

        def flush() -> None:
            nonlocal order
            content = "\n".join(buffer).strip()
            buffer.clear()
            if content:
                blocks.append(
                    TextBlock(text=content, section=current_section, order=order)
                )
                order += 1

        for line in lines:
            if _FENCE.match(line.strip()):
                in_fence = not in_fence
                buffer.append(line)
                if not in_fence:  # 代码块结束，单独成块，避免被后续段落合并
                    flush()
                continue

            if in_fence:
                buffer.append(line)
                continue

            heading_match = _HEADING.match(line)
            if heading_match:
                flush()
                current_section = heading_match.group(2).strip()[:255]
                # 标题本身也作为内容保留，便于“按标题检索”
                blocks.append(
                    TextBlock(text=current_section, section=current_section, order=order)
                )
                order += 1
                continue

            if not line.strip():
                flush()
                continue

            buffer.append(line)

        flush()

        if not blocks:
            raise DocumentParseError("Markdown 文件中没有可提取的文本内容")

        title = blocks[0].text if blocks[0].section == blocks[0].text else path.stem
        return ParsedDocument(blocks=blocks, title=title, encoding=encoding)
