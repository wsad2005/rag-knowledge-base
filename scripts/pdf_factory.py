"""测试用的最小 PDF 生成器。

不引入 reportlab 之类的额外依赖：直接按 PDF 1.4 规范拼装对象、xref 表与 trailer。
生成的是纯 ASCII 文本 PDF（Helvetica 标准字体），足以验证“逐页提取 + 页码保留”逻辑。
"""

from __future__ import annotations


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _content_stream(lines: list[str]) -> bytes:
    parts = ["BT", "/F1 12 Tf", "16 TL", "72 760 Td"]
    for index, line in enumerate(lines):
        if index:
            parts.append("T*")
        parts.append(f"({_escape(line)}) Tj")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1")


def build_pdf(pages: list[list[str]]) -> bytes:
    """``pages`` 是每页的文本行列表，返回完整 PDF 字节。"""
    if not pages:
        raise ValueError("至少需要一页")

    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    # 占位：先分配对象号，再回填引用
    catalog_num = 1
    pages_num = 2
    font_num = 3
    objects.extend([b"", b"", b""])  # 1 catalog, 2 pages, 3 font

    page_refs: list[int] = []
    for lines in pages:
        stream = _content_stream(lines)
        content_num = add(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
        page_num = add(
            f"<< /Type /Page /Parent {pages_num} 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 {font_num} 0 R >> >> "
            f"/Contents {content_num} 0 R >>".encode()
        )
        page_refs.append(page_num)

    objects[catalog_num - 1] = f"<< /Type /Catalog /Pages {pages_num} 0 R >>".encode()
    kids = " ".join(f"{num} 0 R" for num in page_refs)
    objects[pages_num - 1] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_refs)} >>".encode()
    )
    objects[font_num - 1] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode())
        output.extend(body)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_num} 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return bytes(output)


def build_scanned_pdf(page_count: int = 1) -> bytes:
    """没有文本层的 PDF（模拟扫描件）：用于验证解析失败提示。"""
    return build_pdf([[] for _ in range(page_count)])
