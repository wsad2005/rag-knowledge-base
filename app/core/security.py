"""上传文件的安全校验。

三条红线：

1. **不信任客户端文件名**：落盘一律用 UUID，原文件名只作为展示字段入库；
2. **类型白名单**：按扩展名判定，扩展名与解析器一一对应；
3. **大小限制**：流式写入时就检查，绝不先读进内存再判断。
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

from app.core.config import Settings
from app.core.errors import (
    EmptyFileError,
    FileTooLargeError,
    InvalidRequestError,
    UnsupportedFileTypeError,
)

#: 扩展名 -> (规范化的 MIME 提示, 解析器标识)
EXTENSION_REGISTRY: dict[str, tuple[str, str]] = {
    ".pdf": ("application/pdf", "pdf"),
    ".docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"),
    ".txt": ("text/plain", "text"),
    ".md": ("text/markdown", "markdown"),
    ".markdown": ("text/markdown", "markdown"),
}

_UNSAFE_CHARS = re.compile(r"[\\/:*?\"<>|\x00-\x1f]")
_DEFAULT_NAME = "unnamed"


def sanitize_filename(raw_name: str | None, *, max_length: int = 200) -> str:
    """清洗客户端文件名，仅用于展示与扩展名判断。

    - 去掉目录部分（同时兼容 ``/`` 与 ``\\``，防止 ``../evil.txt`` 之类）
    - 去掉控制字符与 Windows 非法字符
    - 压缩连续空白，限制长度
    """
    if not raw_name:
        return _DEFAULT_NAME

    # 统一分隔符后只取最后一段，避免路径穿越
    normalized = raw_name.replace("\\", "/").split("/")[-1]
    normalized = unicodedata.normalize("NFC", normalized)
    normalized = _UNSAFE_CHARS.sub("_", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" .")

    if not normalized or normalized in {".", ".."}:
        return _DEFAULT_NAME

    if len(normalized) > max_length:
        suffix = Path(normalized).suffix[:16]
        normalized = normalized[: max_length - len(suffix)] + suffix

    return normalized


def detect_extension(filename: str) -> str:
    """返回小写扩展名（含点）。"""
    return Path(filename).suffix.lower()


def validate_extension(filename: str, settings: Settings) -> str:
    """校验扩展名是否在白名单内，返回扩展名。"""
    extension = detect_extension(filename)
    if not extension:
        raise UnsupportedFileTypeError(
            f"文件「{filename}」没有扩展名，无法判断类型",
            detail={"allowed_extensions": sorted(settings.allowed_extension_set)},
        )
    if extension not in settings.allowed_extension_set:
        raise UnsupportedFileTypeError(
            f"不支持的文件类型：{extension}",
            detail={"allowed_extensions": sorted(settings.allowed_extension_set)},
        )
    return extension


def content_type_for(extension: str) -> str:
    return EXTENSION_REGISTRY.get(extension, ("application/octet-stream", "unknown"))[0]


def parser_kind_for(extension: str) -> str:
    return EXTENSION_REGISTRY.get(extension, ("application/octet-stream", "unknown"))[1]


def storage_relative_path(extension: str) -> Path:
    """生成不透明的落盘相对路径，形如 ``ab/abcdef...pdf``（按 UUID 前两位分桶）。"""
    import uuid

    token = uuid.uuid4().hex
    return Path(token[:2]) / f"{token}{extension}"


def assert_within_directory(candidate: Path, root: Path) -> Path:
    """防御性检查：确保路径最终解析后仍在指定目录内。"""
    resolved = candidate.resolve()
    root_resolved = root.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise InvalidRequestError("非法的文件路径")
    return resolved


def iter_limited_chunks(
    fileobj: BinaryIO,
    *,
    max_bytes: int,
    chunk_size: int = 1024 * 1024,
) -> Iterator[bytes]:
    """分块读取上传流，超限立即中断。

    这样 20MB 限制不会因为一次性 ``read()`` 而先吃掉 20MB 内存。
    """
    total = 0
    while True:
        chunk = fileobj.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise FileTooLargeError(
                f"文件超过 {max_bytes // (1024 * 1024)}MB 限制",
                detail={"max_bytes": max_bytes},
            )
        yield chunk
    if total == 0:
        raise EmptyFileError("上传的文件内容为空")
