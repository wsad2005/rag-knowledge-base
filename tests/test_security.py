"""上传校验：文件名清洗、类型白名单、大小限制、路径安全检查。"""

from __future__ import annotations

import io

import pytest

from app.core.config import Settings
from app.core.errors import (
    EmptyFileError,
    FileTooLargeError,
    InvalidRequestError,
    UnsupportedFileTypeError,
)
from app.core.security import (
    assert_within_directory,
    detect_extension,
    iter_limited_chunks,
    sanitize_filename,
    storage_relative_path,
    validate_extension,
)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, environment="test", data_dir=tmp_path / "data", upload_max_mb=1)


class TestSanitizeFilename:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("../../etc/passwd.txt", "passwd.txt"),
            (r"C:\Users\me\制度.pdf", "制度.pdf"),
            ("a/b/c/report.docx", "report.docx"),
            ("  空格 开头.md  ", "空格 开头.md"),
            ("bad<>name?.txt", "bad__name_.txt"),
        ],
    )
    def test_strips_directories_and_illegal_chars(self, raw: str, expected: str) -> None:
        assert sanitize_filename(raw) == expected

    def test_empty_name_falls_back(self) -> None:
        assert sanitize_filename(None) == "unnamed"
        assert sanitize_filename("") == "unnamed"
        assert sanitize_filename("..") == "unnamed"

    def test_long_name_is_truncated_but_keeps_suffix(self) -> None:
        name = "很长的文件名" * 60 + ".pdf"
        cleaned = sanitize_filename(name, max_length=50)
        assert len(cleaned) <= 50
        assert cleaned.endswith(".pdf")


class TestExtensionValidation:
    @pytest.mark.parametrize("name", ["a.pdf", "b.DOCX", "c.txt", "d.md", "e.markdown"])
    def test_allowed(self, name: str, settings: Settings) -> None:
        assert validate_extension(name, settings) in settings.allowed_extension_set

    @pytest.mark.parametrize("name", ["evil.exe", "script.sh", "archive.zip"])
    def test_rejected(self, name: str, settings: Settings) -> None:
        with pytest.raises(UnsupportedFileTypeError) as excinfo:
            validate_extension(name, settings)
        assert excinfo.value.status_code == 415
        assert "allowed_extensions" in (excinfo.value.detail or {})

    def test_without_extension(self, settings: Settings) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            validate_extension("README", settings)

    def test_detect_extension_lowercases(self) -> None:
        assert detect_extension("Report.PDF") == ".pdf"


class TestSizeLimit:
    def test_within_limit(self) -> None:
        stream = io.BytesIO(b"x" * 100)
        assert sum(len(chunk) for chunk in iter_limited_chunks(stream, max_bytes=1000)) == 100

    def test_exceeds_limit(self) -> None:
        stream = io.BytesIO(b"x" * 5000)
        with pytest.raises(FileTooLargeError) as excinfo:
            list(iter_limited_chunks(stream, max_bytes=1000, chunk_size=512))
        assert excinfo.value.status_code == 413

    def test_empty_file(self) -> None:
        with pytest.raises(EmptyFileError):
            list(iter_limited_chunks(io.BytesIO(b""), max_bytes=1000))


class TestPathSafety:
    def test_storage_path_is_opaque_and_bucketed(self) -> None:
        relative = storage_relative_path(".pdf")
        assert relative.suffix == ".pdf"
        assert len(relative.parent.name) == 2
        assert len(relative.stem) == 32

    def test_assert_within_directory(self, tmp_path) -> None:
        root = tmp_path / "uploads"
        root.mkdir()
        inside = root / "a" / "b.txt"
        inside.parent.mkdir()
        assert assert_within_directory(inside, root) == inside.resolve()

    def test_rejects_traversal(self, tmp_path) -> None:
        root = tmp_path / "uploads"
        root.mkdir()
        outside = tmp_path / "secret.txt"
        with pytest.raises(InvalidRequestError):
            assert_within_directory(outside, root)
