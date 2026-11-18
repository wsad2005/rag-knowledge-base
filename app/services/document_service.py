"""文档服务：上传落盘、解析、切分、向量化、入库、删除的完整编排。

事务边界刻意分成三段（建档 / 写块 / 标记完成），这样任何一步失败都能把文档标记为
``failed`` 并保留失败原因，而不是留下一个“半成品”文档让用户困惑。
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError

from app.core.config import Settings
from app.core.errors import (
    AppError,
    DocumentNotFoundError,
    DocumentParseError,
    DuplicateDocumentError,
)
from app.core.logging import get_logger
from app.core.security import (
    assert_within_directory,
    content_type_for,
    iter_limited_chunks,
    parser_kind_for,
    sanitize_filename,
    storage_relative_path,
    validate_extension,
)
from app.database.session import Database
from app.models.base import new_id, utcnow
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.rag.chunker import TextChunker
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.loaders.registry import load_document
from app.vector_store.base import VectorRecord, VectorStore

logger = get_logger(__name__)


class DocumentService:
    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        chunker: TextChunker | None = None,
    ) -> None:
        self._settings = settings
        self._db = database
        self._embedder = embedder
        self._store = vector_store
        self._chunker = chunker or TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_chunk_chars=settings.min_chunk_chars,
        )

    # ------------------------------------------------------------------ 查询
    def list_documents(
        self, *, skip: int = 0, limit: int = 20, query: str | None = None
    ) -> tuple[int, list[Document]]:
        with self._db.session() as session:
            statement = select(Document)
            if query:
                # icontains + autoescape：用户输入里的 % 和 _ 会被转义，不会变成通配符
                pattern = query.strip()
                statement = statement.where(Document.filename.icontains(pattern, autoescape=True))
            total = session.scalar(select(func.count()).select_from(statement.subquery())) or 0
            items = list(
                session.scalars(
                    statement.order_by(Document.created_at.desc()).offset(skip).limit(limit)
                )
            )
        return int(total), items

    def get_document(self, document_id: str) -> Document:
        with self._db.session() as session:
            document = session.get(Document, document_id)
            if document is None:
                raise DocumentNotFoundError(f"文档不存在：{document_id}")
            _ = list(document.chunks)  # 触发加载，返回后仍可访问
        return document

    # ------------------------------------------------------------------ 上传
    def ingest_upload(self, filename: str | None, stream: BinaryIO) -> Document:
        """保存上传文件并完成解析入库。"""
        safe_name = sanitize_filename(filename, max_length=self._settings.max_filename_length)
        extension = validate_extension(safe_name, self._settings)

        relative_path = storage_relative_path(extension)
        target = assert_within_directory(
            self._settings.upload_dir / relative_path, self._settings.upload_dir
        )
        target.parent.mkdir(parents=True, exist_ok=True)

        content_hash, file_size = self._save_stream(stream, target)
        document_id = new_id()

        try:
            self._create_document(
                document_id=document_id,
                filename=safe_name,
                content_hash=content_hash,
                extension=extension,
                file_size=file_size,
                relative_path=relative_path.as_posix(),
            )
        except DuplicateDocumentError:
            target.unlink(missing_ok=True)
            raise
        except Exception:
            target.unlink(missing_ok=True)
            raise

        try:
            self._process(document_id, target, extension)
        except AppError as exc:
            self._mark_failed(document_id, str(exc), target)
            raise
        except Exception as exc:  # noqa: BLE001 - 未预期异常也要留下失败记录
            logger.exception("文档 %s 处理失败", document_id)
            self._mark_failed(document_id, f"内部错误：{exc}", target)
            raise DocumentParseError(
                "文档处理失败", detail={"document_id": document_id, "reason": str(exc)}
            ) from exc

        return self.get_document(document_id)

    def _save_stream(self, stream: BinaryIO, target: Path) -> tuple[str, int]:
        """流式写盘并计算 SHA-256，超限时立即中断且不留下半截文件。"""
        digest = hashlib.sha256()
        size = 0
        temp_path = target.with_name(target.name + ".part")
        try:
            with temp_path.open("wb") as handle:
                for piece in iter_limited_chunks(
                    stream, max_bytes=self._settings.upload_max_bytes
                ):
                    digest.update(piece)
                    size += len(piece)
                    handle.write(piece)
            temp_path.replace(target)
        except Exception:
            temp_path.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            raise
        return digest.hexdigest(), size

    def _create_document(
        self,
        *,
        document_id: str,
        filename: str,
        content_hash: str,
        extension: str,
        file_size: int,
        relative_path: str,
    ) -> Document:
        with self._db.session() as session:
            existing = session.scalar(
                select(Document).where(Document.content_hash == content_hash)
            )
            if existing is not None:
                raise DuplicateDocumentError(
                    f"该文件内容已存在：《{existing.filename}》",
                    detail={"document_id": existing.id, "filename": existing.filename},
                )

            document = Document(
                id=document_id,
                filename=filename,
                title=Path(filename).stem,
                content_hash=content_hash,
                file_ext=extension,
                mime_type=content_type_for(extension),
                source_type=parser_kind_for(extension),
                storage_path=relative_path,
                file_size=file_size,
                status=DocumentStatus.PROCESSING.value,
            )
            session.add(document)
            try:
                session.flush()
            except IntegrityError as exc:  # 并发上传同一文件
                raise DuplicateDocumentError("该文件内容已存在") from exc
        return document

    def _process(self, document_id: str, path: Path, extension: str) -> None:
        parsed = load_document(path, source_type=parser_kind_for(extension))
        chunks = self._chunker.split(parsed.blocks)
        if not chunks:
            raise DocumentParseError("切分后没有得到有效文本块")

        logger.info(
            "解析完成：document=%s blocks=%s chunks=%s chars=%s",
            document_id,
            len(parsed.blocks),
            len(chunks),
            parsed.char_count,
        )

        vectors = self._embedder.embed_documents([chunk.content for chunk in chunks])
        chunk_rows = [
            Chunk(
                id=new_id(),
                document_id=document_id,
                chunk_index=chunk.index,
                content=chunk.content,
                char_count=chunk.char_count,
                page_number=chunk.page_number,
                section=chunk.section,
            )
            for chunk in chunks
        ]

        with self._db.session() as session:
            document = session.get(Document, document_id)
            if document is None:
                raise DocumentNotFoundError(f"文档不存在：{document_id}")
            document.title = parsed.title or document.filename
            document.char_count = parsed.char_count
            document.page_count = parsed.page_count
            document.chunk_count = len(chunk_rows)
            session.add_all(chunk_rows)

        records = [
            VectorRecord(
                id=row.id,
                embedding=vector,
                content=row.content,
                document_id=document_id,
                chunk_id=row.id,
                chunk_index=row.chunk_index,
                document_name="",  # 随后由 _attach_document_name 统一填充
                page_number=row.page_number,
                section=row.section,
            )
            for row, vector in zip(chunk_rows, vectors, strict=True)
        ]
        self._attach_document_name(records, document_id)
        self._store.add(records)

        with self._db.session() as session:
            document = session.get(Document, document_id)
            if document is None:
                raise DocumentNotFoundError(f"文档不存在：{document_id}")
            document.status = DocumentStatus.READY.value
            document.processed_at = utcnow()
            document.error_message = None

    def _attach_document_name(self, records: Sequence[VectorRecord], document_id: str) -> None:
        """向量元数据需要文档名（检索时直接给出引用），从库里取一次填充。"""
        with self._db.session() as session:
            document = session.get(Document, document_id)
            name = document.filename if document else document_id
        for record in records:
            record.document_name = name

    def _mark_failed(self, document_id: str, message: str, path: Path) -> None:
        """记录失败原因并清理半成品（向量、文本块、原始文件）。"""
        try:
            self._store.delete_document(document_id)
        except Exception as exc:  # noqa: BLE001 - 清理失败不应掩盖原始错误
            logger.warning("清理向量失败 document=%s: %s", document_id, exc)

        with self._db.session() as session:
            document = session.get(Document, document_id)
            if document is None:
                return
            session.execute(delete(Chunk).where(Chunk.document_id == document_id))
            document.status = DocumentStatus.FAILED.value
            document.error_message = message[:1000]
            document.chunk_count = 0
            document.processed_at = utcnow()

        path.unlink(missing_ok=True)

    # ------------------------------------------------------------------ 删除
    def delete_document(self, document_id: str) -> None:
        with self._db.session() as session:
            document = session.get(Document, document_id)
            if document is None:
                raise DocumentNotFoundError(f"文档不存在：{document_id}")
            storage_path = document.storage_path

        # 先删向量：万一向量库失败，数据库记录还在，用户可以重试删除，不会出现“DB 没了但向量还在”
        self._store.delete_document(document_id)

        with self._db.session() as session:
            document = session.get(Document, document_id)
            if document is not None:
                session.delete(document)

        if storage_path:
            file_path = self._settings.upload_dir / storage_path
            try:
                file_path.unlink(missing_ok=True)
            except OSError as exc:  # 文件被占用时不影响主流程
                logger.warning("删除原始文件失败 %s: %s", file_path, exc)

        logger.info("文档已删除：%s", document_id)
