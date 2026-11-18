"""文档相关接口。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_container, get_document_service
from app.core.container import Container
from app.core.errors import FileTooLargeError
from app.schemas.common import ErrorResponse
from app.schemas.document import (
    DocumentDetail,
    DocumentListResponse,
    DocumentSummary,
    UploadResponse,
)
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["文档管理"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="上传并解析文档",
    description=(
        "支持 PDF / DOCX / TXT / Markdown。服务端会依次完成：类型与大小校验 → 内容哈希查重 → "
        "解析 → 切分 → 向量化 → 写入向量库。解析失败会返回 422 并在文档列表中保留 failed 记录。"
    ),
    responses={
        413: {"model": ErrorResponse, "description": "文件超过大小限制"},
        415: {"model": ErrorResponse, "description": "不支持的文件类型"},
        409: {"model": ErrorResponse, "description": "相同内容的文档已存在"},
        422: {"model": ErrorResponse, "description": "文档解析失败"},
    },
)
async def upload_document(
    file: UploadFile = File(..., description="待上传的文档"),
    service: DocumentService = Depends(get_document_service),
    container: Container = Depends(get_container),
) -> UploadResponse:
    # 早失败：Starlette 已把内容写入临时文件，这里先用已知大小拦截超大文件
    declared_size = getattr(file, "size", None)
    if declared_size and declared_size > container.settings.upload_max_bytes:
        raise FileTooLargeError(
            f"文件超过 {container.settings.upload_max_mb}MB 限制",
            detail={"max_bytes": container.settings.upload_max_bytes},
        )

    document = await run_in_threadpool(service.ingest_upload, file.filename, file.file)
    return UploadResponse(document=DocumentSummary.model_validate(document))


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="文档列表",
    description="按上传时间倒序分页返回；`q` 支持按文件名模糊搜索。",
)
async def list_documents(
    skip: int = Query(0, ge=0, description="跳过条数"),
    limit: int = Query(20, ge=1, le=100, description="每页条数"),
    q: str | None = Query(None, max_length=100, description="文件名关键字"),
    service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    total, items = await run_in_threadpool(
        service.list_documents, skip=skip, limit=limit, query=q
    )
    return DocumentListResponse(
        total=total,
        skip=skip,
        limit=limit,
        items=[DocumentSummary.model_validate(item) for item in items],
    )


@router.get(
    "/{document_id}",
    response_model=DocumentDetail,
    summary="文档详情（含切分块）",
    responses={404: {"model": ErrorResponse, "description": "文档不存在"}},
)
async def get_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentDetail:
    document = await run_in_threadpool(service.get_document, document_id)
    return DocumentDetail.model_validate(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除文档及其向量",
    responses={404: {"model": ErrorResponse, "description": "文档不存在"}},
)
async def delete_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
) -> None:
    await run_in_threadpool(service.delete_document, document_id)
