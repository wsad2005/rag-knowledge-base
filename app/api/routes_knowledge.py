"""知识库检索与问答接口。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_rag_service
from app.schemas.common import ErrorResponse
from app.schemas.knowledge import QueryRequest, QueryResponse, SearchResponse
from app.services.rag_service import RagService

router = APIRouter(prefix="/knowledge", tags=["知识库问答"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="RAG 问答",
    description=(
        "完整链路：问题向量化 → 向量检索 Top-K → 组装带编号的上下文 → 大模型生成 → "
        "返回答案与引用。\n\n"
        "未配置大模型时 `answer_mode` 为 `extractive`（只返回检索到的原文片段）；"
        "检索不到相关内容时 `answer_mode` 为 `no_context`，不会编造答案。"
    ),
    responses={
        400: {"model": ErrorResponse, "description": "top_k 超出允许范围"},
        503: {"model": ErrorResponse, "description": "向量化或向量库不可用"},
    },
)
async def query_knowledge(
    payload: QueryRequest,
    service: RagService = Depends(get_rag_service),
) -> QueryResponse:
    return await run_in_threadpool(
        service.query,
        payload.question,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
    )


@router.get(
    "/search",
    response_model=SearchResponse,
    summary="检索调试（不调用大模型）",
    description="只做向量检索，返回 Top-K 片段与相似度，用于调参和排查召回问题。",
)
async def search_knowledge(
    q: str = Query(..., min_length=1, max_length=1000, description="检索词/问题"),
    top_k: int | None = Query(None, ge=1, le=50, description="召回条数"),
    document_ids: list[str] | None = Query(
        None, description="限定文档 ID，可重复传参：?document_ids=a&document_ids=b"
    ),
    service: RagService = Depends(get_rag_service),
) -> SearchResponse:
    return await run_in_threadpool(
        service.search, q, top_k=top_k, document_ids=document_ids
    )
