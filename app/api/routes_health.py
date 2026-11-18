"""健康检查接口。"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_container
from app.core.container import Container
from app.schemas.knowledge import HealthResponse

router = APIRouter(tags=["系统"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="健康检查",
    description=(
        "返回数据库、向量库、Embedding、大模型配置与文档数量的真实状态。"
        "数据库或向量库异常时 `status` 为 `degraded`，但接口本身仍返回 200，便于探活。"
    ),
)
async def health(container: Container = Depends(get_container)) -> HealthResponse:
    snapshot = await run_in_threadpool(container.health_snapshot)
    return HealthResponse(**snapshot)  # type: ignore[arg-type]
