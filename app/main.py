"""FastAPI 应用入口。

启动顺序：读配置 → 建目录 → 建库表 → 装配 Embedding / 向量库 / LLM → 挂路由与静态页面。
关闭时统一释放 httpx 连接与数据库连接池。
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.core.config import PROJECT_ROOT, Settings
from app.core.container import Container
from app.core.handlers import register_exception_handlers
from app.core.logging import get_logger, request_id_var

logger = get_logger(__name__)

FRONTEND_DIR = PROJECT_ROOT / "frontend"

DESCRIPTION = """
基于 RAG（Retrieval-Augmented Generation）的智能文档知识库。

**能力**：PDF / DOCX / TXT / Markdown 上传解析 → 段落感知切分 → 向量化 → ChromaDB 持久化检索
→ 大模型问答并返回可溯源引用。

**两种运行模式**：
- 未配置大模型：`/api/knowledge/query` 返回检索到的原文片段（`answer_mode=extractive`），
  文档处理与检索功能完全可用；
- 配置 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` 后：返回模型生成的归纳答案。

**接口文档**：`/docs`（Swagger UI）、`/redoc`、`/openapi.json`。
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    """应用工厂。测试通过传入自定义 ``Settings`` 得到隔离的数据目录。"""
    resolved_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = Container(resolved_settings)
        app.state.container = container
        logger.info("%s v%s 已启动", resolved_settings.app_name, resolved_settings.app_version)
        try:
            yield
        finally:
            container.close()

    app = FastAPI(
        title="RAG Knowledge Base API",
        description=DESCRIPTION,
        version=resolved_settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={"name": "rag-knowledge-base"},
        license_info={"name": "MIT"},
    )

    register_exception_handlers(app)
    app.include_router(api_router, prefix=resolved_settings.api_prefix)

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # noqa: ANN001, ANN202
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        if not request.url.path.startswith("/static"):
            logger.info(
                "%s %s -> %s (%.1fms)",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )
        return response

    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    """挂载原生前端页面：``/`` 返回首页，``/static`` 提供 CSS/JS。"""
    if not FRONTEND_DIR.exists():  # pragma: no cover - 仓库始终包含 frontend/
        logger.warning("未找到 frontend 目录，跳过页面挂载：%s", FRONTEND_DIR)
        return

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> JSONResponse:
        # 不额外放二进制资源，直接告诉浏览器没有图标
        return JSONResponse(status_code=204, content=None)


app = create_app()
