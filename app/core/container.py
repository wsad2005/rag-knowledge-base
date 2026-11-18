"""依赖装配容器。

所有组件在应用启动时创建一次，挂在 ``app.state.container`` 上；路由通过依赖注入取用。
测试里直接用测试配置构造 ``Container``，不用 monkeypatch 全局变量。
"""

from __future__ import annotations

import time

from sqlalchemy import func, select, text

from app.core.config import Settings
from app.core.logging import configure_logging, get_logger
from app.database.session import Database
from app.models.document import Document
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.embeddings.factory import build_embedder
from app.rag.llm import LLMClient, build_llm
from app.services.document_service import DocumentService
from app.services.rag_service import RagService
from app.vector_store.base import VectorStore
from app.vector_store.factory import build_vector_store

logger = get_logger(__name__)


class Container:
    """应用级单例，持有数据库、向量库、模型与两个业务服务。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        configure_logging(settings.log_level)
        settings.ensure_directories()

        self.database = Database(settings)
        self.database.create_all()

        self.embedder: EmbeddingProvider = build_embedder(settings)
        self.vector_store: VectorStore = build_vector_store(settings, self.embedder)
        self.llm: LLMClient = build_llm(settings)

        self.documents = DocumentService(
            settings=settings,
            database=self.database,
            embedder=self.embedder,
            vector_store=self.vector_store,
        )
        self.rag = RagService(
            settings=settings,
            database=self.database,
            embedder=self.embedder,
            vector_store=self.vector_store,
            llm=self.llm,
        )
        self.started_at = time.monotonic()
        logger.info("应用装配完成：%s", settings.masked_summary())

    # ------------------------------------------------------------------ 健康检查
    def health_snapshot(self) -> dict[str, object]:
        database_status = "ok"
        document_count = 0
        try:
            with self.database.session() as session:
                session.execute(text("SELECT 1"))
                document_count = int(
                    session.scalar(select(func.count()).select_from(Document)) or 0
                )
        except Exception as exc:  # noqa: BLE001
            database_status = f"error: {exc}"

        try:
            vector_store_status = self.vector_store.health()
        except Exception as exc:  # noqa: BLE001
            vector_store_status = {"provider": "chromadb", "status": f"error: {exc}", "vectors": 0}

        try:
            dimension = self.embedder.dimension
        except Exception:  # noqa: BLE001 - 远程模型未调用前维度未知
            dimension = 0

        llm_configured = bool(getattr(self.llm, "model_name", ""))
        overall = "ok"
        if database_status != "ok" or vector_store_status.get("status") != "ok":
            overall = "degraded"

        return {
            "status": overall,
            "version": self.settings.app_version,
            "environment": self.settings.environment,
            "database": database_status,
            "vector_store": vector_store_status,
            "embedding": {
                "provider": self.embedder.provider_name,
                "model": self.embedder.model_name,
                "dimension": dimension,
            },
            "llm": {
                "configured": llm_configured,
                "model": getattr(self.llm, "model_name", "") or None,
                "fallback": "extractive" if not llm_configured else None,
            },
            "chunking": {
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
            },
            "document_count": document_count,
            "uptime_seconds": round(time.monotonic() - self.started_at, 3),
        }

    # ------------------------------------------------------------------ 释放
    def close(self) -> None:
        for component in (self.llm, self.embedder):
            close = getattr(component, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("释放 %s 失败：%s", type(component).__name__, exc)
        self.database.dispose()
        logger.info("应用资源已释放")
