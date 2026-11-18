"""FastAPI 依赖：从 app.state 取出容器与各服务。"""

from __future__ import annotations

from fastapi import Depends, Request

from app.core.container import Container
from app.services.document_service import DocumentService
from app.services.rag_service import RagService


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


def get_document_service(container: Container = Depends(get_container)) -> DocumentService:
    return container.documents


def get_rag_service(container: Container = Depends(get_container)) -> RagService:
    return container.rag
