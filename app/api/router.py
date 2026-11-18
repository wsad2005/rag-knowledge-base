"""API 路由汇总。"""

from fastapi import APIRouter

from app.api import routes_documents, routes_health, routes_knowledge

api_router = APIRouter()
api_router.include_router(routes_health.router)
api_router.include_router(routes_documents.router)
api_router.include_router(routes_knowledge.router)
