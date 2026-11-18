"""统一异常处理：把各类异常收敛成 ``{"error": {...}}``。

这样前端只需要处理一种错误结构，不用判断是 FastAPI 校验错误、HTTPException，
还是业务异常。同时保留正确的 HTTP 状态码。
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import AppError
from app.core.logging import get_logger

logger = get_logger(__name__)


def _payload(code: str, message: str, detail: object = None) -> dict[str, object]:
    body: dict[str, object] = {"code": code, "message": message}
    if detail is not None:
        body["detail"] = detail
    return {"error": body}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error(
                "业务异常 %s %s -> %s(%s): %s",
                request.method,
                request.url.path,
                exc.code,
                exc.status_code,
                exc.message,
            )
        else:
            logger.info(
                "请求被拒绝 %s %s -> %s(%s): %s",
                request.method,
                request.url.path,
                exc.code,
                exc.status_code,
                exc.message,
            )
        return JSONResponse(status_code=exc.status_code, content={"error": exc.to_payload()})

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        fields = [
            {
                "field": ".".join(str(part) for part in error.get("loc", []) if part != "body"),
                "message": error.get("msg", "参数不合法"),
                "type": error.get("type", "value_error"),
            }
            for error in exc.errors()
        ]
        logger.info("参数校验失败 %s %s: %s", request.method, request.url.path, fields)
        return JSONResponse(
            status_code=422,
            content=_payload("validation_error", "请求参数校验失败", {"fields": fields}),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        message = str(exc.detail) if exc.detail else "请求无法完成"
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(f"http_{exc.status_code}", message),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("未处理异常 %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500, content=_payload("internal_error", "服务器内部错误")
        )
