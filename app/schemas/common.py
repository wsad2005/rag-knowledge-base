"""API 通用响应模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str = Field(description="机器可读的错误码，如 unsupported_file_type")
    message: str = Field(description="面向用户的中文错误说明")
    detail: Any | None = Field(default=None, description="补充信息，如允许的扩展名列表")


class ErrorResponse(BaseModel):
    """所有非 2xx 响应都遵循这一结构。"""

    error: ErrorBody

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": {
                    "code": "unsupported_file_type",
                    "message": "不支持的文件类型：.exe",
                    "detail": {"allowed_extensions": [".pdf", ".docx", ".txt", ".md"]},
                }
            }
        }
    }
