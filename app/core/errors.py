"""领域异常与统一错误码。

约定：业务代码只抛 :class:`AppError` 的子类，由 ``app.core.handlers`` 统一转换为
``{"error": {"code", "message", "detail"}}`` 结构的 JSON 响应，路由层不写 try/except 拼装响应。
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """所有可预期业务异常的基类。"""

    code: str = "internal_error"
    status_code: int = 500
    default_message: str = "服务器内部错误"

    def __init__(
        self,
        message: str | None = None,
        *,
        detail: Any = None,
        code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.detail = detail
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        super().__init__(self.message)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


# --------------------------------------------------------------------- 请求校验
class InvalidRequestError(AppError):
    code = "invalid_request"
    status_code = 400
    default_message = "请求参数不合法"


class UnsupportedFileTypeError(AppError):
    code = "unsupported_file_type"
    status_code = 415
    default_message = "不支持的文件类型"


class FileTooLargeError(AppError):
    code = "file_too_large"
    status_code = 413
    default_message = "文件超过大小限制"


class EmptyFileError(AppError):
    code = "empty_file"
    status_code = 400
    default_message = "文件内容为空"


# --------------------------------------------------------------------- 资源
class DocumentNotFoundError(AppError):
    code = "document_not_found"
    status_code = 404
    default_message = "文档不存在"


class DuplicateDocumentError(AppError):
    code = "duplicate_document"
    status_code = 409
    default_message = "相同内容的文档已存在"


# --------------------------------------------------------------------- 处理
class DocumentParseError(AppError):
    code = "document_parse_error"
    status_code = 422
    default_message = "文档解析失败，无法提取有效文本"


class EmptyKnowledgeBaseError(AppError):
    code = "empty_knowledge_base"
    status_code = 409
    default_message = "知识库中还没有可用于检索的文档"


# --------------------------------------------------------------------- 依赖服务
class EmbeddingError(AppError):
    code = "embedding_error"
    status_code = 503
    default_message = "向量化失败"


class EmbeddingConfigurationError(EmbeddingError):
    code = "embedding_config_error"
    default_message = "Embedding 配置不完整"


class VectorStoreError(AppError):
    code = "vector_store_error"
    status_code = 503
    default_message = "向量库操作失败"


class LLMNotConfiguredError(AppError):
    code = "llm_not_configured"
    status_code = 503
    default_message = "未配置大模型，无法生成回答"


class LLMTimeoutError(AppError):
    code = "llm_timeout"
    status_code = 504
    default_message = "大模型响应超时"


class LLMUpstreamError(AppError):
    code = "llm_upstream_error"
    status_code = 502
    default_message = "大模型服务返回错误"


class LLMEmptyResponseError(AppError):
    code = "llm_empty_response"
    status_code = 502
    default_message = "大模型返回了空内容"
