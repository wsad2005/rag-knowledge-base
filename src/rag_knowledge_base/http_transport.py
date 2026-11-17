"""共享 HTTP 传输层：把 httpx 异常与状态码统一映射为 Rag* 异常。

Embedding 与 Generation 两个真实客户端共用本模块，避免重复的
超时/连接/认证/限流映射逻辑；异常消息只含状态码与 URL，不含密钥。
"""

import logging
from typing import Any

import httpx

from .errors import (
    InvalidResponseError,
    RagAuthError,
    RagConnectionError,
    RagRateLimitError,
    RagServerError,
    RagTimeoutError,
)

logger = logging.getLogger(__name__)


def post_json(
    url: str,
    *,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: int,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """发送 Bearer 认证的 JSON POST 请求并解析为 dict。

    transport 仅供测试注入（httpx.MockTransport），生产不传。
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    logger.info("real 请求开始：POST %s（超时 %ss）", url, timeout_seconds)
    with httpx.Client(
        timeout=httpx.Timeout(timeout_seconds), transport=transport
    ) as client:
        try:
            response = client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise RagTimeoutError(f"请求超过 {timeout_seconds}s 未响应") from exc
        except httpx.TransportError as exc:
            raise RagConnectionError(f"无法连接 {url}") from exc
    logger.info("real 请求结束：HTTP %d", response.status_code)
    _raise_for_status(response.status_code)
    try:
        data = response.json()
    except ValueError as exc:
        raise InvalidResponseError(f"响应不是合法 JSON（HTTP {response.status_code}）") from exc
    if not isinstance(data, dict):
        raise InvalidResponseError(f"响应顶层不是 JSON 对象（HTTP {response.status_code}）")
    return data


def _raise_for_status(status: int) -> None:
    """把非 2xx 状态码映射为具体异常（消息不含密钥与响应体）。"""
    if status in (401, 403):
        raise RagAuthError(f"认证失败（HTTP {status}）：请核对本地 .env 中的 LLM_API_KEY")
    if status == 404:
        raise InvalidResponseError("地址不存在（HTTP 404）：请检查 LLM_BASE_URL 是否包含版本路径")
    if status == 429:
        raise RagRateLimitError("触发限流（HTTP 429）：请稍后重试")
    if status >= 500:
        raise RagServerError(f"服务端错误（HTTP {status}）：请稍后重试")
    if status >= 400:
        raise InvalidResponseError(f"请求被拒绝（HTTP {status}）：请检查请求参数配置")
