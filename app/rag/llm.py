"""大模型客户端（OpenAI 兼容 ``/chat/completions``）。

只依赖一个协议：``generate(messages) -> str``。换成别的厂商只要实现这个协议。
所有异常都映射为项目异常，便于路由层返回 502/504 并有统一错误码。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import httpx
from pydantic import SecretStr

from app.core.config import Settings
from app.core.errors import (
    LLMEmptyResponseError,
    LLMNotConfiguredError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


class LLMClient(Protocol):
    """生成模型协议。"""

    model_name: str

    def generate(self, messages: Sequence[dict[str, str]]) -> str:  # pragma: no cover - 协议
        ...


class NullLLM:
    """未配置模型时的占位实现：调用即抛错，由业务层降级为“仅检索”。"""

    model_name = ""

    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        raise LLMNotConfiguredError("当前未配置大模型，无法生成回答")


class OpenAICompatibleLLM:
    """OpenAI 兼容的对话补全客户端。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: SecretStr | str,
        model: str,
        timeout: float = 30.0,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        client: httpx.Client | None = None,
    ) -> None:
        key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        if not key:
            raise LLMNotConfiguredError("缺少 LLM_API_KEY")
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        payload = {
            "model": self.model_name,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        try:
            response = self._client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"大模型响应超时（>{self._timeout_hint()}s）") from exc
        except httpx.HTTPError as exc:
            raise LLMUpstreamError("无法连接大模型服务", detail={"reason": str(exc)}) from exc

        if response.status_code == 401:
            raise LLMUpstreamError("大模型鉴权失败，请检查 LLM_API_KEY")
        if response.status_code == 429:
            raise LLMUpstreamError("大模型服务限流（429），请稍后重试")
        if response.status_code >= 400:
            raise LLMUpstreamError(
                f"大模型服务返回 HTTP {response.status_code}",
                detail={"body": response.text[:300]},
            )

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMUpstreamError(
                "大模型返回格式不符合 OpenAI 协议", detail={"body": response.text[:300]}
            ) from exc

        content = content.strip()
        if not content:
            raise LLMEmptyResponseError("大模型返回了空内容")
        return content

    def _timeout_hint(self) -> str:
        timeout = self._client.timeout
        return str(getattr(timeout, "read", timeout))


def build_llm(settings: Settings) -> LLMClient:
    """配置齐全才创建真实客户端，否则返回 NullLLM（由业务层降级）。"""
    if not settings.llm_configured:
        logger.info("未配置大模型（LLM_BASE_URL/LLM_API_KEY/LLM_MODEL），问答将仅返回检索结果")
        return NullLLM()

    assert settings.llm_base_url is not None
    assert settings.llm_api_key is not None
    return OpenAICompatibleLLM(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout=settings.llm_timeout_seconds,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
