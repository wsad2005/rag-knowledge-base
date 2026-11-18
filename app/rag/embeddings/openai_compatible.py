"""OpenAI 兼容的远程 Embedding（``POST {base_url}/embeddings``）。

兼容 OpenAI / 智谱 / 硅基流动 / Ollama 等遵循同一协议的服务。
所有网络异常都映射为项目自己的异常，路由层据此返回 502/504，而不是 500。
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx
from pydantic import SecretStr

from app.core.errors import EmbeddingConfigurationError, EmbeddingError
from app.core.logging import get_logger
from app.rag.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)


class OpenAICompatibleEmbedder(EmbeddingProvider):
    provider_name = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: SecretStr | str,
        model: str,
        dimension: int | None = None,
        timeout: float = 30.0,
        batch_size: int = 32,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url or not model:
            raise EmbeddingConfigurationError("远程 Embedding 需要配置 base_url 与模型名")
        key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        if not key:
            raise EmbeddingConfigurationError("远程 Embedding 缺少 API Key")

        self.model_name = model
        self.batch_size = batch_size
        #: 0 表示尚未从响应中得知维度
        self._dimension = int(dimension or 0)
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )

    @property
    def dimension(self) -> int:
        if self._dimension <= 0:
            raise EmbeddingConfigurationError("尚未调用远程接口，向量维度未知")
        return self._dimension

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------ 实现
    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            vectors.extend(self._request_batch(batch))
        return vectors

    def _request_batch(self, batch: list[str]) -> list[list[float]]:
        payload = {"model": self.model_name, "input": batch, "encoding_format": "float"}
        try:
            response = self._client.post("/embeddings", json=payload)
        except httpx.TimeoutException as exc:
            raise EmbeddingError(
                "远程 Embedding 请求超时", detail={"model": self.model_name}
            ) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingError(
                "无法连接远程 Embedding 服务", detail={"reason": str(exc)}
            ) from exc

        if response.status_code == 401:
            raise EmbeddingConfigurationError("远程 Embedding 鉴权失败，请检查 API Key")
        if response.status_code == 429:
            raise EmbeddingError("远程 Embedding 触发限流（429），请稍后重试")
        if response.status_code >= 400:
            raise EmbeddingError(
                f"远程 Embedding 返回 HTTP {response.status_code}",
                detail={"body": response.text[:300]},
            )

        try:
            data = response.json()
            items = sorted(data["data"], key=lambda item: item.get("index", 0))
            embeddings = [[float(value) for value in item["embedding"]] for item in items]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError(
                "远程 Embedding 返回格式不符合 OpenAI 协议", detail={"body": response.text[:300]}
            ) from exc

        if len(embeddings) != len(batch):
            raise EmbeddingError(
                f"远程 Embedding 返回数量不匹配：期望 {len(batch)}，实际 {len(embeddings)}"
            )

        if self._dimension <= 0 and embeddings:
            self._dimension = len(embeddings[0])
            logger.info("远程 Embedding 维度已确定：%s", self._dimension)
        return embeddings
