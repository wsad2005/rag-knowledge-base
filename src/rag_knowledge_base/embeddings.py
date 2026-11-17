"""向量化：接口、确定性 Mock 实现与 OpenAI 兼容真实客户端。

当前默认 Mock（本地 n-gram 哈希，无网络、无费用）；real 模式通过
OpenAI 兼容的 /embeddings 接口调用真实服务（已验证目标：智谱 GLM）。
"""

import math
import zlib
from typing import Protocol, runtime_checkable

import httpx

from .config import Settings
from .errors import InvalidResponseError
from .http_transport import post_json


@runtime_checkable
class EmbeddingClient(Protocol):
    """向量化客户端接口：文本进，定长向量出。"""

    def embed(self, text: str) -> list[float]:
        """把文本编码为向量。"""
        ...


class MockEmbeddingClient:
    """确定性 Mock 向量：字符二元组（bigram）哈希 + L2 归一化。

    特性：相同文本得到相同向量；共享字词越多的文本向量越接近，
    足以演示检索排序，但不代表真实语义质量。
    """

    dim = 64

    def embed(self, text: str) -> list[float]:
        """把文本编码为 64 维归一化向量（用 crc32 保证跨进程稳定）。"""
        vector = [0.0] * self.dim
        normalized = text.lower().strip()
        for position in range(len(normalized) - 1):
            gram = normalized[position : position + 2]
            bucket = zlib.crc32(gram.encode("utf-8")) % self.dim
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


class OpenAICompatibleEmbeddingClient:
    """通过 OpenAI 兼容的 /embeddings 接口生成真实向量（已验证目标：智谱 GLM）。

    官方接口（2026-11 文档确认）：POST {base_url}/embeddings，
    请求体 {"model": ..., "input": ...}，响应取 data[0].embedding。
    """

    provider = "openai-compatible"

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """保存配置；transport 仅供测试注入（httpx.MockTransport），生产不传。"""
        self._settings = settings
        self._transport = transport

    def embed(self, text: str) -> list[float]:
        """把文本编码为真实服务返回的向量。"""
        assert self._settings.api_key is not None  # load_settings 已保证非空
        url = (self._settings.base_url or "").rstrip("/") + "/embeddings"
        payload = {"model": self._settings.embedding_model, "input": text}
        data = post_json(
            url,
            api_key=self._settings.api_key,
            payload=payload,
            timeout_seconds=self._settings.timeout_seconds,
            transport=self._transport,
        )
        try:
            embedding = data["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InvalidResponseError("响应缺少 data[0].embedding 字段") from exc
        if not isinstance(embedding, list) or not embedding:
            raise InvalidResponseError("embedding 字段不是非空数组")
        if not all(isinstance(value, (int, float)) for value in embedding):
            raise InvalidResponseError("embedding 含非数值元素")
        return [float(value) for value in embedding]
