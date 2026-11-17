"""embeddings 模块测试（Mock 向量与 OpenAI 兼容真实客户端，HTTP 全部 Mock）。"""

import json
import math
from collections.abc import Callable

import httpx
import pytest

from rag_knowledge_base.config import load_settings
from rag_knowledge_base.embeddings import (
    EmbeddingClient,
    MockEmbeddingClient,
    OpenAICompatibleEmbeddingClient,
)
from rag_knowledge_base.errors import InvalidResponseError, RagAuthError, RagTimeoutError

BASE_ENV = {
    "RAG_MODE": "real",
    "LLM_BASE_URL": "https://api.example.invalid/v1",  # 测试用假地址
    "LLM_API_KEY": "dummy-key-abcdef123456",  # 测试用假密钥
    "EMBEDDING_MODEL": "dummy-embedding",  # 测试用假模型名
    "GENERATION_MODEL": "dummy-generation",  # 测试用假模型名
}


def make_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> OpenAICompatibleEmbeddingClient:
    settings = load_settings(environ=dict(BASE_ENV))
    return OpenAICompatibleEmbeddingClient(settings, transport=httpx.MockTransport(handler))


def ok_handler(request: httpx.Request) -> httpx.Response:
    assert request.url.path.endswith("/embeddings")
    assert "dummy-key-abcdef123456" not in str(request.url)  # 密钥只在请求头
    assert request.headers["Authorization"] == "Bearer dummy-key-abcdef123456"
    body = json.loads(request.content)
    assert body["model"] == "dummy-embedding"
    assert body["input"] == "向量检索"
    return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.25, -0.5, 1.0]}]})


def test_same_text_gives_same_vector() -> None:
    embedder = MockEmbeddingClient()
    assert embedder.embed("向量检索") == embedder.embed("向量检索")


def test_vector_is_normalized_with_fixed_dim() -> None:
    vector = MockEmbeddingClient().embed("rag 检索增强生成")
    assert len(vector) == MockEmbeddingClient.dim
    assert math.isclose(sum(value * value for value in vector), 1.0, rel_tol=1e-9)


def test_different_texts_give_different_vectors() -> None:
    embedder = MockEmbeddingClient()
    assert embedder.embed("猫") != embedder.embed("数据库")


def test_similar_texts_score_higher_than_unrelated() -> None:
    embedder = MockEmbeddingClient()
    query = embedder.embed("余弦相似度")
    near = embedder.embed("按余弦相似度排序")
    far = embedder.embed("今天天气不错")
    query_dot_near = sum(a * b for a, b in zip(query, near, strict=True))
    query_dot_far = sum(a * b for a, b in zip(query, far, strict=True))
    assert query_dot_near > query_dot_far


def test_blank_text_returns_zero_vector() -> None:
    assert MockEmbeddingClient().embed("   ") == [0.0] * MockEmbeddingClient.dim


def test_embedder_satisfies_protocol() -> None:
    assert isinstance(MockEmbeddingClient(), EmbeddingClient)


def test_real_client_returns_embedding() -> None:
    assert make_client(ok_handler).embed("向量检索") == [0.25, -0.5, 1.0]


def test_real_client_error_message_never_contains_api_key() -> None:
    client = make_client(lambda request: httpx.Response(401, json={"error": {"message": "x"}}))
    with pytest.raises(RagAuthError) as excinfo:
        client.embed("hi")
    assert BASE_ENV["LLM_API_KEY"] not in str(excinfo.value)


def test_real_client_timeout_maps_to_rag_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    with pytest.raises(RagTimeoutError):
        make_client(handler).embed("hi")


def test_real_client_missing_embedding_field() -> None:
    client = make_client(lambda request: httpx.Response(200, json={"data": []}))
    with pytest.raises(InvalidResponseError):
        client.embed("hi")


def test_real_client_non_numeric_embedding() -> None:
    payload = {"data": [{"index": 0, "embedding": ["bad"]}]}
    client = make_client(lambda request: httpx.Response(200, json=payload))
    with pytest.raises(InvalidResponseError):
        client.embed("hi")
