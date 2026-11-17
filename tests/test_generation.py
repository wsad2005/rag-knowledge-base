"""generation 模块测试（Mock 生成与 OpenAI 兼容真实客户端，HTTP 全部 Mock）。"""

import json
from collections.abc import Callable

import httpx
import pytest

from rag_knowledge_base.config import load_settings
from rag_knowledge_base.errors import (
    InvalidResponseError,
    RagAuthError,
    RagConnectionError,
    RagRateLimitError,
    RagServerError,
)
from rag_knowledge_base.generation import (
    GenerationClient,
    MockGenerationClient,
    OpenAICompatibleGenerationClient,
)
from rag_knowledge_base.text_splitter import Chunk
from rag_knowledge_base.vector_store import SearchHit

BASE_ENV = {
    "RAG_MODE": "real",
    "LLM_BASE_URL": "https://api.example.invalid/v1",  # 测试用假地址
    "LLM_API_KEY": "dummy-key-abcdef123456",  # 测试用假密钥
    "EMBEDDING_MODEL": "dummy-embedding",  # 测试用假模型名
    "GENERATION_MODEL": "dummy-generation",  # 测试用假模型名
}


def make_hit(text: str, score: float) -> SearchHit:
    return SearchHit(chunk=Chunk(text=text, source="s", index=0), score=score)


def make_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> OpenAICompatibleGenerationClient:
    settings = load_settings(environ=dict(BASE_ENV))
    return OpenAICompatibleGenerationClient(settings, transport=httpx.MockTransport(handler))


def test_generation_cites_contexts() -> None:
    answer = MockGenerationClient().generate(
        "问题",
        [make_hit("资料一", 0.9), make_hit("资料二", 0.5)],
    )
    assert "2 段资料" in answer
    assert "资料一" in answer


def test_generation_without_contexts_says_so() -> None:
    answer = MockGenerationClient().generate("问题", [])
    assert "未检索到" in answer


def test_generator_satisfies_protocol() -> None:
    assert isinstance(MockGenerationClient(), GenerationClient)


def test_real_client_prompt_includes_question_context_and_rules() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        assert request.url.path.endswith("/chat/completions")
        assert "dummy-key-abcdef123456" not in str(request.url)  # 密钥只在请求头
        assert request.headers["Authorization"] == "Bearer dummy-key-abcdef123456"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "根据资料回答"}}]},
        )

    answer = make_client(handler).generate(
        "什么是 RAG？",
        [make_hit("RAG 是检索增强生成技术", 0.9)],
    )
    assert answer == "根据资料回答"
    body = captured["body"]
    assert body["model"] == "dummy-generation"
    system = body["messages"][0]["content"]
    user = body["messages"][1]["content"]
    assert "资料中没有足够信息" in system  # 防编造规则进入 System Prompt
    assert "不要编造" in system
    assert "什么是 RAG？" in user  # 用户问题进入请求
    assert "RAG 是检索增强生成技术" in user  # 检索到的块进入请求


def test_real_client_rate_limit_maps_to_rag_rate_limit_error() -> None:
    client = make_client(lambda request: httpx.Response(429, json={"error": {"message": "x"}}))
    with pytest.raises(RagRateLimitError):
        client.generate("q", [make_hit("c", 0.5)])


def test_real_client_server_error_maps_to_rag_server_error() -> None:
    client = make_client(lambda request: httpx.Response(500, json={"error": {"message": "x"}}))
    with pytest.raises(RagServerError):
        client.generate("q", [make_hit("c", 0.5)])


def test_real_client_connection_error_maps_to_rag_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(RagConnectionError):
        make_client(handler).generate("q", [make_hit("c", 0.5)])


def test_real_client_missing_choices_maps_to_invalid_response() -> None:
    client = make_client(lambda request: httpx.Response(200, json={"object": "chat.completion"}))
    with pytest.raises(InvalidResponseError):
        client.generate("q", [make_hit("c", 0.5)])


def test_real_client_auth_error_message_never_contains_api_key() -> None:
    client = make_client(lambda request: httpx.Response(401, json={"error": {"message": "x"}}))
    with pytest.raises(RagAuthError) as excinfo:
        client.generate("q", [make_hit("c", 0.5)])
    assert BASE_ENV["LLM_API_KEY"] not in str(excinfo.value)
