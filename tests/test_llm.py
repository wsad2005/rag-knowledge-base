"""大模型客户端测试：正常返回、鉴权失败、限流、超时、空响应、协议不兼容。

所有请求都由 ``httpx.MockTransport`` 拦截：不联网、不产生费用、结果可重复。
"""

from __future__ import annotations

import httpx
import pytest

from app.core.config import Settings
from app.core.errors import (
    LLMEmptyResponseError,
    LLMNotConfiguredError,
    LLMTimeoutError,
    LLMUpstreamError,
)
from app.rag.llm import NullLLM, OpenAICompatibleLLM, build_llm

MESSAGES = [{"role": "user", "content": "你好"}]


def make_llm(handler, **kwargs) -> OpenAICompatibleLLM:  # noqa: ANN001
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://fake-llm.local/v1",
    )
    return OpenAICompatibleLLM(
        base_url="https://fake-llm.local/v1",
        api_key="test-key",
        model="test-model",
        timeout=5,
        client=client,
        **kwargs,
    )


def _ok(content: str = "这是模型生成的答案 [1]") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
            "usage": {"total_tokens": 42},
        },
    )


class TestSuccess:
    def test_returns_content(self) -> None:
        llm = make_llm(lambda request: _ok())
        assert llm.generate(MESSAGES) == "这是模型生成的答案 [1]"

    def test_sends_expected_payload(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.read().decode()))
            return _ok()

        make_llm(handler).generate(MESSAGES)
        assert captured["model"] == "test-model"
        assert captured["messages"] == MESSAGES
        assert captured["stream"] is False

    def test_strips_whitespace(self) -> None:
        llm = make_llm(lambda request: _ok("  答案  \n"))
        assert llm.generate(MESSAGES) == "答案"


class TestFailures:
    def test_unauthorized(self) -> None:
        llm = make_llm(lambda request: httpx.Response(401, json={"error": "invalid key"}))
        with pytest.raises(LLMUpstreamError) as excinfo:
            llm.generate(MESSAGES)
        assert "鉴权" in excinfo.value.message
        assert excinfo.value.status_code == 502

    def test_rate_limited(self) -> None:
        llm = make_llm(lambda request: httpx.Response(429, json={"error": "slow down"}))
        with pytest.raises(LLMUpstreamError) as excinfo:
            llm.generate(MESSAGES)
        assert "限流" in excinfo.value.message

    def test_server_error(self) -> None:
        llm = make_llm(lambda request: httpx.Response(500, text="boom"))
        with pytest.raises(LLMUpstreamError) as excinfo:
            llm.generate(MESSAGES)
        assert excinfo.value.status_code == 502

    def test_timeout(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("timed out", request=request)

        with pytest.raises(LLMTimeoutError) as excinfo:
            make_llm(handler).generate(MESSAGES)
        assert excinfo.value.status_code == 504

    def test_connection_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with pytest.raises(LLMUpstreamError):
            make_llm(handler).generate(MESSAGES)

    def test_empty_content(self) -> None:
        llm = make_llm(lambda request: _ok("   "))
        with pytest.raises(LLMEmptyResponseError):
            llm.generate(MESSAGES)

    def test_malformed_payload(self) -> None:
        llm = make_llm(lambda request: httpx.Response(200, json={"choices": []}))
        with pytest.raises(LLMUpstreamError):
            llm.generate(MESSAGES)

    def test_non_json_body(self) -> None:
        llm = make_llm(lambda request: httpx.Response(200, text="<html>gateway</html>"))
        with pytest.raises(LLMUpstreamError):
            llm.generate(MESSAGES)

    def test_missing_api_key(self) -> None:
        with pytest.raises(LLMNotConfiguredError):
            OpenAICompatibleLLM(base_url="https://x.local/v1", api_key="", model="m")


class TestNullLLM:
    def test_raises_not_configured(self) -> None:
        with pytest.raises(LLMNotConfiguredError):
            NullLLM().generate(MESSAGES)

    def test_model_name_is_empty(self) -> None:
        assert NullLLM().model_name == ""


class TestBuildLLM:
    def test_returns_null_when_not_configured(self, tmp_path) -> None:
        settings = Settings(
            _env_file=None,
            environment="test",
            data_dir=tmp_path,
            llm_base_url=None,
            llm_api_key=None,
        )
        assert isinstance(build_llm(settings), NullLLM)

    def test_returns_client_when_configured(self, tmp_path) -> None:
        settings = Settings(
            _env_file=None,
            environment="test",
            data_dir=tmp_path,
            llm_base_url="https://fake-llm.local/v1",
            llm_api_key="test-key",  # noqa: S106 - 测试用假密钥
            llm_model="test-model",
        )
        llm = build_llm(settings)
        assert isinstance(llm, OpenAICompatibleLLM)
        assert llm.model_name == "test-model"
        llm.close()
