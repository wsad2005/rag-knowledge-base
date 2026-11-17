"""errors 模块测试。"""

import pytest

from rag_knowledge_base.errors import (
    ConfigError,
    InvalidResponseError,
    RagAuthError,
    RagConnectionError,
    RagError,
    RagRateLimitError,
    RagRequestError,
    RagServerError,
    RagTimeoutError,
    friendly_message,
)

REQUEST_ERRORS = [
    RagTimeoutError,
    RagConnectionError,
    RagAuthError,
    RagRateLimitError,
    RagServerError,
    InvalidResponseError,
]


@pytest.mark.parametrize("exc_type", REQUEST_ERRORS)
def test_request_errors_share_base(exc_type: type[RagRequestError]) -> None:
    assert issubclass(exc_type, RagRequestError)
    assert issubclass(exc_type, RagError)


@pytest.mark.parametrize(
    ("exc", "keyword"),
    [
        (RagTimeoutError("t"), "超时"),
        (RagConnectionError("c"), "连接"),
        (RagAuthError("a"), "认证"),
        (RagRateLimitError("r"), "限流"),
        (RagServerError("s"), "服务端错误"),
        (InvalidResponseError("i"), "解析"),
        (ConfigError("x"), "配置"),
    ],
)
def test_friendly_message_by_type(exc: Exception, keyword: str) -> None:
    assert keyword in friendly_message(exc)


def test_config_error_message_includes_detail() -> None:
    assert "LLM_API_KEY 缺少" in friendly_message(ConfigError("LLM_API_KEY 缺少"))


def test_friendly_message_falls_back_for_unknown() -> None:
    assert friendly_message(ValueError("boom")) == "发生意外错误，请查看日志获取详情。"
