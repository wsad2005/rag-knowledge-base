"""异常分类与用户友好提示（与 01 项目同一风格）。

约定：所有异常消息都不包含 API Key 等秘密信息，可以安全进入日志。
"""


class RagError(Exception):
    """本项目所有自定义异常的基类。"""


class ConfigError(RagError):
    """配置缺失或取值非法（环境变量 / .env）。"""


class RagRequestError(RagError):
    """调用模型服务过程中的失败（超时、连接、认证、限流、服务端错误、无效响应）。"""


class RagTimeoutError(RagRequestError):
    """请求在限定时间内未得到响应。"""


class RagConnectionError(RagRequestError):
    """无法连接到服务地址。"""


class RagAuthError(RagRequestError):
    """API Key 无效或缺少权限。"""


class RagRateLimitError(RagRequestError):
    """触发服务商限流（HTTP 429）。"""


class RagServerError(RagRequestError):
    """模型服务端内部错误（HTTP 5xx）。"""


class InvalidResponseError(RagRequestError):
    """服务返回的内容无法解析或不符合约定。"""


_FRIENDLY_MESSAGES: dict[type[Exception], str] = {
    ConfigError: "配置有误：请检查 .env 与环境变量，参考 README 的配置变量说明。",
    RagTimeoutError: "请求超时：服务未在限定时间内响应，可稍后重试或调大 LLM_TIMEOUT_SECONDS。",
    RagConnectionError: "连接失败：无法访问服务地址，请检查网络与 LLM_BASE_URL。",
    RagAuthError: (
        "认证失败：API Key 无效或权限不足，请在本地 .env 中核对（密钥不要提供给任何人）。"
    ),
    RagRateLimitError: "触发限流：请求过于频繁，请稍后重试。",
    RagServerError: "服务端错误：模型服务暂时不可用，请稍后重试。",
    InvalidResponseError: "服务返回了无法解析的内容，请稍后重试或查看日志。",
    RagRequestError: "调用模型服务失败，请稍后重试。",
    RagError: "程序出现问题，请查看日志获取详情。",
}


def friendly_message(exc: Exception) -> str:
    """沿 MRO 返回面向用户的安全提示；ConfigError 附带明细（不含密钥值）。"""
    for exc_type in type(exc).__mro__:
        message = _FRIENDLY_MESSAGES.get(exc_type)
        if message is not None:
            if isinstance(exc, ConfigError):
                return f"{message}详细信息：{exc}"
            return message
    return "发生意外错误，请查看日志获取详情。"
