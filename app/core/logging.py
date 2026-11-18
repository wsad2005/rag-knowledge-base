"""日志配置。

统一格式、统一级别，并为每条日志附上当前请求 ID（由 ``RequestContextMiddleware`` 注入），
方便把一次上传/问答产生的多行日志串起来排查。
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

#: 当前请求 ID，未处于请求上下文时为 "-"
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | req=%(request_id)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class RequestIdFilter(logging.Filter):
    """把 contextvar 中的请求 ID 注入 LogRecord。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    """配置根 logger（幂等：重复调用不会叠加 handler）。"""
    root = logging.getLogger()
    root.setLevel(level.upper())

    for handler in list(root.handlers):
        if getattr(handler, "_ragkb_handler", False):
            root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    handler.addFilter(RequestIdFilter())
    handler._ragkb_handler = True  # type: ignore[attr-defined]
    root.addHandler(handler)

    # 降低第三方库噪音
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
