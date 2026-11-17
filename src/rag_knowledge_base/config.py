"""配置解析：从环境变量（可先经 .env 加载）读取模型服务配置。

约定：
- 真实密钥只来自本地 .env 或系统环境变量，绝不写进代码或仓库；
- mock 模式不需要任何真实配置；
- real 模式必须提供 LLM_BASE_URL、LLM_API_KEY、EMBEDDING_MODEL、GENERATION_MODEL；
- 日志与配置摘要都不包含明文密钥（见 mask_secret 与 Settings.describe）。
"""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import load_dotenv

from .errors import ConfigError

logger = logging.getLogger(__name__)

MOCK_MODE = "mock"
REAL_MODE = "real"
MODES = frozenset({MOCK_MODE, REAL_MODE})
DEFAULT_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class Settings:
    """模型服务配置快照。"""

    mode: str
    base_url: str | None
    api_key: str | None
    embedding_model: str | None
    generation_model: str | None
    timeout_seconds: int

    def describe(self) -> str:
        """返回可安全打印/记录的配置摘要（不含明文秘密）。"""
        return (
            f"mode={self.mode} embedding_model={self.embedding_model or '未设置'} "
            f"generation_model={self.generation_model or '未设置'} "
            f"api_key={mask_secret(self.api_key)} base_url={self.base_url or '未设置'} "
            f"timeout_seconds={self.timeout_seconds}"
        )


def mask_secret(value: str | None) -> str:
    """把秘密值转成可安全展示的掩码；空值显示“未设置”。"""
    if not value:
        return "未设置"
    if len(value) <= 8:
        return "***"
    return f"{value[:2]}***{value[-2:]}"


def _optional(raw: str | None) -> str | None:
    """去掉首尾空白；空字符串视为未设置。"""
    if raw is None:
        return None
    text = raw.strip()
    return text or None


def _parse_timeout(raw: str | None) -> int:
    """解析 LLM_TIMEOUT_SECONDS；空值用默认值，非法值抛 ConfigError。"""
    text = _optional(raw)
    if text is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = int(text)
    except ValueError as exc:
        raise ConfigError(f"LLM_TIMEOUT_SECONDS 必须是正整数秒，当前为 {text!r}") from exc
    if value <= 0:
        raise ConfigError(f"LLM_TIMEOUT_SECONDS 必须是正整数秒，当前为 {text!r}")
    return value


def _env_mapping() -> dict[str, str]:
    """生产读取来源：先加载当前目录的 .env（不覆盖已有环境变量），再读 os.environ。"""
    load_dotenv()
    return dict(os.environ)


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """从环境变量构建 Settings；测试可传入 environ，缺省读取真实环境。"""
    env = dict(environ) if environ is not None else _env_mapping()
    mode = (env.get("RAG_MODE") or MOCK_MODE).strip().lower()
    base_url = _optional(env.get("LLM_BASE_URL"))
    api_key = _optional(env.get("LLM_API_KEY"))
    embedding_model = _optional(env.get("EMBEDDING_MODEL"))
    generation_model = _optional(env.get("GENERATION_MODEL"))
    timeout_seconds = _parse_timeout(env.get("LLM_TIMEOUT_SECONDS"))
    if mode not in MODES:
        raise ConfigError(f"RAG_MODE 必须是 {MOCK_MODE} 或 {REAL_MODE}，当前为 {mode!r}")
    if mode == REAL_MODE:
        values = {
            "LLM_BASE_URL": base_url,
            "LLM_API_KEY": api_key,
            "EMBEDDING_MODEL": embedding_model,
            "GENERATION_MODEL": generation_model,
        }
        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise ConfigError(
                f"RAG_MODE={REAL_MODE} 缺少配置项：{', '.join(missing)}"
                "（密钥只写入本地 .env，绝不提交；参考 README 与 .env.example）"
            )
    settings = Settings(
        mode=mode,
        base_url=base_url,
        api_key=api_key,
        embedding_model=embedding_model,
        generation_model=generation_model,
        timeout_seconds=timeout_seconds,
    )
    logger.info("配置加载完成：%s", settings.describe())
    return settings
