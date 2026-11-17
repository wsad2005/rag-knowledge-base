"""组件工厂：按 Settings.mode 创建 Mock 或真实客户端。"""

from .config import MOCK_MODE, REAL_MODE, Settings
from .embeddings import EmbeddingClient, MockEmbeddingClient, OpenAICompatibleEmbeddingClient
from .errors import ConfigError
from .generation import GenerationClient, MockGenerationClient, OpenAICompatibleGenerationClient

__all__ = ["create_embedder", "create_generator"]


def create_embedder(settings: Settings) -> EmbeddingClient:
    """按配置创建向量化客户端。"""
    if settings.mode == MOCK_MODE:
        return MockEmbeddingClient()
    if settings.mode == REAL_MODE:
        return OpenAICompatibleEmbeddingClient(settings)
    raise ConfigError(f"未知的 RAG_MODE：{settings.mode!r}")


def create_generator(settings: Settings) -> GenerationClient:
    """按配置创建生成客户端。"""
    if settings.mode == MOCK_MODE:
        return MockGenerationClient()
    if settings.mode == REAL_MODE:
        return OpenAICompatibleGenerationClient(settings)
    raise ConfigError(f"未知的 RAG_MODE：{settings.mode!r}")
