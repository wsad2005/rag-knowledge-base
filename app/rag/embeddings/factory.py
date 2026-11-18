"""Embedding 工厂：按配置创建具体实现。"""

from __future__ import annotations

from app.core.config import Settings
from app.core.errors import EmbeddingConfigurationError
from app.core.logging import get_logger
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.embeddings.hashing import HashingEmbedder

logger = get_logger(__name__)


def build_embedder(settings: Settings) -> EmbeddingProvider:
    """根据 ``EMBEDDING_PROVIDER`` 创建向量化实现。

    默认 ``hashing``：零依赖、零下载、离线可用，保证开箱即可演示。
    """
    provider = settings.embedding_provider

    if provider == "hashing":
        return HashingEmbedder(dimension=settings.embedding_dimension)

    if provider == "sentence-transformers":
        # 延迟 import：未安装可选依赖时不影响其他 provider
        from app.rag.embeddings.sentence_transformer import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        )

    if provider == "openai":
        from app.rag.embeddings.openai_compatible import OpenAICompatibleEmbedder

        if not settings.embedding_configured:
            raise EmbeddingConfigurationError(
                "远程 Embedding 配置不完整",
                detail={
                    "required": [
                        "EMBEDDING_API_BASE_URL",
                        "EMBEDDING_API_KEY",
                        "EMBEDDING_API_MODEL",
                    ]
                },
            )
        assert settings.embedding_api_base_url is not None
        assert settings.embedding_api_key is not None
        assert settings.embedding_api_model is not None
        return OpenAICompatibleEmbedder(
            base_url=settings.embedding_api_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_api_model,
            dimension=None,
            timeout=settings.embedding_api_timeout_seconds,
            batch_size=settings.embedding_batch_size,
        )

    raise EmbeddingConfigurationError(f"未知的 Embedding 提供方：{provider}")
