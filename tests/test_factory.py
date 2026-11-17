"""factory 模块测试。"""

from rag_knowledge_base.config import load_settings
from rag_knowledge_base.embeddings import (
    MockEmbeddingClient,
    OpenAICompatibleEmbeddingClient,
)
from rag_knowledge_base.factory import create_embedder, create_generator
from rag_knowledge_base.generation import (
    MockGenerationClient,
    OpenAICompatibleGenerationClient,
)

MOCK_ENV: dict[str, str] = {"RAG_MODE": "mock"}

REAL_ENV = {
    "RAG_MODE": "real",
    "LLM_BASE_URL": "https://api.example.invalid/v1",  # 测试用假地址
    "LLM_API_KEY": "dummy-key-abcdef123456",  # 测试用假密钥
    "EMBEDDING_MODEL": "dummy-embedding",  # 测试用假模型名
    "GENERATION_MODEL": "dummy-generation",  # 测试用假模型名
}


def test_mock_mode_creates_mock_clients() -> None:
    settings = load_settings(environ=dict(MOCK_ENV))
    assert isinstance(create_embedder(settings), MockEmbeddingClient)
    assert isinstance(create_generator(settings), MockGenerationClient)


def test_real_mode_creates_openai_compatible_clients() -> None:
    settings = load_settings(environ=dict(REAL_ENV))
    assert isinstance(create_embedder(settings), OpenAICompatibleEmbeddingClient)
    assert isinstance(create_generator(settings), OpenAICompatibleGenerationClient)
