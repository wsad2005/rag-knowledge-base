"""Embedding 测试：哈希实现的确定性、维度校验，以及远程 API 的 Mock 行为。

远程调用全部通过 ``httpx.MockTransport`` 拦截，不联网、不产生费用。
"""

from __future__ import annotations

import httpx
import pytest

from app.core.config import Settings
from app.core.errors import EmbeddingConfigurationError, EmbeddingError
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.embeddings.factory import build_embedder
from app.rag.embeddings.hashing import HashingEmbedder
from app.rag.embeddings.openai_compatible import OpenAICompatibleEmbedder


class TestHashingEmbedder:
    def test_dimension_and_shape(self) -> None:
        embedder = HashingEmbedder(dimension=128)
        vector = embedder.embed_query("报销流程是怎样的")
        assert embedder.dimension == 128
        assert len(vector) == 128
        assert all(isinstance(value, float) for value in vector)

    def test_l2_normalized(self) -> None:
        vector = HashingEmbedder(dimension=256).embed_query("差旅住宿标准")
        norm = sum(value * value for value in vector) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_deterministic_across_instances(self) -> None:
        first = HashingEmbedder().embed_query("同一个问题")
        second = HashingEmbedder().embed_query("同一个问题")
        assert first == second

    def test_different_text_different_vector(self) -> None:
        embedder = HashingEmbedder()
        assert embedder.embed_query("报销流程") != embedder.embed_query("员工入职")

    def test_similar_text_has_higher_similarity(self) -> None:
        embedder = HashingEmbedder()
        base = embedder.embed_query("一线城市住宿标准为每晚 600 元")
        close = embedder.embed_query("一线城市住宿标准")
        far = embedder.embed_query("员工入职需要提交学历证明")

        def cosine(a: list[float], b: list[float]) -> float:
            return sum(x * y for x, y in zip(a, b, strict=True))

        assert cosine(base, close) > cosine(base, far)

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(EmbeddingError):
            HashingEmbedder().embed_query("   ")

    def test_batch_embedding(self) -> None:
        vectors = HashingEmbedder().embed_documents(["第一段", "第二段", "第三段"])
        assert len(vectors) == 3
        assert all(len(vector) == 512 for vector in vectors)

    def test_empty_batch(self) -> None:
        assert HashingEmbedder().embed_documents([]) == []


class _BrokenProvider(EmbeddingProvider):
    """故意返回错误维度的提供方，验证基类的维度校验。"""

    provider_name = "broken"
    model_name = "broken-model"

    def __init__(self, dimension: int, actual: int) -> None:
        self._dimension = dimension
        self._actual = actual

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(self, texts):  # noqa: ANN001, ANN202
        return [[0.0] * self._actual for _ in texts]


class TestDimensionValidation:
    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(EmbeddingError) as excinfo:
            _BrokenProvider(dimension=8, actual=4).embed_query("abc")
        assert "维度不一致" in excinfo.value.message

    def test_nan_rejected(self) -> None:
        class _NanProvider(_BrokenProvider):
            def _embed(self, texts):  # noqa: ANN001, ANN202
                return [[float("nan")] * self._actual for _ in texts]

        with pytest.raises(EmbeddingError):
            _NanProvider(dimension=4, actual=4).embed_query("abc")

    def test_count_mismatch_raises(self) -> None:
        class _ShortProvider(_BrokenProvider):
            def _embed(self, texts):  # noqa: ANN001, ANN202
                return [[0.0] * 4]

        with pytest.raises(EmbeddingError):
            _ShortProvider(dimension=4, actual=4).embed_documents(["a", "b"])


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://fake-embedding.local/v1",
    )


class TestRemoteEmbedding:
    def test_success_learns_dimension(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path.endswith("/embeddings")
            payload = request.read().decode()
            assert "text-embedding-test" in payload
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [0.0, 1.0, 0.0]},
                        {"index": 0, "embedding": [1.0, 0.0, 0.0]},
                    ]
                },
            )

        embedder = OpenAICompatibleEmbedder(
            base_url="https://fake-embedding.local/v1",
            api_key="test-key",
            model="text-embedding-test",
            client=_mock_client(handler),
        )
        vectors = embedder.embed_documents(["第一条", "第二条"])
        # 服务端返回乱序时按 index 排序
        assert vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        assert embedder.dimension == 3

    def test_unauthorized_raises_configuration_error(self) -> None:
        embedder = OpenAICompatibleEmbedder(
            base_url="https://fake-embedding.local/v1",
            api_key="bad-key",
            model="m",
            client=_mock_client(lambda request: httpx.Response(401, json={"error": "bad key"})),
        )
        with pytest.raises(EmbeddingConfigurationError):
            embedder.embed_query("abc")

    def test_timeout_maps_to_embedding_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("timeout", request=request)

        embedder = OpenAICompatibleEmbedder(
            base_url="https://fake-embedding.local/v1",
            api_key="k",
            model="m",
            client=_mock_client(handler),
        )
        with pytest.raises(EmbeddingError):
            embedder.embed_query("abc")

    def test_malformed_response(self) -> None:
        embedder = OpenAICompatibleEmbedder(
            base_url="https://fake-embedding.local/v1",
            api_key="k",
            model="m",
            client=_mock_client(lambda request: httpx.Response(200, json={"unexpected": True})),
        )
        with pytest.raises(EmbeddingError):
            embedder.embed_query("abc")

    def test_missing_credentials(self) -> None:
        with pytest.raises(EmbeddingConfigurationError):
            OpenAICompatibleEmbedder(base_url="https://x.local/v1", api_key="", model="m")


class TestFactory:
    def test_default_is_hashing(self, tmp_path) -> None:
        settings = Settings(_env_file=None, environment="test", data_dir=tmp_path)
        assert isinstance(build_embedder(settings), HashingEmbedder)

    def test_remote_without_config_raises(self, tmp_path) -> None:
        settings = Settings(
            _env_file=None, environment="test", data_dir=tmp_path, embedding_provider="openai"
        )
        with pytest.raises(EmbeddingConfigurationError):
            build_embedder(settings)

    def test_sentence_transformer_without_package_raises_clear_error(self, tmp_path) -> None:
        settings = Settings(
            _env_file=None,
            environment="test",
            data_dir=tmp_path,
            embedding_provider="sentence-transformers",
        )
        # 本机未安装可选依赖时，应得到带安装指引的配置错误（而不是 ImportError）
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            with pytest.raises(EmbeddingConfigurationError) as excinfo:
                build_embedder(settings)
            assert "requirements-local-embedding" in str(excinfo.value.detail)
        else:  # pragma: no cover - 仅在装了可选依赖的机器上走到
            pytest.skip("本机已安装 sentence-transformers，跳过未安装分支的断言")
