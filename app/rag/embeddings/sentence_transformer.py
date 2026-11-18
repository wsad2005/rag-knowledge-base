"""本地 Sentence Transformers Embedding（可选依赖）。

``sentence-transformers`` 会引入 torch，属于重依赖，因此：

- 只在 ``EMBEDDING_PROVIDER=sentence-transformers`` 时才 import；
- 未安装或模型下载失败都转换成带安装指引的 ``EmbeddingConfigurationError``，
  而不是在服务启动时抛出难以理解的 ImportError；
- 默认不启用，README 也不声称模型已在本机验证（详见 README「已验证 / 未验证」）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.core.errors import EmbeddingConfigurationError, EmbeddingError
from app.core.logging import get_logger
from app.rag.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)

_INSTALL_HINT = "pip install -r requirements-local-embedding.txt"


class SentenceTransformerEmbedder(EmbeddingProvider):
    """基于 sentence-transformers 的本地向量化。"""

    provider_name = "sentence-transformers"

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        *,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # 未安装可选依赖
            raise EmbeddingConfigurationError(
                "未安装 sentence-transformers，无法使用本地模型 Embedding",
                detail={"install": _INSTALL_HINT, "provider": self.provider_name},
            ) from exc

        try:
            self._model: Any = SentenceTransformer(model_name, device=device)
        except Exception as exc:  # noqa: BLE001 - 模型缺失/网络失败/格式错误都归一化
            raise EmbeddingConfigurationError(
                f"本地 Embedding 模型加载失败：{model_name}",
                detail={
                    "reason": str(exc),
                    "hint": "首次使用需要联网下载权重；离线环境请改用 EMBEDDING_PROVIDER=hashing",
                },
            ) from exc

        dimension = self._model.get_sentence_embedding_dimension()
        if not dimension:
            raise EmbeddingConfigurationError(f"无法获取模型维度：{model_name}")
        self._dimension = int(dimension)
        logger.info("本地 Embedding 模型已加载：%s（维度 %s）", model_name, self._dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            matrix = self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(
                "本地模型向量化失败", detail={"model": self.model_name, "reason": str(exc)}
            ) from exc
        return [[float(value) for value in row] for row in matrix.tolist()]
