"""Embedding 接口层。

三个实现共用同一套契约：

- ``embed_documents`` / ``embed_query`` 返回 ``list[float]``，维度必须等于 ``dimension``；
- 任何维度不一致、空向量、非有限值都会抛 :class:`EmbeddingError`，而不是静默塞进向量库；
- ``signature`` 用于给向量集合命名，切换模型时不会把不同语义空间的向量混在一起。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.core.errors import EmbeddingError


class EmbeddingProvider(ABC):
    """向量化提供方抽象基类。"""

    #: 提供方标识，用于日志与调试信息
    provider_name: str = "base"
    #: 模型标识（本地模型名或远程模型名）
    model_name: str = "unknown"

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度。"""

    @property
    def signature(self) -> str:
        """模型指纹：provider + 模型名 + 维度。"""
        safe_model = self.model_name.replace("/", "_").replace(":", "_")
        return f"{self.provider_name}-{safe_model}-{self.dimension}"

    @abstractmethod
    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        """子类实现的批量向量化。"""

    # ------------------------------------------------------------------ 公共入口
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [text.strip() for text in texts]
        if not cleaned:
            return []
        if any(not text for text in cleaned):
            raise EmbeddingError("待向量化文本中存在空字符串")
        vectors = self._embed(cleaned)
        return self._validate(vectors, expected_count=len(cleaned))

    def embed_query(self, text: str) -> list[float]:
        query = text.strip()
        if not query:
            raise EmbeddingError("查询文本不能为空")
        vectors = self._embed([query])
        return self._validate(vectors, expected_count=1)[0]

    def _validate(self, vectors: list[list[float]], *, expected_count: int) -> list[list[float]]:
        if len(vectors) != expected_count:
            raise EmbeddingError(
                f"向量数量与文本数量不一致：期望 {expected_count}，实际 {len(vectors)}"
            )
        for vector in vectors:
            if len(vector) != self.dimension:
                raise EmbeddingError(
                    f"向量维度不一致：期望 {self.dimension}，实际 {len(vector)}",
                    detail={"model": self.model_name},
                )
            if not all(
                isinstance(value, (int, float)) and math.isfinite(value) for value in vector
            ):
                raise EmbeddingError("向量中包含非法数值（NaN/Inf）")
        return vectors
