"""零依赖的确定性哈希 Embedding（默认方案）。

它不是语义模型，而是 **特征哈希（hashing trick）**：把中文单字/二元组、英文词
映射到固定维度向量并做 L2 归一化。作用有两个：

1. 让项目**不下载任何模型、不联网**就能跑通“上传 → 切分 → 向量化 → 检索 → 问答”全链路；
2. 给全部单元测试提供可重复、跨进程稳定的向量（``hash()`` 带盐，不能用，这里用 blake2b）。

局限必须说清楚：它只做字面匹配，同义词（"报销" vs "费用申请"）召回不了，
真实语义检索请切换 ``EMBEDDING_PROVIDER=sentence-transformers`` 或远程 API。
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from app.rag.embeddings.base import EmbeddingProvider

_CJK_RUN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]+")
_WORD = re.compile(r"[a-zA-Z0-9_]+")


def _features(text: str) -> list[str]:
    """抽取特征：中文单字 + 二元组，英文/数字整词（小写）。"""
    features: list[str] = []

    for run in _CJK_RUN.findall(text):
        features.extend(run)  # 单字
        features.extend(run[i : i + 2] for i in range(len(run) - 1))  # 二元组

    features.extend(word.lower() for word in _WORD.findall(text))
    return features


class HashingEmbedder(EmbeddingProvider):
    """确定性特征哈希向量化。"""

    provider_name = "hashing"

    def __init__(self, dimension: int = 512, model_name: str = "char-ngram") -> None:
        if dimension < 32:
            raise ValueError("哈希向量维度至少为 32")
        self._dimension = dimension
        self.model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for feature in _features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            bucket = value % self._dimension
            # 用最高位决定符号，避免哈希冲突全部同向累加造成偏差
            sign = -1.0 if (value >> 63) & 1 else 1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0:  # 全是符号/标点时给一个稳定方向，避免零向量
            vector[0] = 1.0
            return vector
        return [component / norm for component in vector]
