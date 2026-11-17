"""向量存储接口：保存向量并按相似度检索最相关的块。"""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .text_splitter import Chunk


@dataclass(frozen=True)
class VectorRecord:
    """入库记录：一个块与其向量。"""

    chunk: Chunk
    vector: list[float]


@dataclass(frozen=True)
class SearchHit:
    """一条检索命中：块与相似度得分。"""

    chunk: Chunk
    score: float


@runtime_checkable
class VectorStore(Protocol):
    """向量存储接口。"""

    def add(self, records: Iterable[VectorRecord]) -> None:
        """把记录加入存储。"""
        ...

    def search(self, vector: list[float], top_k: int) -> list[SearchHit]:
        """按与 vector 的相似度降序返回最多 top_k 条命中。"""
        ...


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """余弦相似度；任一向量为零向量时返回 0.0。"""
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


class InMemoryVectorStore:
    """纯 Python 内存向量库：余弦相似度 Top-K（离线、零依赖）。

    下一阶段可替换为 FAISS/Chroma 等持久化实现，接口不变。
    """

    def __init__(self) -> None:
        self._records: list[VectorRecord] = []

    def add(self, records: Iterable[VectorRecord]) -> None:
        """把记录追加到内存列表。"""
        self._records.extend(records)

    def search(self, vector: list[float], top_k: int) -> list[SearchHit]:
        """返回余弦相似度最高的 top_k 条记录。"""
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        scored = [
            SearchHit(chunk=record.chunk, score=cosine_similarity(vector, record.vector))
            for record in self._records
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]
