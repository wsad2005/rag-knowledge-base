"""端到端 RAG 流水线：加载 → 切分 → 向量化 → 入库 → 检索 → 生成。

把六个模块按 RAG 流程串起来；所有组件都通过 Protocol 边界注入，可整体替换。
"""

from dataclasses import dataclass

from .document_loader import DocumentLoader
from .embeddings import EmbeddingClient
from .generation import GenerationClient
from .retrieval import Retriever
from .text_splitter import TextSplitter
from .vector_store import VectorRecord, VectorStore


@dataclass
class RagPipeline:
    """可组合的 RAG 流水线。"""

    loader: DocumentLoader
    splitter: TextSplitter
    embedder: EmbeddingClient
    store: VectorStore
    generator: GenerationClient

    def __post_init__(self) -> None:
        self._retriever = Retriever(embedder=self.embedder, store=self.store)

    def ingest(self) -> int:
        """加载全部文档并入库，返回处理的块数。"""
        count = 0
        for document in self.loader.load():
            for chunk in self.splitter.split(document):
                vector = self.embedder.embed(chunk.text)
                self.store.add([VectorRecord(chunk=chunk, vector=vector)])
                count += 1
        return count

    def ask(self, question: str, top_k: int = 3) -> str:
        """检索相关资料并生成回答。"""
        hits = self._retriever.retrieve(question, top_k=top_k)
        return self.generator.generate(question, hits)
