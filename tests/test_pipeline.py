"""端到端流水线测试：真实文件 + 真实切分 + Mock 向量/存储/生成，全程离线。"""

from pathlib import Path

from rag_knowledge_base.document_loader import TextLoader
from rag_knowledge_base.embeddings import MockEmbeddingClient
from rag_knowledge_base.generation import MockGenerationClient
from rag_knowledge_base.pipeline import RagPipeline
from rag_knowledge_base.text_splitter import FixedSizeSplitter
from rag_knowledge_base.vector_store import InMemoryVectorStore

SAMPLE = (
    "RAG（检索增强生成）是一种把检索与生成结合的技术。\n"
    "RAG 先从知识库中找出与问题最相关的资料，再交给大语言模型生成回答。\n"
    "向量检索是 RAG 的核心环节：文本被切成块并转为向量，查询时按余弦相似度排序。\n"
    "本项目的流水线分为六步：文档加载、文本切分、向量化、向量存储、检索、生成。\n"
)


def build_pipeline(tmp_path: Path) -> RagPipeline:
    path = tmp_path / "sample.txt"
    path.write_text(SAMPLE, encoding="utf-8")
    return RagPipeline(
        loader=TextLoader(path),
        splitter=FixedSizeSplitter(chunk_size=40, overlap=5),
        embedder=MockEmbeddingClient(),
        store=InMemoryVectorStore(),
        generator=MockGenerationClient(),
    )


def test_ingest_returns_positive_chunk_count(tmp_path: Path) -> None:
    assert build_pipeline(tmp_path).ingest() > 0


def test_ask_returns_mock_answer_with_context(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)
    pipeline.ingest()
    answer = pipeline.ask("余弦相似度", top_k=1)
    assert answer.startswith("[mock]")
    assert "1 段资料" in answer


def test_retrieval_picks_chunk_containing_query_terms(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)
    pipeline.ingest()
    answer = pipeline.ask("余弦相似度", top_k=1)
    assert "余弦相似度" in answer  # 命中包含查询词的块


def test_ask_is_deterministic(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)
    pipeline.ingest()
    assert pipeline.ask("RAG 是什么", top_k=2) == pipeline.ask("RAG 是什么", top_k=2)
