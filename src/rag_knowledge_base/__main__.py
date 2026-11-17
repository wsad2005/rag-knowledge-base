"""`python -m rag_knowledge_base` 端到端演示：Mock 模式离线免费；real 模式走真实服务。"""

import sys
import tempfile
from pathlib import Path

from .config import load_settings
from .document_loader import TextLoader
from .errors import RagError, friendly_message
from .factory import create_embedder, create_generator
from .pipeline import RagPipeline
from .text_splitter import FixedSizeSplitter
from .vector_store import InMemoryVectorStore

SAMPLE_TEXT = """RAG（检索增强生成）是一种把检索与生成结合的技术。
RAG 先从知识库中找出与问题最相关的资料，再交给大语言模型生成回答。
向量检索是 RAG 的核心环节：文本被切成块并转为向量，查询时按余弦相似度排序。
本项目的流水线分为六步：文档加载、文本切分、向量化、向量存储、检索、生成。
"""


def main() -> None:
    """按当前配置跑通一次完整 RAG 流程；失败时输出友好提示并以非零码退出。"""
    try:
        settings = load_settings()
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_path = Path(temp_dir) / "sample.txt"
            sample_path.write_text(SAMPLE_TEXT, encoding="utf-8")
            pipeline = RagPipeline(
                loader=TextLoader(sample_path),
                splitter=FixedSizeSplitter(chunk_size=60, overlap=10),
                embedder=create_embedder(settings),
                store=InMemoryVectorStore(),
                generator=create_generator(settings),
            )
            chunks = pipeline.ingest()
            print(f"[demo] 配置：{settings.describe()}")
            print(f"[demo] 已入库 {chunks} 个文本块")
            question = "向量检索的核心环节是什么？"
            print(f"[demo] 问题：{question}")
            print(pipeline.ask(question, top_k=2))
    except RagError as exc:
        print(friendly_message(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
