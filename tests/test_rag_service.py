"""RAG 服务层测试：正常生成、LLM 故障降级、无相关资料、检索参数校验、日志落库。"""

from __future__ import annotations

import io

import pytest
from sqlalchemy import func, select

from app.core.errors import InvalidRequestError, LLMUpstreamError
from app.models.document import Document
from app.models.query_log import QueryLog
from app.services.rag_service import RagService
from tests import samples


class FakeLLM:
    """可编程的假模型：记录收到的 messages，按需返回或抛错。"""

    model_name = "fake-llm-1"

    def __init__(self, answer: str = "一线城市住宿标准为每晚 600 元 [1]") -> None:
        self.answer = answer
        self.calls: list[list[dict[str, str]]] = []
        self.error: Exception | None = None

    def generate(self, messages):  # noqa: ANN001, ANN202
        self.calls.append(list(messages))
        if self.error:
            raise self.error
        return self.answer


def make_service(container, llm) -> RagService:  # noqa: ANN001
    return RagService(
        settings=container.settings,
        database=container.database,
        embedder=container.embedder,
        vector_store=container.vector_store,
        llm=llm,
    )


@pytest.fixture
def loaded(container):
    """先入库一份文档，保证检索有内容可召回。"""
    container.documents.ingest_upload(
        "报销制度.txt", io.BytesIO(samples.TXT_CONTENT.encode("utf-8"))
    )
    return container


class TestGeneration:
    def test_llm_answer_with_sources(self, loaded) -> None:
        llm = FakeLLM()
        service = make_service(loaded, llm)
        response = service.query("一线城市住宿标准是多少", top_k=3)

        assert response.answer_mode == "llm"
        assert response.answer == llm.answer
        assert response.notice is None
        assert response.sources
        assert response.sources[0].index == 1
        assert response.sources[0].document_name == "报销制度.txt"
        assert response.debug.llm_model == "fake-llm-1"
        assert response.debug.used_in_context == len(response.sources)

    def test_prompt_contains_context_and_question(self, loaded) -> None:
        llm = FakeLLM()
        make_service(loaded, llm).query("住宿标准", top_k=3)

        messages = llm.calls[0]
        assert messages[0]["role"] == "system"
        assert "只能依据" in messages[0]["content"]
        assert "住宿标准" in messages[1]["content"]
        assert "[1] 《报销制度.txt》" in messages[1]["content"]

    def test_query_log_is_written(self, loaded) -> None:
        service = make_service(loaded, FakeLLM())
        service.query("住宿标准", top_k=2)

        with loaded.database.session() as session:
            log = session.scalar(select(QueryLog).order_by(QueryLog.created_at.desc()))
        assert log is not None
        assert log.question == "住宿标准"
        assert log.answer_mode == "llm"
        assert log.top_k == 2
        assert log.sources_json and "报销制度" in log.sources_json


class TestFallback:
    def test_llm_failure_degrades_to_extractive(self, loaded) -> None:
        llm = FakeLLM()
        llm.error = LLMUpstreamError("大模型服务返回 HTTP 500")
        response = make_service(loaded, llm).query("住宿标准", top_k=3)

        assert response.answer_mode == "extractive"
        assert response.sources
        assert response.notice and "llm_upstream_error" in response.notice
        assert "未配置大模型" in response.answer  # 降级文案

    def test_no_llm_configured(self, loaded) -> None:
        from app.rag.llm import NullLLM

        response = make_service(loaded, NullLLM()).query("住宿标准", top_k=3)
        assert response.answer_mode == "extractive"
        assert response.notice and "未配置大模型" in response.notice

    def test_high_threshold_yields_no_context(self, loaded) -> None:
        strict_settings = loaded.settings.model_copy(
            update={"retrieval_score_threshold": 0.999}
        )
        service = RagService(
            settings=strict_settings,
            database=loaded.database,
            embedder=loaded.embedder,
            vector_store=loaded.vector_store,
            llm=FakeLLM(),
        )
        response = service.query("住宿标准")
        assert response.answer_mode == "no_context"
        assert response.sources == []
        assert response.notice


class TestEmptyKnowledgeBase:
    def test_query_without_documents(self, container) -> None:
        response = make_service(container, FakeLLM()).query("有人吗")

        assert response.answer_mode == "no_context"
        assert response.sources == []
        assert "还没有可检索的文档" in (response.notice or "")
        assert response.debug.retrieved == 0


class TestParameters:
    def test_top_k_over_limit(self, loaded) -> None:
        with pytest.raises(InvalidRequestError) as excinfo:
            make_service(loaded, FakeLLM()).query("住宿标准", top_k=999)
        assert excinfo.value.status_code == 400

    def test_top_k_default_from_settings(self, loaded) -> None:
        response = make_service(loaded, FakeLLM()).query("住宿标准")
        assert response.debug.top_k == loaded.settings.retrieval_top_k

    def test_document_ids_filter(self, loaded) -> None:
        service = make_service(loaded, FakeLLM())
        response = service.search("住宿标准", document_ids=["not-a-real-id"])
        assert response.hits == []

    def test_search_returns_similarity_desc(self, loaded) -> None:
        response = make_service(loaded, FakeLLM()).search("住宿标准 600 元", top_k=5)
        scores = [hit.score for hit in response.hits]
        assert scores == sorted(scores, reverse=True)
        assert all(-1.0 <= score <= 1.0 for score in scores)

    def test_search_does_not_call_llm(self, loaded) -> None:
        llm = FakeLLM()
        make_service(loaded, llm).search("住宿标准")
        assert llm.calls == []


class TestDocumentCount:
    def test_document_persisted(self, loaded) -> None:
        with loaded.database.session() as session:
            assert session.scalar(select(func.count()).select_from(Document)) == 1
