"""RAG 问答服务：检索 → 组装上下文 → 生成 → 返回引用。

关键行为：

- **可降级**：没配大模型时返回 ``answer_mode=extractive``（原文片段 + 提示），
  而不是报错让整个功能不可用；
- **不伪造**：检索不到就明确返回 ``no_context``，绝不让模型凭空回答；
- **可追溯**：每个 ``sources[n]`` 都带文档名、页码、块号、相似度，答案里的 ``[n]`` 能对上；
- **可排障**：``debug`` 字段给出召回条数、进入 Prompt 的条数、上下文长度、耗时。
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence

from app.core.config import Settings
from app.core.errors import AppError, InvalidRequestError
from app.core.logging import get_logger
from app.database.session import Database
from app.models.query_log import QueryLog
from app.rag.embeddings.base import EmbeddingProvider
from app.rag.llm import LLMClient, NullLLM
from app.rag.prompt import (
    NO_CONTEXT_MESSAGE,
    ContextBundle,
    build_context,
    build_extractive_answer,
    build_messages,
)
from app.schemas.knowledge import (
    QueryResponse,
    RetrievalDebug,
    SearchResponse,
    Source,
)
from app.vector_store.base import VectorMatch, VectorStore

logger = get_logger(__name__)

_EMPTY_KB_MESSAGE = (
    "知识库中还没有可检索的文档。请先上传 PDF / DOCX / TXT / Markdown 文件，再提问。"
)
_NO_MATCH_MESSAGE = (
    "没有在知识库中检索到与问题相关的内容。可以尝试换用文档中出现过的具体名词，"
    "或先确认相关文档已上传且解析成功。"
)


class RagService:
    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        llm: LLMClient,
    ) -> None:
        self._settings = settings
        self._db = database
        self._embedder = embedder
        self._store = vector_store
        self._llm = llm

    # ------------------------------------------------------------------ 检索
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> SearchResponse:
        """只做检索，不调用大模型（供调试与前端“检索结果”面板使用）。"""
        resolved_k = self._resolve_top_k(top_k)
        started = time.perf_counter()
        matches = self._retrieve(query, top_k=resolved_k, document_ids=document_ids)
        latency_ms = int((time.perf_counter() - started) * 1000)
        hits = self._to_sources(matches)

        return SearchResponse(
            query=query,
            top_k=resolved_k,
            total=len(hits),
            hits=hits,
            embedding_provider=self._embedder.provider_name,
            embedding_model=self._embedder.model_name,
            embedding_dimension=self._safe_dimension(),
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------ 问答
    def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> QueryResponse:
        resolved_k = self._resolve_top_k(top_k)
        started = time.perf_counter()

        matches = self._retrieve(question, top_k=resolved_k, document_ids=document_ids)
        matches = [
            match
            for match in matches
            if match.score >= self._settings.retrieval_score_threshold
        ]
        bundle = build_context(self._to_sources(matches), self._settings.max_context_chars)

        answer_mode = "no_context"
        notice: str | None = None
        answer = NO_CONTEXT_MESSAGE

        if not matches:
            notice = _EMPTY_KB_MESSAGE if self._is_knowledge_base_empty() else _NO_MATCH_MESSAGE
        elif bundle.is_empty:
            notice = "检索到的片段内容为空，请检查文档解析结果。"
        else:
            answer, answer_mode, notice = self._answer(question, bundle)

        latency_ms = int((time.perf_counter() - started) * 1000)
        debug = RetrievalDebug(
            top_k=resolved_k,
            retrieved=len(matches),
            used_in_context=len(bundle.used_sources),
            context_chars=bundle.total_chars,
            embedding_provider=self._embedder.provider_name,
            embedding_model=self._embedder.model_name,
            embedding_dimension=self._safe_dimension(),
            llm_model=(self._llm.model_name or None) if answer_mode == "llm" else None,
            latency_ms=latency_ms,
        )
        response = QueryResponse(
            question=question,
            answer=answer,
            answer_mode=answer_mode,  # type: ignore[arg-type]
            notice=notice,
            sources=bundle.used_sources,
            debug=debug,
        )
        self._log_query(response, resolved_k, latency_ms)
        return response

    # ------------------------------------------------------------------ 内部
    def _answer(
        self, question: str, bundle: ContextBundle
    ) -> tuple[str, str, str | None]:
        """返回 ``(answer, answer_mode, notice)``。"""
        if isinstance(self._llm, NullLLM):
            return (
                build_extractive_answer(bundle.used_sources),
                "extractive",
                "当前未配置大模型（LLM_API_KEY 等），已降级为检索结果展示。",
            )

        messages = build_messages(question, bundle)
        try:
            answer = self._llm.generate(messages)
        except AppError as exc:
            # 大模型不可用不应让整个问答失败：降级为检索结果，并如实告知原因
            logger.warning(
                "大模型调用失败，降级为检索结果：code=%s message=%s", exc.code, exc.message
            )
            return (
                build_extractive_answer(bundle.used_sources),
                "extractive",
                f"大模型调用失败（{exc.code}），已降级为检索结果：{exc.message}",
            )
        return answer, "llm", None

    def _retrieve(
        self,
        text: str,
        *,
        top_k: int,
        document_ids: Sequence[str] | None,
    ) -> list[VectorMatch]:
        vector = self._embedder.embed_query(text)
        return self._store.search(vector, top_k=top_k, document_ids=document_ids)

    def _to_sources(self, matches: Sequence[VectorMatch]) -> list[Source]:
        return [
            Source(
                index=position,
                document_id=match.document_id,
                document_name=match.document_name or "未知文档",
                chunk_id=match.chunk_id,
                chunk_index=match.chunk_index,
                page_number=match.page_number,
                section=match.section,
                score=round(match.score, 4),
                snippet=match.content,
            )
            for position, match in enumerate(matches, start=1)
        ]

    def _resolve_top_k(self, top_k: int | None) -> int:
        resolved = top_k if top_k is not None else self._settings.retrieval_top_k
        if resolved < 1:
            raise InvalidRequestError("top_k 必须大于等于 1")
        if resolved > self._settings.retrieval_max_top_k:
            raise InvalidRequestError(
                f"top_k 不能超过 {self._settings.retrieval_max_top_k}",
                detail={"max_top_k": self._settings.retrieval_max_top_k},
            )
        return resolved

    def _safe_dimension(self) -> int:
        try:
            return self._embedder.dimension
        except AppError:  # 远程 Embedding 尚未调用过，维度未知
            return 0

    def _is_knowledge_base_empty(self) -> bool:
        try:
            return self._store.count() == 0
        except AppError:
            return False

    def _log_query(self, response: QueryResponse, top_k: int, latency_ms: int) -> None:
        """写问答日志；日志失败只告警，不影响用户拿到答案。"""
        sources_json = json.dumps(
            [
                {
                    "index": source.index,
                    "document_id": source.document_id,
                    "document_name": source.document_name,
                    "page_number": source.page_number,
                    "chunk_index": source.chunk_index,
                    "score": source.score,
                }
                for source in response.sources
            ],
            ensure_ascii=False,
        )
        try:
            with self._db.session() as session:
                session.add(
                    QueryLog(
                        question=response.question,
                        answer=response.answer,
                        answer_mode=response.answer_mode,
                        top_k=top_k,
                        source_count=len(response.sources),
                        latency_ms=latency_ms,
                        sources_json=sources_json,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("写入问答日志失败：%s", exc)
