"""检索与问答的请求/响应模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AnswerMode = Literal["llm", "extractive", "no_context"]


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="用户问题，1~1000 字",
        examples=["这份文档讲了哪些内容？"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="召回条数；留空使用服务端默认值，上限受 RETRIEVAL_MAX_TOP_K 约束",
    )
    document_ids: list[str] | None = Field(
        default=None,
        max_length=50,
        description="只在指定文档范围内检索，留空表示全库检索",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"question": "RAG 的核心流程是什么？", "top_k": 5, "document_ids": None}
        }
    )


class Source(BaseModel):
    """答案引用的来源片段。"""

    index: int = Field(description="引用编号，与答案中的 [n] 对应")
    document_id: str
    document_name: str
    chunk_id: str
    chunk_index: int
    page_number: int | None = None
    section: str | None = None
    score: float = Field(description="余弦相似度，越大越相关")
    snippet: str = Field(description="片段内容（截断展示）")


class RetrievalDebug(BaseModel):
    """检索调试信息：面试演示时可以解释“为什么召回这些片段”。"""

    top_k: int
    retrieved: int = Field(description="向量库返回条数")
    used_in_context: int = Field(description="实际进入 Prompt 的条数")
    context_chars: int
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    llm_model: str | None = None
    latency_ms: int


class QueryResponse(BaseModel):
    question: str
    answer: str
    answer_mode: AnswerMode = Field(
        description="llm=大模型生成；extractive=未配置模型时的检索摘要；no_context=库中无相关资料"
    )
    notice: str | None = Field(default=None, description="降级或异常时的提示文案")
    sources: list[Source] = Field(default_factory=list)
    debug: RetrievalDebug


class SearchResponse(BaseModel):
    query: str
    top_k: int
    total: int
    hits: list[Source]
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    latency_ms: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    database: str = Field(description="ok / error: ...")
    vector_store: dict[str, object]
    embedding: dict[str, object]
    llm: dict[str, object]
    chunking: dict[str, int]
    document_count: int
    uptime_seconds: float
