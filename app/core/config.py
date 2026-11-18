"""应用配置。

所有可调参数集中在此，通过环境变量或项目根目录的 ``.env`` 覆盖。
业务代码只依赖 :class:`Settings`，不直接读 ``os.environ``，便于测试时整体替换。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: 项目根目录（app/core/config.py -> app/core -> app -> 项目根）
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """运行时配置。字段名即环境变量名（大小写不敏感）。"""

    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ 应用
    app_name: str = "RAG Knowledge Base"
    app_version: str = "1.0.0"
    environment: Literal["development", "test", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"
    api_prefix: str = "/api"

    # ------------------------------------------------------------------ 存储
    data_dir: Path = Path("data")
    database_url: str | None = None
    database_echo: bool = False

    # ------------------------------------------------------------------ 上传限制
    upload_max_mb: int = Field(default=20, ge=1, le=200)
    allowed_extensions: str = ".pdf,.docx,.txt,.md,.markdown"
    max_filename_length: int = Field(default=200, ge=16, le=512)

    # ------------------------------------------------------------------ 切分
    chunk_size: int = Field(default=500, ge=80, le=4000)
    chunk_overlap: int = Field(default=80, ge=0, le=2000)
    min_chunk_chars: int = Field(default=20, ge=0, le=500)

    # ------------------------------------------------------------------ Embedding
    embedding_provider: Literal["hashing", "sentence-transformers", "openai"] = "hashing"
    #: 哈希 Embedding 的向量维度；使用 sentence-transformers 时以模型自身维度为准
    embedding_dimension: int = Field(default=512, ge=32, le=4096)
    # 占位默认：仅 EMBEDDING_PROVIDER=sentence-transformers 时才会下载/调用该本地模型；
    # 默认 provider=hashing 不涉及任何模型下载或调用。
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    embedding_api_base_url: str | None = None
    embedding_api_key: SecretStr | None = None
    embedding_api_model: str | None = None
    embedding_api_timeout_seconds: float = Field(default=30.0, gt=0, le=300)

    # ------------------------------------------------------------------ 向量库
    vector_store_provider: Literal["chroma"] = "chroma"
    chroma_dir: Path | None = None
    chroma_collection: str = "knowledge_chunks"

    # ------------------------------------------------------------------ 检索
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    retrieval_max_top_k: int = Field(default=20, ge=1, le=100)
    retrieval_score_threshold: float = Field(default=0.0, ge=-1.0, le=1.0)
    max_context_chars: int = Field(default=4000, ge=200, le=32000)

    # ------------------------------------------------------------------ 大模型
    # 占位默认：LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 三项齐全才会真实调用大模型；
    # 缺任一项时 build_llm 返回 NullLLM，问答自动降级为 extractive，不发起任何网络请求。
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    llm_model: str = "glm-4-flash"
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, ge=64, le=8192)

    # ------------------------------------------------------------------ 校验
    @field_validator("data_dir", "chroma_dir")
    @classmethod
    def _resolve_relative_to_project(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        return value.upper()

    @model_validator(mode="after")
    def _check_chunking(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE")
        if self.retrieval_top_k > self.retrieval_max_top_k:
            raise ValueError("RETRIEVAL_TOP_K 不能大于 RETRIEVAL_MAX_TOP_K")
        return self

    # ------------------------------------------------------------------ 派生配置
    @property
    def upload_dir(self) -> Path:
        """原始上传文件目录（不对外暴露，文件名一律使用 UUID）。"""
        return self.data_dir / "uploads"

    @property
    def chroma_path(self) -> Path:
        return self.chroma_dir or (self.data_dir / "chroma")

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        db_file = (self.data_dir / "app.db").as_posix()
        return f"sqlite:///{db_file}"

    @property
    def is_sqlite(self) -> bool:
        return self.resolved_database_url.startswith("sqlite")

    @property
    def allowed_extension_set(self) -> frozenset[str]:
        return frozenset(
            ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
            for ext in self.allowed_extensions.split(",")
            if ext.strip()
        )

    @property
    def upload_max_bytes(self) -> int:
        return self.upload_max_mb * 1024 * 1024

    @property
    def llm_configured(self) -> bool:
        """是否具备真实调用大模型的条件（缺任一项则降级为仅检索模式）。"""
        return bool(self.llm_base_url and self.llm_api_key and self.llm_model)

    @property
    def embedding_configured(self) -> bool:
        """远程 Embedding 是否配置完整。哈希/本地模型模式恒为 True。"""
        if self.embedding_provider != "openai":
            return True
        return bool(
            self.embedding_api_base_url and self.embedding_api_key and self.embedding_api_model
        )

    def ensure_directories(self) -> None:
        """创建运行期必需的目录（幂等）。"""
        for path in (self.data_dir, self.upload_dir, self.chroma_path):
            path.mkdir(parents=True, exist_ok=True)

    def masked_summary(self) -> dict[str, object]:
        """用于启动日志的配置摘要，绝不包含密钥明文。"""
        return {
            "environment": self.environment,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "vector_store": self.vector_store_provider,
            "database": self.resolved_database_url.split("://", 1)[0] + "://…",
            "llm_configured": self.llm_configured,
            "llm_model": self.llm_model if self.llm_configured else None,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
