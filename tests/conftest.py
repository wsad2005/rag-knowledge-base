"""Pytest 公共夹具。

原则：每个测试用独立的临时数据目录（SQLite + Chroma 都落在 tmp_path），
绝不读写仓库里的 ``data/``，避免测试互相污染或把测试数据提交进 Git。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.container import Container
from app.main import create_app


@pytest.fixture
def settings(tmp_path) -> Settings:
    """测试配置：小切分块、哈希 Embedding、不读本机 .env。"""
    return Settings(
        _env_file=None,
        environment="test",
        data_dir=tmp_path / "data",
        log_level="WARNING",
        chunk_size=200,
        chunk_overlap=40,
        min_chunk_chars=10,
        upload_max_mb=1,
        retrieval_top_k=3,
        retrieval_max_top_k=10,
        max_context_chars=800,
    )


@pytest.fixture
def container(settings: Settings):
    """直接操作服务层的容器（不起 HTTP）。"""
    instance = Container(settings)
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture
def client(settings: Settings):
    """带 lifespan 的 HTTP 客户端。"""
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
