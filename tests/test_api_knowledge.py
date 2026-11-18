"""知识库 HTTP 接口测试 + 健康检查。"""

from __future__ import annotations

import pytest

from tests import samples


def upload(client, name: str, content: bytes):
    return client.post("/api/documents/upload", files={"file": (name, content, "text/plain")})


class TestHealth:
    def test_health_shape(self, client) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200
        payload = response.json()

        assert payload["status"] == "ok"
        assert payload["database"] == "ok"
        assert payload["vector_store"]["status"] == "ok"
        assert payload["vector_store"]["vectors"] == 0
        assert payload["embedding"]["provider"] == "hashing"
        assert payload["embedding"]["dimension"] == 512
        assert payload["llm"]["configured"] is False
        assert payload["llm"]["fallback"] == "extractive"
        assert payload["document_count"] == 0
        assert payload["chunking"]["chunk_size"] == 200

    def test_health_reflects_documents(self, client) -> None:
        upload(client, "制度.txt", samples.TXT_CONTENT.encode("utf-8"))
        payload = client.get("/api/health").json()
        assert payload["document_count"] == 1
        assert payload["vector_store"]["vectors"] > 0

    def test_openapi_available(self, client) -> None:
        schema = client.get("/openapi.json").json()
        assert schema["info"]["title"] == "RAG Knowledge Base API"
        assert "/api/documents/upload" in schema["paths"]
        assert "/api/knowledge/query" in schema["paths"]

    def test_index_page_served(self, client) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "RAG 知识库" in response.text

    def test_static_assets_served(self, client) -> None:
        assert client.get("/static/styles.css").status_code == 200
        assert client.get("/static/app.js").status_code == 200

    def test_unknown_route_uses_error_envelope(self, client) -> None:
        response = client.get("/api/nope")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "http_404"


class TestQuery:
    def test_empty_knowledge_base(self, client) -> None:
        response = client.post("/api/knowledge/query", json={"question": "报销流程是什么"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["answer_mode"] == "no_context"
        assert payload["sources"] == []
        assert "还没有可检索的文档" in payload["notice"]

    def test_extractive_answer_after_upload(self, client) -> None:
        upload(client, "制度.txt", samples.TXT_CONTENT.encode("utf-8"))
        response = client.post(
            "/api/knowledge/query",
            json={"question": "一线城市住宿标准是多少？", "top_k": 3},
        )
        assert response.status_code == 200

        payload = response.json()
        assert payload["answer_mode"] == "extractive"
        assert payload["sources"], "应当召回至少一个片段"
        assert "未配置大模型" in payload["answer"]
        assert payload["sources"][0]["document_name"] == "制度.txt"
        assert payload["debug"]["retrieved"] >= 1
        assert payload["debug"]["top_k"] == 3

    def test_question_validation(self, client) -> None:
        response = client.post("/api/knowledge/query", json={"question": ""})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"
        fields = response.json()["error"]["detail"]["fields"]
        assert any("question" in field["field"] for field in fields)

    def test_question_too_long(self, client) -> None:
        response = client.post("/api/knowledge/query", json={"question": "长" * 1001})
        assert response.status_code == 422

    def test_top_k_out_of_range(self, client) -> None:
        assert (
            client.post("/api/knowledge/query", json={"question": "x", "top_k": 0}).status_code
            == 422
        )
        assert (
            client.post("/api/knowledge/query", json={"question": "x", "top_k": 51}).status_code
            == 422
        )

    def test_top_k_over_server_limit(self, client) -> None:
        # 测试配置的 RETRIEVAL_MAX_TOP_K=10，schema 允许到 50，服务层应拒绝
        response = client.post("/api/knowledge/query", json={"question": "x", "top_k": 20})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"

    def test_document_ids_filter(self, client) -> None:
        upload(client, "制度.txt", samples.TXT_CONTENT.encode("utf-8"))
        response = client.post(
            "/api/knowledge/query",
            json={"question": "住宿标准", "document_ids": ["missing-id"]},
        )
        assert response.status_code == 200
        assert response.json()["answer_mode"] == "no_context"


class TestSearch:
    def test_search_hits(self, client) -> None:
        upload(client, "制度.txt", samples.TXT_CONTENT.encode("utf-8"))
        response = client.get("/api/knowledge/search", params={"q": "住宿标准", "top_k": 2})
        assert response.status_code == 200

        payload = response.json()
        assert payload["top_k"] == 2
        assert payload["total"] >= 1
        hit = payload["hits"][0]
        assert hit["document_name"] == "制度.txt"
        assert hit["chunk_index"] >= 0
        assert -1 <= hit["score"] <= 1
        assert payload["embedding_provider"] == "hashing"

    def test_search_requires_query(self, client) -> None:
        assert client.get("/api/knowledge/search").status_code == 422
        assert client.get("/api/knowledge/search", params={"q": ""}).status_code == 422

    def test_search_empty_result(self, client) -> None:
        payload = client.get("/api/knowledge/search", params={"q": "完全没有的词汇"}).json()
        assert payload["total"] == 0
        assert payload["hits"] == []

    @pytest.mark.parametrize("top_k", [0, 51, -1])
    def test_search_top_k_validation(self, client, top_k: int) -> None:
        response = client.get("/api/knowledge/search", params={"q": "x", "top_k": top_k})
        assert response.status_code == 422
