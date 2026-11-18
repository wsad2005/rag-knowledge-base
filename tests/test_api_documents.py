"""文档接口测试：上传（四种格式）、查重、列表、详情、删除、各类错误码。"""

from __future__ import annotations

import pytest

from tests import samples


def upload(client, name: str, content: bytes, content_type: str = "application/octet-stream"):
    return client.post(
        "/api/documents/upload",
        files={"file": (name, content, content_type)},
    )


class TestUploadSuccess:
    def test_upload_txt(self, client) -> None:
        response = upload(client, "报销制度.txt", samples.TXT_CONTENT.encode("utf-8"))
        assert response.status_code == 201, response.text

        payload = response.json()
        document = payload["document"]
        assert document["filename"] == "报销制度.txt"
        assert document["status"] == "ready"
        assert document["source_type"] == "text"
        assert document["chunk_count"] > 0
        assert document["char_count"] > 100
        assert document["error_message"] is None

    def test_upload_markdown(self, client) -> None:
        response = upload(client, "部署手册.md", samples.MD_CONTENT.encode("utf-8"))
        assert response.status_code == 201
        assert response.json()["document"]["source_type"] == "markdown"

    def test_upload_docx(self, client) -> None:
        response = upload(client, "需求.docx", samples.docx_bytes())
        assert response.status_code == 201
        assert response.json()["document"]["source_type"] == "docx"

    def test_upload_pdf_keeps_page_count(self, client) -> None:
        response = upload(client, "overview.pdf", samples.pdf_bytes())
        assert response.status_code == 201
        document = response.json()["document"]
        assert document["source_type"] == "pdf"
        assert document["page_count"] == 2

    def test_all_formats_coexist(self, client) -> None:
        for name, content in [
            ("a.txt", samples.TXT_CONTENT.encode("utf-8")),
            ("b.md", samples.MD_CONTENT.encode("utf-8")),
            ("c.docx", samples.docx_bytes()),
            ("d.pdf", samples.pdf_bytes()),
        ]:
            assert upload(client, name, content).status_code == 201, name

        listing = client.get("/api/documents").json()
        assert listing["total"] == 4
        assert {item["source_type"] for item in listing["items"]} == {
            "text",
            "markdown",
            "docx",
            "pdf",
        }


class TestUploadRejections:
    def test_duplicate_content(self, client) -> None:
        content = samples.TXT_CONTENT.encode("utf-8")
        first = upload(client, "制度.txt", content)
        assert first.status_code == 201

        second = upload(client, "制度-副本.txt", content)
        assert second.status_code == 409
        error = second.json()["error"]
        assert error["code"] == "duplicate_document"
        assert error["detail"]["document_id"] == first.json()["document"]["id"]

        assert client.get("/api/documents").json()["total"] == 1

    def test_unsupported_extension(self, client) -> None:
        response = upload(client, "virus.exe", b"MZ....")
        assert response.status_code == 415
        assert response.json()["error"]["code"] == "unsupported_file_type"

    def test_missing_extension(self, client) -> None:
        response = upload(client, "README", b"hello")
        assert response.status_code == 415

    def test_empty_file(self, client) -> None:
        response = upload(client, "empty.txt", b"")
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "empty_file"

    def test_file_too_large(self, client) -> None:
        oversized = b"a" * (1024 * 1024 + 1024)  # 测试配置上限为 1MB
        response = upload(client, "big.txt", oversized)
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "file_too_large"

    def test_missing_file_field(self, client) -> None:
        response = client.post("/api/documents/upload")
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    def test_unparsable_pdf_is_recorded_as_failed(self, client) -> None:
        response = upload(client, "scan.pdf", samples.scanned_pdf_bytes())
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "document_parse_error"

        listing = client.get("/api/documents").json()
        failed = [item for item in listing["items"] if item["status"] == "failed"]
        assert len(failed) == 1
        assert "扫描" in failed[0]["error_message"]


class TestListingAndDetail:
    @pytest.fixture(autouse=True)
    def _seed(self, client) -> None:
        upload(client, "制度.txt", samples.TXT_CONTENT.encode("utf-8"))
        upload(client, "手册.md", samples.MD_CONTENT.encode("utf-8"))

    def test_pagination(self, client) -> None:
        page = client.get("/api/documents", params={"limit": 1, "skip": 0}).json()
        assert page["total"] == 2
        assert len(page["items"]) == 1
        assert page["limit"] == 1

        second = client.get("/api/documents", params={"limit": 1, "skip": 1}).json()
        assert second["items"][0]["id"] != page["items"][0]["id"]

    def test_filter_by_filename(self, client) -> None:
        result = client.get("/api/documents", params={"q": "制度"}).json()
        assert result["total"] == 1
        assert result["items"][0]["filename"] == "制度.txt"

    def test_filter_escapes_wildcards(self, client) -> None:
        # 用户输入 % 不应被当作通配符
        assert client.get("/api/documents", params={"q": "%"}).json()["total"] == 0

    def test_detail_contains_chunks(self, client) -> None:
        document_id = client.get("/api/documents").json()["items"][0]["id"]
        detail = client.get(f"/api/documents/{document_id}").json()
        assert detail["id"] == document_id
        assert len(detail["chunks"]) == detail["chunk_count"]
        assert detail["chunks"][0]["content"]

    def test_detail_not_found(self, client) -> None:
        response = client.get("/api/documents/does-not-exist")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "document_not_found"

    def test_list_validation(self, client) -> None:
        assert client.get("/api/documents", params={"limit": 0}).status_code == 422
        assert client.get("/api/documents", params={"limit": 1000}).status_code == 422
        assert client.get("/api/documents", params={"skip": -1}).status_code == 422


class TestDelete:
    def test_delete_removes_document_and_vectors(self, client) -> None:
        created = upload(client, "制度.txt", samples.TXT_CONTENT.encode("utf-8"))
        document = created.json()["document"]
        assert client.get("/api/health").json()["vector_store"]["vectors"] > 0

        response = client.delete(f"/api/documents/{document['id']}")
        assert response.status_code == 204
        assert client.get("/api/documents").json()["total"] == 0
        assert client.get("/api/health").json()["vector_store"]["vectors"] == 0

        again = client.delete(f"/api/documents/{document['id']}")
        assert again.status_code == 404

    def test_delete_keeps_other_documents(self, client) -> None:
        first = upload(client, "a.txt", samples.TXT_CONTENT.encode("utf-8")).json()["document"]
        second = upload(client, "b.md", samples.MD_CONTENT.encode("utf-8")).json()["document"]

        client.delete(f"/api/documents/{first['id']}")
        remaining = client.get("/api/documents").json()
        assert remaining["total"] == 1
        assert remaining["items"][0]["id"] == second["id"]
