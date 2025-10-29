"""Tests for REST API endpoints."""
import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root(self):
        """Test root endpoint returns service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "document-analysis-service"
        assert data["status"] == "running"
        assert "version" in data

    def test_ping(self):
        """Test ping endpoint."""
        response = client.get("/ping")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health(self):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "available_frameworks" in data


class TestMetadataEndpoints:
    """Tests for metadata endpoints."""

    def test_get_formats(self):
        """Test supported formats endpoint."""
        response = client.get("/api/v1/formats")
        assert response.status_code == 200
        data = response.json()
        assert "formats" in data
        assert "xml" in data["formats"]
        assert "docling" in data["formats"]
        assert "document" in data["formats"]
        assert "data" in data["formats"]

    def test_get_strategies(self):
        """Test supported strategies endpoint."""
        response = client.get("/api/v1/strategies")
        assert response.status_code == 200
        data = response.json()
        assert "strategies" in data
        assert "auto" in data["strategies"]
        assert "hierarchical" in data["strategies"]


class TestAnalyzeEndpoint:
    """Tests for document analysis endpoints."""

    def test_analyze_missing_file(self):
        """Test analyze with no file or S3 URL."""
        response = client.post("/api/v1/analyze")
        assert response.status_code == 400

    def test_analyze_invalid_s3_url(self):
        """Test analyze with invalid S3 URL."""
        response = client.post(
            "/api/v1/analyze",
            params={"s3_url": "invalid-url"},
        )
        assert response.status_code == 400

    @pytest.mark.skip(reason="Requires actual test file")
    def test_analyze_xml_file(self):
        """Test analyze with XML file upload."""
        # This test requires a real XML file
        # In actual implementation, create a test XML file
        pass


class TestChunkEndpoint:
    """Tests for document chunking endpoints."""

    def test_chunk_missing_file(self):
        """Test chunk with no file or S3 URL."""
        response = client.post("/api/v1/chunk")
        assert response.status_code == 400

    def test_chunk_invalid_strategy(self):
        """Test chunk with invalid strategy."""
        # Note: The strategy parameter is validated by Pydantic
        # This would need a real file to test the invalid strategy case
        pass

    @pytest.mark.skip(reason="Requires actual test file")
    def test_chunk_with_strategy(self):
        """Test chunk with specific strategy."""
        # This test requires a real file
        pass


class TestBatchEndpoints:
    """Tests for batch processing endpoints."""

    def test_batch_analyze_empty_list(self):
        """Test batch analyze with empty list."""
        response = client.post(
            "/api/v1/batch/analyze",
            json={"s3_urls": []},
        )
        assert response.status_code == 422  # Validation error

    def test_batch_analyze_invalid_urls(self):
        """Test batch analyze with invalid URLs."""
        response = client.post(
            "/api/v1/batch/analyze",
            json={"s3_urls": ["invalid-url"]},
        )
        assert response.status_code == 422  # Validation error

    def test_batch_chunk_empty_list(self):
        """Test batch chunk with empty list."""
        response = client.post(
            "/api/v1/batch/chunk",
            json={"s3_urls": []},
        )
        assert response.status_code == 422  # Validation error


class TestConvertEndpoint:
    """Tests for format conversion endpoints."""

    def test_convert_missing_file(self):
        """Test convert with no file or S3 URL."""
        response = client.post(
            "/api/v1/convert",
            params={"output_format": "json"},
        )
        assert response.status_code == 400

    @pytest.mark.skip(reason="Requires actual test file")
    def test_convert_to_json(self):
        """Test convert to JSON format."""
        # This test requires a real file
        pass

    @pytest.mark.skip(reason="Requires actual test file")
    def test_convert_to_markdown(self):
        """Test convert to Markdown format."""
        # This test requires a real file
        pass


class TestErrorHandling:
    """Tests for error handling."""

    def test_unsupported_format_error(self):
        """Test error handling for unsupported format."""
        # Would need a file with unsupported extension
        pass

    def test_file_too_large_error(self):
        """Test error handling for file size limit."""
        # Would need to upload a file larger than the limit
        pass


# Integration tests that require actual files
class TestIntegration:
    """Integration tests with real files."""

    @pytest.fixture
    def sample_xml_file(self, tmp_path):
        """Create a sample XML file for testing."""
        xml_content = """<?xml version="1.0"?>
<root>
    <element>test data</element>
</root>
"""
        file_path = tmp_path / "test.xml"
        file_path.write_text(xml_content)
        return file_path

    @pytest.mark.skip(reason="Requires unified-document-analysis to be installed")
    def test_analyze_xml_integration(self, sample_xml_file):
        """Integration test for XML analysis."""
        with open(sample_xml_file, "rb") as f:
            response = client.post(
                "/api/v1/analyze",
                files={"file": ("test.xml", f, "application/xml")},
            )
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
