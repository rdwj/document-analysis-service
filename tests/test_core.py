"""Tests for core service logic."""
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from src.core.analyzer import DocumentAnalyzer
from src.core.chunker import DocumentChunker
from src.core.file_handler import FileHandler
from src.utils.s3 import S3Client


class TestS3Client:
    """Tests for S3Client."""

    def test_parse_s3_url_with_s3_scheme(self):
        """Test parsing S3 URL with s3:// scheme."""
        client = S3Client()
        bucket, key = client.parse_s3_url("s3://my-bucket/path/to/file.xml")
        assert bucket == "my-bucket"
        assert key == "path/to/file.xml"

    def test_parse_s3_url_with_https(self):
        """Test parsing S3 URL with https:// scheme."""
        client = S3Client()
        bucket, key = client.parse_s3_url("https://my-bucket.s3.amazonaws.com/file.xml")
        assert bucket == "my-bucket"
        assert key == "file.xml"

    def test_parse_s3_url_invalid_scheme(self):
        """Test parsing S3 URL with invalid scheme."""
        client = S3Client()
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client.parse_s3_url("ftp://my-bucket/file.xml")

    def test_parse_s3_url_invalid_format(self):
        """Test parsing malformed S3 URL."""
        client = S3Client()
        with pytest.raises(ValueError, match="Could not parse bucket/key"):
            client.parse_s3_url("s3://")


class TestFileHandler:
    """Tests for FileHandler."""

    @pytest.fixture
    def handler(self):
        """Create a FileHandler instance."""
        return FileHandler(max_upload_size_mb=1, max_s3_size_mb=10)

    @pytest.mark.asyncio
    async def test_handle_upload_success(self, handler):
        """Test successful file upload handling."""
        # Mock UploadFile
        mock_file = Mock()
        mock_file.filename = "test.xml"
        mock_file.read = Mock(return_value=b"<root>test</root>")

        file_path = await handler.handle_upload(mock_file)

        assert os.path.exists(file_path)
        assert file_path.endswith(".xml")
        assert file_path in handler._temp_files

        # Cleanup
        handler.cleanup()
        assert not os.path.exists(file_path)

    @pytest.mark.asyncio
    async def test_handle_upload_too_large(self, handler):
        """Test upload with file exceeding size limit."""
        # Create mock file with content larger than 1MB
        large_content = b"x" * (2 * 1024 * 1024)  # 2MB
        mock_file = Mock()
        mock_file.filename = "large.xml"
        mock_file.read = Mock(return_value=large_content)

        with pytest.raises(ValueError, match="exceeds maximum"):
            await handler.handle_upload(mock_file)

    @pytest.mark.asyncio
    async def test_get_file_path_no_source(self, handler):
        """Test get_file_path with no source provided."""
        with pytest.raises(ValueError, match="Must provide either"):
            await handler.get_file_path()

    @pytest.mark.asyncio
    async def test_get_file_path_both_sources(self, handler):
        """Test get_file_path with both sources provided."""
        mock_file = Mock()
        with pytest.raises(ValueError, match="not both"):
            await handler.get_file_path(upload_file=mock_file, s3_url="s3://bucket/key")

    def test_cleanup_removes_temp_files(self, handler):
        """Test cleanup removes all temporary files."""
        # Create some temp files
        fd1, path1 = tempfile.mkstemp()
        fd2, path2 = tempfile.mkstemp()
        os.close(fd1)
        os.close(fd2)

        handler._temp_files = [path1, path2]

        handler.cleanup()

        assert not os.path.exists(path1)
        assert not os.path.exists(path2)
        assert len(handler._temp_files) == 0


class TestDocumentAnalyzer:
    """Tests for DocumentAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a DocumentAnalyzer instance."""
        return DocumentAnalyzer()

    def test_get_supported_formats(self, analyzer):
        """Test getting supported formats."""
        formats = analyzer.get_supported_formats()
        assert "xml" in formats
        assert "docling" in formats
        assert "document" in formats
        assert "data" in formats
        assert ".xml" in formats["xml"]
        assert ".pdf" in formats["docling"]

    @pytest.mark.skip(reason="Requires unified-document-analysis installed")
    def test_analyze_with_real_file(self, analyzer, tmp_path):
        """Test analysis with a real XML file."""
        # Create test XML file
        xml_file = tmp_path / "test.xml"
        xml_file.write_text('<?xml version="1.0"?><root><item>test</item></root>')

        result = analyzer.analyze(str(xml_file))

        assert isinstance(result, dict)
        assert "document_type" in result or "framework" in result

    def test_get_available_frameworks(self, analyzer):
        """Test getting available frameworks."""
        frameworks = analyzer.get_available_frameworks()
        assert isinstance(frameworks, dict)
        # At minimum, we know these framework names should be present
        assert "xml" in frameworks
        assert "docling" in frameworks


class TestDocumentChunker:
    """Tests for DocumentChunker."""

    @pytest.fixture
    def chunker(self):
        """Create a DocumentChunker instance."""
        return DocumentChunker()

    def test_get_supported_strategies(self, chunker):
        """Test getting supported strategies."""
        strategies = chunker.get_supported_strategies()
        assert "auto" in strategies
        assert "hierarchical" in strategies
        assert "sliding_window" in strategies
        assert "content_aware" in strategies

    @pytest.mark.skip(reason="Requires unified-document-analysis installed")
    def test_chunk_with_real_file(self, chunker, tmp_path):
        """Test chunking with a real file."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document with some content.")

        chunks = chunker.chunk(str(test_file), strategy="auto")

        assert isinstance(chunks, list)
        assert len(chunks) > 0

    @pytest.mark.skip(reason="Requires unified-document-analysis installed")
    def test_batch_chunk(self, chunker, tmp_path):
        """Test batch chunking."""
        # Create test files
        file1 = tmp_path / "test1.txt"
        file2 = tmp_path / "test2.txt"
        file1.write_text("Test content 1")
        file2.write_text("Test content 2")

        results = chunker.batch_chunk([str(file1), str(file2)])

        assert isinstance(results, list)
        assert len(results) == 2
        assert all("status" in r for r in results)


class TestIntegrationScenarios:
    """Integration tests for common scenarios."""

    @pytest.mark.skip(reason="Requires unified-document-analysis installed")
    def test_analyze_and_chunk_workflow(self, tmp_path):
        """Test complete workflow: analyze then chunk."""
        # Create test file
        xml_file = tmp_path / "test.xml"
        xml_content = """<?xml version="1.0"?>
<document>
    <section>
        <title>Section 1</title>
        <content>This is the content of section 1.</content>
    </section>
    <section>
        <title>Section 2</title>
        <content>This is the content of section 2.</content>
    </section>
</document>
"""
        xml_file.write_text(xml_content)

        # Analyze
        analyzer = DocumentAnalyzer()
        analysis = analyzer.analyze(str(xml_file))
        assert analysis is not None

        # Chunk
        chunker = DocumentChunker()
        chunks = chunker.chunk(str(xml_file), strategy="hierarchical")
        assert len(chunks) > 0

    @pytest.mark.skip(reason="Requires S3 setup")
    def test_s3_download_and_analyze(self):
        """Test downloading from S3 and analyzing."""
        # This would require actual S3 credentials and test bucket
        pass
