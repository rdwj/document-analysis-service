"""File handling for uploads and S3 downloads."""
import os
import tempfile
from typing import Optional, Union
from pathlib import Path

from fastapi import UploadFile

from ..utils.s3 import S3Client


class FileHandler:
    """Handles file uploads and S3 downloads."""

    def __init__(
        self,
        max_upload_size_mb: int = 100,
        max_s3_size_mb: int = 1000,
        s3_endpoint: Optional[str] = None,
        s3_access_key: Optional[str] = None,
        s3_secret_key: Optional[str] = None,
    ):
        """
        Initialize file handler.

        Args:
            max_upload_size_mb: Maximum upload file size in MB
            max_s3_size_mb: Maximum S3 file size in MB
            s3_endpoint: S3 endpoint URL
            s3_access_key: S3 access key
            s3_secret_key: S3 secret key
        """
        self.max_upload_size_bytes = max_upload_size_mb * 1024 * 1024
        self.max_s3_size_bytes = max_s3_size_mb * 1024 * 1024
        self.s3_client = S3Client(
            endpoint_url=s3_endpoint,
            access_key=s3_access_key,
            secret_key=s3_secret_key,
        )
        self._temp_files: list[str] = []

    async def handle_upload(self, file: UploadFile) -> str:
        """
        Save uploaded file to temporary location.

        Args:
            file: FastAPI UploadFile object

        Returns:
            Path to temporary file

        Raises:
            ValueError: If file is too large
        """
        # Check file size
        content = await file.read()
        if len(content) > self.max_upload_size_bytes:
            raise ValueError(
                f"File size ({len(content)} bytes) exceeds maximum "
                f"({self.max_upload_size_bytes} bytes)"
            )

        # Create temporary file with original extension
        suffix = Path(file.filename or "").suffix or ".tmp"
        fd, temp_path = tempfile.mkstemp(suffix=suffix)

        try:
            os.write(fd, content)
        finally:
            os.close(fd)

        # Track temp file for cleanup
        self._temp_files.append(temp_path)

        return temp_path

    def handle_s3(self, s3_url: str) -> str:
        """
        Download file from S3 to temporary location.

        Args:
            s3_url: S3 URL to download

        Returns:
            Path to temporary file

        Raises:
            ValueError: If file is too large or doesn't exist
        """
        # Check file exists
        if not self.s3_client.check_file_exists(s3_url):
            raise ValueError(f"File not found in S3: {s3_url}")

        # Check file size
        file_size = self.s3_client.get_file_size(s3_url)
        if file_size > self.max_s3_size_bytes:
            raise ValueError(
                f"File size ({file_size} bytes) exceeds maximum "
                f"({self.max_s3_size_bytes} bytes)"
            )

        # Download to temp
        temp_path = self.s3_client.download_to_temp(s3_url)

        # Track temp file for cleanup
        self._temp_files.append(temp_path)

        return temp_path

    async def get_file_path(
        self,
        upload_file: Optional[UploadFile] = None,
        s3_url: Optional[str] = None,
    ) -> str:
        """
        Get file path from either upload or S3.

        Args:
            upload_file: Uploaded file
            s3_url: S3 URL

        Returns:
            Path to file (temporary if uploaded/downloaded)

        Raises:
            ValueError: If neither or both sources provided
        """
        if upload_file and s3_url:
            raise ValueError("Provide either upload_file or s3_url, not both")

        if not upload_file and not s3_url:
            raise ValueError("Must provide either upload_file or s3_url")

        if upload_file:
            return await self.handle_upload(upload_file)

        if s3_url:
            return self.handle_s3(s3_url)

        raise ValueError("Invalid file source")

    def cleanup(self):
        """Remove all temporary files created during this session."""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                # Best effort cleanup
                pass
        self._temp_files.clear()

    def __del__(self):
        """Cleanup on object destruction."""
        self.cleanup()
