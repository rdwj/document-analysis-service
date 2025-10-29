"""S3 utility for downloading files from object storage."""
import os
import tempfile
from typing import Optional
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError


class S3Client:
    """Client for interacting with S3-compatible object storage."""

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: str = "us-east-1",
    ):
        """
        Initialize S3 client.

        Args:
            endpoint_url: S3 endpoint URL (for S3-compatible services)
            access_key: AWS access key ID
            secret_key: AWS secret access key
            region: AWS region
        """
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=secret_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region,
        )

    def parse_s3_url(self, s3_url: str) -> tuple[str, str]:
        """
        Parse S3 URL into bucket and key.

        Args:
            s3_url: S3 URL (s3://bucket/key or https://bucket.s3.amazonaws.com/key)

        Returns:
            Tuple of (bucket, key)

        Raises:
            ValueError: If URL format is invalid
        """
        parsed = urlparse(s3_url)

        if parsed.scheme == "s3":
            # s3://bucket/key/path
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
        elif parsed.scheme in ("http", "https"):
            # https://bucket.s3.amazonaws.com/key/path or
            # https://s3.amazonaws.com/bucket/key/path
            if ".s3." in parsed.netloc or ".s3-" in parsed.netloc:
                bucket = parsed.netloc.split(".")[0]
                key = parsed.path.lstrip("/")
            else:
                raise ValueError(f"Invalid S3 URL format: {s3_url}")
        else:
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

        if not bucket or not key:
            raise ValueError(f"Could not parse bucket/key from URL: {s3_url}")

        return bucket, key

    def download_to_temp(self, s3_url: str) -> str:
        """
        Download file from S3 to temporary location.

        Args:
            s3_url: S3 URL to download

        Returns:
            Path to downloaded temporary file

        Raises:
            ClientError: If S3 download fails
            ValueError: If URL is invalid
        """
        bucket, key = self.parse_s3_url(s3_url)

        # Create temporary file with same extension
        _, ext = os.path.splitext(key)
        fd, temp_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)

        try:
            self.client.download_file(bucket, key, temp_path)
            return temp_path
        except ClientError as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def check_file_exists(self, s3_url: str) -> bool:
        """
        Check if file exists in S3.

        Args:
            s3_url: S3 URL to check

        Returns:
            True if file exists, False otherwise
        """
        try:
            bucket, key = self.parse_s3_url(s3_url)
            self.client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False

    def get_file_size(self, s3_url: str) -> int:
        """
        Get file size in bytes.

        Args:
            s3_url: S3 URL

        Returns:
            File size in bytes

        Raises:
            ClientError: If file doesn't exist or access denied
        """
        bucket, key = self.parse_s3_url(s3_url)
        response = self.client.head_object(Bucket=bucket, Key=key)
        return response["ContentLength"]
