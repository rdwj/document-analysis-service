"""Pydantic models for API requests and responses."""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class FileSource(str, Enum):
    """Source of the file to analyze."""

    UPLOAD = "upload"
    S3 = "s3"


class ChunkStrategy(str, Enum):
    """Chunking strategy to use."""

    AUTO = "auto"
    HIERARCHICAL = "hierarchical"
    SLIDING_WINDOW = "sliding_window"
    CONTENT_AWARE = "content_aware"
    SEMANTIC = "semantic"


class OutputFormat(str, Enum):
    """Output format for conversion."""

    JSON = "json"
    MARKDOWN = "markdown"
    TEXT = "text"


class AnalyzeRequest(BaseModel):
    """Request model for document analysis."""

    s3_url: Optional[str] = Field(None, description="S3 URL to analyze (if source=s3)")
    framework_hint: Optional[str] = Field(
        None, description="Framework to use (xml, docling, document, data)"
    )

    @validator("s3_url")
    def validate_s3_url(cls, v, values):
        """Validate S3 URL is provided when needed."""
        if v and not v.startswith(("s3://", "http://", "https://")):
            raise ValueError("s3_url must start with s3://, http://, or https://")
        return v


class ChunkRequest(BaseModel):
    """Request model for document chunking."""

    s3_url: Optional[str] = Field(None, description="S3 URL to chunk (if source=s3)")
    strategy: ChunkStrategy = Field(
        ChunkStrategy.AUTO, description="Chunking strategy to use"
    )
    framework_hint: Optional[str] = Field(
        None, description="Framework to use (xml, docling, document, data)"
    )
    max_chunk_size: Optional[int] = Field(
        None, description="Maximum chunk size in tokens"
    )
    overlap: Optional[int] = Field(None, description="Overlap between chunks in tokens")

    @validator("s3_url")
    def validate_s3_url(cls, v):
        """Validate S3 URL format."""
        if v and not v.startswith(("s3://", "http://", "https://")):
            raise ValueError("s3_url must start with s3://, http://, or https://")
        return v


class BatchAnalyzeRequest(BaseModel):
    """Request model for batch analysis."""

    s3_urls: List[str] = Field(..., description="List of S3 URLs to analyze")
    framework_hint: Optional[str] = Field(
        None, description="Framework to use for all files"
    )

    @validator("s3_urls")
    def validate_urls(cls, v):
        """Validate all URLs are valid."""
        if not v:
            raise ValueError("Must provide at least one URL")
        for url in v:
            if not url.startswith(("s3://", "http://", "https://")):
                raise ValueError(
                    f"Invalid URL format: {url}. Must start with s3://, http://, or https://"
                )
        return v


class BatchChunkRequest(BaseModel):
    """Request model for batch chunking."""

    s3_urls: List[str] = Field(..., description="List of S3 URLs to chunk")
    strategy: ChunkStrategy = Field(
        ChunkStrategy.AUTO, description="Chunking strategy to use"
    )
    framework_hint: Optional[str] = Field(
        None, description="Framework to use for all files"
    )

    @validator("s3_urls")
    def validate_urls(cls, v):
        """Validate all URLs are valid."""
        if not v:
            raise ValueError("Must provide at least one URL")
        for url in v:
            if not url.startswith(("s3://", "http://", "https://")):
                raise ValueError(f"Invalid URL: {url}")
        return v


class ConvertRequest(BaseModel):
    """Request model for format conversion."""

    s3_url: Optional[str] = Field(None, description="S3 URL to convert (if source=s3)")
    output_format: OutputFormat = Field(..., description="Desired output format")

    @validator("s3_url")
    def validate_s3_url(cls, v):
        """Validate S3 URL format."""
        if v and not v.startswith(("s3://", "http://", "https://")):
            raise ValueError("s3_url must start with s3://, http://, or https://")
        return v


class AnalysisResponse(BaseModel):
    """Response model for analysis."""

    file_path: str = Field(..., description="Path to analyzed file")
    result: Dict[str, Any] = Field(..., description="Analysis result")


class ChunkResponse(BaseModel):
    """Response model for chunking."""

    file_path: str = Field(..., description="Path to chunked file")
    chunks: List[Dict[str, Any]] = Field(..., description="List of chunks")
    chunk_count: int = Field(..., description="Total number of chunks")


class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis."""

    results: List[Dict[str, Any]] = Field(..., description="List of analysis results")
    total: int = Field(..., description="Total files processed")
    successful: int = Field(..., description="Number of successful analyses")
    failed: int = Field(..., description="Number of failed analyses")


class BatchChunkResponse(BaseModel):
    """Response model for batch chunking."""

    results: List[Dict[str, Any]] = Field(..., description="List of chunking results")
    total: int = Field(..., description="Total files processed")
    successful: int = Field(..., description="Number of successful chunks")
    failed: int = Field(..., description="Number of failed chunks")


class ConversionResponse(BaseModel):
    """Response model for format conversion."""

    file_path: str = Field(..., description="Path to converted file")
    output_format: str = Field(..., description="Output format used")
    content: str = Field(..., description="Converted content")


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    available_frameworks: Dict[str, bool] = Field(
        ..., description="Available analysis frameworks"
    )


class FormatsResponse(BaseModel):
    """Response model for supported formats."""

    formats: Dict[str, List[str]] = Field(
        ..., description="Supported formats by framework"
    )


class StrategiesResponse(BaseModel):
    """Response model for supported strategies."""

    strategies: List[str] = Field(..., description="Supported chunking strategies")
