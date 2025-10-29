"""REST API endpoints for document analysis service."""
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from ..core.analyzer import DocumentAnalyzer
from ..core.chunker import DocumentChunker
from ..core.file_handler import FileHandler
from .models import (
    AnalysisResponse,
    AnalyzeRequest,
    BatchAnalysisResponse,
    BatchAnalyzeRequest,
    BatchChunkRequest,
    BatchChunkResponse,
    ChunkRequest,
    ChunkResponse,
    ConversionResponse,
    ConvertRequest,
    ErrorResponse,
    FormatsResponse,
    HealthResponse,
    StrategiesResponse,
)

router = APIRouter()

# Initialize service components
file_handler = FileHandler()
analyzer = DocumentAnalyzer()
chunker = DocumentChunker()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns service status and available frameworks.
    """
    return HealthResponse(
        status="healthy",
        available_frameworks=analyzer.get_available_frameworks(),
    )


@router.get("/formats", response_model=FormatsResponse)
async def get_formats():
    """
    Get supported file formats by framework.

    Returns a mapping of framework names to supported file extensions.
    """
    return FormatsResponse(formats=analyzer.get_supported_formats())


@router.get("/strategies", response_model=StrategiesResponse)
async def get_strategies():
    """
    Get supported chunking strategies.

    Returns a list of available chunking strategies.
    """
    return StrategiesResponse(strategies=chunker.get_supported_strategies())


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_document(
    file: Optional[UploadFile] = File(None),
    s3_url: Optional[str] = None,
    framework_hint: Optional[str] = None,
):
    """
    Analyze a document.

    Accepts either an uploaded file or an S3 URL.
    Auto-detects document type and applies appropriate analysis framework.

    Args:
        file: Uploaded file (multipart/form-data)
        s3_url: S3 URL to analyze
        framework_hint: Optional framework to use (xml, docling, document, data)

    Returns:
        Analysis result with document metadata, schema, and specialized analysis
    """
    try:
        # Get file path
        file_path = await file_handler.get_file_path(
            upload_file=file, s3_url=s3_url
        )

        # Analyze
        result = analyzer.analyze(file_path, framework_hint=framework_hint)

        return AnalysisResponse(file_path=file_path, result=result)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )
    finally:
        file_handler.cleanup()


@router.post("/chunk", response_model=ChunkResponse)
async def chunk_document(
    file: Optional[UploadFile] = File(None),
    s3_url: Optional[str] = None,
    strategy: str = "auto",
    framework_hint: Optional[str] = None,
    max_chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
):
    """
    Chunk a document for LLM/RAG processing.

    Accepts either an uploaded file or an S3 URL.
    Supports multiple chunking strategies optimized for different document types.

    Args:
        file: Uploaded file (multipart/form-data)
        s3_url: S3 URL to chunk
        strategy: Chunking strategy (auto, hierarchical, sliding_window, content_aware, semantic)
        framework_hint: Optional framework to use
        max_chunk_size: Maximum chunk size in tokens
        overlap: Overlap between chunks in tokens

    Returns:
        List of document chunks with metadata
    """
    try:
        # Get file path
        file_path = await file_handler.get_file_path(
            upload_file=file, s3_url=s3_url
        )

        # Build kwargs
        kwargs = {}
        if max_chunk_size:
            kwargs["max_chunk_size"] = max_chunk_size
        if overlap:
            kwargs["overlap"] = overlap

        # Chunk
        chunks = chunker.chunk(
            file_path, strategy=strategy, framework_hint=framework_hint, **kwargs
        )

        return ChunkResponse(
            file_path=file_path, chunks=chunks, chunk_count=len(chunks)
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chunking failed: {str(e)}",
        )
    finally:
        file_handler.cleanup()


@router.post("/batch/analyze", response_model=BatchAnalysisResponse)
async def batch_analyze(request: BatchAnalyzeRequest):
    """
    Analyze multiple documents in batch.

    Accepts a list of S3 URLs and processes them all.
    Returns results for all files, including any errors.

    Args:
        request: Batch analysis request with list of S3 URLs

    Returns:
        Aggregated results with success/failure counts
    """
    try:
        # Download all files
        file_paths = []
        for s3_url in request.s3_urls:
            try:
                path = file_handler.handle_s3(s3_url)
                file_paths.append(path)
            except Exception as e:
                # Continue with other files
                file_paths.append(None)

        # Analyze all files
        results = analyzer.batch_analyze(
            [p for p in file_paths if p], framework_hint=request.framework_hint
        )

        # Count successes/failures
        successful = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - successful

        return BatchAnalysisResponse(
            results=results,
            total=len(results),
            successful=successful,
            failed=failed,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch analysis failed: {str(e)}",
        )
    finally:
        file_handler.cleanup()


@router.post("/batch/chunk", response_model=BatchChunkResponse)
async def batch_chunk(request: BatchChunkRequest):
    """
    Chunk multiple documents in batch.

    Accepts a list of S3 URLs and chunks them all using the specified strategy.
    Returns results for all files, including any errors.

    Args:
        request: Batch chunk request with list of S3 URLs and strategy

    Returns:
        Aggregated results with success/failure counts
    """
    try:
        # Download all files
        file_paths = []
        for s3_url in request.s3_urls:
            try:
                path = file_handler.handle_s3(s3_url)
                file_paths.append(path)
            except Exception as e:
                file_paths.append(None)

        # Chunk all files
        results = chunker.batch_chunk(
            [p for p in file_paths if p],
            strategy=request.strategy.value,
            framework_hint=request.framework_hint,
        )

        # Count successes/failures
        successful = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - successful

        return BatchChunkResponse(
            results=results,
            total=len(results),
            successful=successful,
            failed=failed,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch chunking failed: {str(e)}",
        )
    finally:
        file_handler.cleanup()


@router.post("/convert", response_model=ConversionResponse)
async def convert_format(
    file: Optional[UploadFile] = File(None),
    s3_url: Optional[str] = None,
    output_format: str = "json",
):
    """
    Convert document to different format.

    Accepts either an uploaded file or an S3 URL.
    Extracts and converts document content to the specified format.

    Args:
        file: Uploaded file (multipart/form-data)
        s3_url: S3 URL to convert
        output_format: Output format (json, markdown, text)

    Returns:
        Converted document content
    """
    try:
        # Get file path
        file_path = await file_handler.get_file_path(
            upload_file=file, s3_url=s3_url
        )

        # Analyze to get structured data
        result = analyzer.analyze(file_path)

        # Convert based on output format
        if output_format == "json":
            import json

            content = json.dumps(result, indent=2)
        elif output_format == "markdown":
            # Basic markdown conversion
            content = f"# Document Analysis\n\n"
            content += f"**Type:** {result.get('document_type', {}).get('type_name', 'Unknown')}\n\n"
            if "schema" in result:
                content += "## Schema\n\n"
                content += f"```json\n{json.dumps(result['schema'], indent=2)}\n```\n"
        elif output_format == "text":
            # Plain text conversion
            content = f"Document Type: {result.get('document_type', {}).get('type_name', 'Unknown')}\n"
            content += f"Framework: {result.get('framework', 'Unknown')}\n"
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        return ConversionResponse(
            file_path=file_path, output_format=output_format, content=content
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Conversion failed: {str(e)}",
        )
    finally:
        file_handler.cleanup()
