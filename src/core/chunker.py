"""Document chunking using unified-document-analysis framework."""
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from unified_document_analysis import UnifiedAnalyzer
from unified_document_analysis.exceptions import (
    ChunkingError,
    FrameworkNotInstalledError,
)


def _convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types and dataclasses to native Python types."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif is_dataclass(obj) and not isinstance(obj, type):
        # Convert dataclass to dict and recursively process
        return _convert_numpy_types(asdict(obj))
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    return obj


class DocumentChunker:
    """Wrapper around unified-document-analysis chunking."""

    def __init__(self):
        """Initialize the unified analyzer."""
        self.analyzer = UnifiedAnalyzer()

    def chunk(
        self,
        file_path: str,
        strategy: str = "auto",
        framework_hint: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document.

        Args:
            file_path: Path to file to chunk
            strategy: Chunking strategy (auto, hierarchical, sliding_window, content_aware)
            framework_hint: Optional framework to use
            **kwargs: Additional arguments passed to chunker

        Returns:
            List of chunk dictionaries

        Raises:
            FrameworkNotInstalledError: If required framework not installed
            ChunkingError: If chunking fails
        """
        try:
            # Remove strategy and framework_hint from kwargs to avoid duplicate args
            kwargs_clean = {k: v for k, v in kwargs.items() if k not in ('strategy', 'framework_hint')}

            # Chunk document - frameworks handle their own analysis internally
            chunks = self.analyzer.chunk(
                file_path,
                strategy=strategy,
                framework_hint=framework_hint,
                **kwargs_clean,
            )

            # Convert chunks to dicts
            result = []
            for chunk in chunks:
                if hasattr(chunk, "to_dict"):
                    chunk_dict = chunk.to_dict()
                elif is_dataclass(chunk):
                    chunk_dict = asdict(chunk)
                else:
                    chunk_dict = dict(chunk)

                # Recursively convert numpy types to native Python types
                result.append(_convert_numpy_types(chunk_dict))

            return result

        except FrameworkNotInstalledError as e:
            raise FrameworkNotInstalledError(
                f"Required framework not installed: {e}. "
                f"Install with: pip install unified-document-analysis[all]"
            )
        except ChunkingError:
            # Re-raise as-is since it requires specific arguments
            raise
        except Exception as e:
            raise RuntimeError(f"Chunking failed: {e}")

    def batch_chunk(
        self,
        file_paths: List[str],
        strategy: str = "auto",
        framework_hint: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.

        Args:
            file_paths: List of file paths to chunk
            strategy: Chunking strategy
            framework_hint: Optional framework to use for all files
            **kwargs: Additional arguments passed to chunker

        Returns:
            List of results, each containing file_path, status, chunks, and error

        Raises:
            FrameworkNotInstalledError: If required framework not installed
        """
        results = []

        for file_path in file_paths:
            try:
                chunks = self.chunk(file_path, strategy, framework_hint, **kwargs)
                results.append(
                    {
                        "file_path": file_path,
                        "status": "success",
                        "chunks": chunks,
                        "chunk_count": len(chunks),
                        "error": None,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "file_path": file_path,
                        "status": "error",
                        "chunks": [],
                        "chunk_count": 0,
                        "error": str(e),
                    }
                )

        return results

    def get_supported_strategies(self) -> List[str]:
        """
        Get supported chunking strategies.

        Returns:
            List of strategy names
        """
        return [
            "auto",
            "hierarchical",
            "sliding_window",
            "content_aware",
            "semantic",
        ]

    def analyze_and_chunk(
        self,
        file_path: str,
        strategy: str = "auto",
        framework_hint: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Analyze and chunk a document in one call.

        Args:
            file_path: Path to file
            strategy: Chunking strategy
            framework_hint: Optional framework to use
            **kwargs: Additional arguments

        Returns:
            Dictionary with both analysis and chunks

        Raises:
            FrameworkNotInstalledError: If required framework not installed
        """
        # First analyze
        from .analyzer import DocumentAnalyzer

        analyzer = DocumentAnalyzer()
        analysis = analyzer.analyze(file_path, framework_hint, **kwargs)

        # Then chunk
        chunks = self.chunk(file_path, strategy, framework_hint, **kwargs)

        return {
            "file_path": file_path,
            "analysis": analysis,
            "chunks": chunks,
            "chunk_count": len(chunks),
        }
