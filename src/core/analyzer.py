"""Document analyzer using unified-document-analysis framework."""
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from unified_document_analysis import UnifiedAnalyzer
from unified_document_analysis.exceptions import (
    FrameworkNotInstalledError,
    UnsupportedFileTypeError,
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


class DocumentAnalyzer:
    """Wrapper around unified-document-analysis for service use."""

    def __init__(self):
        """Initialize the unified analyzer."""
        self.analyzer = UnifiedAnalyzer()

    def analyze(
        self,
        file_path: str,
        framework_hint: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Analyze a document.

        Args:
            file_path: Path to file to analyze
            framework_hint: Optional framework to use (xml, docling, document, data)
            **kwargs: Additional arguments passed to analyzer

        Returns:
            Analysis result as dictionary

        Raises:
            FrameworkNotInstalledError: If required framework not installed
            UnsupportedFormatError: If file format not supported
        """
        try:
            result = self.analyzer.analyze(
                file_path,
                framework_hint=framework_hint,
                **kwargs,
            )

            # Convert result to dict for JSON serialization
            if hasattr(result, "to_dict"):
                dict_result = result.to_dict()
            elif is_dataclass(result):
                dict_result = asdict(result)
            else:
                dict_result = dict(result)

            # Recursively convert numpy types to native Python types
            return _convert_numpy_types(dict_result)

        except FrameworkNotInstalledError as e:
            raise FrameworkNotInstalledError(
                f"Required framework not installed: {e}. "
                f"Install with: pip install unified-document-analysis[all]"
            )
        except UnsupportedFileTypeError as e:
            raise UnsupportedFileTypeError(f"Unsupported file format: {e}")

    def batch_analyze(
        self,
        file_paths: List[str],
        framework_hint: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple documents.

        Args:
            file_paths: List of file paths to analyze
            framework_hint: Optional framework to use for all files
            **kwargs: Additional arguments passed to analyzer

        Returns:
            List of analysis results

        Raises:
            FrameworkNotInstalledError: If required framework not installed
            UnsupportedFormatError: If file format not supported
        """
        results = []
        errors = []

        for file_path in file_paths:
            try:
                result = self.analyze(file_path, framework_hint, **kwargs)
                results.append(
                    {
                        "file_path": file_path,
                        "status": "success",
                        "result": result,
                        "error": None,
                    }
                )
            except Exception as e:
                errors.append(
                    {
                        "file_path": file_path,
                        "status": "error",
                        "result": None,
                        "error": str(e),
                    }
                )
                results.append(errors[-1])

        return results

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """
        Get supported file formats by framework.

        Returns:
            Dictionary mapping framework names to supported extensions
        """
        return {
            "xml": [".xml"],
            "docling": [".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".tiff"],
            "document": [
                ".py",
                ".js",
                ".ts",
                ".tsx",
                ".jsx",
                ".java",
                ".cpp",
                ".c",
                ".h",
                ".cs",
                ".go",
                ".rs",
                ".rb",
                ".php",
                ".swift",
                ".kt",
                ".scala",
                ".r",
                ".sql",
                ".sh",
                ".bash",
                ".ps1",
                ".yaml",
                ".yml",
                ".json",
                ".toml",
                ".ini",
                ".cfg",
                ".conf",
                ".md",
                ".txt",
                ".rst",
                ".tex",
                ".log",
                ".csv",
                ".html",
                ".css",
                ".scss",
                ".less",
            ],
            "data": [".csv", ".parquet", ".db", ".sqlite", ".sqlite3"],
        }

    def detect_framework(self, file_path: str) -> tuple[str, float]:
        """
        Detect which framework should handle this file.

        Args:
            file_path: Path to file

        Returns:
            Tuple of (framework_name, confidence_score)
        """
        from unified_document_analysis.router import detect_framework

        return detect_framework(file_path)

    def get_available_frameworks(self) -> Dict[str, bool]:
        """
        Get which frameworks are currently installed.

        Returns:
            Dictionary mapping framework names to availability
        """
        available = self.analyzer.get_available_frameworks()
        # Convert list to dict with availability status
        all_frameworks = ["xml", "docling", "document", "data"]
        return {fw: fw in available for fw in all_frameworks}
