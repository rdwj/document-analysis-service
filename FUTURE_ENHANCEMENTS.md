# Future Enhancements

## VLM Pipeline Support for Malformed PDFs

### Overview

Currently, the document analysis service uses the standard Docling pipeline for PDF processing. This works well for most documents but may struggle with malformed, complex, or poorly structured PDFs.

### Enhancement Proposal

Add an optional API route parameter to use the Visual Language Model (VLM) pipeline for cases where the standard pipeline encounters issues.

### Technical Details

**Current Implementation:**
- Standard Docling pipeline with layout detection, table extraction, and OCR
- Uses RapidOCR for text extraction
- Models pre-downloaded: layout, tableformer, code_formula, picture_classifier

**Proposed VLM Enhancement:**
- Add VLM models (already included in container):
  - GraniteDocling (258M parameters)
  - GraniteDocling MLX
  - SmolVlm
  - SmolDocling
  - SmolDocling MLX
  - Granite Vision

**VLM Pipeline Benefits:**
- Better handling of malformed PDFs with irregular layouts
- Improved understanding of visual document structure
- More robust text extraction from complex documents
- Enhanced table and figure recognition in difficult cases

### API Design Proposal

Add an optional query parameter to the `/api/v1/analyze` endpoint:

```python
@router.post("/analyze")
async def analyze_document(
    file: UploadFile,
    pipeline: str = Query("standard", enum=["standard", "vlm"]),
    vlm_model: str = Query("granite_docling", enum=["granite_docling", "smolvlm", "smoldocling"])
):
    """
    Analyze a document using Docling.

    Args:
        file: The document file to analyze
        pipeline: Processing pipeline to use (default: "standard")
            - "standard": Standard Docling pipeline (faster, most documents)
            - "vlm": Visual Language Model pipeline (slower, malformed/complex PDFs)
        vlm_model: VLM model to use when pipeline="vlm" (default: "granite_docling")
    """
```

### Example Usage

**Standard Pipeline (current):**
```bash
curl -X POST "https://service-url/api/v1/analyze" \
  -F "file=@document.pdf"
```

**VLM Pipeline (proposed):**
```bash
# Use VLM for malformed or complex PDF
curl -X POST "https://service-url/api/v1/analyze?pipeline=vlm&vlm_model=granite_docling" \
  -F "file=@malformed-document.pdf"
```

### Command Line Equivalent

From Docling CLI documentation:

```bash
# Standard pipeline
docling document.pdf --to html --to md

# VLM pipeline with GraniteDocling
docling document.pdf --to html --to md --pipeline vlm --vlm-model granite_docling

# With layout visualization
docling document.pdf --to html_split_page --show-layout --pipeline vlm --vlm-model granite_docling
```

### Implementation Considerations

1. **Performance**: VLM pipeline is slower than standard - add appropriate timeouts
2. **Resource Usage**: VLM models require more memory - may need pod resource adjustments
3. **Default Behavior**: Keep standard pipeline as default for backward compatibility
4. **Error Handling**: Consider fallback from standard to VLM on specific error types
5. **Model Selection**: Allow users to choose which VLM model (granite_docling, smolvlm, etc.)

### Related Files

- `src/routers/analysis.py` - Main analysis endpoint (lines 45-85)
- `Containerfile` - VLM models already pre-downloaded (line 41-43)
- Environment variable: `DOCLING_SERVE_ARTIFACTS_PATH` already configured

### Priority

Medium - Implement after confirming standard pipeline works reliably in air-gapped deployment.

### References

- Docling VLM Documentation: https://github.com/docling-project/docling
- GraniteDocling Model: https://huggingface.co/ibm-granite/granite-docling-258M
- Example arxiv paper used in docs: https://arxiv.org/pdf/2501.17887
