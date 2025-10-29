# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **OpenShift-ready REST API and MCP server** for analyzing and chunking documents using the unified-document-analysis framework. The service provides a unified interface to four specialized document analysis frameworks (XML, Docling, Document, and Data), with smart routing, multi-format support, and optimized chunking for LLM/RAG workflows.

## Architecture

**Layered Architecture:**
```
REST API Layer (src/api/rest.py)
    ↓
Business Logic Layer (src/core/)
    ↓
Unified Document Analysis Framework (external dependency)
```

**Key Components:**
- `src/main.py`: FastAPI application with settings management and lifespan hooks
- `src/api/rest.py`: All REST endpoints (analyze, chunk, batch operations, conversions)
- `src/api/models.py`: Pydantic request/response models
- `src/core/analyzer.py`: Document analysis logic wrapping UnifiedAnalyzer
- `src/core/chunker.py`: Document chunking logic with multiple strategies
- `src/core/file_handler.py`: File upload and S3 download handling
- `src/mcp/`: MCP server implementation (FastMCP v2-based, separate subproject)

## Common Development Commands

### Environment Setup
```bash
# Create and activate venv
python3 -m venv venv
source venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Install all optional frameworks for full functionality
pip install unified-document-analysis[all]
```

### Running Locally
```bash
# Run with hot reload
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Access API documentation
open http://localhost:8000/docs
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py

# Run specific test class
pytest tests/test_api.py::TestHealthEndpoints

# Run specific test method
pytest tests/test_api.py::TestHealthEndpoints::test_health

# Skip integration tests (require actual files/frameworks)
pytest -m "not skip"
```

### Code Quality
```bash
# Format code (line length: 100 chars)
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Container Operations
```bash
# Build for OpenShift (ALWAYS use --platform on Mac)
podman build --platform linux/amd64 -t document-analysis-service:latest -f Containerfile . --no-cache

# Run locally with podman-compose
podman-compose up --build

# Run container manually
podman run -p 8000:8000 document-analysis-service:latest
```

### OpenShift Deployment
```bash
# Create namespace (if needed)
oc new-project document-analysis-dev

# Build using OpenShift BuildConfig (preferred)
oc new-build https://github.com/rdwj/document-analysis-service \
  --name=document-analysis-service \
  --strategy=docker
oc start-build document-analysis-service --follow

# Deploy to environment (use -n for namespace)
oc apply -k manifests/overlays/dev -n document-analysis-dev
oc apply -k manifests/overlays/prod -n document-analysis-prod

# Check deployment status
oc get pods -n document-analysis-dev
oc get route -n document-analysis-dev

# View logs
oc logs -f deployment/document-analysis-service -n document-analysis-dev

# Create S3 credentials secret
oc create secret generic s3-credentials \
  --from-literal=access-key-id=YOUR_KEY \
  --from-literal=secret-access-key=YOUR_SECRET \
  -n document-analysis-dev
```

## Configuration

### Environment Variables
All environment variables use the `APP_` prefix:

**Service Configuration:**
- `APP_SERVICE_NAME`: Service identifier (default: "document-analysis-service")
- `APP_SERVICE_VERSION`: Version (default: "1.0.0")
- `APP_DEBUG`: Enable debug logging (default: false)

**File Size Limits:**
- `APP_MAX_UPLOAD_SIZE_MB`: Max file upload size (default: 100MB)
- `APP_MAX_S3_SIZE_MB`: Max S3 file size (default: 1000MB)

**S3 Configuration (Optional):**
- `APP_S3_ENDPOINT`: S3 endpoint URL
- `AWS_ACCESS_KEY_ID`: AWS/S3 access key
- `AWS_SECRET_ACCESS_KEY`: AWS/S3 secret key

**CORS:**
- `APP_CORS_ORIGINS`: Allowed CORS origins (default: ["*"])

Configuration is managed via:
- Local: Environment variables
- OpenShift: ConfigMap (`manifests/base/configmap.yaml`) and Secrets

## Important Architectural Decisions

### 1. Unified Framework Approach
The service provides a single API interface that routes to four specialized analysis frameworks:
- **xml-analysis-framework**: 29+ specialized XML handlers (SCAP, RSS, Maven, Spring, etc.)
- **docling-analysis-framework**: PDFs, Office docs (DOCX, PPTX, XLSX), images
- **document-analysis-framework**: Code files, markdown, YAML, JSON, logs
- **data-analysis-framework**: CSV, Parquet, databases

Auto-detection handles routing, but you can provide framework hints if needed.

### 2. Offline-First Container Design (Air-Gapped/Disconnected Environment)
The Containerfile uses a multi-stage build that **pre-downloads ALL models and dependencies** during build time for **complete offline operation with NO network access at runtime**:

**Pre-cached Models:**
- **Docling Models** (layout, tableformer, code_formula, picture_classifier)
  - Location: `/opt/app-root/src/.cache/docling/`
  - Total size: ~500 MB (RT-DETR, TableFormer, EfficientNet-B0, etc.)

- **EasyOCR Models** (default OCR backend)
  - Location: `/opt/app-root/src/.cache/easyocr/`
  - Models: CRAFT detection (83.2 MB) + english_g2 recognition
  - Total size: ~100 MB

- **RapidOCR Models** (alternative OCR backend - smaller and faster)
  - Location: Embedded in rapidocr-onnxruntime package
  - Models: PP-OCRv5 detection + recognition + PP-OCRv4 classification
  - Total size: ~15 MB (significantly smaller than EasyOCR)

- **HuggingFace Models** (transformer-based models)
  - Location: `/opt/app-root/src/.cache/huggingface/`
  - Auto-cached during model downloads

**Critical Environment Variables for Offline Operation:**
```bash
# Model cache locations
HF_HOME=/opt/app-root/src/.cache/huggingface
TRANSFORMERS_CACHE=/opt/app-root/src/.cache/huggingface
HF_DATASETS_CACHE=/opt/app-root/src/.cache/huggingface
DOCLING_SERVE_ARTIFACTS_PATH=/opt/app-root/src/.cache/docling
EASYOCR_MODULE_PATH=/opt/app-root/src/.cache/easyocr

# Offline mode flags (CRITICAL - prevent network calls)
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

**OCR Backend Comparison:**
- **EasyOCR** (default): Better accuracy, supports 80+ languages, slower on CPU (~100 MB)
- **RapidOCR** (alternative): Faster, smaller models (~15 MB), optimized for offline, but less accurate

**Build Process:**
1. **Builder stage**: Downloads ALL models with network access
2. **Runtime stage**: Copies cached models, sets offline flags, NO network access
3. **Verification**: Cache directories are read-only (755 permissions) at runtime

This design ensures the container works in FIPS-enabled OpenShift clusters with **zero external network connectivity**.

### 3. Two Deployment Modes
- **REST API**: Standard HTTP endpoints for general integration
- **MCP Server**: AI agent integration via Model Context Protocol (see `src/mcp/`)

### 4. Security-First Deployment
- Non-root container (UID 1001)
- No privilege escalation
- All capabilities dropped
- Red Hat UBI 9 base image (FIPS-compatible)
- TLS termination at OpenShift Route level

### 5. Chunking Strategies
Five strategies available for LLM/RAG optimization:
- `auto`: Auto-selects based on document type
- `hierarchical`: Preserves document structure (sections → paragraphs)
- `sliding_window`: Fixed-size windows with configurable overlap
- `content_aware`: Respects semantic boundaries (paragraphs, sentences)
- `semantic`: Uses embeddings for coherence (future enhancement)

## API Endpoint Reference

**Health & Metadata:**
- `GET /`: Root health check
- `GET /ping`: Simple ping
- `GET /api/v1/health`: Detailed health with available frameworks
- `GET /api/v1/formats`: Supported formats by framework
- `GET /api/v1/strategies`: Available chunking strategies

**Document Operations:**
- `POST /api/v1/analyze`: Analyze single document (upload or S3)
- `POST /api/v1/chunk`: Chunk single document with strategy selection
- `POST /api/v1/batch/analyze`: Batch analyze multiple documents
- `POST /api/v1/batch/chunk`: Batch chunk multiple documents
- `POST /api/v1/convert`: Convert analysis results to JSON/Markdown/Text

**File Input Options:**
1. Upload: `curl -F "file=@document.xml"`
2. S3 URL: `curl "...?s3_url=s3://bucket/key"`

## Testing Strategy

**Test Organization:**
- `tests/test_api.py`: API endpoint integration tests using FastAPI TestClient
- `tests/test_core.py`: Core logic unit tests

**Some tests are marked with `@pytest.mark.skip`** because they require:
- Actual document files
- All frameworks installed (`pip install unified-document-analysis[all]`)
- S3 connectivity

To run integration tests, install all frameworks and remove skip markers or use `-m "not skip"`.

## Troubleshooting Common Issues

### Service Not Starting
```bash
# Check logs for detailed error messages
oc logs -f deployment/document-analysis-service -n document-analysis-dev

# Verify pod status
oc get pods -n document-analysis-dev
oc describe pod <pod-name> -n document-analysis-dev
```

### Framework Not Available
The `/api/v1/health` endpoint shows which frameworks are installed:
```bash
curl http://localhost:8000/api/v1/health
```

If a framework is missing, install it:
```bash
pip install unified-document-analysis[all]
```

### S3 Connection Issues
```bash
# Verify secret exists
oc get secret s3-credentials -n document-analysis-dev -o yaml

# Test S3 connectivity
aws s3 ls s3://your-bucket --endpoint-url $APP_S3_ENDPOINT
```

### Memory Issues
- Check pod resource limits in deployment manifest
- Increase `memory.limit` in overlay kustomization
- Or reduce `APP_MAX_UPLOAD_SIZE_MB` in ConfigMap

### Model Download Issues (Container Build)
If model downloads fail during container build:
1. Check network connectivity during build
2. Consider building in stages with checkpoints
3. Pre-download models to a cache volume

### Offline/Air-Gapped Deployment Issues

**Problem: Models not found at runtime**
```bash
# Verify models were cached during build
oc exec <pod-name> -n <namespace> -- ls -lh /opt/app-root/src/.cache/docling
oc exec <pod-name> -n <namespace> -- ls -lh /opt/app-root/src/.cache/easyocr
oc exec <pod-name> -n <namespace> -- ls -lh /opt/app-root/src/.cache/huggingface
```

**Problem: Service trying to download models at runtime**
Check environment variables are set correctly:
```bash
oc exec <pod-name> -n <namespace> -- env | grep -E "HF_|DOCLING_|EASYOCR|TRANSFORMERS"
```

Should show:
- `HF_DATASETS_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `DOCLING_SERVE_ARTIFACTS_PATH=/opt/app-root/src/.cache/docling`
- `EASYOCR_MODULE_PATH=/opt/app-root/src/.cache/easyocr`

**Problem: Permission denied accessing cache directories**
```bash
# Verify cache permissions (should be 755, owned by 1001:0)
oc exec <pod-name> -n <namespace> -- ls -la /opt/app-root/src/.cache
```

**Problem: Container image too large**
The offline image is larger due to pre-cached models (~700-800 MB total):
- Docling models: ~500 MB
- EasyOCR models: ~100 MB
- RapidOCR models: ~15 MB
- Python packages: ~100 MB

To reduce size:
- Remove EasyOCR and use only RapidOCR (saves ~100 MB)
- Remove picture_classifier or code_formula if not needed (saves ~100 MB each)
- Use multi-stage build optimization (already implemented)

**Switching from EasyOCR to RapidOCR:**
If you need smaller models and faster processing, configure Docling to use RapidOCR:
```python
from docling.pipeline.standard_pdf_pipeline import PdfPipelineOptions
from docling_core.types.doc import PictureClassificationMode
from docling.datamodel.pipeline_options import RapidOcrOptions

pipeline_options = PdfPipelineOptions()
pipeline_options.ocr_options = RapidOcrOptions()
```

**Testing Offline Operation Locally:**
```bash
# Build the container with --no-cache to ensure fresh build
podman build --platform linux/amd64 -f Containerfile -t document-analysis-service:offline . --no-cache

# Run without network access to verify offline operation
podman run --network=none -p 8000:8000 document-analysis-service:offline

# Test the service
curl http://localhost:8000/api/v1/health
```

## File Handling & Cleanup

**Important:** The service creates temporary files for uploads and S3 downloads. Cleanup happens automatically via:
- `FileHandler.handle_upload()`: Deletes temp file after processing
- `FileHandler.handle_s3()`: Deletes downloaded file after processing

Both methods use `try/finally` blocks to ensure cleanup even on errors.

## Future Enhancements

See `FUTURE_ENHANCEMENTS.md` for planned features:
- **VLM Pipeline Support**: Optional Visual Language Model pipeline for malformed/complex PDFs
- VLM models already included in container (GraniteDocling, SmolVlm, etc.)
- Will add `pipeline` parameter to allow switching between standard and VLM processing

## Dependencies & Version Requirements

**Python:** 3.11+ (3.12 supported)

**Core Dependencies:**
- FastAPI ≥0.100.0
- Uvicorn ≥0.23.0
- Pydantic ≥2.0.0
- unified-document-analysis[all] ≥1.0.0
- boto3 ≥1.28.0

**Dev Dependencies:**
- pytest ≥7.0
- pytest-asyncio ≥0.21.0
- pytest-cov ≥4.0
- black ≥23.0 (line length: 100)
- flake8 ≥6.0
- mypy ≥1.0

## OpenShift Manifest Structure

```
manifests/
├── base/
│   ├── deployment.yaml       # Pod template with health checks, resources
│   ├── service.yaml          # ClusterIP service
│   ├── route.yaml            # External access with TLS
│   ├── configmap.yaml        # Environment configuration
│   └── kustomization.yaml
└── overlays/
    ├── dev/                  # 1 replica, 1Gi memory
    ├── staging/              # (customize as needed)
    └── prod/                 # 3-10 replicas, 2-4Gi memory, autoscaling
```

**Kustomize Usage:**
- Base manifests contain common configuration
- Overlays patch base for environment-specific needs
- Use `oc apply -k manifests/overlays/<env> -n <namespace>`

## MCP Server Implementation

The `/src/mcp/` directory contains an MCP server implementation (FastMCP v2-based) that exposes the same functionality as the REST API but via the Model Context Protocol for AI agent integration.

**Available MCP Tools:**
- `analyze_document`: Single document analysis
- `chunk_document`: Single document chunking
- `analyze_and_chunk`: Combined operation
- `batch_analyze` / `batch_chunk`: Bulk operations
- `convert_format`: Format conversion
- `list_formats` / `list_strategies`: Metadata retrieval
- `check_health`: Health verification

See `src/mcp/MCP_TOOLS_SPEC.md` for complete specification.

## Code Style & Conventions

- **Line Length:** 100 characters (black formatter)
- **Docstrings:** All public functions have comprehensive docstrings
- **Type Hints:** Used throughout codebase
- **Error Handling:** Custom exceptions from unified-document-analysis framework
- **Numpy Arrays:** Converted to lists for JSON serialization
- **File Paths:** Use absolute paths in responses for clarity

## Adding New Features

When adding functionality:

1. **Core logic first**: Add to `src/core/` (analyzer, chunker, or new module)
2. **API models**: Add Pydantic models to `src/api/models.py`
3. **REST endpoint**: Add to `src/api/rest.py` with proper error handling
4. **Tests**: Add to `tests/` mirroring source structure
5. **MCP integration**: Update `src/mcp/` if feature should be agent-callable
6. **Documentation**: Update README.md and docstrings

## Resource Requirements

**Development:**
- Memory: 1Gi request
- CPU: 250m request

**Production:**
- Memory: 2Gi request / 4Gi limit
- CPU: 500m request / 2000m limit
- Replicas: 3-10 (autoscaling on CPU/memory)
- Autoscaling: 70% CPU or memory triggers scale-up

## Links & References

- **unified-document-analysis**: https://github.com/rdwj/unified-document-analysis
- **xml-analysis-framework**: https://github.com/rdwj/xml-analysis-framework
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Model Context Protocol**: https://modelcontextprotocol.io/
- **OpenShift Documentation**: https://docs.openshift.com/
- This project is designed to work on a FIPS mode OpenShift cluster in a disconnected environment.