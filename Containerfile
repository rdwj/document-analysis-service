# Multi-stage build for document-analysis-service
# Uses Red Hat UBI 9 with Python 3.12 as base image
# Pre-downloads ALL models and dependencies for air-gapped/offline deployment
# CRITICAL: This container MUST work in disconnected environments with NO external network access

# Stage 1: Builder
FROM registry.access.redhat.com/ubi9/python-312:latest AS builder

USER root

# Install system dependencies (including OpenGL for OpenCV/OCR)
RUN dnf install -y \
    gcc \
    python3-devel \
    file-libs \
    mesa-libGL \
    && dnf clean all

# Create cache directories as root so user 1001 can write to them
RUN mkdir -p /opt/app-root/src/.cache/huggingface \
             /opt/app-root/src/.cache/docling \
             /opt/app-root/src/.cache/easyocr && \
    chown -R 1001:0 /opt/app-root/src/.cache

USER 1001

# Set working directory
WORKDIR /opt/app-root/src

# Set cache locations for all frameworks (build stage - allow downloads)
ENV HF_HOME=/opt/app-root/src/.cache/huggingface \
    TRANSFORMERS_CACHE=/opt/app-root/src/.cache/huggingface \
    HF_DATASETS_CACHE=/opt/app-root/src/.cache/huggingface \
    EASYOCR_MODULE_PATH=/opt/app-root/src/.cache/easyocr

# Copy all files needed for build
COPY --chown=1001:0 pyproject.toml README.md ./
COPY --chown=1001:0 src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

# Pre-download Docling models for standard pipeline for offline usage
# Models: layout (RT-DETR), tableformer, code_formula, picture_classifier
# Note: RapidOCR is specified but models must be cached separately (see below)
RUN docling-tools models download layout tableformer code_formula picture_classifier \
    -o /opt/app-root/src/.cache/docling && \
    echo "Docling standard pipeline models pre-downloaded successfully"

# Pre-cache EasyOCR models (default OCR backend for Docling)
# This triggers model download during build so they're available offline
# Models: CRAFT detection (83.2 MB) + english_g2 recognition
RUN python3 -c "import easyocr; reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='/opt/app-root/src/.cache/easyocr', download_enabled=True); print('EasyOCR models cached successfully')"

# Pre-cache RapidOCR models (alternative OCR backend - smaller and faster)
# Models: PP-OCRv5 detection + recognition + PP-OCRv4 classification (~15 MB total)
RUN python3 -c "from rapidocr_onnxruntime import RapidOCR; ocr = RapidOCR(); print('RapidOCR models cached successfully')"

# Stage 2: Runtime
FROM registry.access.redhat.com/ubi9/python-312:latest

USER root

# Install runtime dependencies (including OpenGL for OpenCV/OCR)
RUN dnf install -y \
    file-libs \
    mesa-libGL \
    && dnf clean all

USER 1001

# Set working directory
WORKDIR /opt/app-root/src

# Copy Python dependencies from builder (includes all installed packages)
COPY --from=builder --chown=1001:0 /opt/app-root/lib/python3.12/site-packages /opt/app-root/lib/python3.12/site-packages
COPY --from=builder --chown=1001:0 /opt/app-root/bin /opt/app-root/bin

# Copy ALL pre-downloaded model caches from builder
COPY --from=builder --chown=1001:0 /opt/app-root/src/.cache /opt/app-root/src/.cache

# Fix permissions on cache directory to ensure all nested directories are readable
# Make cache directories read-only at runtime to prevent modification in air-gapped env
USER root
RUN chmod -R 755 /opt/app-root/src/.cache && \
    chown -R 1001:0 /opt/app-root/src/.cache
USER 1001

# Copy application code
COPY --chown=1001:0 src/ ./src/
COPY --chown=1001:0 pyproject.toml ./

# Set environment variables for OFFLINE operation
# CRITICAL: These settings ensure NO network calls are made at runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/app-root/bin:${PATH}" \
    PYTHONPATH="/opt/app-root/src:${PYTHONPATH}" \
    HF_HOME=/opt/app-root/src/.cache/huggingface \
    TRANSFORMERS_CACHE=/opt/app-root/src/.cache/huggingface \
    HF_DATASETS_CACHE=/opt/app-root/src/.cache/huggingface \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DOCLING_SERVE_ARTIFACTS_PATH=/opt/app-root/src/.cache/docling \
    EASYOCR_MODULE_PATH=/opt/app-root/src/.cache/easyocr

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ping')"

# Run the application
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
