# Multi-stage build for document-analysis-service
# Uses Red Hat UBI 9 with Python 3.12 as base image
# Pre-downloads all Docling models for air-gapped/offline deployment

# Stage 1: Builder
FROM registry.access.redhat.com/ubi9/python-312:latest AS builder

USER root

# Install system dependencies (including OpenGL for OpenCV/RapidOCR)
RUN dnf install -y \
    gcc \
    python3-devel \
    file-libs \
    mesa-libGL \
    && dnf clean all

# Create cache directories as root so user 1001 can write to them
RUN mkdir -p /opt/app-root/src/.cache/huggingface /opt/app-root/src/.cache/docling && \
    chown -R 1001:0 /opt/app-root/src/.cache

USER 1001

# Set working directory
WORKDIR /opt/app-root/src

# Set HuggingFace cache location for builder stage
ENV HF_HOME=/opt/app-root/src/.cache/huggingface \
    TRANSFORMERS_CACHE=/opt/app-root/src/.cache/huggingface \
    HF_DATASETS_CACHE=/opt/app-root/src/.cache/huggingface

# Copy all files needed for build
COPY --chown=1001:0 pyproject.toml README.md ./
COPY --chown=1001:0 src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

# Pre-download Docling models for standard pipeline with RapidOCR for offline usage
# VLM models (smolvlm, granitedocling, etc.) documented in FUTURE_ENHANCEMENTS.md for later addition
RUN docling-tools models download layout tableformer code_formula picture_classifier rapidocr \
    -o /opt/app-root/src/.cache/docling && \
    echo "Docling standard pipeline models with RapidOCR pre-downloaded successfully"

# Pre-download RapidOCR models by initializing the library (triggers model download to package dir)
# This ensures models are cached during build rather than attempted at runtime
RUN python3 -c "from rapidocr import RapidOCR; ocr = RapidOCR()" && \
    echo "RapidOCR models pre-downloaded successfully"

# Stage 2: Runtime
FROM registry.access.redhat.com/ubi9/python-312:latest

USER root

# Install runtime dependencies (including OpenGL for OpenCV/Docling)
RUN dnf install -y \
    file-libs \
    mesa-libGL \
    && dnf clean all

USER 1001

# Set working directory
WORKDIR /opt/app-root/src

# Copy Python dependencies from builder
COPY --from=builder --chown=1001:0 /opt/app-root/lib/python3.12/site-packages /opt/app-root/lib/python3.12/site-packages
COPY --from=builder --chown=1001:0 /opt/app-root/bin /opt/app-root/bin

# Copy pre-downloaded HuggingFace cache from builder
COPY --from=builder --chown=1001:0 /opt/app-root/src/.cache /opt/app-root/src/.cache

# Fix permissions on cache directory to ensure all nested directories are writable
# Also fix RapidOCR models directory permissions for runtime updates
USER root
RUN chmod -R 775 /opt/app-root/src/.cache && \
    chown -R 1001:0 /opt/app-root/src/.cache && \
    chmod -R 775 /opt/app-root/lib/python3.12/site-packages/rapidocr/models && \
    chown -R 1001:0 /opt/app-root/lib/python3.12/site-packages/rapidocr/models
USER 1001

# Copy application code
COPY --chown=1001:0 src/ ./src/
COPY --chown=1001:0 pyproject.toml ./

# Set environment variables
# Models are pre-downloaded and cached - HuggingFace will use cache automatically
# Removed HF_HUB_OFFLINE=1 as it prevents proper cache lookups with revision metadata
# DOCLING_SERVE_ARTIFACTS_PATH points to pre-downloaded Docling models for offline use
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/app-root/bin:${PATH}" \
    PYTHONPATH="/opt/app-root/src:${PYTHONPATH}" \
    HF_HOME=/opt/app-root/src/.cache/huggingface \
    TRANSFORMERS_CACHE=/opt/app-root/src/.cache/huggingface \
    HF_DATASETS_CACHE=/opt/app-root/src/.cache/huggingface \
    DOCLING_SERVE_ARTIFACTS_PATH=/opt/app-root/src/.cache/docling

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ping')"

# Run the application
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
