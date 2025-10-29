"""Main FastAPI application entry point."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings

from .api.rest import router as api_router


class Settings(BaseSettings):
    """Application settings."""

    # Service configuration
    service_name: str = "document-analysis-service"
    service_version: str = "1.0.0"
    debug: bool = False

    # File size limits
    max_upload_size_mb: int = 100
    max_s3_size_mb: int = 1000

    # S3 configuration
    s3_endpoint: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None

    # CORS
    cors_origins: list[str] = ["*"]

    class Config:
        env_prefix = "APP_"


settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print(f"Starting {settings.service_name} v{settings.service_version}")
    print(f"Debug mode: {settings.debug}")
    print(f"Max upload size: {settings.max_upload_size_mb}MB")
    print(f"Max S3 size: {settings.max_s3_size_mb}MB")

    yield

    # Shutdown
    print(f"Shutting down {settings.service_name}")


# Create FastAPI app
app = FastAPI(
    title="Document Analysis Service",
    description=(
        "OpenShift service for analyzing and chunking documents using "
        "unified-document-analysis framework"
    ),
    version=settings.service_version,
    lifespan=lifespan,
    debug=settings.debug,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api/v1", tags=["Document Analysis"])


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "status": "running",
        "docs": "/docs",
        "api": "/api/v1",
    }


@app.get("/ping")
async def ping():
    """Simple ping endpoint for basic health checks."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug",
    )
