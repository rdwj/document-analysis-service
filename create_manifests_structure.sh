#!/bin/bash

# Create Kustomize manifest directory structure for document-analysis-service

mkdir -p manifests/base
mkdir -p manifests/overlays/dev
mkdir -p manifests/overlays/staging
mkdir -p manifests/overlays/prod

echo "Created manifest directory structure:"
tree manifests/ 2>/dev/null || find manifests/ -type d
