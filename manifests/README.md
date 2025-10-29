# Document Analysis Service - OpenShift Deployment Manifests

This directory contains Kustomize-based deployment manifests for deploying the Document Analysis Service to Red Hat OpenShift.

## Structure

```
manifests/
├── base/               # Base configuration for all environments
│   ├── configmap.yaml      # Application configuration
│   ├── deployment.yaml     # Deployment and ServiceAccount
│   ├── service.yaml        # Kubernetes Service
│   ├── route.yaml          # OpenShift Route
│   └── kustomization.yaml  # Base Kustomize config
└── overlays/           # Environment-specific configurations
    ├── dev/            # Development environment
    ├── staging/        # Staging environment
    └── prod/           # Production environment
```

## Container Image

The manifests deploy from the Quay.io registry:
- **Image**: `quay.io/wjackson/document-analysis:latest`
- **Base Image**: Red Hat UBI 9 with Python 3.11
- **Frameworks**: Includes unified-document-analysis with all frameworks (xml, docling, document, data)

## Environment Configurations

### Development (`overlays/dev`)
- **Namespace**: `document-analysis-dev`
- **Replicas**: 1
- **Resources**: 1Gi/250m (requests), 2Gi/1000m (limits)
- **Debug**: Enabled
- **Upload Limits**: 50MB (upload), 500MB (S3)

### Staging (`overlays/staging`)
- **Namespace**: `document-analysis-staging`
- **Replicas**: 2
- **Resources**: 1.5Gi/375m (requests), 3Gi/1500m (limits)
- **Debug**: Disabled
- **Upload Limits**: 75MB (upload), 750MB (S3)

### Production (`overlays/prod`)
- **Namespace**: `document-analysis-prod`
- **Replicas**: 3 (with HPA up to 10)
- **Resources**: 2Gi/500m (requests), 4Gi/2000m (limits)
- **Debug**: Disabled
- **Upload Limits**: 100MB (upload), 1000MB (S3)
- **Additional Features**:
  - HorizontalPodAutoscaler (CPU 70%, Memory 80%)
  - PodDisruptionBudget (minAvailable: 2)

## Deployment Instructions

### Prerequisites

1. Access to an OpenShift cluster
2. Logged in via `oc login`
3. Appropriate permissions to create namespaces and resources

### Deploy to Development

```bash
# Preview what will be created
oc kustomize manifests/overlays/dev

# Create the namespace if it doesn't exist
oc get namespace document-analysis-dev || oc new-project document-analysis-dev

# Deploy the application
oc apply -k manifests/overlays/dev -n document-analysis-dev

# Verify deployment
oc get pods -n document-analysis-dev
oc get route -n document-analysis-dev
```

### Deploy to Staging

```bash
# Preview what will be created
oc kustomize manifests/overlays/staging

# Create the namespace if it doesn't exist
oc get namespace document-analysis-staging || oc new-project document-analysis-staging

# Deploy the application
oc apply -k manifests/overlays/staging -n document-analysis-staging

# Verify deployment
oc get pods -n document-analysis-staging
oc get route -n document-analysis-staging
```

### Deploy to Production

```bash
# Preview what will be created
oc kustomize manifests/overlays/prod

# Create the namespace if it doesn't exist
oc get namespace document-analysis-prod || oc new-project document-analysis-prod

# Deploy the application
oc apply -k manifests/overlays/prod -n document-analysis-prod

# Verify deployment
oc get pods -n document-analysis-prod
oc get route -n document-analysis-prod
oc get hpa -n document-analysis-prod
```

## Accessing the Service

After deployment, get the route URL:

```bash
# For dev
oc get route dev-document-analysis-service -n document-analysis-dev -o jsonpath='{.spec.host}'

# For staging
oc get route staging-document-analysis-service -n document-analysis-staging -o jsonpath='{.spec.host}'

# For prod
oc get route prod-document-analysis-service -n document-analysis-prod -o jsonpath='{.spec.host}'
```

Test the service:

```bash
# Health check
curl https://<route-url>/ping

# Service info
curl https://<route-url>/
```

## Customization

### Change Namespace

To deploy to a different namespace, override the namespace in your overlay:

```yaml
# In your overlay's kustomization.yaml
namespace: my-custom-namespace
```

### Adjust Resources

To change resource limits/requests, add or modify patches in your overlay:

```yaml
patches:
- patch: |-
    - op: replace
      path: /spec/template/spec/containers/0/resources/requests/memory
      value: "3Gi"
  target:
    kind: Deployment
    name: document-analysis-service
```

### Use Different Image Tag

To use a specific image tag instead of `latest`:

```yaml
# In your overlay's kustomization.yaml
images:
- name: quay.io/wjackson/document-analysis
  newTag: v1.2.3
```

## Health Checks

The deployment includes health probes:

- **Liveness Probe**: `GET /ping` (port 8000)
  - Initial delay: 30s
  - Period: 30s
  - Timeout: 10s

- **Readiness Probe**: `GET /ping` (port 8000)
  - Initial delay: 20s
  - Period: 10s
  - Timeout: 5s

## Security

The deployment follows OpenShift security best practices:

- Runs as non-root user (UID 1001)
- Drops all capabilities
- Disables privilege escalation
- Uses seccomp profile
- TLS-enabled routes with edge termination
- Automatic HTTP to HTTPS redirect

## Configuration

Application configuration is managed via ConfigMap. Key settings:

- `APP_SERVICE_NAME`: Service identifier
- `APP_SERVICE_VERSION`: Service version
- `APP_DEBUG`: Enable/disable debug mode
- `APP_MAX_UPLOAD_SIZE_MB`: Maximum file upload size
- `APP_MAX_S3_SIZE_MB`: Maximum S3 file size
- `APP_CORS_ORIGINS`: CORS allowed origins (JSON array)

To override in your environment, use `configMapGenerator` in the overlay:

```yaml
configMapGenerator:
- name: document-analysis-service-config
  behavior: merge
  literals:
  - APP_DEBUG=true
  - APP_MAX_UPLOAD_SIZE_MB=200
```

## Troubleshooting

### View Logs

```bash
# Get pod name
POD=$(oc get pods -n document-analysis-dev -l app=document-analysis-service -o name | head -1)

# View logs
oc logs -n document-analysis-dev $POD

# Follow logs
oc logs -n document-analysis-dev $POD -f
```

### Check Events

```bash
oc get events -n document-analysis-dev --sort-by='.lastTimestamp'
```

### Debug Pod Issues

```bash
# Describe the deployment
oc describe deployment dev-document-analysis-service -n document-analysis-dev

# Check pod status
oc describe pod -n document-analysis-dev -l app=document-analysis-service
```

### Image Pull Issues

If the pod fails to pull the image:

```bash
# Verify the image exists and is accessible
podman pull quay.io/wjackson/document-analysis:latest

# Check if OpenShift can access Quay.io (public images should work)
oc run test --image=quay.io/wjackson/document-analysis:latest --dry-run=client
```

## Updating the Deployment

To update to a new version:

```bash
# Update the image tag in the overlay or push new image with :latest tag
# Then restart the deployment
oc rollout restart deployment/dev-document-analysis-service -n document-analysis-dev

# Watch the rollout
oc rollout status deployment/dev-document-analysis-service -n document-analysis-dev
```

## Cleanup

To remove the deployment:

```bash
# Delete all resources
oc delete -k manifests/overlays/dev -n document-analysis-dev

# Optionally delete the namespace
oc delete namespace document-analysis-dev
```
