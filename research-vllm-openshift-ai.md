# vLLM Model Serving on OpenShift AI: Configuration Guide

**Research Date:** October 29, 2025
**Target Use Case:** Deploying IBM Granite Docling 258M model with vLLM on OpenShift AI
**Critical Requirement:** Configure `--revision=untied` parameter for Granite Docling model

---

## Executive Summary

This research provides a complete guide to configuring vLLM model serving on Red Hat OpenShift AI (RHOAI) for the IBM Granite Docling 258M model. The key findings show that:

1. **Custom arguments like `--revision=untied`** are added to the ServingRuntime YAML in the `containers.args` section
2. **GPU time-slicing** is configured via a ConfigMap in the `nvidia-gpu-operator` namespace
3. **Authentication** uses service account tokens with Bearer token authorization
4. **Endpoint discovery** is managed through OpenShift AI dashboard with ConfigMap/Secret integration for client applications
5. **HuggingFace models** can be loaded directly using `storageUri` or via the `--model` argument

---

## 1. vLLM ServingRuntime Configuration

### Base ServingRuntime YAML Structure

The ServingRuntime defines how vLLM runs within OpenShift AI. Here's a complete template with custom arguments:

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: ServingRuntime
metadata:
  name: vllm-granite-docling
  annotations:
    openshift.io/display-name: "vLLM Granite Docling Runtime"
    opendatahub.io/recommended-accelerators: '["nvidia.com/gpu"]'
  labels:
    opendatahub.io/dashboard: "true"
spec:
  builtInAdapter:
    modelLoadingTimeoutMillis: 90000

  containers:
    - name: kserve-container
      image: quay.io/rh-aiservices-bu/vllm-openai-ubi9:0.4.2

      # CRITICAL: Custom arguments section where --revision is added
      args:
        - --model
        - /mnt/models/
        - --revision
        - untied                    # ← GRANITE DOCLING SPECIFIC
        - --download-dir
        - /models-cache
        - --port
        - "8080"
        - --max-model-len
        - "6144"
        - --dtype
        - float16                   # Memory optimization

      # Environment variables for offline/air-gapped environments
      env:
        - name: HF_HOME
          value: /tmp/hf_home
        - name: HF_HUB_OFFLINE
          value: "1"                # For disconnected environments
        - name: TRANSFORMERS_CACHE
          value: /tmp/hf_home
        - name: TIKTOKEN_RS_CACHE_DIR
          value: /.cache/tiktoken-rs-cache

      ports:
        - containerPort: 8080
          name: http1
          protocol: TCP

      # Resource requests/limits
      resources:
        requests:
          cpu: "2"
          memory: 8Gi
          nvidia.com/gpu: "1"       # Request 1 GPU slice if time-slicing enabled
        limits:
          cpu: "6"
          memory: 16Gi
          nvidia.com/gpu: "1"

  # GPU tolerations for dedicated GPU nodes
  tolerations:
    - effect: NoSchedule
      key: nvidia.com/gpu
      operator: Exists

  multiModel: false
  supportedModelFormats:
    - autoSelect: true
      name: pytorch
```

### Key Configuration Points

1. **`--revision=untied`**: Critical for Granite Docling model - specifies the HuggingFace branch/tag to load
2. **`--max-model-len`**: Context window size (adjust based on GPU memory)
3. **`--dtype`**: Use `float16` or `bfloat16` to reduce memory footprint
4. **Environment Variables**: Set `HF_HUB_OFFLINE=1` for FIPS/disconnected environments
5. **Model Loading Timeout**: 90 seconds should be sufficient for 258M model

### Additional vLLM Arguments Reference

Based on vLLM documentation, other useful arguments include:

- `--tensor-parallel-size N`: For multi-GPU tensor parallelism
- `--max-num-batched-tokens N`: Control batch size for throughput
- `--gpu-memory-utilization 0.9`: GPU memory fraction to use
- `--trust-remote-code`: Required for some HuggingFace models
- `--enable-auto-tool-choice`: For function calling support
- `--served-model-name NAME`: Override model name in API

**Source:** vLLM Engine Arguments documentation (https://docs.vllm.ai/en/latest/configuration/engine_args.html)

---

## 2. InferenceService Configuration

### InferenceService YAML for Granite Docling

The InferenceService links a model to the ServingRuntime:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: granite-docling-258m
  annotations:
    openshift.io/display-name: "Granite Docling 258M"
    serving.knative.openshift.io/enablePassthrough: "true"
    serving.kserve.io/deploymentMode: Serverless
    sidecar.istio.io/inject: "true"
    sidecar.istio.io/rewriteAppHTTPProbers: "true"
  labels:
    opendatahub.io/dashboard: "true"
  namespace: document-analysis-dev
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 3

    model:
      # Reference the custom ServingRuntime
      runtime: vllm-granite-docling

      # Model format
      modelFormat:
        name: pytorch

      # Storage configuration - Option 1: HuggingFace Hub
      storageUri: hf://ibm-granite/granite-docling-258m

      # Storage configuration - Option 2: S3 via Data Connection
      # storage:
      #   key: aws-connection-my-storage
      #   path: models/granite-docling-258m/

      # Storage configuration - Option 3: PVC
      # storageUri: pvc://granite-docling-pvc/models/granite-docling-258m

      # Resource allocation with GPU time-slicing
      resources:
        requests:
          cpu: "2"
          memory: 8Gi
          nvidia.com/gpu: "1"       # Request 1 GPU slice
        limits:
          cpu: "6"
          memory: 16Gi
          nvidia.com/gpu: "1"

      # Environment variables specific to this model
      env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: huggingface-token
              key: token
              optional: false        # Required for gated models

      # GPU tolerations
      tolerations:
        - effect: NoSchedule
          key: nvidia.com/gpu
          operator: Exists
```

### Storage Options Explained

**Option 1: HuggingFace Hub (Direct)**
- Format: `hf://model-owner/model-name`
- Requires: `HF_TOKEN` in Secret if model is gated
- Revision specified in ServingRuntime args (`--revision=untied`)
- Best for: Public or gated HuggingFace models

**Option 2: S3-Compatible Storage (Data Connection)**
- Uses OpenShift AI Data Connection
- Requires: S3 credentials in Secret
- Best for: Enterprise/air-gapped environments with model registry

**Option 3: PVC (Persistent Volume Claim)**
- Format: `pvc://pvc-name/path/to/model`
- Best for: Pre-downloaded models in shared storage

**Source:** Red Hat OpenShift AI 2.22 Serving Models documentation

---

## 3. GPU Time-Slicing Configuration

GPU time-slicing allows multiple inference pods to share a single physical GPU by dividing processing time into alternating slots.

### ConfigMap for Time-Slicing

Create this ConfigMap in the `nvidia-gpu-operator` namespace:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: nvidia-gpu-operator
data:
  # Configuration for Tesla T4 GPUs
  tesla-t4: |-
    version: v1
    flags:
      migStrategy: none
    sharing:
      timeSlicing:
        renameByDefault: false
        failRequestsGreaterThanOne: false
        resources:
          - name: nvidia.com/gpu
            replicas: 4             # Create 4 GPU slices per physical GPU

  # Configuration for NVIDIA A100 GPUs
  a100-sxm4-40gb: |-
    version: v1
    sharing:
      timeSlicing:
        resources:
          - name: nvidia.com/gpu
            replicas: 8             # Create 8 GPU slices per physical GPU
```

### Applying Time-Slicing Configuration

**Step 1: Create the ConfigMap**

```bash
oc create -f time-slicing-config.yaml -n nvidia-gpu-operator
```

**Step 2: Patch the ClusterPolicy**

```bash
oc patch clusterpolicy gpu-cluster-policy \
  -n nvidia-gpu-operator \
  --type merge \
  -p '{"spec": {"devicePlugin": {"config": {"name": "time-slicing-config"}}}}'
```

**Step 3: Label GPU Nodes**

Label nodes to activate time-slicing for specific GPU types:

```bash
# For Tesla T4 nodes
oc label node <node-name> nvidia.com/device-plugin.config=tesla-t4

# For A100 nodes
oc label node <node-name> nvidia.com/device-plugin.config=a100-sxm4-40gb
```

**Step 4: Verify Time-Slicing**

```bash
# Check available GPU resources on node
oc describe node <node-name> | grep nvidia.com/gpu

# Output should show replicas * physical_gpus
# Example: 4 replicas × 2 physical GPUs = 8 allocatable GPUs
```

### Time-Slicing Considerations

**Advantages:**
- Maximize GPU utilization for inference workloads
- Cost-effective for lightweight models like Granite Docling 258M
- No code changes required - works with existing InferenceServices

**Limitations:**
- **No memory isolation**: All slices share GPU memory (can cause OOM)
- **No fault isolation**: One pod crash can affect others
- **Time-sharing overhead**: Slight latency increase due to context switching
- **Best for**: Batch inference, async processing, low-concurrency scenarios

**Performance Guidance:**
- Granite Docling 258M (~500MB model): Can support 4-8 replicas on 16GB GPU
- Monitor GPU memory usage: `nvidia-smi` or OpenShift AI metrics
- Start with 2-4 replicas, increase based on actual usage patterns

**Source:** Red Hat OpenShift AI 2.22 Working with Accelerators + NVIDIA GPU Operator documentation

---

## 4. Authentication and Token Management

OpenShift AI model serving uses service account-based authentication with Bearer tokens.

### Token Authorization Configuration

**Enable Token Authorization in InferenceService:**

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: granite-docling-258m
  annotations:
    security.opendatahub.io/enable-auth: "true"  # Enable token auth
spec:
  predictor:
    # ... rest of config
```

### Service Account Setup

**Create Service Account for Inference Clients:**

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: granite-docling-client
  namespace: document-analysis-dev
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: granite-docling-client-binding
  namespace: document-analysis-dev
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: view                          # Need 'get' on InferenceServices
subjects:
  - kind: ServiceAccount
    name: granite-docling-client
    namespace: document-analysis-dev
```

### Retrieving Service Account Token

**Method 1: From Pod (Recommended for Applications)**

When your application pod uses the service account, the token is automatically mounted:

```bash
# Inside the pod
TOKEN=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)

# Use in API call
curl -H "Authorization: Bearer $TOKEN" \
  https://<inference-endpoint>/v1/chat/completions \
  -d '{"model": "granite-docling-258m", "messages": [...]}'
```

**Method 2: Generate Token Manually (For Testing)**

```bash
# Create long-lived token (OpenShift 4.11+)
oc create token granite-docling-client \
  -n document-analysis-dev \
  --duration=8760h  # 1 year

# Or use short-lived token (default: 1 hour)
TOKEN=$(oc create token granite-docling-client -n document-analysis-dev)
```

### Endpoint Discovery

**Get Inference Endpoint URL:**

```bash
# From command line
oc get inferenceservice granite-docling-258m \
  -n document-analysis-dev \
  -o jsonpath='{.status.url}'

# Output example:
# https://granite-docling-258m-document-analysis-dev.apps.cluster.example.com
```

**Dashboard Method:**
1. Navigate to OpenShift AI Dashboard → Data Science Projects
2. Select your project → Models tab
3. Click on the model name
4. Copy the "Inference endpoint" URL
5. If using internal endpoints, click "Internal endpoint details" for the restUrl

### Token Storage for Client Applications

**ConfigMap for Endpoint URL:**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: granite-docling-config
  namespace: document-analysis-dev
data:
  GRANITE_DOCLING_ENDPOINT: "https://granite-docling-258m-document-analysis-dev.apps.cluster.example.com"
  GRANITE_DOCLING_MODEL: "granite-docling-258m"
```

**Secret for Service Account Token:**

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: granite-docling-token
  namespace: document-analysis-dev
type: Opaque
stringData:
  token: "<service-account-token>"
```

**Mount in Application Deployment:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: document-analysis-service
  namespace: document-analysis-dev
spec:
  template:
    spec:
      serviceAccountName: granite-docling-client  # Auto-mount token
      containers:
        - name: api
          image: document-analysis-service:latest
          env:
            - name: GRANITE_DOCLING_ENDPOINT
              valueFrom:
                configMapKeyRef:
                  name: granite-docling-config
                  key: GRANITE_DOCLING_ENDPOINT
            - name: GRANITE_DOCLING_TOKEN
              valueFrom:
                secretKeyRef:
                  name: granite-docling-token
                  key: token
```

**Source:** Red Hat OpenShift AI 2.22 Serving Models documentation + Authorino integration guide

---

## 5. Complete Deployment Example

### Directory Structure for Kustomize

```
manifests/
├── base/
│   ├── namespace.yaml
│   ├── serving-runtime.yaml      # vLLM ServingRuntime
│   ├── inference-service.yaml    # InferenceService
│   ├── service-account.yaml      # Client SA
│   ├── configmap.yaml           # Endpoint URL
│   └── kustomization.yaml
└── overlays/
    ├── dev/
    │   ├── inference-service-patch.yaml
    │   └── kustomization.yaml
    └── prod/
        ├── inference-service-patch.yaml
        └── kustomization.yaml
```

### Base Kustomization

**`manifests/base/kustomization.yaml`:**

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: document-analysis-dev

resources:
  - namespace.yaml
  - serving-runtime.yaml
  - inference-service.yaml
  - service-account.yaml
  - configmap.yaml

configMapGenerator:
  - name: granite-docling-config
    literals:
      - GRANITE_DOCLING_MODEL=granite-docling-258m
```

### Overlay Patches

**`manifests/overlays/dev/inference-service-patch.yaml`:**

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: granite-docling-258m
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 2
    model:
      resources:
        requests:
          nvidia.com/gpu: "1"  # Use time-sliced GPU
```

**`manifests/overlays/prod/inference-service-patch.yaml`:**

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: granite-docling-258m
spec:
  predictor:
    minReplicas: 2
    maxReplicas: 10
    model:
      resources:
        requests:
          nvidia.com/gpu: "1"  # Dedicated GPU or time-sliced
        limits:
          memory: 24Gi
```

### Deployment Commands

**Deploy to Dev:**

```bash
# Create namespace first
oc new-project document-analysis-dev

# Apply GPU time-slicing (if not already configured)
oc apply -f time-slicing-config.yaml -n nvidia-gpu-operator
oc patch clusterpolicy gpu-cluster-policy -n nvidia-gpu-operator \
  --type merge \
  -p '{"spec": {"devicePlugin": {"config": {"name": "time-slicing-config"}}}}'

# Label GPU nodes
oc label node <gpu-node-name> nvidia.com/device-plugin.config=tesla-t4

# Create HuggingFace token secret (if needed)
oc create secret generic huggingface-token \
  --from-literal=token=<YOUR_HF_TOKEN> \
  -n document-analysis-dev

# Deploy model serving
oc apply -k manifests/overlays/dev -n document-analysis-dev

# Wait for deployment
oc wait --for=condition=Ready inferenceservice/granite-docling-258m \
  -n document-analysis-dev \
  --timeout=5m

# Get endpoint URL
oc get inferenceservice granite-docling-258m \
  -n document-analysis-dev \
  -o jsonpath='{.status.url}'
```

**Verify Deployment:**

```bash
# Check pod status
oc get pods -n document-analysis-dev -l serving.kserve.io/inferenceservice=granite-docling-258m

# Check logs
oc logs -f deployment/granite-docling-258m-predictor-00001-deployment \
  -n document-analysis-dev

# Test inference endpoint
TOKEN=$(oc create token granite-docling-client -n document-analysis-dev)

curl -X POST \
  "https://granite-docling-258m-document-analysis-dev.apps.cluster.example.com/v1/completions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite-docling-258m",
    "prompt": "Extract text from document",
    "max_tokens": 100
  }'
```

---

## 6. Client Application Integration

### Python Client Example

**Install Dependencies:**

```bash
pip install openai kubernetes
```

**`client/granite_docling_client.py`:**

```python
import os
from openai import OpenAI
from kubernetes import client, config

class GraniteDoclingClient:
    def __init__(self):
        # Load Kubernetes config (in-cluster or kubeconfig)
        try:
            config.load_incluster_config()  # Running inside pod
        except:
            config.load_kube_config()       # Running locally

        # Get endpoint URL from ConfigMap
        v1 = client.CoreV1Api()
        namespace = os.getenv("NAMESPACE", "document-analysis-dev")

        configmap = v1.read_namespaced_config_map(
            name="granite-docling-config",
            namespace=namespace
        )

        endpoint_url = configmap.data["GRANITE_DOCLING_ENDPOINT"]
        model_name = configmap.data["GRANITE_DOCLING_MODEL"]

        # Get token from service account (auto-mounted)
        with open("/var/run/secrets/kubernetes.io/serviceaccount/token") as f:
            token = f.read().strip()

        # Initialize OpenAI client with custom endpoint
        self.client = OpenAI(
            base_url=f"{endpoint_url}/v1",
            api_key=token,  # Use service account token as API key
        )
        self.model = model_name

    def analyze_document(self, document_text: str, max_tokens: int = 512):
        """
        Analyze document using Granite Docling model.
        """
        response = self.client.completions.create(
            model=self.model,
            prompt=f"Analyze this document:\n{document_text}",
            max_tokens=max_tokens,
            temperature=0.1
        )

        return response.choices[0].text.strip()

    def chat_completion(self, messages: list, max_tokens: int = 512):
        """
        Chat completion for conversational document analysis.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1
        )

        return response.choices[0].message.content

# Usage example
if __name__ == "__main__":
    client = GraniteDoclingClient()

    result = client.analyze_document(
        "This is a sample document for analysis."
    )

    print(f"Analysis result: {result}")
```

### Environment Variable Configuration

For applications that prefer environment variables over Kubernetes API calls:

```python
import os
from openai import OpenAI

# Read from environment (set by ConfigMap/Secret)
endpoint_url = os.getenv("GRANITE_DOCLING_ENDPOINT")
token = os.getenv("GRANITE_DOCLING_TOKEN")
model = os.getenv("GRANITE_DOCLING_MODEL", "granite-docling-258m")

client = OpenAI(
    base_url=f"{endpoint_url}/v1",
    api_key=token,
)

response = client.completions.create(
    model=model,
    prompt="Extract key information from this document...",
    max_tokens=256
)
```

---

## 7. Monitoring and Troubleshooting

### Health Checks

**Check Model Server Health:**

```bash
TOKEN=$(oc create token granite-docling-client -n document-analysis-dev)
ENDPOINT=$(oc get isvc granite-docling-258m -n document-analysis-dev -o jsonpath='{.status.url}')

# Health endpoint
curl -H "Authorization: Bearer $TOKEN" "${ENDPOINT}/health"

# Models endpoint
curl -H "Authorization: Bearer $TOKEN" "${ENDPOINT}/v1/models"
```

### Common Issues and Solutions

**Issue 1: Model Pod Not Starting**

```bash
# Check pod status
oc get pods -n document-analysis-dev -l serving.kserve.io/inferenceservice=granite-docling-258m

# Check events
oc describe inferenceservice granite-docling-258m -n document-analysis-dev

# Common causes:
# - GPU not available: Check GPU node labels and time-slicing config
# - HuggingFace token missing: Verify secret exists
# - Model download timeout: Increase modelLoadingTimeoutMillis in ServingRuntime
```

**Issue 2: 401 Unauthorized Errors**

```bash
# Verify service account has correct permissions
oc get rolebinding -n document-analysis-dev | grep granite-docling

# Check if token is expired
TOKEN=$(oc create token granite-docling-client -n document-analysis-dev --duration=1h)

# Test with fresh token
curl -H "Authorization: Bearer $TOKEN" "${ENDPOINT}/v1/models"
```

**Issue 3: GPU Memory Exhausted (OOM)**

```bash
# Check GPU usage on node
oc debug node/<gpu-node-name>
chroot /host
nvidia-smi

# Solutions:
# - Reduce --max-model-len in ServingRuntime
# - Reduce number of time-slicing replicas
# - Use --dtype bfloat16 instead of float16
# - Reduce minReplicas in InferenceService
```

**Issue 4: `--revision=untied` Not Applied**

```bash
# Verify ServingRuntime has correct args
oc get servingruntime vllm-granite-docling -n document-analysis-dev -o yaml | grep -A 20 "args:"

# Check pod environment
oc exec -it <predictor-pod-name> -n document-analysis-dev -- env | grep -i vllm

# Verify vLLM server logs show revision parameter
oc logs <predictor-pod-name> -n document-analysis-dev | grep revision
```

### Metrics and Observability

**Enable Prometheus Metrics:**

Add annotation to InferenceService:

```yaml
metadata:
  annotations:
    prometheus.io/path: /metrics
    prometheus.io/port: '3000'
    prometheus.io/scrape: 'true'
```

**Query GPU Utilization:**

```promql
# GPU utilization percentage
DCGM_FI_DEV_GPU_UTIL{namespace="document-analysis-dev"}

# GPU memory used
DCGM_FI_DEV_FB_USED{namespace="document-analysis-dev"}

# Inference requests per second
rate(vllm_request_success_total[5m])
```

---

## 8. Limitations and Caveats

### OpenShift AI Specific Limitations

1. **Model Format Support**: vLLM runtime only supports PyTorch format (`.safetensors` or `.bin` files)

2. **Storage Protocols**:
   - Single-node serving: Supports S3, PVC, HuggingFace Hub (`hf://`)
   - Multi-node serving: Only PVC supported

3. **GPU Requirements**:
   - NVIDIA GPU Operator must be installed
   - Node Feature Discovery (NFD) operator required
   - GPU support must be enabled in OpenShift AI dashboard

4. **vLLM Version**: OpenShift AI ships with specific vLLM versions (e.g., 0.4.2)
   - Custom vLLM images: Build your own for newer versions
   - Check compatibility with Granite Docling model

5. **FIPS Mode**:
   - vLLM may have issues in FIPS-enabled clusters
   - Test thoroughly in your specific environment
   - Some crypto operations may fail

### Time-Slicing Specific Limitations

1. **No Memory Isolation**: All replicas share GPU memory
   - Risk of OOM if one replica uses too much memory
   - Monitor memory usage carefully

2. **No Fault Isolation**: One crashed pod can affect others
   - Use PodDisruptionBudgets for high availability
   - Implement health checks and auto-restart policies

3. **Performance Variability**:
   - Response latency increases with more replicas
   - Not suitable for real-time or low-latency requirements
   - Best for async/batch processing

4. **Replica Limits**:
   - Don't exceed 8-10 replicas per GPU for inference
   - Granite Docling 258M: Recommend 4-6 replicas max on 16GB GPU

### Authentication Limitations

1. **Token Expiration**:
   - Service account tokens expire (default: 1 hour for `oc create token`)
   - Use long-lived tokens or auto-refresh mechanism
   - Pods with mounted SA get auto-refreshed tokens

2. **RBAC Requirements**:
   - Service account needs `get` permission on InferenceService
   - Admin/editor roles have this by default
   - Custom roles must explicitly grant it

3. **External Access**:
   - Routes are cluster-external by default
   - Service mesh may require additional configuration
   - Corporate firewalls may block inference endpoints

---

## 9. Recommendations for Document Analysis Service

### Architecture Decision: Optional VLM Pipeline

Based on your project's `FUTURE_ENHANCEMENTS.md`, you're planning optional VLM pipeline support. Here's the recommended approach:

**ConfigMap Structure:**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: document-analysis-config
  namespace: document-analysis-dev
data:
  # VLM pipeline feature flag
  ENABLE_VLM_PIPELINE: "false"  # Default: false (standard pipeline)

  # VLM endpoint configuration (only used if ENABLE_VLM_PIPELINE=true)
  VLM_ENDPOINT: "https://granite-docling-258m-document-analysis-dev.apps.cluster.example.com"
  VLM_MODEL_NAME: "granite-docling-258m"
  VLM_TIMEOUT_SECONDS: "30"
  VLM_MAX_TOKENS: "2048"
```

**Secret for VLM Token:**

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: vlm-credentials
  namespace: document-analysis-dev
type: Opaque
stringData:
  token: ""  # Empty by default, populate when VLM is enabled
```

**Kustomize Overlay Strategy:**

```
manifests/
├── base/
│   ├── deployment.yaml
│   ├── configmap.yaml (VLM disabled)
│   ├── secret.yaml (empty token)
│   └── kustomization.yaml
├── overlays/
│   ├── dev/
│   │   └── kustomization.yaml (VLM disabled)
│   ├── dev-vlm/
│   │   ├── configmap-vlm-enabled.yaml
│   │   ├── inference-service.yaml (Granite Docling)
│   │   ├── serving-runtime.yaml (vLLM)
│   │   └── kustomization.yaml
│   └── prod-vlm/
│       └── kustomization.yaml
```

**Deployment Commands:**

```bash
# Standard deployment (no VLM)
oc apply -k manifests/overlays/dev -n document-analysis-dev

# With VLM enabled
oc apply -k manifests/overlays/dev-vlm -n document-analysis-dev
oc create token granite-docling-client -n document-analysis-dev > /tmp/token
oc create secret generic vlm-credentials \
  --from-file=token=/tmp/token \
  -n document-analysis-dev \
  --dry-run=client -o yaml | oc apply -f -
```

### GPU Resource Strategy

For the Granite Docling 258M model in your environment:

**Dev Environment:**
- Use GPU time-slicing with 4 replicas per GPU
- Request 1 GPU slice: `nvidia.com/gpu: "1"`
- Set `minReplicas: 1, maxReplicas: 2`
- Memory: 8Gi request, 16Gi limit

**Prod Environment:**
- Consider dedicated GPU (no time-slicing) for consistent latency
- Or use time-slicing with monitoring and alerting
- Set `minReplicas: 2, maxReplicas: 10` for high availability
- Memory: 16Gi request, 24Gi limit

### Integration with Document Analysis Service

**Add VLM client to `src/core/vlm_client.py`:**

```python
import os
from typing import Optional
from openai import OpenAI

class GraniteDoclingClient:
    """
    Client for Granite Docling 258M VLM inference on OpenShift AI.
    Only instantiated if VLM pipeline is enabled.
    """

    def __init__(self):
        if not self._is_enabled():
            raise RuntimeError("VLM pipeline not enabled. Set ENABLE_VLM_PIPELINE=true")

        self.endpoint = os.getenv("VLM_ENDPOINT")
        self.model = os.getenv("VLM_MODEL_NAME", "granite-docling-258m")
        self.timeout = int(os.getenv("VLM_TIMEOUT_SECONDS", "30"))
        self.max_tokens = int(os.getenv("VLM_MAX_TOKENS", "2048"))

        # Get token from mounted secret
        token_path = "/var/run/secrets/vlm/token"
        if os.path.exists(token_path):
            with open(token_path) as f:
                token = f.read().strip()
        else:
            token = os.getenv("VLM_TOKEN", "")

        self.client = OpenAI(
            base_url=f"{self.endpoint}/v1",
            api_key=token,
            timeout=self.timeout
        )

    @staticmethod
    def _is_enabled() -> bool:
        return os.getenv("ENABLE_VLM_PIPELINE", "false").lower() == "true"

    def analyze_document(self, image_path: str, prompt: str) -> str:
        """
        Analyze document using VLM pipeline.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"file://{image_path}"}}
                    ]
                }
            ],
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content
```

**Update `src/main.py` settings:**

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # ... existing settings

    # VLM Pipeline (optional)
    enable_vlm_pipeline: bool = False
    vlm_endpoint: Optional[str] = None
    vlm_model_name: str = "granite-docling-258m"
    vlm_timeout_seconds: int = 30
    vlm_max_tokens: int = 2048

    class Config:
        env_prefix = "APP_"
        env_file = ".env"
```

**Add VLM endpoint to `src/api/rest.py`:**

```python
from src.core.vlm_client import GraniteDoclingClient

@app.post("/api/v1/analyze/vlm")
async def analyze_with_vlm(file: UploadFile = File(...)):
    """
    Analyze document using VLM pipeline (Granite Docling).
    Only available if ENABLE_VLM_PIPELINE=true.
    """
    if not settings.enable_vlm_pipeline:
        raise HTTPException(
            status_code=501,
            detail="VLM pipeline not enabled. Set ENABLE_VLM_PIPELINE=true"
        )

    # Save uploaded file temporarily
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        client = GraniteDoclingClient()
        result = client.analyze_document(
            image_path=temp_path,
            prompt="Extract all text and structure from this document"
        )

        return {"status": "success", "result": result}
    finally:
        os.remove(temp_path)
```

---

## 10. Key Takeaways

### Critical Success Factors

1. **`--revision=untied` is mandatory** for Granite Docling - add to ServingRuntime args
2. **GPU time-slicing is cost-effective** for 258M model but requires monitoring
3. **Service account tokens** provide seamless authentication for in-cluster clients
4. **ConfigMap + Secret pattern** keeps configuration flexible across environments
5. **Kustomize overlays** enable optional VLM deployment without base manifest changes

### Next Steps

1. **Create base manifests** for ServingRuntime and InferenceService
2. **Test in dev environment** with GPU time-slicing enabled
3. **Validate `--revision=untied`** parameter in vLLM logs
4. **Implement client code** with service account token authentication
5. **Add health checks** and monitoring for production readiness
6. **Document VLM pipeline** setup in project README

### Additional Resources

- **Red Hat OpenShift AI Documentation**: https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.22/
- **vLLM Documentation**: https://docs.vllm.ai/en/latest/
- **NVIDIA GPU Operator**: https://docs.nvidia.com/datacenter/cloud-native/openshift/latest/
- **rh-aiservices-bu GitHub**: https://github.com/rh-aiservices-bu/llm-on-openshift
- **KServe Documentation**: https://kserve.github.io/website/

---

**Research Completed:** October 29, 2025
**Confidence Level:** High - Based on official Red Hat documentation (2.22), NVIDIA operator docs, and validated examples from rh-aiservices-bu
**Recommended Action:** Proceed with implementation using provided YAML templates
