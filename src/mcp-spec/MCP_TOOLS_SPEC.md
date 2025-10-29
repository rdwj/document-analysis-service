# MCP Tools Specification for Document Analysis Service

This document specifies the MCP (Model Context Protocol) tools that should be implemented for the document-analysis-service. These tools will allow AI agents to analyze and chunk documents programmatically.

## Overview

The MCP server should expose tools that mirror the REST API functionality, enabling AI agents to:
- Analyze documents to extract metadata, schema, and specialized insights
- Chunk documents for LLM/RAG processing
- Process multiple documents in batch
- Query supported formats and strategies
- Convert documents between formats

## Core Implementation Notes

**Reuse REST API Logic**: The MCP tools should delegate to the existing REST API implementation in `src/api/rest.py` to avoid code duplication. This ensures:
- Consistent behavior between REST and MCP interfaces
- Single source of truth for business logic
- Easier maintenance and testing

**File Handling**: MCP tools should accept:
- Local file paths (most common for MCP usage)
- S3 URLs (for cloud-based workflows)

**Error Handling**: All tools should return structured errors that include:
- Error type (validation, framework not installed, unsupported format, etc.)
- Clear error message
- Suggestions for resolution when applicable

---

## Tool Specifications

### 1. analyze_document

**Description**: Analyze a single document to extract metadata, schema, and specialized analysis.

**Purpose**: Enables AI agents to understand document structure, detect document type, and extract key information for further processing.

**Parameters**:
- `file_path` (string, required): Path to local file or S3 URL
- `framework_hint` (string, optional): Framework to use (xml, docling, document, data). If not provided, auto-detects.

**Returns**:
```json
{
  "file_path": "path/to/file.xml",
  "result": {
    "document_type": {
      "type_name": "SCAP Security Report",
      "confidence": 0.95,
      "format": "xml"
    },
    "framework": "xml",
    "schema": { ... },
    "specialized_analysis": { ... },
    "ai_use_cases": [ ... ]
  }
}
```

**Error Cases**:
- File not found
- Unsupported file format
- Required framework not installed
- File too large
- Invalid S3 URL

**Example Use Case**:
An AI agent analyzing a folder of documents to understand what types of files are present before deciding how to process them.

---

### 2. chunk_document

**Description**: Split a document into chunks suitable for LLM/RAG processing.

**Purpose**: Enables AI agents to break large documents into manageable pieces that fit within LLM context windows while preserving semantic meaning and structure.

**Parameters**:
- `file_path` (string, required): Path to local file or S3 URL
- `strategy` (string, optional): Chunking strategy (auto, hierarchical, sliding_window, content_aware, semantic). Default: auto
- `framework_hint` (string, optional): Framework to use
- `max_chunk_size` (integer, optional): Maximum chunk size in tokens
- `overlap` (integer, optional): Overlap between chunks in tokens

**Returns**:
```json
{
  "file_path": "path/to/file.pdf",
  "chunks": [
    {
      "chunk_id": "chunk_0",
      "content": "...",
      "metadata": {
        "start_position": 0,
        "end_position": 500,
        "token_count": 120,
        "element_path": "document > section[0] > paragraph[0]"
      }
    },
    ...
  ],
  "chunk_count": 15
}
```

**Error Cases**:
- File not found
- Unsupported file format
- Invalid strategy
- Required framework not installed

**Example Use Case**:
An AI agent preparing documents for ingestion into a vector database for RAG applications.

---

### 3. analyze_and_chunk

**Description**: Analyze and chunk a document in one operation.

**Purpose**: Convenience tool that combines analysis and chunking, useful when both operations are needed and you want to avoid two separate calls.

**Parameters**:
- `file_path` (string, required): Path to local file or S3 URL
- `strategy` (string, optional): Chunking strategy. Default: auto
- `framework_hint` (string, optional): Framework to use
- `max_chunk_size` (integer, optional): Maximum chunk size in tokens
- `overlap` (integer, optional): Overlap between chunks in tokens

**Returns**:
```json
{
  "file_path": "path/to/file.xml",
  "analysis": { ... },
  "chunks": [ ... ],
  "chunk_count": 8
}
```

**Error Cases**: Same as analyze_document and chunk_document combined

**Example Use Case**:
An AI agent that needs to understand document structure and immediately chunk it for processing.

---

### 4. batch_analyze

**Description**: Analyze multiple documents in batch.

**Purpose**: Enables efficient processing of multiple documents, returning results for all files even if some fail.

**Parameters**:
- `file_paths` (array of strings, required): List of local file paths or S3 URLs
- `framework_hint` (string, optional): Framework to use for all files

**Returns**:
```json
{
  "results": [
    {
      "file_path": "file1.xml",
      "status": "success",
      "result": { ... },
      "error": null
    },
    {
      "file_path": "file2.pdf",
      "status": "error",
      "result": null,
      "error": "Unsupported format"
    }
  ],
  "total": 2,
  "successful": 1,
  "failed": 1
}
```

**Error Cases**:
- Empty file list
- Invalid file paths
- Batch size exceeds limit (configurable, default 10)

**Example Use Case**:
An AI agent processing an entire directory of documents to build a catalog or index.

---

### 5. batch_chunk

**Description**: Chunk multiple documents in batch.

**Purpose**: Enables efficient chunking of multiple documents for bulk RAG ingestion.

**Parameters**:
- `file_paths` (array of strings, required): List of local file paths or S3 URLs
- `strategy` (string, optional): Chunking strategy for all files. Default: auto
- `framework_hint` (string, optional): Framework to use for all files

**Returns**:
```json
{
  "results": [
    {
      "file_path": "file1.xml",
      "status": "success",
      "chunks": [ ... ],
      "chunk_count": 5,
      "error": null
    },
    ...
  ],
  "total": 3,
  "successful": 2,
  "failed": 1
}
```

**Error Cases**: Same as batch_analyze

**Example Use Case**:
An AI agent preparing multiple documents for RAG ingestion in a single operation.

---

### 6. convert_format

**Description**: Convert document to different output format.

**Purpose**: Enables extraction and conversion of document content to formats suitable for different purposes (JSON for structured processing, Markdown for readability, plain text for simple analysis).

**Parameters**:
- `file_path` (string, required): Path to local file or S3 URL
- `output_format` (string, required): Output format (json, markdown, text)

**Returns**:
```json
{
  "file_path": "path/to/file.xml",
  "output_format": "markdown",
  "content": "# Document Analysis\n\n**Type:** SCAP Security Report\n\n..."
}
```

**Error Cases**:
- File not found
- Unsupported output format
- Conversion failed

**Example Use Case**:
An AI agent converting technical XML documents to Markdown for inclusion in a report or documentation.

---

### 7. list_formats

**Description**: Get list of supported file formats by framework.

**Purpose**: Enables AI agents to discover what file types can be processed and which framework handles each type.

**Parameters**: None

**Returns**:
```json
{
  "formats": {
    "xml": [".xml"],
    "docling": [".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".tiff"],
    "document": [".py", ".js", ".ts", ".md", ".txt", ".yaml", ...],
    "data": [".csv", ".parquet", ".db", ".sqlite"]
  }
}
```

**Error Cases**: None (always succeeds)

**Example Use Case**:
An AI agent checking if it can process a specific file type before attempting analysis.

---

### 8. list_strategies

**Description**: Get list of supported chunking strategies.

**Purpose**: Enables AI agents to discover available chunking strategies and choose the appropriate one for their use case.

**Parameters**: None

**Returns**:
```json
{
  "strategies": [
    "auto",
    "hierarchical",
    "sliding_window",
    "content_aware",
    "semantic"
  ]
}
```

**Error Cases**: None (always succeeds)

**Example Use Case**:
An AI agent discovering which chunking strategies are available before processing a document.

---

### 9. check_health

**Description**: Check service health and framework availability.

**Purpose**: Enables AI agents to verify the service is running and which frameworks are currently available (installed).

**Parameters**: None

**Returns**:
```json
{
  "status": "healthy",
  "available_frameworks": {
    "xml": true,
    "docling": true,
    "document": true,
    "data": false
  }
}
```

**Error Cases**: None (always returns current state)

**Example Use Case**:
An AI agent checking which frameworks are available before attempting to process specific document types.

---

## Implementation Guidelines

### Transport Protocol

**Recommended**: HTTP (streamable-http) for production deployment
- Allows service to run in OpenShift
- Enables external access via routes
- Supports standard HTTP infrastructure (load balancers, monitoring, etc.)

**Alternative**: STDIO for local development/testing
- Simpler for command-line usage
- Good for development and testing
- Not suitable for OpenShift deployment

### Authentication

**OpenShift Deployment**: Use OpenShift OAuth2/OIDC
- Service account tokens
- OAuth proxy for external access
- Integrate with OpenShift RBAC

**Development**: Optional basic auth or no auth for local testing

### Configuration

The MCP server should respect these environment variables:
- `APP_MAX_UPLOAD_SIZE_MB`: Maximum file size for uploads (default: 100)
- `APP_MAX_S3_SIZE_MB`: Maximum file size for S3 downloads (default: 1000)
- `APP_S3_ENDPOINT`: S3 endpoint URL
- `AWS_ACCESS_KEY_ID`: S3 access key
- `AWS_SECRET_ACCESS_KEY`: S3 secret key

### Error Response Format

All errors should follow this structure:
```json
{
  "error": "Brief error message",
  "error_type": "ValidationError | FrameworkNotInstalledError | UnsupportedFormatError | ...",
  "detail": "Detailed explanation of what went wrong",
  "suggestion": "How to fix it (when applicable)"
}
```

### Logging

Tools should log:
- Tool invocations with parameters (excluding sensitive data)
- Processing time for each operation
- Errors with full stack traces
- File sizes and types processed

### Performance Considerations

- Implement timeouts for long-running operations
- Consider streaming responses for large chunk sets
- Add progress indicators for batch operations
- Cache analysis results when safe to do so

---

## Testing the MCP Server

### Test Cases

1. **Basic Analysis**: Analyze a simple XML file
2. **Chunking**: Chunk a PDF with different strategies
3. **Batch Processing**: Analyze 5 files at once
4. **Error Handling**: Try unsupported format, missing file
5. **S3 Integration**: Analyze file from S3 URL
6. **Framework Detection**: Auto-detect various file types
7. **Format Conversion**: Convert XML to Markdown

### Integration with AI Agents

The MCP server should work seamlessly with:
- LangChain agents (via MCP integration)
- LlamaStack agents
- Claude Desktop (for development/testing)
- Any MCP-compatible client

---

## Deployment Notes

**Container Image**: The MCP server should be packaged in the same container as the REST API but with a different command/entry point.

**OpenShift Service**: Expose both REST API and MCP server through same service but different paths:
- `/api/v1/*` ’ REST API
- `/mcp` ’ MCP Server (streamable-http endpoint)

**Resource Limits**: Same as REST API (2Gi memory request, 4Gi limit)

**Health Checks**: MCP server should respond to standard health endpoints

---

## Future Enhancements

Consider adding these tools in future versions:
- `extract_entities`: Extract named entities from documents
- `summarize_document`: Generate document summaries
- `compare_documents`: Compare two documents for differences
- `validate_schema`: Validate document against schema
- `get_statistics`: Get statistical analysis of document

---

## Questions for Implementation

When implementing the MCP server, consider:
1. Should tools accept both local paths and S3 URLs, or separate tools for each?
2. Should batch operations have size limits? What's reasonable?
3. Should there be a tool to stream large chunk sets vs. returning all at once?
4. Should there be progress callbacks for long-running operations?
5. How should the server handle multiple concurrent requests?

---

## References

- **MCP Specification**: https://modelcontextprotocol.io/
- **FastMCP v2 Documentation**: https://github.com/jlowin/fastmcp
- **unified-document-analysis**: https://github.com/rdwj/unified-document-analysis
- **REST API Implementation**: See `src/api/rest.py` in this repository
