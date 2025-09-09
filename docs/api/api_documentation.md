# MemFuse API Documentation

This comprehensive API documentation covers all endpoints, request/response formats, authentication, error handling, and integration examples for the MemFuse system.

## Table of Contents

1. [API Overview](#api-overview)
2. [Authentication](#authentication)
3. [Core Endpoints](#core-endpoints)
4. [Memory Layer APIs](#memory-layer-apis)
5. [Gateway APIs](#gateway-apis)
6. [Monitoring APIs](#monitoring-apis)
7. [Error Handling](#error-handling)
8. [Rate Limiting](#rate-limiting)
9. [Integration Examples](#integration-examples)

## API Overview

### Base URL
```
Production: https://api.memfuse.com/v1
Development: http://localhost:8000/v1
```

### API Characteristics
- **Protocol**: REST API with JSON payloads
- **Authentication**: JWT-based authentication
- **Rate Limiting**: Configurable per endpoint
- **Versioning**: URL-based versioning (v1, v2, etc.)
- **Content Type**: `application/json`
- **Character Encoding**: UTF-8

### Response Format
All API responses follow a consistent format:

```json
{
  "success": true,
  "data": {
    // Response data
  },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123456789",
    "processing_time_ms": 45,
    "version": "1.0.0"
  },
  "debug": {
    // Debug information (only in development)
    "cache_stats": {
      "hits": 15,
      "misses": 3
    },
    "performance": {
      "gateway_duration_ms": 12,
      "buffer_duration_ms": 8,
      "database_duration_ms": 25
    }
  }
}
```

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "content",
      "reason": "Content cannot be empty"
    }
  },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_123456789"
  }
}
```

## Authentication

### JWT Authentication

MemFuse uses JWT (JSON Web Tokens) for authentication. Include the token in the Authorization header:

```http
Authorization: Bearer <jwt_token>
```

### Obtaining a Token

**Endpoint**: `POST /auth/login`

**Request**:
```json
{
  "username": "user@example.com",
  "password": "secure_password"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "expires_in": 3600,
    "refresh_token": "refresh_token_here"
  }
}
```

### Token Refresh

**Endpoint**: `POST /auth/refresh`

**Request**:
```json
{
  "refresh_token": "refresh_token_here"
}
```

## Core Endpoints

### Health Check

**Endpoint**: `GET /health`

**Description**: Check system health and status

**Response**:
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00Z",
    "version": "1.0.0",
    "uptime_seconds": 86400,
    "checks": {
      "database": "healthy",
      "redis": "healthy",
      "memory": "healthy",
      "disk": "healthy"
    }
  }
}
```

### System Information

**Endpoint**: `GET /info`

**Description**: Get system information and capabilities

**Response**:
```json
{
  "success": true,
  "data": {
    "system": {
      "name": "MemFuse",
      "version": "1.0.0",
      "environment": "production"
    },
    "capabilities": {
      "memory_layers": ["M0", "M1", "M2", "M3", "MSMG"],
      "filters": ["sensitive_word", "pii_redaction", "length_limit"],
      "semantic_validation": true,
      "performance_monitoring": true
    },
    "limits": {
      "max_content_length": 1000000,
      "max_requests_per_minute": 1000,
      "max_concurrent_requests": 100
    }
  }
}
```

## Memory Layer APIs

### Store Memory

**Endpoint**: `POST /memory/store`

**Description**: Store content in the appropriate memory layer

**Request**:
```json
{
  "content": "This is the content to store",
  "metadata": {
    "user_id": "user_123",
    "session_id": "session_456",
    "content_type": "text",
    "priority": "normal",
    "tags": ["important", "work"]
  },
  "layer_hint": "M1",
  "processing_options": {
    "enable_semantic_analysis": true,
    "enable_quality_scoring": true,
    "enable_deduplication": true
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "item_id": "item_789",
    "layer_assigned": "M1",
    "processing_results": {
      "semantic_score": 0.85,
      "quality_score": 0.92,
      "duplicate_detected": false,
      "content_hash": "sha256:abc123..."
    },
    "storage_metadata": {
      "created_at": "2024-01-15T10:30:00Z",
      "size_bytes": 1024,
      "encoding": "utf-8"
    }
  }
}
```

### Query Memory

**Endpoint**: `POST /memory/query`

**Description**: Query across memory layers for relevant content

**Request**:
```json
{
  "query": "Find information about machine learning",
  "parameters": {
    "top_k": 10,
    "similarity_threshold": 0.7,
    "layers": ["M1", "M2", "M3"],
    "filters": {
      "user_id": "user_123",
      "date_range": {
        "start": "2024-01-01T00:00:00Z",
        "end": "2024-01-15T23:59:59Z"
      },
      "tags": ["work", "research"]
    }
  },
  "options": {
    "include_metadata": true,
    "include_scores": true,
    "enable_reranking": true
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "item_id": "item_789",
        "content": "Machine learning is a subset of artificial intelligence...",
        "layer": "M2",
        "similarity_score": 0.95,
        "quality_score": 0.88,
        "metadata": {
          "user_id": "user_123",
          "created_at": "2024-01-10T15:30:00Z",
          "tags": ["ml", "ai", "research"]
        }
      }
    ],
    "query_metadata": {
      "total_results": 25,
      "results_returned": 10,
      "query_time_ms": 45,
      "layers_searched": ["M1", "M2", "M3"],
      "cache_hit": true
    }
  }
}
```

### Retrieve Memory Item

**Endpoint**: `GET /memory/items/{item_id}`

**Description**: Retrieve a specific memory item by ID

**Response**:
```json
{
  "success": true,
  "data": {
    "item_id": "item_789",
    "content": "Full content of the memory item...",
    "layer": "M1",
    "metadata": {
      "user_id": "user_123",
      "session_id": "session_456",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T10:30:00Z",
      "content_type": "text",
      "size_bytes": 1024,
      "tags": ["important", "work"]
    },
    "processing_info": {
      "semantic_score": 0.85,
      "quality_score": 0.92,
      "content_hash": "sha256:abc123...",
      "processing_version": "1.0.0"
    }
  }
}
```

### Update Memory Item

**Endpoint**: `PUT /memory/items/{item_id}`

**Description**: Update an existing memory item

**Request**:
```json
{
  "content": "Updated content...",
  "metadata": {
    "tags": ["updated", "important"],
    "priority": "high"
  },
  "processing_options": {
    "reprocess_semantic": true,
    "update_quality_score": true
  }
}
```

### Delete Memory Item

**Endpoint**: `DELETE /memory/items/{item_id}`

**Description**: Delete a memory item

**Response**:
```json
{
  "success": true,
  "data": {
    "item_id": "item_789",
    "deleted_at": "2024-01-15T10:30:00Z",
    "layer": "M1"
  }
}
```

## Gateway APIs

### Filter Configuration

**Endpoint**: `GET /gateway/filters`

**Description**: Get current filter configuration

**Response**:
```json
{
  "success": true,
  "data": {
    "inbound_filters": [
      {
        "name": "request_validator",
        "enabled": true,
        "config": {
          "max_content_length": 1000000,
          "required_fields": ["content"]
        }
      },
      {
        "name": "input_sanitizer",
        "enabled": true,
        "config": {
          "strip_html": true,
          "normalize_unicode": true
        }
      }
    ],
    "outbound_filters": [
      {
        "name": "sensitive_word",
        "enabled": true,
        "config": {
          "action": "mask",
          "patterns_file": "sensitive_words.txt"
        }
      }
    ]
  }
}
```

### Update Filter Configuration

**Endpoint**: `PUT /gateway/filters/{filter_name}`

**Description**: Update filter configuration

**Request**:
```json
{
  "enabled": true,
  "config": {
    "action": "flag",
    "sensitivity_level": "high"
  }
}
```

### Filter Statistics

**Endpoint**: `GET /gateway/filters/stats`

**Description**: Get filter performance statistics

**Response**:
```json
{
  "success": true,
  "data": {
    "cache_stats": {
      "regex_cache": {
        "hits": 15420,
        "misses": 234,
        "hit_rate_percent": 98.5,
        "cache_size": 856,
        "evictions": 12
      },
      "content_cache": {
        "hits": 8934,
        "misses": 1205,
        "hit_rate_percent": 88.1,
        "cache_size": 4567
      }
    },
    "filter_performance": {
      "sensitive_word": {
        "total_processed": 10000,
        "avg_duration_ms": 0.89,
        "matches_found": 45
      },
      "pii_redaction": {
        "total_processed": 10000,
        "avg_duration_ms": 2.3,
        "items_redacted": 123
      }
    }
  }
}
```

## Monitoring APIs

### System Metrics

**Endpoint**: `GET /metrics`

**Description**: Get Prometheus-format metrics

**Response**: (Prometheus format)
```
# HELP memfuse_requests_total Total number of requests
# TYPE memfuse_requests_total counter
memfuse_requests_total{method="POST",endpoint="/memory/store"} 1234

# HELP memfuse_request_duration_seconds Request duration in seconds
# TYPE memfuse_request_duration_seconds histogram
memfuse_request_duration_seconds_bucket{le="0.1"} 8934
memfuse_request_duration_seconds_bucket{le="0.5"} 9876
memfuse_request_duration_seconds_bucket{le="1.0"} 9999
memfuse_request_duration_seconds_bucket{le="+Inf"} 10000
```

### Performance Statistics

**Endpoint**: `GET /stats/performance`

**Description**: Get detailed performance statistics

**Response**:
```json
{
  "success": true,
  "data": {
    "request_stats": {
      "total_requests": 50000,
      "requests_per_second": 125.5,
      "avg_response_time_ms": 45.2,
      "p95_response_time_ms": 120.0,
      "p99_response_time_ms": 250.0
    },
    "cache_performance": {
      "overall_hit_rate": 0.92,
      "regex_cache_performance": {
        "hit_time_avg_us": 0.89,
        "miss_time_avg_us": 47.06,
        "improvement_factor": 52.6
      }
    },
    "memory_usage": {
      "current_mb": 512,
      "peak_mb": 678,
      "gc_collections": 234,
      "memory_growth_rate_mb_per_hour": 0.0
    }
  }
}
```

## Error Handling

### Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `VALIDATION_ERROR` | Request validation failed | 400 |
| `AUTHENTICATION_ERROR` | Authentication failed | 401 |
| `AUTHORIZATION_ERROR` | Insufficient permissions | 403 |
| `NOT_FOUND` | Resource not found | 404 |
| `RATE_LIMIT_EXCEEDED` | Rate limit exceeded | 429 |
| `INTERNAL_ERROR` | Internal server error | 500 |
| `SERVICE_UNAVAILABLE` | Service temporarily unavailable | 503 |

### Error Response Examples

**Validation Error**:
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "field": "content",
      "reason": "Content exceeds maximum length of 1000000 characters",
      "provided_length": 1500000
    }
  }
}
```

**Rate Limit Error**:
```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 1000,
      "window": "1 minute",
      "retry_after": 45
    }
  }
}
```

## Rate Limiting

### Rate Limit Headers

All responses include rate limiting headers:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248600
X-RateLimit-Window: 60
```

### Rate Limit Configuration

Different endpoints have different rate limits:

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/memory/store` | 100/min | 1 minute |
| `/memory/query` | 500/min | 1 minute |
| `/memory/items/*` | 1000/min | 1 minute |
| `/gateway/*` | 200/min | 1 minute |
| `/metrics` | 60/min | 1 minute |

## Integration Examples

### Python Client Example

```python
import requests
import json

class MemFuseClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def store_memory(self, content, metadata=None):
        """Store content in MemFuse"""
        payload = {
            'content': content,
            'metadata': metadata or {}
        }
        
        response = requests.post(
            f'{self.base_url}/memory/store',
            headers=self.headers,
            json=payload
        )
        
        return response.json()
    
    def query_memory(self, query, top_k=10):
        """Query MemFuse for relevant content"""
        payload = {
            'query': query,
            'parameters': {'top_k': top_k}
        }
        
        response = requests.post(
            f'{self.base_url}/memory/query',
            headers=self.headers,
            json=payload
        )
        
        return response.json()

# Usage example
client = MemFuseClient('https://api.memfuse.com/v1', 'your_api_key')

# Store memory
result = client.store_memory(
    content="Machine learning is transforming industries...",
    metadata={'tags': ['ml', 'ai'], 'category': 'research'}
)

# Query memory
results = client.query_memory("machine learning applications")
```

### JavaScript Client Example

```javascript
class MemFuseClient {
    constructor(baseUrl, apiKey) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        };
    }
    
    async storeMemory(content, metadata = {}) {
        const response = await fetch(`${this.baseUrl}/memory/store`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                content: content,
                metadata: metadata
            })
        });
        
        return await response.json();
    }
    
    async queryMemory(query, topK = 10) {
        const response = await fetch(`${this.baseUrl}/memory/query`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                query: query,
                parameters: { top_k: topK }
            })
        });
        
        return await response.json();
    }
}

// Usage
const client = new MemFuseClient('https://api.memfuse.com/v1', 'your_api_key');

// Store and query
client.storeMemory('AI research findings...', {tags: ['ai', 'research']})
    .then(result => console.log('Stored:', result));

client.queryMemory('artificial intelligence')
    .then(results => console.log('Results:', results));
```

This comprehensive API documentation provides complete coverage of all MemFuse endpoints, authentication, error handling, and integration examples for developers.
