# MemFuse Buffer Architecture

## Overview

The MemFuse Buffer system provides intelligent message buffering and batch processing capabilities for high-throughput conversation management. This document outlines the architectural design, component interactions, and implementation strategies for the buffer subsystem.

## Core Architecture

### System Components

```mermaid
graph TB
    subgraph "Client Layer"
        A[Client Request] --> B[API Gateway]
    end
    
    subgraph "Service Layer"
        B --> C[BufferService]
        C --> D[WriteBuffer]
        C --> E[QueryBuffer]
    end
    
    subgraph "Buffer Components"
        D --> F[RoundBuffer]
        D --> G[HybridBuffer]
        E --> H[Query Cache]
        E --> I[Result Processor]
    end
    
    subgraph "Storage Layer"
        F --> J[Memory Storage]
        G --> K[Chunk Storage]
        G --> L[Vector Store]
        G --> M[Keyword Store]
    end
    
    subgraph "Processing Layer"
        K --> N[MemoryService]
        N --> O[ChunkStrategy]
        O --> P[Persistent Storage]
    end
```

### Architectural Principles

```mermaid
graph LR
    subgraph "Design Principles"
        A[Unified Entry Point<br/>WriteBuffer] --> B[Component Isolation<br/>Clear Boundaries]
        B --> C[Async Processing<br/>Non-blocking Operations]
        C --> D[Threshold-based<br/>Intelligent Batching]
        D --> E[Configurable<br/>Flexible Parameters]
    end
```

## Buffer Components

### WriteBuffer - Unified Entry Point

```mermaid
graph TB
    subgraph "WriteBuffer Architecture"
        A[Client Messages] --> B[WriteBuffer]
        B --> C[RoundBuffer Management]
        B --> D[HybridBuffer Management]
        B --> E[Component Coordination]
        
        C --> F[Token-based FIFO]
        D --> G[Dual-format Storage]
        E --> H[Transfer Orchestration]
        
        F --> I[Automatic Transfer]
        G --> J[Chunk Processing]
        H --> K[Storage Handlers]
    end
```

**Key Responsibilities**:
- Unified message entry point
- Component lifecycle management
- Transfer coordination between buffers
- Statistics collection and monitoring

**Interface Design**:
```python
class WriteBuffer:
    async def add(self, messages: MessageList, session_id: str = None) -> Dict[str, Any]
    async def add_batch(self, message_batch_list: MessageBatchList, session_id: str = None) -> Dict[str, Any]
    
    def get_round_buffer(self) -> RoundBuffer
    def get_hybrid_buffer(self) -> HybridBuffer
    
    async def flush_all(self) -> Dict[str, Any]
    def get_stats(self) -> Dict[str, Any]
```

### RoundBuffer - Token-based FIFO

```mermaid
graph LR
    subgraph "RoundBuffer Flow"
        A[MessageList Input] --> B[Token Counting]
        B --> C{Token Limit<br/>Exceeded?}
        C -->|No| D[Add to Buffer]
        C -->|Yes| E[Transfer Trigger]
        
        D --> F[Accumulate Messages]
        E --> G[Transfer to HybridBuffer]
        G --> H[Clear Buffer]
        
        F --> I{Size Limit<br/>Exceeded?}
        I -->|Yes| E
        I -->|No| F
    end
```

**Configuration Parameters**:
- `max_tokens`: Token threshold for transfer (default: 800)
- `max_size`: Maximum number of rounds (default: 5)
- `token_model`: Model for token counting (default: "gpt-4o-mini")

**Transfer Triggers**:
1. **Token Limit**: When accumulated tokens exceed threshold
2. **Size Limit**: When number of rounds exceeds maximum
3. **Manual Flush**: Explicit transfer request

### HybridBuffer - Dual-format Storage

```mermaid
graph TB
    subgraph "HybridBuffer Architecture"
        A[RoundBuffer Transfer] --> B[Dual Storage]
        B --> C[Chunk Format]
        B --> D[Round Format]
        
        C --> E[ChunkStrategy Processing]
        D --> F[Original Round Preservation]
        
        E --> G[Embedding Generation]
        F --> H[Metadata Enhancement]
        
        G --> I[Vector Storage Ready]
        H --> J[Query Processing Ready]
        
        I --> K{FIFO Limit<br/>Exceeded?}
        J --> K
        K -->|Yes| L[Flush to Storage]
        K -->|No| M[Memory Buffer]
    end
```

**Storage Formats**:
- **Chunks**: Processed through ChunkStrategy for semantic search
- **Rounds**: Original message structure for context preservation

**FIFO Management**:
- Automatic eviction when size limit exceeded
- Configurable flush behavior (manual/automatic)
- Preservation of recent data for fast access

### QueryBuffer - Intelligent Query Processing

```mermaid
graph TB
    subgraph "QueryBuffer Architecture"
        A[Query Request] --> B[Cache Check]
        B -->|Hit| C[Cached Results]
        B -->|Miss| D[Multi-source Query]
        
        D --> E[Storage Query]
        D --> F[HybridBuffer Query]
        
        E --> G[Storage Results]
        F --> H[Buffer Results]
        
        G --> I[Result Combination]
        H --> I
        I --> J[Sorting & Ranking]
        J --> K[Cache Update]
        K --> L[Final Results]
        C --> L
    end
```

**Query Sources**:
1. **Persistent Storage**: Long-term data via MemoryService
2. **HybridBuffer**: Recent data in memory buffer
3. **Cache**: Previously computed results

**Sorting Options**:
- `score`: Relevance-based ranking (default)
- `timestamp`: Temporal ordering

## Data Flow Architecture

### Message Processing Pipeline

```mermaid
sequenceDiagram
    participant Client
    participant BufferService
    participant WriteBuffer
    participant RoundBuffer
    participant HybridBuffer
    participant MemoryService
    participant Storage

    Client->>BufferService: add(MessageList)
    BufferService->>WriteBuffer: add(MessageList)
    WriteBuffer->>RoundBuffer: add(MessageList)
    
    Note over RoundBuffer: Accumulate until threshold
    
    RoundBuffer->>RoundBuffer: check_thresholds()
    RoundBuffer->>HybridBuffer: transfer_and_clear()
    HybridBuffer->>HybridBuffer: process_rounds()
    
    Note over HybridBuffer: FIFO limit exceeded
    
    HybridBuffer->>MemoryService: flush_to_storage()
    MemoryService->>Storage: persist_data()
    
    Storage-->>MemoryService: Success
    MemoryService-->>HybridBuffer: Success
    HybridBuffer-->>WriteBuffer: Transfer Complete
    WriteBuffer-->>BufferService: Success
    BufferService-->>Client: 200 OK
```

### Query Processing Pipeline

```mermaid
sequenceDiagram
    participant Client
    participant BufferService
    participant QueryBuffer
    participant HybridBuffer
    participant MemoryService
    participant Cache

    Client->>BufferService: query(text, top_k)
    BufferService->>QueryBuffer: query(text, top_k, hybrid_buffer)
    QueryBuffer->>Cache: check_cache(query_hash)
    
    alt Cache Hit
        Cache-->>QueryBuffer: cached_results
    else Cache Miss
        par Parallel Query
            QueryBuffer->>MemoryService: query_storage(text, top_k)
            QueryBuffer->>HybridBuffer: query_buffer(text, top_k)
        end
        
        MemoryService-->>QueryBuffer: storage_results
        HybridBuffer-->>QueryBuffer: buffer_results
        
        QueryBuffer->>QueryBuffer: combine_and_sort()
        QueryBuffer->>Cache: update_cache(query, results)
    end
    
    QueryBuffer-->>BufferService: final_results
    BufferService-->>Client: query_response
```

## Configuration Architecture

### Hierarchical Configuration

```mermaid
graph TB
    subgraph "Configuration Hierarchy"
        A[config/buffer/default.yaml] --> B[Buffer Configuration]
        B --> C[RoundBuffer Config]
        B --> D[HybridBuffer Config]
        B --> E[QueryBuffer Config]
        
        C --> F[max_tokens: 800<br/>max_size: 5<br/>token_model: gpt-4o-mini]
        D --> G[max_size: 5<br/>chunk_strategy: message<br/>embedding_model: all-MiniLM-L6-v2]
        E --> H[max_size: 15<br/>cache_size: 100<br/>default_sort_by: score]
    end
```

### Configuration Schema

```yaml
buffer:
  enabled: true
  
  # RoundBuffer configuration
  round_buffer:
    max_tokens: 800               # Token threshold for transfer
    max_size: 5                   # Maximum rounds before transfer
    token_model: "gpt-4o-mini"    # Model for token counting
  
  # HybridBuffer configuration
  hybrid_buffer:
    max_size: 5                   # FIFO buffer size
    chunk_strategy: "message"     # Chunking strategy
    embedding_model: "all-MiniLM-L6-v2"  # Embedding model
  
  # QueryBuffer configuration
  query:
    max_size: 15                  # Maximum results per query
    cache_size: 100               # Query cache size
    default_sort_by: "score"      # Default sorting method
    default_order: "desc"         # Default sort order
```

## Performance Architecture

### Throughput Optimization

```mermaid
graph TB
    subgraph "Performance Optimizations"
        A[Batch Processing] --> B[Reduced I/O Operations]
        C[Async Operations] --> D[Non-blocking Processing]
        E[Memory Buffering] --> F[Fast Access Patterns]
        G[Intelligent Caching] --> H[Query Acceleration]
        
        B --> I[High Throughput]
        D --> I
        F --> I
        H --> I
    end
```

### Latency Characteristics

| Operation | Latency | Description |
|-----------|---------|-------------|
| Message Add | <5ms | Add to RoundBuffer |
| Buffer Transfer | <50ms | RoundBuffer → HybridBuffer |
| Storage Flush | <200ms | HybridBuffer → Persistent Storage |
| Query (Cached) | <10ms | Cache hit response |
| Query (Cold) | <100ms | Multi-source query |

### Memory Management

```mermaid
graph LR
    subgraph "Memory Usage Pattern"
        A[RoundBuffer<br/>~1MB] --> B[HybridBuffer<br/>~5MB]
        B --> C[QueryCache<br/>~2MB]
        C --> D[Total Memory<br/>~8MB]
    end
    
    subgraph "Garbage Collection"
        E[FIFO Eviction] --> F[Automatic Cleanup]
        F --> G[Memory Efficiency]
    end
```

## Error Handling & Resilience

### Fault Tolerance

```mermaid
graph TB
    subgraph "Error Handling Strategy"
        A[Component Failure] --> B{Failure Type}
        B -->|Storage Error| C[Retry with Backoff]
        B -->|Memory Error| D[Graceful Degradation]
        B -->|Network Error| E[Circuit Breaker]
        
        C --> F[Error Recovery]
        D --> G[Fallback Mode]
        E --> H[Service Protection]
        
        F --> I[System Resilience]
        G --> I
        H --> I
    end
```

### Recovery Mechanisms

1. **Automatic Retry**: Transient failure recovery
2. **Circuit Breaker**: Prevent cascade failures
3. **Graceful Degradation**: Reduced functionality under stress
4. **Data Persistence**: No data loss during failures

## Monitoring & Observability

### Metrics Collection

```mermaid
graph TB
    subgraph "Buffer Metrics"
        A[WriteBuffer Stats] --> B[total_writes<br/>total_transfers<br/>component_health]
        C[RoundBuffer Stats] --> D[current_tokens<br/>round_count<br/>transfer_triggers]
        E[HybridBuffer Stats] --> F[chunk_count<br/>round_count<br/>flush_operations]
        G[QueryBuffer Stats] --> H[cache_hits<br/>cache_misses<br/>query_latency]
    end
```

### Health Indicators

| Metric | Healthy Range | Alert Threshold |
|--------|---------------|-----------------|
| Transfer Rate | 10-100/min | >500/min |
| Memory Usage | <50MB | >100MB |
| Query Latency | <100ms | >500ms |
| Cache Hit Rate | >80% | <50% |

## Integration Patterns

### Service Integration

```mermaid
graph TB
    subgraph "Service Integration"
        A[BufferService] --> B[WriteBuffer Integration]
        A --> C[QueryBuffer Integration]
        
        B --> D[Unified Add Interface]
        C --> E[Unified Query Interface]
        
        D --> F["Component Access<br/>get_round_buffer()<br/>get_hybrid_buffer()"]
        E --> G[Multi-source Querying<br/>Storage + Buffer]
    end
```

### API Compatibility

```python
# BufferService maintains full API compatibility
class BufferService:
    async def add(self, messages: MessageList, session_id: str = None) -> Dict[str, Any]
    async def add_batch(self, message_batch_list: MessageBatchList, session_id: str = None) -> Dict[str, Any]
    async def query(self, query: str, top_k: int = 10, **kwargs) -> Dict[str, Any]
    async def get_messages_by_session(self, session_id: str, buffer_only: bool = None, **kwargs) -> Dict[str, Any]
```

## Design Benefits

### Architectural Advantages

1. **Unified Entry Point**: WriteBuffer provides clean abstraction
2. **Component Isolation**: Clear separation of concerns
3. **Configurable Behavior**: Flexible parameter tuning
4. **Performance Optimization**: Intelligent batching and caching
5. **Fault Tolerance**: Robust error handling and recovery

### Scalability Features

- **Horizontal Scaling**: Stateless component design
- **Memory Efficiency**: FIFO-based memory management
- **Load Distribution**: Async processing capabilities
- **Resource Optimization**: Intelligent threshold management

## Future Enhancements

### Short-term Improvements

- **Dynamic Thresholds**: Adaptive threshold adjustment
- **Advanced Caching**: Multi-level cache hierarchy
- **Compression**: Memory usage optimization

### Long-term Vision

- **Distributed Buffering**: Multi-node buffer coordination
- **ML-based Optimization**: Intelligent parameter tuning
- **Stream Processing**: Real-time data processing capabilities

This buffer architecture provides a robust, scalable foundation for high-throughput message processing in the MemFuse system, ensuring optimal performance while maintaining data integrity and system reliability.
