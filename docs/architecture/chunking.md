# MemFuse Chunking Architecture

## Overview

The MemFuse chunking system transforms conversation-based messages into retrievable chunks for efficient storage and semantic search. This document outlines the architectural design, component interactions, and implementation strategies for the chunking pipeline.

## Core Architecture

### System Components

```mermaid
graph TB
    subgraph "Input Layer"
        A[Client Request] --> B[MessageList]
        B --> C[MessageBatchList]
    end
    
    subgraph "Processing Layer"
        C --> D[ChunkStrategy]
        D --> E[ChunkData Generation]
        E --> F[Metadata Enhancement]
    end
    
    subgraph "Storage Layer"
        F --> G[VectorStore]
        F --> H[KeywordStore]
        F --> I[GraphStore]
    end
    
    subgraph "Retrieval Layer"
        G --> J[Semantic Search]
        H --> K[Keyword Search]
        I --> L[Graph Traversal]
        J --> M[Unified Results]
        K --> M
        L --> M
    end
```

### Data Flow Pipeline with HybridBuffer Integration

```mermaid
sequenceDiagram
    participant Client
    participant BufferService
    participant RoundBuffer
    participant HybridBuffer
    participant ChunkStrategy
    participant MemoryService
    participant SQLite
    participant Qdrant

    Client->>BufferService: add(MessageList)
    BufferService->>RoundBuffer: add(MessageList)

    Note over RoundBuffer: Accumulate until token/size threshold

    RoundBuffer->>HybridBuffer: transfer_and_clear()

    Note over HybridBuffer: Immediate Processing Phase
    HybridBuffer->>ChunkStrategy: create_chunks(rounds)
    ChunkStrategy-->>HybridBuffer: List[ChunkData]
    HybridBuffer->>HybridBuffer: generate embeddings
    HybridBuffer->>HybridBuffer: store to VectorCache
    HybridBuffer->>HybridBuffer: add rounds to RoundQueue

    Note over HybridBuffer: Queue size check (max_size = 5)

    alt Queue Full
        Note over HybridBuffer: Batch Write Phase
        par Sequential Storage
            HybridBuffer->>MemoryService: write rounds to SQLite
            MemoryService->>SQLite: store with updated_at refresh
        and
            HybridBuffer->>Qdrant: write pre-calculated embeddings
        end

        Note over HybridBuffer: Clear all queues after successful write
        HybridBuffer->>HybridBuffer: clear RoundQueue + VectorCache
    else Queue Not Full
        Note over HybridBuffer: Keep data in memory for fast retrieval
    end

    HybridBuffer-->>BufferService: Success Response
    BufferService-->>Client: 200 OK
```

## Data Structures

### Core Types

```python
# Type definitions
MessageList = List[Dict[str, Any]]      # List of conversation messages
MessageBatchList = List[MessageList]    # Batch of message lists

@dataclass
class ChunkData:
    """Unified chunk representation across all stores."""
    content: str                        # Text content of the chunk
    chunk_id: str                      # Unique identifier
    metadata: Dict[str, Any]           # Enhanced metadata
```

### Enhanced Metadata Structure

```python
# Comprehensive chunk metadata
chunk_metadata = {
    # Strategy Information
    "strategy": "contextual",          # Chunking strategy used
    "message_count": 2,                # Number of messages in chunk
    "source": "message_list",          # Source type

    # Context Information
    "user_id": "user_123",            # User identifier
    "session_id": "session_456",      # Conversation session
    "round_id": "round_789",          # Conversation round
    "agent_id": "agent_abc",          # AI agent identifier

    # Temporal Information
    "created_at": "2025-01-01T12:00:00Z",  # Creation timestamp
    "batch_index": 0,                 # Position in batch

    # Content Information
    "roles": ["user", "assistant"],   # Message roles
    "type": "chunk",                  # Data type

    # Contextual Enhancement Information
    "has_context": true,              # Whether chunk has contextual information
    "gpt_enhanced": true,             # LLM enhancement flag
    "context_window_size": 2,         # Number of previous chunks used as context
    "context_chunk_ids": ["chunk_1", "chunk_2"],  # IDs of context chunks
    "contextual_description": "...",   # LLM-generated contextual description
}
```

## Chunking Strategies

### Strategy Interface

```python
class ChunkStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    async def create_chunks(self, message_batch_list: MessageBatchList) -> List[ChunkData]:
        """Transform message batches into chunks."""
        pass
```

### Available Strategies

#### MessageChunkStrategy
**Purpose**: One chunk per MessageList, preserving conversation structure.

```mermaid
graph LR
    A[MessageList 1<br/>User + Assistant] --> B[Chunk 1]
    C[MessageList 2<br/>User + Assistant] --> D[Chunk 2]
    E[MessageList 3<br/>User + Assistant] --> F[Chunk 3]
    
    B --> G[Combined Content]
    D --> G
    F --> G
```

**Use Cases**:
- Short conversations
- Preserving message pairs
- Simple processing requirements

#### ContextualChunkStrategy (Recommended)
**Purpose**: Advanced contextual chunking with LLM enhancement and sliding window context.

```mermaid
graph TB
    A[MessageBatchList] --> B[Message Grouping]
    B --> C[Word Count Analysis]
    C --> D[Smart Overflow Handling]
    D --> E[Role-based Formatting]
    E --> F[Contextual Enhancement]
    F --> G[Sliding Window Context]
    G --> H[LLM Description Generation]
    H --> I[Final Enhanced ChunkData]
```

**Features**:
- CJK language support (Chinese, Japanese, Korean)
- Intelligent message grouping by word count
- LLM-powered contextual descriptions with XAI grok-3-mini
- Sliding window context retrieval (configurable window size)
- Async parallel processing for enhanced chunks
- Previous chunk context retrieval from vector store
- Contextual metadata preservation

**Use Cases**:
- Long conversation threads requiring context
- Semantic understanding enhancement
- Multi-turn conversation analysis
- Context-aware retrieval scenarios

#### CharacterChunkStrategy
**Purpose**: Fixed-size chunks with configurable overlap.

**Use Cases**:
- Uniform chunk sizes
- Predictable processing
- Overlap support for continuity

### Strategy Selection Guide

```mermaid
flowchart TD
    A[Input: MessageBatchList] --> B{Content Length?}
    B -->|Short| C{Preserve Message Boundaries?}
    B -->|Long| D{Need Context Awareness?}
    
    C -->|Yes| E[MessageChunkStrategy<br/>✓ Simple structure<br/>✓ Fast processing<br/>✓ Message integrity]
    C -->|No| F[CharacterChunkStrategy<br/>✓ Uniform sizes<br/>✓ Configurable overlap<br/>✓ Predictable output]
    
    D -->|Yes| G[ContextualChunkStrategy<br/>✓ Intelligent splitting<br/>✓ Context preservation<br/>✓ Semantic boundaries]
    D -->|No| F
    
    style E fill:#e1f5fe
    style F fill:#f3e5f5
    style G fill:#e8f5e8
```

## Storage Integration with HybridBuffer

### HybridBuffer Chunking Architecture

```mermaid
graph TB
    subgraph "HybridBuffer Chunking Flow"
        A[RoundBuffer Transfer] --> B[Immediate Chunking]
        B --> C["ChunkStrategy.create_chunks()"]
        C --> D[Embedding Generation]
        D --> E[VectorCache]

        A --> F[RoundQueue]
        F --> G[Original Rounds Storage]

        E --> H{Queue Size Check}
        G --> H
        H -->|≥ max_size| I[Batch Write Trigger]
        H -->|< max_size| J[Keep in Memory]

        I --> K[SQLite: Original Messages]
        I --> L[Qdrant: Pre-calculated Embeddings]

        K --> M[Clear All Queues]
        L --> M
    end
```

**Key Benefits**:
- **Immediate Availability**: Chunks and embeddings ready for retrieval instantly via VectorCache
- **No Recomputation**: Embeddings calculated once, reused for storage
- **Clear Separation**: RoundQueue for persistence, VectorCache for retrieval
- **Batch Efficiency**: Sequential writes prevent database conflicts
- **Memory Safety**: Complete queue clearing after successful writes

### Unified Store Interface

```python
class ChunkStoreInterface(ABC):
    """Unified interface for all chunk storage implementations."""

    # CRUD Operations
    async def add(self, chunks: List[ChunkData]) -> List[str]
    async def read(self, chunk_ids: List[str]) -> List[ChunkData]
    async def update(self, chunk_id: str, chunk: ChunkData) -> bool
    async def delete(self, chunk_ids: List[str]) -> List[bool]

    # Query Operations
    async def query(self, query: Query, top_k: int = 5) -> List[ChunkData]

    # Business Operations
    async def get_chunks_by_session(self, session_id: str) -> List[ChunkData]
    async def get_chunks_by_round(self, round_id: str) -> List[ChunkData]
    async def get_chunks_stats(self) -> Dict[str, Any]
```

### Store Implementations

#### VectorStore (Qdrant)
```mermaid
graph LR
    A[ChunkData] --> B[Content Extraction]
    B --> C[Embedding Generation]
    C --> D[Vector Storage]
    D --> E[Metadata Indexing]
    E --> F[Similarity Search]
```

**Capabilities**:
- Semantic similarity search
- High-dimensional vector operations
- Metadata filtering
- Scalable vector indexing

#### KeywordStore (SQLite FTS)
```mermaid
graph LR
    A[ChunkData] --> B[Text Tokenization]
    B --> C[BM25 Indexing]
    C --> D[Full-Text Search]
    D --> E[Relevance Scoring]
    E --> F[Keyword Matching]
```

**Capabilities**:
- Full-text search
- BM25 relevance scoring
- Fast keyword matching
- Lightweight storage

#### GraphStore (NetworkX)
```mermaid
graph LR
    A[ChunkData] --> B[Node Creation]
    B --> C[Relationship Extraction]
    C --> D[Graph Construction]
    D --> E[Traversal Algorithms]
    E --> F[Connected Results]
```

**Capabilities**:
- Relationship modeling
- Graph traversal
- Connected component analysis
- Structural queries

## Advanced Features

### LLM Integration

```mermaid
graph TB
    subgraph "LLM Enhancement Pipeline"
        A[ChunkData] --> B[Context Retrieval]
        B --> C[Previous Chunks]
        C --> D[Prompt Generation]
        D --> E[LLM Provider]
        E --> F[Contextual Description]
        F --> G[Enhanced ChunkData]
    end
    
    subgraph "Provider Support"
        H[OpenAI Provider]
        I[Mock Provider]
        J[Custom Provider]
    end
    
    E --> H
    E --> I
    E --> J
```

### Contextual Retrieval

```mermaid
graph TB
    subgraph "Three-Layer Retrieval"
        A[Query] --> B[Layer 1: Similar Chunks]
        A --> C[Layer 2: Connected Contextual]
        A --> D[Layer 3: Similar Contextual]
        
        B --> E[Content Similarity]
        C --> F[Description Retrieval]
        D --> G[Contextual Similarity]
        
        E --> H[Comprehensive Context]
        F --> H
        G --> H
    end
```

### Parallel Processing

```mermaid
graph TB
    subgraph "Enhanced Processing Pipeline"
        A[MessageBatchList] --> B[Session/Round Preparation]
        B --> C[Parallel Tasks]
        
        C --> D[Message Storage Task]
        C --> E[Chunk Processing Task]
        
        D --> F[Store Original Messages]
        E --> G[Create Chunks]
        E --> H[Store Enhanced Chunks]
        
        F --> I[asyncio.gather]
        H --> I
        I --> J[Combined Results]
    end
    
    subgraph "Performance Improvement"
        K[Traditional Sequential: ~200ms]
        L[Enhanced Parallel: ~120ms]
        M[40% Latency Reduction]
    end
```

## API Design

### RESTful Endpoints

```http
# Session-based chunk retrieval
GET /api/v1/sessions/{session_id}/chunks
GET /api/v1/sessions/{session_id}/chunks?limit=20&sort_by=created_at&order=desc

# Round-based chunk retrieval
GET /api/v1/rounds/{round_id}/chunks
GET /api/v1/rounds/{round_id}/chunks?limit=20&sort_by=created_at&order=desc

# Chunk statistics
GET /api/v1/chunks/stats?user_id=user_123&store_type=hybrid
```

### Query Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `limit` | integer | Maximum chunks to return | 20 |
| `sort_by` | string | Sort field (`created_at`, `chunk_id`) | `created_at` |
| `order` | string | Sort order (`asc`, `desc`) | `desc` |
| `store_type` | string | Store filter (`vector`, `keyword`, `graph`, `hybrid`) | `hybrid` |

### Response Format

```json
{
    "status": "success",
    "data": {
        "chunks": [
            {
                "chunk_id": "chunk_123",
                "content": "Conversation content...",
                "metadata": {
                    "strategy": "message",
                    "session_id": "session_456",
                    "user_id": "user_123",
                    "created_at": "2025-01-01T12:00:00Z"
                }
            }
        ],
        "total_count": 1,
        "session_id": "session_456"
    },
    "message": "Retrieved 1 chunks for session session_456"
}
```

## Performance Characteristics

### Current Metrics

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Chunk Creation | <50ms | 1000 chunks/sec |
| Vector Search | <100ms | 500 queries/sec |
| Keyword Search | <20ms | 2000 queries/sec |
| Graph Traversal | <80ms | 800 queries/sec |

### Optimization Strategies

1. **Parallel Processing**: Simultaneous storage operations
2. **Batch Operations**: Bulk chunk processing
3. **Caching**: Frequently accessed chunks
4. **Indexing**: Optimized database indexes

## Design Principles

### Core Principles

1. **Unified Interface**: Consistent API across all stores
2. **Separation of Concerns**: Clear distinction between CRUD and query operations
3. **Extensibility**: Pluggable chunking strategies
4. **Performance**: Optimized for high-throughput scenarios
5. **Metadata Rich**: Comprehensive context preservation

### Quality Attributes

- **Scalability**: Horizontal scaling support
- **Reliability**: Robust error handling
- **Maintainability**: Clean, modular architecture
- **Testability**: Comprehensive test coverage
- **Observability**: Detailed logging and metrics

## Future Enhancements

### Short-term Goals

- **Chunk Versioning**: Track modifications over time
- **Advanced Filtering**: Sophisticated query capabilities
- **Bulk Operations**: Batch processing optimizations

### Long-term Vision

- **Distributed Processing**: Multi-node chunk processing
- **Streaming Capabilities**: Real-time chunk updates
- **ML-Enhanced Chunking**: Intelligent strategy selection
- **Advanced Analytics**: Usage patterns and optimization insights

## Implementation Guidelines

### Best Practices

1. **Strategy Selection**: Choose appropriate chunking strategy based on content characteristics
2. **Metadata Design**: Include comprehensive context information
3. **Error Handling**: Implement robust error recovery
4. **Testing**: Comprehensive unit and integration tests
5. **Monitoring**: Track performance metrics and usage patterns

### Common Patterns

```python
# Strategy initialization
strategy = ContextualChunkStrategy(
    max_words_per_chunk=800,
    enable_contextual=True,
    context_window_size=2,
    gpt_model="grok-3-mini",
    llm_provider=xai_provider
)

# Chunk processing with contextual enhancement
chunks = await strategy.create_chunks(message_batch_list)

# Store operations
chunk_ids = await vector_store.add(chunks)
retrieved_chunks = await vector_store.read(chunk_ids)
```

This architecture provides a robust, scalable foundation for conversation chunking and retrieval in the MemFuse system, supporting diverse use cases while maintaining high performance and extensibility.
