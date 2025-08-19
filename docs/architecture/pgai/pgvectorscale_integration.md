# MemFuse PgVectorScale Integration

This document describes the complete integration of pgvectorscale with MemFuse, implementing the simplified M0/M1 memory layer architecture with high-performance vector similarity search.

## Overview

The integration provides:

- **M0 Layer**: Raw streaming messages with metadata and lineage tracking
- **M1 Layer**: Intelligent chunks with embeddings and StreamingDiskANN indexing
- **Integrated Chunking**: Reuses existing MemFuse chunking strategies
- **Global Embedding Models**: Avoids repeated model loading for better performance
- **Normalized Similarity Scores**: 0-1 range for cross-system compatibility

## Architecture

```
Streaming Data → M0 Layer (Raw Messages) → Intelligent Chunking → M1 Layer (Chunks + Embeddings) → pgvectorscale StreamingDiskANN → Normalized Similarity Search
```

### Key Components

1. **PgVectorScaleStore**: Vector store implementation with M0/M1 support
2. **IntegratedChunkingProcessor**: Reuses existing chunking strategies
3. **PgVectorScaleMemoryLayer**: Memory layer implementation
4. **StreamingDataProcessor**: Complete data flow pipeline

## Quick Start

### 1. Start the Database

```bash
# Start pgvectorscale container
poetry run python scripts/memfuse_launcher.py
```

This will:
- Start the pgvectorscale-enabled PostgreSQL container on port 5432
- Initialize the M0/M1 schema with StreamingDiskANN indexes
- Apply database optimizations for vector workloads

### 2. Run the Demo

```bash
# Quick functionality test
poetry run python scripts/pgvectorscale_demo.py --quick

# Full integration demo
poetry run python scripts/pgvectorscale_demo.py
```

### 3. Run Integration Tests

```bash
# Run pytest integration tests
pytest tests/integration/test_pgvectorscale_integration.py -v

# Or run standalone tests
poetry run python tests/integration/test_pgvectorscale_integration.py
```

## Usage Examples

### Basic Usage

```python
from src.memfuse_core.hierarchy.streaming_pipeline import StreamingDataProcessor

# Initialize processor
processor = StreamingDataProcessor(
    user_id="your_user_id",
    db_config={
        'host': 'localhost',
        'port': 5432,
        'database': 'memfuse',
        'user': 'postgres',
        'password': 'postgres'
    }
)

await processor.initialize()

# Process streaming messages
messages = [
    {"content": "Hello, I want to learn about machine learning", "role": "user"},
    {"content": "I'd be happy to help you learn about ML!", "role": "assistant"}
]

result = await processor.process_streaming_messages(messages, session_id="session_001")

# Query with vector similarity search
query_result = await processor.query_processed_data(
    query="machine learning algorithms",
    top_k=5
)

print(f"Found {len(query_result['results'])} similar chunks")
```

### Advanced Usage with Memory Layer

```python
from src.memfuse_core.hierarchy.pgvectorscale_memory_layer import PgVectorScaleMemoryLayer

# Direct memory layer usage
memory_layer = PgVectorScaleMemoryLayer(user_id="user_001")
await memory_layer.initialize()

# Write message batches
message_batches = [[
    {"content": "First message", "role": "user"},
    {"content": "Second message", "role": "assistant"}
]]

write_result = await memory_layer.write(
    message_batch_list=message_batches,
    session_id="session_001"
)

# Query with advanced parameters
query_result = await memory_layer.query(
    query="your search query",
    top_k=10,
    similarity_threshold=0.2
)
```

## Configuration

### Database Configuration

The system uses the following database configuration by default:

```python
db_config = {
    'host': 'localhost',
    'port': 5432,
    'database': 'memfuse',
    'user': 'postgres',
    'password': 'postgres'
}
```

### Chunking Configuration

The integrated chunking processor supports multiple strategies:

```python
from src.memfuse_core.store.vector_store.chunking_processor import TokenBasedChunkingProcessor

# Token-based chunking (default)
processor = TokenBasedChunkingProcessor(
    max_tokens_per_chunk=200  # Adjust based on your needs
)

# Or use the integrated processor with different strategies
from src.memfuse_core.store.vector_store.chunking_processor import IntegratedChunkingProcessor

processor = IntegratedChunkingProcessor(
    strategy_name="contextual",  # or "message", "character"
    max_words_per_chunk=200,
    enable_contextual=False  # Disable LLM enhancement for performance
)
```

## Performance Features

### StreamingDiskANN Benefits

- **2-5x faster queries** compared to standard HNSW indexes
- **75% memory reduction** through SBQ compression
- **Incremental updates** for streaming data
- **Memory-optimized storage layout**

### Global Model Management

- **Single model loading** at startup
- **Model sharing** across all users and services
- **Reduced memory footprint**
- **Faster initialization**

### Normalized Similarity Scores

All similarity scores are normalized to 0-1 range:
- **1.0**: Identical content
- **0.0**: Completely different content
- **0.7+**: High similarity
- **0.3-0.7**: Moderate similarity
- **<0.3**: Low similarity

## Database Schema

### M0 Layer (Raw Messages)

```sql
CREATE TABLE m0_messages (
    message_id UUID PRIMARY KEY,
    content TEXT NOT NULL,
    role VARCHAR(20) NOT NULL,
    conversation_id UUID NOT NULL,
    sequence_number INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE,
    processing_status VARCHAR(20),
    chunk_assignments UUID[]
);
```

### M1 Layer (Intelligent Chunks)

```sql
CREATE TABLE m1_chunks (
    chunk_id UUID PRIMARY KEY,
    content TEXT NOT NULL,
    chunking_strategy VARCHAR(50) NOT NULL,
    token_count INTEGER NOT NULL,
    embedding vector(384),
    m0_raw_ids UUID[] NOT NULL,
    conversation_id UUID NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE,
    embedding_generated_at TIMESTAMP WITH TIME ZONE
);
```

### Vector Index

```sql
CREATE INDEX idx_m1_embedding_diskann 
ON m1_chunks 
USING diskann (embedding vector_cosine_ops)
WITH (
    storage_layout = 'memory_optimized',
    num_neighbors = 50,
    search_list_size = 100,
    max_alpha = 1.2,
    num_dimensions = 384,
    num_bits_per_dimension = 2
);
```

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Ensure pgvectorscale container is running on port 5432
   - Check database credentials
   - Verify network connectivity

2. **Extension Not Found**
   - Ensure pgvectorscale extension is installed
   - Check container logs for initialization errors

3. **Slow Query Performance**
   - Verify StreamingDiskANN index is created
   - Check database optimization settings
   - Monitor memory usage

4. **Embedding Model Loading Issues**
   - Ensure global model manager is initialized
   - Check model cache and memory availability
   - Verify sentence-transformers installation

### Debug Commands

```bash
# Check container status
docker ps | grep memfuse

# Check database connectivity
docker exec memfuse-pgai-postgres pg_isready -U postgres -d memfuse

# Check extensions
docker exec memfuse-pgai-postgres psql -U postgres -d memfuse -c "SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'vectorscale');"

# Check table status
docker exec memfuse-pgai-postgres psql -U postgres -d memfuse -c "SELECT * FROM get_data_lineage_stats();"
```

## Performance Benchmarks

Expected performance characteristics:

| Metric | Value |
|--------|-------|
| Query Latency | 2-4ms |
| Memory Usage | 75% reduction vs HNSW |
| Throughput | 250-500 QPS |
| Index Build Time | 25% faster |
| Update Latency | Incremental (100x faster) |

## Next Steps

1. **Production Deployment**: Configure for production workloads
2. **Monitoring**: Set up performance monitoring and alerting
3. **Scaling**: Implement horizontal scaling strategies
4. **Advanced Features**: Add multi-modal embeddings and federated search

For more information, see the [simplified architecture documentation](architecture/pgai/simplified.md).
