# MemFuse pgvectorscale Integration Tests

Complete end-to-end integration testing suite for MemFuse memory layer architecture using pgvectorscale with StreamingDiskANN for high-performance vector similarity search.

## Overview

This integration test suite demonstrates the complete MemFuse memory layer workflow:

- **M0 Layer**: Raw streaming messages with metadata and lineage tracking
- **M1 Layer**: Intelligent chunking with high-performance vector embeddings
- **pgvectorscale**: StreamingDiskANN for optimized vector similarity search
- **Normalized Scoring**: 0-1 range similarity scores for cross-system comparison

## Architecture

```
Streaming Data → M0 Messages → M1 Chunks → Vector Search
                     ↓             ↓           ↓
                 Raw Storage   Embeddings  StreamingDiskANN
                 Metadata      384-dim     SBQ Compression
                 Lineage       Vectors     Normalized Scores
```

## Features

### High-Performance Vector Search
- **StreamingDiskANN**: 2-5x faster than standard HNSW
- **SBQ Compression**: 75% memory reduction with 2 bits per dimension
- **Memory-Optimized**: Efficient storage layout for large-scale deployments
- **Incremental Updates**: Real-time index updates for streaming data

### Normalized Similarity Scoring
- **0-1 Range**: Consistent scoring across different systems
- **Cross-Compatible**: Easy comparison with other similarity metrics
- **Intuitive**: 1.0 = identical, 0.0 = completely different

### Complete Data Lineage
- **M0 → M1 Tracking**: Full traceability from raw messages to chunks
- **Conversation Context**: Maintains conversation flow and context
- **Quality Metrics**: Chunk quality scoring and validation

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- At least 4GB available memory

### Installation

1. **Clone and navigate to the test directory**:
   ```bash
   cd tests/integration/pgai
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the complete end-to-end demo**:
   ```bash
   ./run_pgvectorscale_e2e_demo.sh
   ```

### Usage Options

```bash
# Run complete end-to-end demo
./run_pgvectorscale_e2e_demo.sh

# Verify existing data integrity
./run_pgvectorscale_e2e_demo.sh --verify

# Clean up environment
./run_pgvectorscale_e2e_demo.sh --cleanup

# Show help
./run_pgvectorscale_e2e_demo.sh --help
```

## Components

### 1. Database Environment (`docker-compose.pgvectorscale.yml`)
- TimescaleDB HA pg17 with pgvectorscale installation support
- Extensions installed via `CREATE EXTENSION` commands (pgai + pgvectorscale)
- Optimized PostgreSQL configuration for vector workloads
- Persistent data volumes for development
- Health checks and resource limits

**Note**: Uses `timescale/timescaledb-ha:pg17` for reliable pgvectorscale support.

### 2. Database Schema (`init-pgvectorscale.sql`)
- M0 and M1 table definitions with proper constraints
- StreamingDiskANN indexes with SBQ compression
- Utility functions for normalized similarity search
- Performance monitoring views

### 3. Python Demo (`pgvectorscale_e2e_demo.py`)
- Streaming conversation data generation
- Token-based intelligent chunking
- High-performance embedding generation
- Vector similarity search with normalized scores
- Complete data integrity validation

### 4. Shell Script (`run_pgvectorscale_e2e_demo.sh`)
- Automated environment setup and teardown
- Prerequisites checking
- Complete workflow orchestration
- Error handling and logging

## Performance Characteristics

### StreamingDiskANN vs Standard HNSW

| Feature | Standard HNSW | StreamingDiskANN |
|---------|---------------|------------------|
| Memory Usage | 100% | **25%** (SBQ compression) |
| Query Speed | Baseline | **2-5x faster** |
| Index Updates | Static rebuild | **Incremental** |
| Storage Layout | Standard | **Memory-optimized** |
| Compression | None | **SBQ (2 bits/dim)** |

### Normalized Similarity Scores

- **Range**: 0.0 to 1.0
- **Interpretation**: 1.0 = identical, 0.0 = completely different
- **Conversion**: `similarity = 1.0 - (cosine_distance / 2.0)`
- **Benefits**: Cross-system compatibility and intuitive interpretation

## Configuration

### Database Configuration
- **Host**: localhost
- **Port**: 5434
- **Database**: memfuse
- **User**: postgres
- **Password**: memfuse_secure_password

### Vector Configuration
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Dimensions**: 384
- **Index Type**: StreamingDiskANN with SBQ compression
- **Chunking Strategy**: Token-based (200 tokens per chunk)

### Environment Variables
```bash
export PGVECTORSCALE_HOST="localhost"
export PGVECTORSCALE_PORT="5434"
export PGVECTORSCALE_DB="memfuse"
export PGVECTORSCALE_USER="postgres"
export PGVECTORSCALE_PASSWORD="memfuse_secure_password"
```

## Expected Output

### Successful Demo Run
```
🚀 MemFuse pgvectorscale End-to-End Integration Demo
======================================================================
✅ Connected to pgvectorscale database
✅ pgvectorscale extension version: 0.8.0
🧠 Loading sentence-transformers model...
✅ Embedding model loaded successfully
📊 Generating streaming conversation data...
💾 Writing M0 raw message data...
✅ Inserted 36 M0 message records
🧠 Creating M1 chunks (token-based strategy)...
✅ Created 1 M1 chunks
💾 Inserting M1 chunks with embeddings...
✅ Inserted 1 M1 chunk records with embeddings
🔍 Performing high-performance vector similarity search...

Query: 'Python machine learning algorithms'
✅ StreamingDiskANN (pgvectorscale) returned 1 results
  1. Similarity: 0.8705 (Distance: 0.2591)

🎯 Demo Summary:
======================================================================
✅ Streaming data ingestion: Success
✅ M0 message storage: Success
✅ M1 intelligent chunking: Success (multi-message chunks)
✅ StreamingDiskANN vector indexing: Success
✅ Normalized similarity search (0-1 range): Success
✅ Data lineage tracking: Success

🚀 MemFuse pgvectorscale End-to-End Demo Complete!
🎉 All functionality verified with StreamingDiskANN optimization!
```

## Troubleshooting

### Common Issues

1. **Docker not running**:
   ```bash
   # Start Docker service
   sudo systemctl start docker  # Linux
   # Or start Docker Desktop on macOS/Windows
   ```

2. **Port 5434 already in use**:
   ```bash
   # Check what's using the port
   lsof -i :5434
   # Kill the process or change the port in docker-compose.yml
   ```

3. **Python dependencies missing**:
   ```bash
   # Install in virtual environment
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Database connection failed**:
   ```bash
   # Check container status
   docker ps
   # Check logs
   docker logs memfuse-pgvectorscale-e2e
   ```

### Performance Tuning

1. **Increase memory allocation**:
   - Edit `docker-compose.pgvectorscale.yml`
   - Increase memory limits in deploy section

2. **Optimize vector index parameters**:
   - Edit `init-pgvectorscale.sql`
   - Adjust `num_neighbors`, `search_list_size` parameters

3. **Tune PostgreSQL settings**:
   - Edit `postgresql.conf`
   - Adjust `shared_buffers`, `work_mem` based on available memory

## Development

### Running Individual Components

1. **Start database only**:
   ```bash
   docker-compose -f tests/integration/pgai/docker-compose.pgvectorscale.yml up -d
   ```

2. **Run Python demo only**:
   ```bash
   python3 pgvectorscale_e2e_demo.py
   ```

3. **Connect to database**:
   ```bash
   docker exec -it memfuse-pgvectorscale-e2e psql -U postgres -d memfuse
   ```

### Testing Custom Queries

```sql
-- Test similarity search with custom query
SELECT * FROM search_similar_chunks(
    '[0.1, 0.2, ...]'::vector,  -- Your query embedding
    0.1,                        -- Similarity threshold
    5                           -- Max results
);

-- Check index statistics
SELECT * FROM vector_index_stats;

-- Analyze data lineage
SELECT * FROM data_lineage_summary;
```

## License

This integration test suite is part of the MemFuse project and follows the same licensing terms.
