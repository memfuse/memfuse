-- MemFuse pgvectorscale Database Initialization
-- ===============================================
--
-- This script initializes the MemFuse memory layer database with pgvectorscale
-- support for high-performance vector similarity search using StreamingDiskANN.
--
-- Architecture:
-- - M0 Layer: Raw streaming messages with metadata and lineage tracking
-- - M1 Layer: Intelligent chunks with embeddings and optimized vector indexes
-- - pgvectorscale: StreamingDiskANN for 2-5x faster vector search with SBQ compression
--
-- Features:
-- - Normalized similarity scores (0-1 range) for cross-system comparison
-- - Memory-optimized storage layout with SBQ compression
-- - Incremental index updates for streaming data
-- - Complete data lineage tracking from M0 to M1
--
-- Usage:
--   This script is automatically executed during Docker container initialization.
--   Manual execution: psql -U postgres -d memfuse -f init-pgvectorscale.sql

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS ai CASCADE;
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;

-- Set optimal configuration for vector workloads
ALTER DATABASE memfuse SET diskann.query_search_list_size = 100;
ALTER DATABASE memfuse SET diskann.query_rescore = 50;
ALTER DATABASE memfuse SET maintenance_work_mem = '256MB';

-- ============================================================================
-- M0 Layer: Raw Streaming Messages
-- ============================================================================
-- Stores raw streaming conversation messages with complete metadata
-- and lineage tracking for downstream processing.

CREATE TABLE IF NOT EXISTS m0_raw (
    -- Primary identification
    message_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Content and metadata
    content TEXT NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),

    -- Streaming context
    conversation_id UUID NOT NULL,
    sequence_number INTEGER NOT NULL,

    -- Token analysis for chunking decisions
    token_count INTEGER NOT NULL DEFAULT 0,

    -- Temporal tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE,

    -- Processing status
    processing_status VARCHAR(20) DEFAULT 'pending'
        CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),

    -- Lineage tracking
    chunk_assignments UUID[] DEFAULT '{}',

    -- Performance optimization
    CONSTRAINT unique_conversation_sequence
        UNIQUE (conversation_id, sequence_number)
);

-- Indexes for M0 layer performance
CREATE INDEX IF NOT EXISTS idx_m0_conversation_sequence
    ON m0_raw (conversation_id, sequence_number);

CREATE INDEX IF NOT EXISTS idx_m0_processing_status
    ON m0_raw (processing_status)
    WHERE processing_status != 'completed';

CREATE INDEX IF NOT EXISTS idx_m0_created_at
    ON m0_raw (created_at DESC);

-- ============================================================================
-- M1 Layer: Intelligent Chunks with Embeddings
-- ============================================================================
-- Stores intelligently chunked content with high-performance vector embeddings
-- optimized for StreamingDiskANN similarity search.

CREATE TABLE IF NOT EXISTS m1_episodic (
    -- Primary identification
    chunk_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Chunked content
    content TEXT NOT NULL,

    -- Chunking strategy metadata
    chunking_strategy VARCHAR(50) NOT NULL DEFAULT 'token_based'
        CHECK (chunking_strategy IN ('token_based', 'semantic', 'conversation_turn', 'hybrid')),

    -- Token analysis
    token_count INTEGER NOT NULL DEFAULT 0,

    -- Vector embedding (384 dimensions for sentence-transformers/all-MiniLM-L6-v2)
    embedding vector(384),

    -- M0 lineage tracking
    m0_raw_ids UUID[] NOT NULL DEFAULT '{}',
    conversation_id UUID NOT NULL,

    -- Temporal tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    embedding_generated_at TIMESTAMP WITH TIME ZONE,

    -- Quality metrics
    embedding_model VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    chunk_quality_score FLOAT DEFAULT 0.0,

    -- Constraints
    CONSTRAINT m1_chunks_embedding_not_null
        CHECK (embedding IS NOT NULL),
    CONSTRAINT m1_chunks_m0_lineage_not_empty
        CHECK (array_length(m0_raw_ids, 1) > 0)
);

-- ============================================================================
-- High-Performance Vector Indexes
-- ============================================================================

-- StreamingDiskANN index for optimal vector similarity search
-- Features:
-- - memory_optimized storage layout with SBQ compression
-- - 75% memory reduction compared to standard HNSW
-- - 2-5x faster query performance
-- - Incremental updates for streaming data
CREATE INDEX IF NOT EXISTS idx_m1_embedding_diskann
    ON m1_episodic
    USING diskann (embedding vector_cosine_ops)
    WITH (
        storage_layout = 'memory_optimized',  -- Enable SBQ compression
        num_neighbors = 50,                   -- Optimal for 384-dim embeddings
        search_list_size = 100,               -- Balance speed vs accuracy
        max_alpha = 1.2,                      -- Algorithm optimization parameter
        num_dimensions = 384,                 -- Match embedding dimensions
        num_bits_per_dimension = 2            -- SBQ compression: 2 bits per dimension
    );

-- Additional indexes for M1 layer performance
CREATE INDEX IF NOT EXISTS idx_m1_conversation_id
    ON m1_episodic (conversation_id);

CREATE INDEX IF NOT EXISTS idx_m1_chunking_strategy
    ON m1_episodic (chunking_strategy);

CREATE INDEX IF NOT EXISTS idx_m1_created_at
    ON m1_episodic (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_m1_token_count
    ON m1_episodic (token_count);

-- GIN index for M0 message ID arrays (lineage queries)
CREATE INDEX IF NOT EXISTS idx_m1_m0_raw_ids_gin
    ON m1_episodic USING gin (m0_raw_ids);

-- ============================================================================
-- Utility Functions
-- ============================================================================

-- Function to normalize similarity scores to 0-1 range
-- Converts cosine distance (0-2) to similarity score (0-1)
-- where 1 = identical, 0 = completely different
CREATE OR REPLACE FUNCTION normalize_cosine_similarity(distance FLOAT)
RETURNS FLOAT AS $$
BEGIN
    -- Cosine distance range: [0, 2]
    -- Convert to similarity: similarity = 1 - (distance / 2)
    -- Result range: [0, 1] where 1 = identical, 0 = opposite
    RETURN GREATEST(0.0, LEAST(1.0, 1.0 - (distance / 2.0)));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function for high-performance vector similarity search with normalized scores
CREATE OR REPLACE FUNCTION search_similar_chunks(
    query_embedding vector(384),
    similarity_threshold FLOAT DEFAULT 0.1,
    max_results INTEGER DEFAULT 10
)
RETURNS TABLE (
    chunk_id UUID,
    content TEXT,
    similarity_score FLOAT,
    distance FLOAT,
    m0_message_count INTEGER,
    chunking_strategy VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.chunk_id,
        c.content,
        normalize_cosine_similarity(c.embedding <=> query_embedding) as similarity_score,
        (c.embedding <=> query_embedding) as distance,
        array_length(c.m0_raw_ids, 1) as m0_message_count,
        c.chunking_strategy,
        c.created_at
    FROM m1_episodic c
    WHERE normalize_cosine_similarity(c.embedding <=> query_embedding) >= similarity_threshold
    ORDER BY c.embedding <=> query_embedding ASC
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Function to get data lineage statistics
CREATE OR REPLACE FUNCTION get_data_lineage_stats()
RETURNS TABLE (
    layer VARCHAR(20),
    record_count BIGINT,
    table_size TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        'M0 Messages'::VARCHAR(20) as layer,
        COUNT(*) as record_count,
        pg_size_pretty(pg_total_relation_size('m0_raw')) as table_size
    FROM m0_raw
    UNION ALL
    SELECT
        'M1 Chunks'::VARCHAR(20) as layer,
        COUNT(*) as record_count,
        pg_size_pretty(pg_total_relation_size('m1_episodic')) as table_size
    FROM m1_episodic
    UNION ALL
    SELECT
        'M1 Embeddings'::VARCHAR(20) as layer,
        COUNT(*) as record_count,
        'N/A'::TEXT as table_size
    FROM m1_episodic
    WHERE embedding IS NOT NULL;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Performance Monitoring Views
-- ============================================================================

-- View for monitoring vector index performance
CREATE OR REPLACE VIEW vector_index_stats AS
SELECT 
    schemaname,
    tablename,
    indexname,
    CASE 
        WHEN indexdef LIKE '%diskann%' THEN 'StreamingDiskANN (pgvectorscale)'
        WHEN indexdef LIKE '%hnsw%' THEN 'HNSW (pgvector)'
        ELSE 'Other'
    END as index_type,
    indexdef
FROM pg_indexes
WHERE tablename IN ('m1_episodic')
AND indexdef LIKE '%vector%';

-- View for data lineage analysis
CREATE OR REPLACE VIEW data_lineage_summary AS
SELECT
    COUNT(DISTINCT c.chunk_id) as total_chunks,
    AVG(array_length(c.m0_raw_ids, 1))::NUMERIC(10,4) as avg_m0_per_chunk,
    MIN(array_length(c.m0_raw_ids, 1)) as min_m0_per_chunk,
    MAX(array_length(c.m0_raw_ids, 1)) as max_m0_per_chunk,
    (SELECT COUNT(*) FROM m0_raw) as total_m0_messages
FROM m1_episodic c;

-- ============================================================================
-- Initial Data Validation
-- ============================================================================

-- Verify extensions are properly loaded
DO $$
BEGIN
    -- Check vector extension
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        RAISE EXCEPTION 'vector extension is not installed';
    END IF;
    
    -- Check vectorscale extension
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vectorscale') THEN
        RAISE EXCEPTION 'vectorscale extension is not installed';
    END IF;
    
    RAISE NOTICE 'All required extensions are properly installed';
END $$;

-- Log successful initialization
INSERT INTO m0_raw (content, role, conversation_id, sequence_number, token_count, processing_status)
VALUES (
    'MemFuse pgvectorscale database initialized successfully with StreamingDiskANN support',
    'system',
    uuid_generate_v4(),
    1,
    12,
    'completed'
);

-- Display initialization summary
SELECT 
    'MemFuse pgvectorscale Database Initialized' as status,
    CURRENT_TIMESTAMP as initialized_at,
    version() as postgresql_version,
    (SELECT extversion FROM pg_extension WHERE extname = 'vector') as vector_version,
    (SELECT extversion FROM pg_extension WHERE extname = 'vectorscale') as vectorscale_version;
