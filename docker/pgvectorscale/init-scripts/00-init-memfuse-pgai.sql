-- MemFuse pgvectorscale Complete Initialization Script
-- This script sets up the complete MemFuse environment with TimescaleDB + pgvector + optional pgvectorscale
-- Optimized for immediate trigger system and M0 episodic memory layer

-- =============================================================================
-- SECTION 1: Extensions and Configuration
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable pgvectorscale for enhanced vector performance (if available)
DO $$
BEGIN
    BEGIN
        CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
        RAISE NOTICE 'pgvectorscale extension enabled successfully';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'pgvectorscale extension not available, using standard pgvector';
    END;
END $$;

-- Configure PostgreSQL for optimal vector operations
-- Note: shared_preload_libraries is handled by Docker command line configuration

-- Optimize PostgreSQL settings for vector operations
ALTER SYSTEM SET max_connections = 100;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '64MB';
ALTER SYSTEM SET maintenance_work_mem = '256MB';
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;
ALTER SYSTEM SET autovacuum_max_workers = 3;
ALTER SYSTEM SET autovacuum_naptime = '1min';

-- Reload configuration
SELECT pg_reload_conf();

-- Set up search path
ALTER DATABASE memfuse SET search_path TO public, timescaledb_information;

-- =============================================================================
-- SECTION 2: M0 Raw Data Memory Schema (Simplified Demo-Compatible)
-- =============================================================================

-- Create the M0 raw data memory table matching demo schema
CREATE TABLE IF NOT EXISTS m0_raw (
    -- Primary identification
    message_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Content and metadata
    content TEXT NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),

    -- Streaming context
    session_id UUID NOT NULL,
    sequence_number INTEGER NOT NULL,

    -- Token analysis for chunking decisions
    token_count INTEGER NOT NULL DEFAULT 0,

    -- Temporal tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE,

    -- Processing status
    processing_status VARCHAR(20) DEFAULT 'pending'
        CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),

    -- Lineage tracking
    chunk_assignments UUID[] DEFAULT '{}',

    -- Performance optimization
    CONSTRAINT unique_session_sequence
        UNIQUE (session_id, sequence_number)
);

-- Create function to update updated_at timestamp for M0
CREATE OR REPLACE FUNCTION update_m0_raw_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic updated_at updates on M0
DROP TRIGGER IF EXISTS m0_raw_updated_at_trigger ON m0_raw;
CREATE TRIGGER m0_raw_updated_at_trigger
    BEFORE UPDATE ON m0_raw
    FOR EACH ROW
    EXECUTE FUNCTION update_m0_raw_updated_at();

-- =============================================================================
-- SECTION 3: M1 Episodic Memory Schema (Simplified Demo-Compatible)
-- =============================================================================

-- Create the M1 episodic memory table matching demo schema
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

    -- Embedding processing status
    needs_embedding BOOLEAN DEFAULT TRUE,

    -- M0 lineage tracking
    m0_message_ids UUID[] NOT NULL DEFAULT '{}',
    session_id UUID NOT NULL,

    -- Temporal tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    embedding_generated_at TIMESTAMP WITH TIME ZONE,

    -- Quality metrics
    embedding_model VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    chunk_quality_score FLOAT DEFAULT 0.0,

    -- Constraints (removed embedding NOT NULL constraint to support async processing)
    CONSTRAINT m1_chunks_m0_lineage_not_empty
        CHECK (array_length(m0_message_ids, 1) > 0)
);

-- Create function to update updated_at timestamp for M1
CREATE OR REPLACE FUNCTION update_m1_episodic_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic updated_at updates on M1
DROP TRIGGER IF EXISTS m1_episodic_updated_at_trigger ON m1_episodic;
CREATE TRIGGER m1_episodic_updated_at_trigger
    BEFORE UPDATE ON m1_episodic
    FOR EACH ROW
    EXECUTE FUNCTION update_m1_episodic_updated_at();

-- =============================================================================
-- SECTION 4: Optimized Indexes
-- =============================================================================

-- M0 Raw Data Indexes (Demo-Compatible)
-- Indexes for M0 layer performance
CREATE INDEX IF NOT EXISTS idx_m0_session_sequence
    ON m0_raw (session_id, sequence_number);

CREATE INDEX IF NOT EXISTS idx_m0_processing_status
    ON m0_raw (processing_status)
    WHERE processing_status != 'completed';

CREATE INDEX IF NOT EXISTS idx_m0_created_at
    ON m0_raw (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_m0_role
    ON m0_raw (role);

CREATE INDEX IF NOT EXISTS idx_m0_token_count
    ON m0_raw (token_count);

-- GIN index for chunk assignments (lineage queries)
CREATE INDEX IF NOT EXISTS idx_m0_chunk_assignments_gin
    ON m0_raw USING gin (chunk_assignments);

-- M1 Episodic Memory Indexes (Demo-Compatible)
-- Additional indexes for M1 layer performance
CREATE INDEX IF NOT EXISTS idx_m1_session_id
    ON m1_episodic (session_id);

CREATE INDEX IF NOT EXISTS idx_m1_chunking_strategy
    ON m1_episodic (chunking_strategy);

CREATE INDEX IF NOT EXISTS idx_m1_created_at
    ON m1_episodic (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_m1_token_count
    ON m1_episodic (token_count);

CREATE INDEX IF NOT EXISTS idx_m1_chunk_quality_score
    ON m1_episodic (chunk_quality_score);

-- Index for embedding processing status
CREATE INDEX IF NOT EXISTS idx_m1_needs_embedding
    ON m1_episodic (needs_embedding)
    WHERE needs_embedding = TRUE;

-- GIN index for M0 message ID arrays (lineage queries)
CREATE INDEX IF NOT EXISTS idx_m1_m0_message_ids_gin
    ON m1_episodic USING gin (m0_message_ids);

-- =============================================================================
-- SECTION 5: High-Performance Vector Indexes
-- =============================================================================

-- StreamingDiskANN index for optimal vector similarity search on M1 layer
-- Features:
-- - memory_optimized storage layout with SBQ compression
-- - 75% memory reduction compared to standard HNSW
-- - 2-5x faster query performance
-- - Incremental updates for streaming data

-- Try to create StreamingDiskANN index if vectorscale is available, fallback to HNSW
DO $$
BEGIN
    -- Try to create StreamingDiskANN index
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vectorscale') THEN
        BEGIN
            EXECUTE 'CREATE INDEX IF NOT EXISTS idx_m1_embedding_diskann
                ON m1_episodic
                USING diskann (embedding vector_cosine_ops)
                WITH (
                    storage_layout = ''memory_optimized'',
                    num_neighbors = 50,
                    search_list_size = 100,
                    max_alpha = 1.2,
                    num_dimensions = 384,
                    num_bits_per_dimension = 2
                )';
            RAISE NOTICE 'StreamingDiskANN index created successfully';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'StreamingDiskANN index creation failed, falling back to HNSW: %', SQLERRM;
            EXECUTE 'CREATE INDEX IF NOT EXISTS idx_m1_embedding_hnsw
                ON m1_episodic
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)';
        END;
    ELSE
        -- Create standard HNSW index
        EXECUTE 'CREATE INDEX IF NOT EXISTS idx_m1_embedding_hnsw
            ON m1_episodic
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)';
        RAISE NOTICE 'Standard HNSW index created (vectorscale not available)';
    END IF;
END $$;

-- =============================================================================
-- SECTION 6: Utility Functions (Demo-Compatible)
-- =============================================================================

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
        array_length(c.m0_message_ids, 1) as m0_message_count,
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

-- =============================================================================
-- SECTION 7: Performance Monitoring Views
-- =============================================================================

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
    AVG(array_length(c.m0_message_ids, 1))::NUMERIC(10,4) as avg_m0_per_chunk,
    MIN(array_length(c.m0_message_ids, 1)) as min_m0_per_chunk,
    MAX(array_length(c.m0_message_ids, 1)) as max_m0_per_chunk,
    (SELECT COUNT(*) FROM m0_raw) as total_m0_messages
FROM m1_episodic c;

-- =============================================================================
-- SECTION 8: Initial Data Validation
-- =============================================================================

-- Verify extensions are properly loaded
DO $$
BEGIN
    -- Check vector extension
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        RAISE EXCEPTION 'vector extension is not installed';
    END IF;

    -- Check vectorscale extension (optional)
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vectorscale') THEN
        RAISE NOTICE 'vectorscale extension is not installed, using standard pgvector';
    ELSE
        RAISE NOTICE 'vectorscale extension is available';
    END IF;

    RAISE NOTICE 'Required extensions are properly installed';
END $$;

-- Log successful initialization
INSERT INTO m0_raw (content, role, session_id, sequence_number, token_count, processing_status)
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
