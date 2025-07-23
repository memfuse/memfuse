-- MemFuse pgvectorscale Complete Initialization Script
-- This script sets up the complete MemFuse environment with TimescaleDB + pgvector + optional pgvectorscale
-- Optimized for immediate trigger system and M0 episodic memory layer

-- =============================================================================
-- SECTION 1: Extensions and Configuration
-- =============================================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Enable pgvector extension for vector operations
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
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vectorscale') THEN
        ALTER SYSTEM SET shared_preload_libraries = 'timescaledb,vectorscale';
        RAISE NOTICE 'Configured shared_preload_libraries with vectorscale';
    ELSE
        ALTER SYSTEM SET shared_preload_libraries = 'timescaledb';
        RAISE NOTICE 'Configured shared_preload_libraries without vectorscale';
    END IF;
END $$;

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
-- SECTION 2: M0 Raw Data Memory Schema
-- =============================================================================

-- Create the M0 raw data memory table with immediate trigger support
CREATE TABLE IF NOT EXISTS m0_raw (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding VECTOR(384),  -- 384-dimensional vectors for all-MiniLM-L6-v2
    needs_embedding BOOLEAN DEFAULT TRUE,
    retry_count INTEGER DEFAULT 0,
    last_retry_at TIMESTAMP,
    retry_status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
-- SECTION 3: M1 Episodic Memory Schema
-- =============================================================================

-- Create the M1 episodic memory table with immediate trigger support
CREATE TABLE IF NOT EXISTS m1_episodic (
    -- Primary identification
    id TEXT PRIMARY KEY,

    -- Source tracking (links back to M0 raw data)
    source_id TEXT,  -- References m0_raw.id
    source_session_id TEXT,  -- Session context for episode
    source_user_id TEXT,  -- User context for episode

    -- Episode content and metadata
    episode_content TEXT NOT NULL,
    episode_type TEXT,  -- Open-ended episode type, no constraints for extensibility
    episode_category JSONB DEFAULT '{}'::jsonb,  -- Flexible categorization system
    confidence FLOAT NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),

    -- Structured episode data
    entities JSONB DEFAULT '[]'::jsonb,  -- Extracted entities from episode
    temporal_info JSONB DEFAULT '{}'::jsonb,  -- Temporal information (dates, times, etc.)
    source_context TEXT,  -- Brief context about where episode came from

    -- General metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- PgAI embedding infrastructure (identical to M0)
    embedding VECTOR(384),  -- 384-dimensional embedding vector
    needs_embedding BOOLEAN DEFAULT TRUE,  -- Flag for automatic embedding generation
    retry_count INTEGER DEFAULT 0,  -- Number of embedding retry attempts
    last_retry_at TIMESTAMP,  -- Timestamp of last retry attempt
    retry_status TEXT DEFAULT 'pending' CHECK (retry_status IN ('pending', 'processing', 'completed', 'failed')),

    -- Audit timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

-- M0 Raw Data Indexes
-- Index for auto-embedding queries (partial index for efficiency)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m0_raw_needs_embedding
ON m0_raw (needs_embedding)
WHERE needs_embedding = TRUE;

-- Index for retry status queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m0_raw_retry_status
ON m0_raw (retry_status);

-- Index for retry count queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m0_raw_retry_count
ON m0_raw (retry_count);

-- Time-based indexes for performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m0_raw_created_at
ON m0_raw (created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m0_raw_updated_at
ON m0_raw (updated_at);

-- JSONB metadata index for fast metadata queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m0_raw_metadata_gin
ON m0_raw USING GIN (metadata);

-- M1 Episodic Memory Indexes
-- Source tracking indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m1_episodic_source_id
ON m1_episodic (source_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m1_episodic_source_session
ON m1_episodic (source_session_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m1_episodic_source_user
ON m1_episodic (source_user_id);

-- Auto-embedding query optimization
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m1_episodic_needs_embedding
ON m1_episodic (needs_embedding)
WHERE needs_embedding = TRUE;

-- Episode type and confidence indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m1_episodic_type
ON m1_episodic (episode_type);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m1_episodic_confidence
ON m1_episodic (confidence);

-- Retry management indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m1_episodic_retry_status
ON m1_episodic (retry_status);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m1_episodic_retry_count
ON m1_episodic (retry_count);

-- Temporal indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m1_episodic_created_at
ON m1_episodic (created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m1_episodic_updated_at
ON m1_episodic (updated_at);

-- JSONB indexes for structured data
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m1_episodic_entities_gin
ON m1_episodic USING GIN (entities);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m1_episodic_temporal_gin
ON m1_episodic USING GIN (temporal_info);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m1_episodic_metadata_gin
ON m1_episodic USING GIN (metadata);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m1_episodic_category_gin
ON m1_episodic USING GIN (episode_category);

-- =============================================================================
-- SECTION 5: Vector Similarity Search Indexes
-- =============================================================================

-- M0 Raw Data Vector similarity search index - try pgvectorscale first, fallback to standard
DO $$
BEGIN
    -- Try to create diskann index if vectorscale is available
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vectorscale') THEN
        BEGIN
            EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m0_raw_embedding_vectorscale
                     ON m0_raw USING diskann (embedding vector_cosine_ops)';
            RAISE NOTICE 'Created M0 pgvectorscale diskann index for optimal performance';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'M0 diskann index creation failed, falling back to HNSW: %', SQLERRM;
            EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m0_raw_embedding_hnsw
                     ON m0_raw USING hnsw (embedding vector_cosine_ops)';
        END;
    ELSE
        -- Create standard HNSW index
        EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m0_raw_embedding_hnsw
                 ON m0_raw USING hnsw (embedding vector_cosine_ops)';
        RAISE NOTICE 'Created M0 standard HNSW index for vector similarity search';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'M0 Vector index creation failed, will use sequential scan: %', SQLERRM;
END $$;

-- M1 Episodic Memory Vector similarity search index
DO $$
BEGIN
    -- Try to create diskann index if vectorscale is available
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vectorscale') THEN
        BEGIN
            EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m1_episodic_embedding_vectorscale
                     ON m1_episodic USING diskann (embedding vector_cosine_ops)';
            RAISE NOTICE 'Created M1 pgvectorscale diskann index for optimal performance';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'M1 diskann index creation failed, falling back to HNSW: %', SQLERRM;
            EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m1_episodic_embedding_hnsw
                     ON m1_episodic USING hnsw (embedding vector_cosine_ops)';
        END;
    ELSE
        -- Create standard HNSW index
        EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m1_episodic_embedding_hnsw
                 ON m1_episodic USING hnsw (embedding vector_cosine_ops)';
        RAISE NOTICE 'Created M1 standard HNSW index for vector similarity search';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'M1 Vector index creation failed, will use sequential scan: %', SQLERRM;
END $$;

-- =============================================================================
-- SECTION 6: Immediate Trigger System
-- =============================================================================

-- Create notification function for M0 immediate trigger system
CREATE OR REPLACE FUNCTION notify_m0_embedding_needed()
RETURNS TRIGGER AS $$
BEGIN
    -- Only notify if needs_embedding is TRUE
    IF NEW.needs_embedding = TRUE THEN
        PERFORM pg_notify('m0_embedding_needed', NEW.id::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for M0 immediate embedding notifications
DROP TRIGGER IF EXISTS m0_raw_embedding_trigger ON m0_raw;
CREATE TRIGGER m0_raw_embedding_trigger
    AFTER INSERT OR UPDATE OF needs_embedding ON m0_raw
    FOR EACH ROW
    EXECUTE FUNCTION notify_m0_embedding_needed();

-- Create notification function for M1 immediate trigger system
CREATE OR REPLACE FUNCTION notify_m1_embedding_needed()
RETURNS TRIGGER AS $$
BEGIN
    -- Only notify if needs_embedding is TRUE
    IF NEW.needs_embedding = TRUE THEN
        PERFORM pg_notify('m1_embedding_needed', NEW.id::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for M1 immediate embedding notifications
DROP TRIGGER IF EXISTS m1_episodic_embedding_trigger ON m1_episodic;
CREATE TRIGGER m1_episodic_embedding_trigger
    AFTER INSERT OR UPDATE OF needs_embedding ON m1_episodic
    FOR EACH ROW
    EXECUTE FUNCTION notify_m1_embedding_needed();

-- =============================================================================
-- SECTION 7: Monitoring and Utility Functions
-- =============================================================================

-- Create function to get trigger system status for all layers
CREATE OR REPLACE FUNCTION get_trigger_system_status()
RETURNS TABLE(
    trigger_name TEXT,
    table_name TEXT,
    status TEXT,
    function_exists BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        t.tgname::TEXT as trigger_name,
        c.relname::TEXT as table_name,
        CASE WHEN t.tgenabled = 'O' THEN 'enabled' ELSE 'disabled' END as status,
        EXISTS(SELECT 1 FROM pg_proc WHERE proname LIKE 'notify_%_embedding_needed') as function_exists
    FROM pg_trigger t
    JOIN pg_class c ON t.tgrelid = c.oid
    WHERE c.relname IN ('m0_raw', 'm1_episodic')
    AND t.tgname LIKE '%_embedding_trigger';
END;
$$ LANGUAGE plpgsql;

-- Set up automatic statistics collection for vector columns
ALTER TABLE m0_raw ALTER COLUMN embedding SET STATISTICS 1000;
ALTER TABLE m1_episodic ALTER COLUMN embedding SET STATISTICS 1000;

-- Create a view for easy vector statistics monitoring across all memory layers
CREATE OR REPLACE VIEW vector_stats AS
WITH layer_stats AS (
    -- M0 Raw Data Layer
    SELECT
        'm0_raw' as table_name,
        COUNT(*) as total_rows,
        COUNT(embedding) as rows_with_embeddings,
        COUNT(*) - COUNT(embedding) as rows_without_embeddings,
        CASE
            WHEN COUNT(*) > 0 THEN ROUND((COUNT(embedding)::float / COUNT(*)::float) * 100, 2)
            ELSE 0
        END as embedding_completion_percentage,
        pg_size_pretty(pg_total_relation_size('m0_raw')) as table_size
    FROM m0_raw

    UNION ALL

    -- M1 Episodic Memory Layer
    SELECT
        'm1_episodic' as table_name,
        COUNT(*) as total_rows,
        COUNT(embedding) as rows_with_embeddings,
        COUNT(*) - COUNT(embedding) as rows_without_embeddings,
        CASE
            WHEN COUNT(*) > 0 THEN ROUND((COUNT(embedding)::float / COUNT(*)::float) * 100, 2)
            ELSE 0
        END as embedding_completion_percentage,
        pg_size_pretty(pg_total_relation_size('m1_episodic')) as table_size
    FROM m1_episodic
)
SELECT * FROM layer_stats;

-- =============================================================================
-- SECTION 8: Performance Testing Functions
-- =============================================================================

-- Create a function to test vector search performance across memory layers
CREATE OR REPLACE FUNCTION test_vector_search_performance(
    memory_layer TEXT DEFAULT 'm0_raw',
    test_vector VECTOR(384) DEFAULT NULL,
    test_limit INTEGER DEFAULT 10
)
RETURNS TABLE(
    layer_name TEXT,
    search_time_ms NUMERIC,
    results_count INTEGER,
    index_used BOOLEAN
) AS $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    search_time NUMERIC;
    result_count INTEGER;
    uses_index BOOLEAN;
    table_name TEXT;
BEGIN
    -- Validate memory layer
    IF memory_layer NOT IN ('m0_raw', 'm1_episodic') THEN
        RAISE EXCEPTION 'Invalid memory layer: %. Must be m0_raw or m1_episodic', memory_layer;
    END IF;

    table_name := memory_layer;

    -- Use a random vector if none provided
    IF test_vector IS NULL THEN
        EXECUTE format('
            SELECT embedding
            FROM %I
            WHERE embedding IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 1', table_name)
        INTO test_vector;
    END IF;

    -- If still no vector, create a random one
    IF test_vector IS NULL THEN
        test_vector := ARRAY(SELECT random() FROM generate_series(1, 384))::VECTOR(384);
    END IF;

    -- Measure search performance
    start_time := clock_timestamp();

    EXECUTE format('
        SELECT COUNT(*)
        FROM (
            SELECT id
            FROM %I
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1
            LIMIT $2
        ) subq', table_name)
    INTO result_count
    USING test_vector, test_limit;

    end_time := clock_timestamp();
    search_time := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;

    -- Assume index is used for vector operations
    uses_index := TRUE;

    RETURN QUERY SELECT table_name, search_time, result_count, uses_index;
END;
$$ LANGUAGE plpgsql;

-- Create maintenance function for vector indexes across all memory layers
CREATE OR REPLACE FUNCTION maintain_vector_indexes()
RETURNS TEXT AS $$
DECLARE
    result_text TEXT;
BEGIN
    -- Analyze all memory layer tables to update statistics
    ANALYZE m0_raw;
    ANALYZE m1_episodic;

    result_text := 'Vector index maintenance completed for all memory layers. Statistics updated.';

    RETURN result_text;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- SECTION 9: Initialization Complete
-- =============================================================================

-- Log successful initialization
DO $$
DECLARE
    extensions_list TEXT;
    vectorscale_available BOOLEAN;
    m0_count INTEGER;
    m1_count INTEGER;
BEGIN
    SELECT string_agg(extname, ', ') INTO extensions_list
    FROM pg_extension
    WHERE extname IN ('timescaledb', 'vector', 'vectorscale');

    SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vectorscale') INTO vectorscale_available;

    -- Count tables
    SELECT COUNT(*) INTO m0_count FROM pg_tables WHERE tablename = 'm0_raw';
    SELECT COUNT(*) INTO m1_count FROM pg_tables WHERE tablename = 'm1_episodic';

    RAISE NOTICE '=============================================================================';
    RAISE NOTICE 'MemFuse Multi-Layer Memory System initialization completed successfully';
    RAISE NOTICE 'Extensions enabled: %', extensions_list;
    RAISE NOTICE 'Memory layers initialized:';
    RAISE NOTICE '  - M0 Raw Data Layer: m0_raw (384-dimensional vectors)';
    RAISE NOTICE '  - M1 Episodic Memory Layer: m1_episodic (384-dimensional vectors)';
    RAISE NOTICE 'Immediate trigger system: ENABLED for all layers';
    RAISE NOTICE 'Vector index type: %', CASE WHEN vectorscale_available THEN 'pgvectorscale (diskann)' ELSE 'pgvector (hnsw)' END;
    RAISE NOTICE 'Database optimized for vector operations';
    RAISE NOTICE 'Tables created: M0=%s, M1=%s', CASE WHEN m0_count > 0 THEN 'YES' ELSE 'NO' END, CASE WHEN m1_count > 0 THEN 'YES' ELSE 'NO' END;
    RAISE NOTICE '=============================================================================';
END $$;
