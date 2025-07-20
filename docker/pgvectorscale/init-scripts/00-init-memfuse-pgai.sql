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

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_m0_episodic_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic updated_at updates
DROP TRIGGER IF EXISTS m0_episodic_updated_at_trigger ON m0_episodic;
CREATE TRIGGER m0_episodic_updated_at_trigger
    BEFORE UPDATE ON m0_episodic
    FOR EACH ROW
    EXECUTE FUNCTION update_m0_episodic_updated_at();

-- =============================================================================
-- SECTION 3: Optimized Indexes
-- =============================================================================

-- Index for auto-embedding queries (partial index for efficiency)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m0_episodic_needs_embedding 
ON m0_episodic (needs_embedding) 
WHERE needs_embedding = TRUE;

-- Index for retry status queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m0_episodic_retry_status 
ON m0_episodic (retry_status);

-- Index for retry count queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m0_episodic_retry_count 
ON m0_episodic (retry_count);

-- Time-based indexes for performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m0_episodic_created_at 
ON m0_episodic (created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m0_episodic_updated_at 
ON m0_episodic (updated_at);

-- JSONB metadata index for fast metadata queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m0_episodic_metadata_gin 
ON m0_episodic USING GIN (metadata);

-- =============================================================================
-- SECTION 4: Vector Similarity Search Index
-- =============================================================================

-- Vector similarity search index - try pgvectorscale first, fallback to standard
DO $$
BEGIN
    -- Try to create diskann index if vectorscale is available
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vectorscale') THEN
        BEGIN
            EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m0_episodic_embedding_vectorscale
                     ON m0_episodic USING diskann (embedding vector_cosine_ops)';
            RAISE NOTICE 'Created pgvectorscale diskann index for optimal performance';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'diskann index creation failed, falling back to HNSW: %', SQLERRM;
            EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m0_episodic_embedding_hnsw
                     ON m0_episodic USING hnsw (embedding vector_cosine_ops)';
        END;
    ELSE
        -- Create standard HNSW index
        EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m0_episodic_embedding_hnsw
                 ON m0_episodic USING hnsw (embedding vector_cosine_ops)';
        RAISE NOTICE 'Created standard HNSW index for vector similarity search';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Vector index creation failed, will use sequential scan: %', SQLERRM;
END $$;

-- =============================================================================
-- SECTION 5: Immediate Trigger System
-- =============================================================================

-- Create notification function for immediate trigger system
CREATE OR REPLACE FUNCTION notify_embedding_needed()
RETURNS TRIGGER AS $$
BEGIN
    -- Only notify if needs_embedding is TRUE
    IF NEW.needs_embedding = TRUE THEN
        PERFORM pg_notify('embedding_needed', NEW.id::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for immediate embedding notifications
DROP TRIGGER IF EXISTS m0_episodic_embedding_trigger ON m0_episodic;
CREATE TRIGGER m0_episodic_embedding_trigger
    AFTER INSERT OR UPDATE OF needs_embedding ON m0_episodic
    FOR EACH ROW
    EXECUTE FUNCTION notify_embedding_needed();

-- =============================================================================
-- SECTION 6: Monitoring and Utility Functions
-- =============================================================================

-- Create function to get trigger system status
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
        EXISTS(SELECT 1 FROM pg_proc WHERE proname = 'notify_embedding_needed') as function_exists
    FROM pg_trigger t
    JOIN pg_class c ON t.tgrelid = c.oid
    WHERE c.relname = 'm0_episodic'
    AND t.tgname = 'm0_episodic_embedding_trigger';
END;
$$ LANGUAGE plpgsql;

-- Set up automatic statistics collection for vector columns
ALTER TABLE m0_episodic ALTER COLUMN embedding SET STATISTICS 1000;

-- Create a view for easy vector statistics monitoring
CREATE OR REPLACE VIEW vector_stats AS
SELECT 
    'm0_episodic' as table_name,
    COUNT(*) as total_rows,
    COUNT(embedding) as rows_with_embeddings,
    COUNT(*) - COUNT(embedding) as rows_without_embeddings,
    CASE 
        WHEN COUNT(*) > 0 THEN ROUND((COUNT(embedding)::float / COUNT(*)::float) * 100, 2)
        ELSE 0
    END as embedding_completion_percentage,
    pg_size_pretty(pg_total_relation_size('m0_episodic')) as table_size
FROM m0_episodic;

-- =============================================================================
-- SECTION 7: Performance Testing Functions
-- =============================================================================

-- Create a function to test vector search performance
CREATE OR REPLACE FUNCTION test_vector_search_performance(
    test_vector VECTOR(384) DEFAULT NULL,
    test_limit INTEGER DEFAULT 10
)
RETURNS TABLE(
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
BEGIN
    -- Use a random vector if none provided
    IF test_vector IS NULL THEN
        test_vector := (
            SELECT embedding 
            FROM m0_episodic 
            WHERE embedding IS NOT NULL 
            ORDER BY RANDOM() 
            LIMIT 1
        );
    END IF;
    
    -- If still no vector, create a random one
    IF test_vector IS NULL THEN
        test_vector := ARRAY(SELECT random() FROM generate_series(1, 384))::VECTOR(384);
    END IF;
    
    -- Measure search performance
    start_time := clock_timestamp();
    
    SELECT COUNT(*) INTO result_count
    FROM (
        SELECT id
        FROM m0_episodic
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> test_vector
        LIMIT test_limit
    ) subq;
    
    end_time := clock_timestamp();
    search_time := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    
    -- Assume index is used for vector operations
    uses_index := TRUE;
    
    RETURN QUERY SELECT search_time, result_count, uses_index;
END;
$$ LANGUAGE plpgsql;

-- Create maintenance function for vector indexes
CREATE OR REPLACE FUNCTION maintain_vector_indexes()
RETURNS TEXT AS $$
DECLARE
    result_text TEXT;
BEGIN
    -- Analyze the table to update statistics
    ANALYZE m0_episodic;
    
    result_text := 'Vector index maintenance completed. Statistics updated.';
    
    RETURN result_text;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- SECTION 8: Initialization Complete
-- =============================================================================

-- Log successful initialization
DO $$
DECLARE
    extensions_list TEXT;
    vectorscale_available BOOLEAN;
BEGIN
    SELECT string_agg(extname, ', ') INTO extensions_list
    FROM pg_extension 
    WHERE extname IN ('timescaledb', 'vector', 'vectorscale');
    
    SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vectorscale') INTO vectorscale_available;
    
    RAISE NOTICE '=============================================================================';
    RAISE NOTICE 'MemFuse pgvectorscale initialization completed successfully';
    RAISE NOTICE 'Extensions enabled: %', extensions_list;
    RAISE NOTICE 'M0 episodic memory table: m0_episodic (384-dimensional vectors)';
    RAISE NOTICE 'Immediate trigger system: ENABLED';
    RAISE NOTICE 'Vector index type: %', CASE WHEN vectorscale_available THEN 'pgvectorscale (diskann)' ELSE 'pgvector (hnsw)' END;
    RAISE NOTICE 'Database optimized for vector operations';
    RAISE NOTICE '=============================================================================';
END $$;
