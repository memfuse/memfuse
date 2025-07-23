-- M0 Raw Data Memory Layer Schema
-- This schema defines the m0_raw table for storing raw data
-- with automatic embedding generation support

-- =============================================================================
-- M0 RAW DATA TABLE DEFINITION
-- =============================================================================

CREATE TABLE IF NOT EXISTS m0_raw (
    -- Primary identification
    id TEXT PRIMARY KEY,
    
    -- Raw content (original data without processing)
    content TEXT NOT NULL,
    
    -- Flexible metadata storage (includes session_id, user_id, etc.)
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- PgAI embedding infrastructure
    embedding VECTOR(384),  -- 384-dimensional embedding vector
    needs_embedding BOOLEAN DEFAULT TRUE,  -- Flag for automatic embedding generation
    retry_count INTEGER DEFAULT 0,  -- Number of embedding retry attempts
    last_retry_at TIMESTAMP,  -- Timestamp of last retry attempt
    retry_status TEXT DEFAULT 'pending' CHECK (retry_status IN ('pending', 'processing', 'completed', 'failed')),
    
    -- Audit timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- PERFORMANCE INDEXES
-- =============================================================================

-- Primary key index (automatic)
-- CREATE UNIQUE INDEX m0_raw_pkey ON m0_raw USING btree (id);

-- Metadata indexes for common queries
CREATE INDEX IF NOT EXISTS idx_m0_raw_session_id ON m0_raw USING btree ((metadata->>'session_id'));
CREATE INDEX IF NOT EXISTS idx_m0_raw_user_id ON m0_raw USING btree ((metadata->>'user_id'));

-- Embedding-related indexes
CREATE INDEX IF NOT EXISTS idx_m0_raw_needs_embedding ON m0_raw (needs_embedding) WHERE needs_embedding = TRUE;
CREATE INDEX IF NOT EXISTS idx_m0_raw_retry_status ON m0_raw (retry_status);
CREATE INDEX IF NOT EXISTS idx_m0_raw_retry_count ON m0_raw (retry_count);

-- Vector similarity index (HNSW for fast similarity search)
CREATE INDEX IF NOT EXISTS idx_m0_raw_embedding_hnsw ON m0_raw USING hnsw (embedding vector_cosine_ops);

-- Timestamp indexes for temporal queries
CREATE INDEX IF NOT EXISTS idx_m0_raw_created_at ON m0_raw (created_at);
CREATE INDEX IF NOT EXISTS idx_m0_raw_updated_at ON m0_raw (updated_at);

-- =============================================================================
-- AUTOMATIC TIMESTAMP UPDATES
-- =============================================================================

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_m0_raw_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update updated_at on row changes
DROP TRIGGER IF EXISTS trigger_update_m0_raw_updated_at ON m0_raw;
CREATE TRIGGER trigger_update_m0_raw_updated_at
    BEFORE UPDATE ON m0_raw
    FOR EACH ROW
    EXECUTE FUNCTION update_m0_raw_updated_at();

-- =============================================================================
-- PGAI IMMEDIATE TRIGGER SUPPORT
-- =============================================================================

-- Function to handle immediate embedding notifications
CREATE OR REPLACE FUNCTION notify_m0_raw_embedding_needed()
RETURNS TRIGGER AS $$
BEGIN
    -- Only notify if the record actually needs embedding
    IF NEW.needs_embedding = TRUE AND NEW.content IS NOT NULL THEN
        PERFORM pg_notify('embedding_needed_m0_raw', NEW.id);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for immediate embedding notifications
DROP TRIGGER IF EXISTS trigger_m0_raw_embedding_notification ON m0_raw;
CREATE TRIGGER trigger_m0_raw_embedding_notification
    AFTER INSERT OR UPDATE OF needs_embedding ON m0_raw
    FOR EACH ROW
    EXECUTE FUNCTION notify_m0_raw_embedding_needed();

-- =============================================================================
-- COMMENTS AND DOCUMENTATION
-- =============================================================================

COMMENT ON TABLE m0_raw IS 'M0 Raw Data Memory Layer - Stores original unprocessed data with automatic embedding generation';
COMMENT ON COLUMN m0_raw.id IS 'Unique identifier for the raw data record';
COMMENT ON COLUMN m0_raw.content IS 'Original raw content without any processing';
COMMENT ON COLUMN m0_raw.metadata IS 'Flexible JSONB metadata including session_id, user_id, and other context';
COMMENT ON COLUMN m0_raw.embedding IS '384-dimensional vector embedding for similarity search';
COMMENT ON COLUMN m0_raw.needs_embedding IS 'Flag indicating if this record needs embedding generation';
COMMENT ON COLUMN m0_raw.retry_count IS 'Number of embedding generation retry attempts';
COMMENT ON COLUMN m0_raw.last_retry_at IS 'Timestamp of the last embedding retry attempt';
COMMENT ON COLUMN m0_raw.retry_status IS 'Current status of embedding generation process';
