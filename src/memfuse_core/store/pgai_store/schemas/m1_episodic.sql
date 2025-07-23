-- M1 Episodic Memory Layer Schema
-- This schema defines the m1_episodic table for storing episodic memories
-- with automatic embedding generation support

-- =============================================================================
-- M1 EPISODIC TABLE DEFINITION
-- =============================================================================

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

-- =============================================================================
-- PERFORMANCE INDEXES
-- =============================================================================

-- Primary key index (automatic)
-- CREATE UNIQUE INDEX m1_episodic_pkey ON m1_episodic USING btree (id);

-- Source tracking indexes
CREATE INDEX IF NOT EXISTS idx_m1_source_id ON m1_episodic (source_id);
CREATE INDEX IF NOT EXISTS idx_m1_source_session ON m1_episodic (source_session_id);
CREATE INDEX IF NOT EXISTS idx_m1_source_user ON m1_episodic (source_user_id);

-- Fact classification indexes (flexible for any fact_type values)
CREATE INDEX IF NOT EXISTS idx_m1_fact_type ON m1_episodic (fact_type) WHERE fact_type IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_m1_fact_category_gin ON m1_episodic USING gin (fact_category);
CREATE INDEX IF NOT EXISTS idx_m1_confidence ON m1_episodic (confidence);

-- Embedding processing indexes (for automatic embedding generation)
CREATE INDEX IF NOT EXISTS idx_m1_needs_embedding ON m1_episodic (needs_embedding)
    WHERE needs_embedding = TRUE;
CREATE INDEX IF NOT EXISTS idx_m1_retry_status ON m1_episodic (retry_status);
CREATE INDEX IF NOT EXISTS idx_m1_retry_count ON m1_episodic (retry_count);

-- Temporal indexes
CREATE INDEX IF NOT EXISTS idx_m1_created_at ON m1_episodic (created_at);
CREATE INDEX IF NOT EXISTS idx_m1_updated_at ON m1_episodic (updated_at);

-- JSONB indexes for structured data
CREATE INDEX IF NOT EXISTS idx_m1_entities_gin ON m1_episodic USING gin (entities);
CREATE INDEX IF NOT EXISTS idx_m1_temporal_gin ON m1_episodic USING gin (temporal_info);
CREATE INDEX IF NOT EXISTS idx_m1_metadata_gin ON m1_episodic USING gin (metadata);

-- Vector similarity search index (will be created after data insertion)
-- CREATE INDEX IF NOT EXISTS idx_m1_embedding_cosine ON m1_episodic
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- =============================================================================
-- AUTOMATIC TIMESTAMP UPDATE TRIGGER
-- =============================================================================

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_m1_episodic_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update updated_at on row changes
DROP TRIGGER IF EXISTS trigger_update_m1_episodic_updated_at ON m1_episodic;
CREATE TRIGGER trigger_update_m1_episodic_updated_at
    BEFORE UPDATE ON m1_episodic
    FOR EACH ROW
    EXECUTE FUNCTION update_m1_episodic_updated_at();

-- =============================================================================
-- EMBEDDING NOTIFICATION TRIGGER (for immediate embedding generation)
-- =============================================================================

-- Function to notify embedding system when new episodes need embeddings
CREATE OR REPLACE FUNCTION notify_m1_embedding_needed()
RETURNS TRIGGER AS $$
BEGIN
    -- Only notify if needs_embedding is TRUE and content exists
    IF NEW.needs_embedding = TRUE AND NEW.fact_content IS NOT NULL THEN
        PERFORM pg_notify('embedding_needed', 'm1_episodic:' || NEW.id::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for immediate embedding notification
DROP TRIGGER IF EXISTS trigger_m1_embedding_notification ON m1_episodic;
CREATE TRIGGER trigger_m1_embedding_notification
    AFTER INSERT OR UPDATE OF needs_embedding ON m1_episodic
    FOR EACH ROW
    EXECUTE FUNCTION notify_m1_embedding_needed();

-- =============================================================================
-- DATA VALIDATION CONSTRAINTS
-- =============================================================================

-- Additional check constraints for data quality
ALTER TABLE m1_episodic
    ADD CONSTRAINT check_fact_content_not_empty
    CHECK (length(trim(fact_content)) > 0);

ALTER TABLE m1_episodic
    ADD CONSTRAINT check_retry_count_non_negative
    CHECK (retry_count >= 0);

-- =============================================================================
-- COMMENTS FOR DOCUMENTATION
-- =============================================================================

COMMENT ON TABLE m1_episodic IS 'M1 Episodic Memory Layer - stores episodic memories from M0 raw data with automatic embedding generation';

COMMENT ON COLUMN m1_episodic.id IS 'Unique identifier for the episodic memory';
COMMENT ON COLUMN m1_episodic.source_id IS 'Reference to the M0 raw data record this episode was extracted from';
COMMENT ON COLUMN m1_episodic.fact_content IS 'The episodic memory content';
COMMENT ON COLUMN m1_episodic.fact_type IS 'Open-ended classification of episodic memory type, extensible for any categorization system';
COMMENT ON COLUMN m1_episodic.fact_category IS 'Flexible JSONB categorization system for complex episodic memory classification';
COMMENT ON COLUMN m1_episodic.confidence IS 'Confidence score for episodic memory extraction (0.0 to 1.0)';
COMMENT ON COLUMN m1_episodic.entities IS 'JSON array of entities mentioned in the episodic memory';
COMMENT ON COLUMN m1_episodic.temporal_info IS 'JSON object containing temporal information (dates, times, etc.)';
COMMENT ON COLUMN m1_episodic.embedding IS '384-dimensional vector embedding of the episodic memory content';
COMMENT ON COLUMN m1_episodic.needs_embedding IS 'Flag indicating if this record needs embedding generation';
COMMENT ON COLUMN m1_episodic.retry_count IS 'Number of times embedding generation has been attempted';
COMMENT ON COLUMN m1_episodic.retry_status IS 'Current status of embedding generation process';