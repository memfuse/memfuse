-- M2 Semantic Memory Layer Schema
-- This schema defines the m2_semantic table for storing semantic facts
-- with high-performance vector embeddings optimized for similarity search

-- =============================================================================
-- M2 SEMANTIC TABLE DEFINITION
-- =============================================================================

CREATE TABLE IF NOT EXISTS m2_semantic (
    -- Primary identification
    fact_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Fact content
    text TEXT NOT NULL,

    -- Idempotency hash for duplicate detection
    hash TEXT UNIQUE,

    -- Vector embedding (384 dimensions for sentence-transformers/all-MiniLM-L6-v2)
    embedding vector(384),

    -- Confidence score
    confidence FLOAT NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),

    -- Status management
    status VARCHAR(20) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'deprecated')),

    -- Source tracking (links back to M1 chunks)
    chunk_ids UUID[] NOT NULL DEFAULT '{}',

    -- User context
    user_id UUID NOT NULL,

    -- Policy versioning for extraction tracking
    policy_version TEXT,

    -- Temporal tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    embedding_generated_at TIMESTAMP WITH TIME ZONE,

    -- Quality metrics
    embedding_model VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',

    -- General metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Constraints
    CONSTRAINT m2_semantic_chunk_lineage_not_empty
        CHECK (array_length(chunk_ids, 1) > 0)
);

-- =============================================================================
-- HIGH-PERFORMANCE VECTOR INDEXES
-- =============================================================================

-- HNSW index for optimal vector similarity search
-- Features:
-- - Fast approximate nearest neighbor search
-- - Optimized for 384-dimensional embeddings
-- - Cosine distance for semantic similarity
CREATE INDEX IF NOT EXISTS idx_m2_embedding_hnsw
    ON m2_semantic
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Additional indexes for M2 layer performance
CREATE INDEX IF NOT EXISTS idx_m2_user_id
    ON m2_semantic (user_id);

CREATE INDEX IF NOT EXISTS idx_m2_status
    ON m2_semantic (status);

CREATE INDEX IF NOT EXISTS idx_m2_confidence
    ON m2_semantic (confidence);

CREATE INDEX IF NOT EXISTS idx_m2_created_at
    ON m2_semantic (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_m2_updated_at
    ON m2_semantic (updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_m2_policy_version
    ON m2_semantic (policy_version);

-- Hash index for duplicate detection
CREATE INDEX IF NOT EXISTS idx_m2_hash
    ON m2_semantic (hash);

-- GIN index for chunk ID arrays (lineage queries)
CREATE INDEX IF NOT EXISTS idx_m2_chunk_ids_gin
    ON m2_semantic USING gin (chunk_ids);

-- GIN index for metadata queries
CREATE INDEX IF NOT EXISTS idx_m2_metadata_gin
    ON m2_semantic USING gin (metadata);

-- =============================================================================
-- AUTOMATIC TIMESTAMP UPDATE TRIGGER
-- =============================================================================

-- Function to update the embedding_generated_at timestamp when embedding is set
CREATE OR REPLACE FUNCTION update_m2_embedding_generated_at()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.embedding IS NOT NULL AND OLD.embedding IS NULL THEN
        NEW.embedding_generated_at = CURRENT_TIMESTAMP;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update embedding_generated_at when embedding is created
DROP TRIGGER IF EXISTS trigger_update_m2_embedding_generated_at ON m2_semantic;
CREATE TRIGGER trigger_update_m2_embedding_generated_at
    BEFORE UPDATE ON m2_semantic
    FOR EACH ROW
    EXECUTE FUNCTION update_m2_embedding_generated_at();

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_m2_semantic_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update updated_at on changes
DROP TRIGGER IF EXISTS trigger_update_m2_semantic_updated_at ON m2_semantic;
CREATE TRIGGER trigger_update_m2_semantic_updated_at
    BEFORE UPDATE ON m2_semantic
    FOR EACH ROW
    EXECUTE FUNCTION update_m2_semantic_updated_at();

-- =============================================================================
-- EMBEDDING NOTIFICATION TRIGGER (for immediate embedding generation)
-- =============================================================================

-- Function to notify embedding system when new facts need embeddings
CREATE OR REPLACE FUNCTION notify_m2_embedding_needed()
RETURNS TRIGGER AS $$
BEGIN
    -- Notify embedding system when new facts are created without embeddings
    IF NEW.embedding IS NULL AND NEW.text IS NOT NULL THEN
        PERFORM pg_notify('embedding_needed', 'm2_semantic:' || NEW.fact_id::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for immediate embedding notification
DROP TRIGGER IF EXISTS trigger_m2_embedding_notification ON m2_semantic;
CREATE TRIGGER trigger_m2_embedding_notification
    AFTER INSERT ON m2_semantic
    FOR EACH ROW
    EXECUTE FUNCTION notify_m2_embedding_needed();

-- =============================================================================
-- DATA VALIDATION CONSTRAINTS
-- =============================================================================

-- Additional check constraints for data quality
ALTER TABLE m2_semantic
    ADD CONSTRAINT check_text_not_empty
    CHECK (length(trim(text)) > 0);

ALTER TABLE m2_semantic
    ADD CONSTRAINT check_confidence_range
    CHECK (confidence >= 0.0 AND confidence <= 1.0);

-- =============================================================================
-- COMMENTS FOR DOCUMENTATION
-- =============================================================================

COMMENT ON TABLE m2_semantic IS 'M2 Semantic Memory Layer - stores semantic facts with high-performance vector embeddings';

COMMENT ON COLUMN m2_semantic.fact_id IS 'Unique identifier for the semantic fact';
COMMENT ON COLUMN m2_semantic.text IS 'Semantic fact content optimized for search and retrieval';
COMMENT ON COLUMN m2_semantic.hash IS 'Hash for idempotency and duplicate detection';
COMMENT ON COLUMN m2_semantic.embedding IS '384-dimensional vector embedding for similarity search';
COMMENT ON COLUMN m2_semantic.confidence IS 'Confidence score for fact extraction (0.0 to 1.0)';
COMMENT ON COLUMN m2_semantic.status IS 'Status of the fact: active or deprecated';
COMMENT ON COLUMN m2_semantic.chunk_ids IS 'Array of M1 chunk IDs that contributed to this fact (lineage tracking)';
COMMENT ON COLUMN m2_semantic.user_id IS 'User identifier for multi-tenant isolation';
COMMENT ON COLUMN m2_semantic.policy_version IS 'Version of extraction policy used to generate this fact';
COMMENT ON COLUMN m2_semantic.created_at IS 'Timestamp when fact was created';
COMMENT ON COLUMN m2_semantic.updated_at IS 'Timestamp when fact was last updated';
COMMENT ON COLUMN m2_semantic.embedding_generated_at IS 'Timestamp when embedding was generated';
COMMENT ON COLUMN m2_semantic.embedding_model IS 'Model used for embedding generation';
COMMENT ON COLUMN m2_semantic.metadata IS 'Additional metadata for the semantic fact';