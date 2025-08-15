-- M1 Chunks Memory Layer Schema
-- This schema defines the m1_episodic table for storing intelligent chunks
-- with high-performance vector embeddings optimized for similarity search

-- =============================================================================
-- M1 CHUNKS TABLE DEFINITION
-- =============================================================================

CREATE TABLE IF NOT EXISTS m1_episodic (
    -- Primary identification
    chunk_id TEXT PRIMARY KEY,

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
    m0_message_ids TEXT[] NOT NULL DEFAULT '{}',
    conversation_id TEXT NOT NULL,

    -- Temporal tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    embedding_generated_at TIMESTAMP WITH TIME ZONE,

    -- Quality metrics
    embedding_model VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    chunk_quality_score FLOAT DEFAULT 0.0,

    -- Constraints
    CONSTRAINT m1_chunks_embedding_not_null
        CHECK (embedding IS NOT NULL),
    CONSTRAINT m1_chunks_m0_lineage_not_empty
        CHECK (array_length(m0_message_ids, 1) > 0)
);

-- =============================================================================
-- HIGH-PERFORMANCE VECTOR INDEXES
-- =============================================================================

-- HNSW index for optimal vector similarity search
-- Features:
-- - Fast approximate nearest neighbor search
-- - Optimized for 384-dimensional embeddings
-- - Cosine distance for semantic similarity
CREATE INDEX IF NOT EXISTS idx_m1_embedding_hnsw
    ON m1_episodic
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Additional indexes for M1 layer performance
CREATE INDEX IF NOT EXISTS idx_m1_conversation_id
    ON m1_episodic (conversation_id);

CREATE INDEX IF NOT EXISTS idx_m1_chunking_strategy
    ON m1_episodic (chunking_strategy);

CREATE INDEX IF NOT EXISTS idx_m1_created_at
    ON m1_episodic (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_m1_token_count
    ON m1_episodic (token_count);

CREATE INDEX IF NOT EXISTS idx_m1_chunk_quality_score
    ON m1_episodic (chunk_quality_score);

-- GIN index for M0 message ID arrays (lineage queries)
CREATE INDEX IF NOT EXISTS idx_m1_m0_message_ids_gin
    ON m1_episodic USING gin (m0_message_ids);

-- =============================================================================
-- AUTOMATIC TIMESTAMP UPDATE TRIGGER
-- =============================================================================

-- Function to update the embedding_generated_at timestamp when embedding is set
CREATE OR REPLACE FUNCTION update_m1_embedding_generated_at()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.embedding IS NOT NULL AND OLD.embedding IS NULL THEN
        NEW.embedding_generated_at = CURRENT_TIMESTAMP;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update embedding_generated_at when embedding is created
DROP TRIGGER IF EXISTS trigger_update_m1_embedding_generated_at ON m1_episodic;
CREATE TRIGGER trigger_update_m1_embedding_generated_at
    BEFORE UPDATE ON m1_episodic
    FOR EACH ROW
    EXECUTE FUNCTION update_m1_embedding_generated_at();

-- =============================================================================
-- EMBEDDING NOTIFICATION TRIGGER (for immediate embedding generation)
-- =============================================================================

-- Function to notify embedding system when new chunks need embeddings
CREATE OR REPLACE FUNCTION notify_m1_embedding_needed()
RETURNS TRIGGER AS $$
BEGIN
    -- Notify embedding system when new chunks are created without embeddings
    IF NEW.embedding IS NULL AND NEW.content IS NOT NULL THEN
        PERFORM pg_notify('embedding_needed', 'm1_episodic:' || NEW.chunk_id::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for immediate embedding notification
DROP TRIGGER IF EXISTS trigger_m1_embedding_notification ON m1_episodic;
CREATE TRIGGER trigger_m1_embedding_notification
    AFTER INSERT ON m1_episodic
    FOR EACH ROW
    EXECUTE FUNCTION notify_m1_embedding_needed();

-- =============================================================================
-- DATA VALIDATION CONSTRAINTS
-- =============================================================================

-- Additional check constraints for data quality
ALTER TABLE m1_episodic
    ADD CONSTRAINT check_content_not_empty
    CHECK (length(trim(content)) > 0);

ALTER TABLE m1_episodic
    ADD CONSTRAINT check_token_count_non_negative
    CHECK (token_count >= 0);

-- =============================================================================
-- COMMENTS FOR DOCUMENTATION
-- =============================================================================

COMMENT ON TABLE m1_episodic IS 'M1 Chunks Memory Layer - stores intelligent chunks with high-performance vector embeddings';

COMMENT ON COLUMN m1_episodic.chunk_id IS 'Unique identifier for the chunk';
COMMENT ON COLUMN m1_episodic.content IS 'Chunked content optimized for semantic search';
COMMENT ON COLUMN m1_episodic.chunking_strategy IS 'Strategy used for chunking: token_based, semantic, conversation_turn, or hybrid';
COMMENT ON COLUMN m1_episodic.token_count IS 'Number of tokens in the chunk';
COMMENT ON COLUMN m1_episodic.embedding IS '384-dimensional vector embedding for similarity search';
COMMENT ON COLUMN m1_episodic.m0_message_ids IS 'Array of M0 message IDs that contributed to this chunk (lineage tracking)';
COMMENT ON COLUMN m1_episodic.conversation_id IS 'Conversation context identifier';
COMMENT ON COLUMN m1_episodic.embedding_generated_at IS 'Timestamp when embedding was generated';
COMMENT ON COLUMN m1_episodic.embedding_model IS 'Model used for embedding generation';
COMMENT ON COLUMN m1_episodic.chunk_quality_score IS 'Quality score for the chunk (0.0 to 1.0)';