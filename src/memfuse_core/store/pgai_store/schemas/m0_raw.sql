-- M0 Raw Data Memory Layer Schema
-- This schema defines the m0_raw table for storing raw streaming messages
-- M0 layer stores original messages WITHOUT embeddings (embeddings are in M1)

-- =============================================================================
-- M0 RAW MESSAGES TABLE DEFINITION
-- =============================================================================

CREATE TABLE IF NOT EXISTS m0_raw (
    -- Primary identification
    message_id TEXT PRIMARY KEY,

    -- Content and metadata
    content TEXT NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),

    -- Streaming context
    conversation_id TEXT NOT NULL,
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

    -- Lineage tracking (which M1 chunks were created from this message)
    chunk_assignments TEXT[] DEFAULT '{}',

    -- Performance optimization
    CONSTRAINT unique_conversation_sequence
        UNIQUE (conversation_id, sequence_number)
);

-- =============================================================================
-- PERFORMANCE INDEXES
-- =============================================================================

-- Indexes for M0 layer performance
CREATE INDEX IF NOT EXISTS idx_m0_conversation_sequence
    ON m0_raw (conversation_id, sequence_number);

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

-- =============================================================================
-- AUTOMATIC TIMESTAMP UPDATES
-- =============================================================================

-- Function to update the processed_at timestamp when status changes to completed
CREATE OR REPLACE FUNCTION update_m0_processed_at()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.processing_status = 'completed' AND OLD.processing_status != 'completed' THEN
        NEW.processed_at = CURRENT_TIMESTAMP;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update processed_at when processing completes
DROP TRIGGER IF EXISTS trigger_update_m0_processed_at ON m0_raw;
CREATE TRIGGER trigger_update_m0_processed_at
    BEFORE UPDATE ON m0_raw
    FOR EACH ROW
    EXECUTE FUNCTION update_m0_processed_at();

-- =============================================================================
-- M1 PROCESSING TRIGGER SUPPORT
-- =============================================================================

-- Function to handle M1 processing notifications
CREATE OR REPLACE FUNCTION notify_m1_processing_needed()
RETURNS TRIGGER AS $$
BEGIN
    -- Notify M1 layer that new raw data is available for processing
    IF NEW.content IS NOT NULL THEN
        PERFORM pg_notify('m1_processing_needed', NEW.message_id::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for M1 processing notification
DROP TRIGGER IF EXISTS trigger_m0_m1_processing_notification ON m0_raw;
CREATE TRIGGER trigger_m0_m1_processing_notification
    AFTER INSERT ON m0_raw
    FOR EACH ROW
    EXECUTE FUNCTION notify_m1_processing_needed();

-- =============================================================================
-- COMMENTS AND DOCUMENTATION
-- =============================================================================

COMMENT ON TABLE m0_raw IS 'M0 Raw Messages Layer - Stores original streaming messages WITHOUT embeddings (embeddings are in M1)';
COMMENT ON COLUMN m0_raw.message_id IS 'Unique identifier for the raw message';
COMMENT ON COLUMN m0_raw.content IS 'Original message content without any processing';
COMMENT ON COLUMN m0_raw.role IS 'Message role: user, assistant, or system';
COMMENT ON COLUMN m0_raw.conversation_id IS 'Conversation context identifier';
COMMENT ON COLUMN m0_raw.sequence_number IS 'Message sequence number within conversation';
COMMENT ON COLUMN m0_raw.token_count IS 'Token count for chunking decisions';
COMMENT ON COLUMN m0_raw.processing_status IS 'Processing status for M1 chunking pipeline';
COMMENT ON COLUMN m0_raw.chunk_assignments IS 'Array of M1 chunk IDs created from this message';
COMMENT ON COLUMN m0_raw.processed_at IS 'Timestamp when M1 processing completed';
