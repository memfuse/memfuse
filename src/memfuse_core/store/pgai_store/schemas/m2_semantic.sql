-- M2 Semantic Memory Layer Schema
-- This schema defines the m2_semantic table for storing semantic facts
-- with automatic embedding generation support

-- =============================================================================
-- M2 SEMANTIC TABLE DEFINITION
-- =============================================================================

CREATE TABLE IF NOT EXISTS m2_semantic (
    -- Primary identification
    id TEXT PRIMARY KEY,
    
    -- Source tracking (links back to M1 episodic memory)
    source_id TEXT,  -- References m1_episodic.id
    source_session_id TEXT,  -- Session context for fact
    source_user_id TEXT,  -- User context for fact
    
    -- Fact content and metadata
    fact_content TEXT NOT NULL,
    fact_type TEXT,  -- Open-ended fact type, no constraints for extensibility
    fact_category JSONB DEFAULT '{}'::jsonb,  -- Flexible categorization system
    confidence FLOAT NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    
    -- Structured fact data
    entities JSONB DEFAULT '[]'::jsonb,  -- Extracted entities from fact
    temporal_info JSONB DEFAULT '{}'::jsonb,  -- Temporal information (dates, times, etc.)
    source_context TEXT,  -- Brief context about where fact came from
    
    -- Semantic-specific fields
    is_verified BOOLEAN DEFAULT FALSE,  -- Whether fact has been verified
    verification_method TEXT,  -- How the fact was verified
    verification_source TEXT,  -- Source of verification
    verification_confidence FLOAT,  -- Confidence in verification (0.0 to 1.0)
    
    -- Conflict management
    conflict_status TEXT DEFAULT 'none',  -- none, potential, confirmed
    conflicting_facts JSONB DEFAULT '[]'::jsonb,  -- References to conflicting facts
    resolution_status TEXT,  -- How conflicts were resolved
    
    -- General metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- PgAI embedding infrastructure (identical to M0/M1)
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
-- CREATE UNIQUE INDEX m2_semantic_pkey ON m2_semantic USING btree (id);

-- Source tracking indexes
CREATE INDEX IF NOT EXISTS idx_m2_source_id ON m2_semantic (source_id);
CREATE INDEX IF NOT EXISTS idx_m2_source_session ON m2_semantic (source_session_id);
CREATE INDEX IF NOT EXISTS idx_m2_source_user ON m2_semantic (source_user_id);

-- Fact type and confidence indexes
CREATE INDEX IF NOT EXISTS idx_m2_fact_type ON m2_semantic (fact_type);
CREATE INDEX IF NOT EXISTS idx_m2_confidence ON m2_semantic (confidence);

-- Verification indexes
CREATE INDEX IF NOT EXISTS idx_m2_is_verified ON m2_semantic (is_verified);
CREATE INDEX IF NOT EXISTS idx_m2_verification_confidence ON m2_semantic (verification_confidence);

-- Conflict management indexes
CREATE INDEX IF NOT EXISTS idx_m2_conflict_status ON m2_semantic (conflict_status);
CREATE INDEX IF NOT EXISTS idx_m2_resolution_status ON m2_semantic (resolution_status);

-- Auto-embedding query optimization
CREATE INDEX IF NOT EXISTS idx_m2_needs_embedding ON m2_semantic (needs_embedding) 
WHERE needs_embedding = TRUE;

-- Retry management indexes
CREATE INDEX IF NOT EXISTS idx_m2_retry_status ON m2_semantic (retry_status);
CREATE INDEX IF NOT EXISTS idx_m2_retry_count ON m2_semantic (retry_count);

-- Temporal indexes
CREATE INDEX IF NOT EXISTS idx_m2_created_at ON m2_semantic (created_at);
CREATE INDEX IF NOT EXISTS idx_m2_updated_at ON m2_semantic (updated_at);

-- JSONB indexes for structured data
CREATE INDEX IF NOT EXISTS idx_m2_entities_gin ON m2_semantic USING gin (entities);
CREATE INDEX IF NOT EXISTS idx_m2_temporal_gin ON m2_semantic USING gin (temporal_info);
CREATE INDEX IF NOT EXISTS idx_m2_metadata_gin ON m2_semantic USING gin (metadata);
CREATE INDEX IF NOT EXISTS idx_m2_category_gin ON m2_semantic USING gin (fact_category);
CREATE INDEX IF NOT EXISTS idx_m2_conflicting_facts_gin ON m2_semantic USING gin (conflicting_facts);

-- Vector similarity search index (will be created after data insertion)
-- CREATE INDEX IF NOT EXISTS idx_m2_embedding_cosine ON m2_semantic
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- =============================================================================
-- TRIGGERS
-- =============================================================================

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_m2_semantic_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic updated_at updates
DROP TRIGGER IF EXISTS m2_semantic_updated_at_trigger ON m2_semantic;
CREATE TRIGGER m2_semantic_updated_at_trigger
    BEFORE UPDATE ON m2_semantic
    FOR EACH ROW
    EXECUTE FUNCTION update_m2_semantic_updated_at();

-- Create notification function for immediate embedding generation
CREATE OR REPLACE FUNCTION notify_m2_embedding_needed()
RETURNS TRIGGER AS $$
BEGIN
    -- Only notify if needs_embedding is TRUE
    IF NEW.needs_embedding = TRUE THEN
        PERFORM pg_notify('m2_embedding_needed', NEW.id::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for immediate embedding notifications
DROP TRIGGER IF EXISTS m2_semantic_embedding_trigger ON m2_semantic;
CREATE TRIGGER m2_semantic_embedding_trigger
    AFTER INSERT OR UPDATE OF needs_embedding ON m2_semantic
    FOR EACH ROW
    EXECUTE FUNCTION notify_m2_embedding_needed();

-- =============================================================================
-- LINEAGE TABLE
-- =============================================================================

-- Create lineage table to track fact provenance
CREATE TABLE IF NOT EXISTS m2_lineage (
    id TEXT PRIMARY KEY,
    fact_id TEXT NOT NULL REFERENCES m2_semantic(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,  -- m1_episodic, external, inference, etc.
    source_id TEXT,  -- ID of source record
    confidence FLOAT NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes on lineage table
CREATE INDEX IF NOT EXISTS idx_m2_lineage_fact_id ON m2_lineage (fact_id);
CREATE INDEX IF NOT EXISTS idx_m2_lineage_source_type ON m2_lineage (source_type);
CREATE INDEX IF NOT EXISTS idx_m2_lineage_source_id ON m2_lineage (source_id);

-- =============================================================================
-- CONFLICTS TABLE
-- =============================================================================

-- Create conflicts table to track and manage conflicting facts
CREATE TABLE IF NOT EXISTS m2_conflicts (
    id TEXT PRIMARY KEY,
    fact_id_1 TEXT NOT NULL REFERENCES m2_semantic(id) ON DELETE CASCADE,
    fact_id_2 TEXT NOT NULL REFERENCES m2_semantic(id) ON DELETE CASCADE,
    conflict_type TEXT NOT NULL,  -- contradiction, partial, temporal, etc.
    conflict_description TEXT,
    resolution_status TEXT DEFAULT 'unresolved',  -- unresolved, resolved_1, resolved_2, merged, etc.
    resolution_method TEXT,  -- llm, user, rule, etc.
    resolution_confidence FLOAT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

-- Create indexes on conflicts table
CREATE INDEX IF NOT EXISTS idx_m2_conflicts_fact_id_1 ON m2_conflicts (fact_id_1);
CREATE INDEX IF NOT EXISTS idx_m2_conflicts_fact_id_2 ON m2_conflicts (fact_id_2);
CREATE INDEX IF NOT EXISTS idx_m2_conflicts_resolution_status ON m2_conflicts (resolution_status);
CREATE INDEX IF NOT EXISTS idx_m2_conflicts_created_at ON m2_conflicts (created_at);

-- =============================================================================
-- COMMENTS FOR DOCUMENTATION
-- =============================================================================

COMMENT ON TABLE m2_semantic IS 'M2 Semantic Memory Layer - stores semantic facts extracted from M1 episodic memories with automatic embedding generation';

COMMENT ON COLUMN m2_semantic.id IS 'Unique identifier for the semantic fact';
COMMENT ON COLUMN m2_semantic.source_id IS 'Reference to the M1 episodic memory record this fact was extracted from';
COMMENT ON COLUMN m2_semantic.fact_content IS 'The semantic fact content';
COMMENT ON COLUMN m2_semantic.fact_type IS 'Open-ended classification of fact type, extensible for any categorization system';
COMMENT ON COLUMN m2_semantic.fact_category IS 'Flexible JSONB categorization system for complex fact classification';
COMMENT ON COLUMN m2_semantic.confidence IS 'Confidence score for fact extraction (0.0 to 1.0)';
COMMENT ON COLUMN m2_semantic.entities IS 'JSON array of entities mentioned in the fact';
COMMENT ON COLUMN m2_semantic.temporal_info IS 'JSON object containing temporal information (dates, times, etc.)';
COMMENT ON COLUMN m2_semantic.is_verified IS 'Whether this fact has been verified';
COMMENT ON COLUMN m2_semantic.conflict_status IS 'Status of any conflicts with other facts';
COMMENT ON COLUMN m2_semantic.embedding IS '384-dimensional vector embedding of the fact content';
COMMENT ON COLUMN m2_semantic.needs_embedding IS 'Flag indicating if this record needs embedding generation';

COMMENT ON TABLE m2_lineage IS 'Tracks the provenance and lineage of facts in the M2 semantic memory layer';
COMMENT ON TABLE m2_conflicts IS 'Manages conflicts between facts in the M2 semantic memory layer';
