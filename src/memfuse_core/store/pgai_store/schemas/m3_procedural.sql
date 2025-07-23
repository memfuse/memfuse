-- M3 Procedural Memory Layer Schema
-- This schema defines the M3 procedural memory tables for storing skills, behaviors, and patterns
-- with automatic embedding generation support

-- =============================================================================
-- M3 PROCEDURAL PATTERNS TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS m3_procedural (
    -- Primary identification
    id TEXT PRIMARY KEY,
    
    -- Source tracking (links back to M2 semantic memory)
    source_id TEXT,  -- References m2_semantic.id
    source_session_id TEXT,  -- Session context for pattern
    source_user_id TEXT,  -- User context for pattern
    
    -- Pattern content and metadata
    pattern_content TEXT NOT NULL,
    pattern_type TEXT,  -- workflow, decision_tree, algorithm, etc.
    pattern_category JSONB DEFAULT '{}'::jsonb,  -- Flexible categorization system
    confidence FLOAT NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    
    -- Pattern structure
    steps JSONB DEFAULT '[]'::jsonb,  -- Ordered list of steps in the pattern
    conditions JSONB DEFAULT '[]'::jsonb,  -- Conditions for pattern execution
    outcomes JSONB DEFAULT '[]'::jsonb,  -- Expected outcomes
    dependencies JSONB DEFAULT '[]'::jsonb,  -- Dependencies on other patterns/skills
    
    -- Performance metrics
    success_rate FLOAT,  -- Historical success rate (0.0 to 1.0)
    execution_count INTEGER DEFAULT 0,  -- Number of times executed
    last_executed_at TIMESTAMP,  -- Last execution timestamp
    average_duration INTERVAL,  -- Average execution time
    
    -- General metadata
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
-- M3 SKILLS TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS m3_skills (
    -- Primary identification
    id TEXT PRIMARY KEY,
    
    -- Skill definition
    skill_name TEXT NOT NULL,
    skill_description TEXT,
    skill_type TEXT,  -- cognitive, motor, social, technical, etc.
    skill_category JSONB DEFAULT '{}'::jsonb,
    
    -- Skill parameters
    parameters JSONB DEFAULT '{}'::jsonb,  -- Input parameters
    outputs JSONB DEFAULT '{}'::jsonb,  -- Expected outputs
    preconditions JSONB DEFAULT '[]'::jsonb,  -- Prerequisites
    postconditions JSONB DEFAULT '[]'::jsonb,  -- Post-execution conditions
    
    -- Skill relationships
    parent_skills JSONB DEFAULT '[]'::jsonb,  -- Skills this depends on
    child_skills JSONB DEFAULT '[]'::jsonb,  -- Skills that depend on this
    related_patterns JSONB DEFAULT '[]'::jsonb,  -- Related procedural patterns
    
    -- Performance metrics
    proficiency_level FLOAT DEFAULT 0.0,  -- Current proficiency (0.0 to 1.0)
    learning_rate FLOAT,  -- How quickly this skill improves
    decay_rate FLOAT,  -- How quickly this skill degrades without use
    last_used_at TIMESTAMP,  -- Last time skill was used
    
    -- General metadata
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
-- M3 BEHAVIORS TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS m3_behaviors (
    -- Primary identification
    id TEXT PRIMARY KEY,
    
    -- Behavior definition
    behavior_name TEXT NOT NULL,
    behavior_description TEXT,
    behavior_type TEXT,  -- reactive, proactive, adaptive, etc.
    behavior_category JSONB DEFAULT '{}'::jsonb,
    
    -- Behavior triggers and conditions
    triggers JSONB DEFAULT '[]'::jsonb,  -- What triggers this behavior
    conditions JSONB DEFAULT '[]'::jsonb,  -- Conditions for behavior activation
    actions JSONB DEFAULT '[]'::jsonb,  -- Actions taken when behavior is active
    
    -- Behavior parameters
    priority FLOAT DEFAULT 0.5,  -- Priority level (0.0 to 1.0)
    activation_threshold FLOAT DEFAULT 0.5,  -- Threshold for activation
    duration_limit INTERVAL,  -- Maximum duration for behavior
    cooldown_period INTERVAL,  -- Cooldown before reactivation
    
    -- Learning and adaptation
    adaptation_rate FLOAT DEFAULT 0.1,  -- How quickly behavior adapts
    reinforcement_history JSONB DEFAULT '[]'::jsonb,  -- History of reinforcements
    success_patterns JSONB DEFAULT '[]'::jsonb,  -- Patterns of successful execution
    
    -- Performance metrics
    activation_count INTEGER DEFAULT 0,  -- Number of times activated
    success_count INTEGER DEFAULT 0,  -- Number of successful executions
    last_activated_at TIMESTAMP,  -- Last activation timestamp
    
    -- General metadata
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

-- M3 Procedural Patterns Indexes
CREATE INDEX IF NOT EXISTS idx_m3_procedural_source_id ON m3_procedural (source_id);
CREATE INDEX IF NOT EXISTS idx_m3_procedural_pattern_type ON m3_procedural (pattern_type);
CREATE INDEX IF NOT EXISTS idx_m3_procedural_confidence ON m3_procedural (confidence);
CREATE INDEX IF NOT EXISTS idx_m3_procedural_success_rate ON m3_procedural (success_rate);
CREATE INDEX IF NOT EXISTS idx_m3_procedural_needs_embedding ON m3_procedural (needs_embedding) WHERE needs_embedding = TRUE;
CREATE INDEX IF NOT EXISTS idx_m3_procedural_created_at ON m3_procedural (created_at);

-- M3 Skills Indexes
CREATE INDEX IF NOT EXISTS idx_m3_skills_skill_name ON m3_skills (skill_name);
CREATE INDEX IF NOT EXISTS idx_m3_skills_skill_type ON m3_skills (skill_type);
CREATE INDEX IF NOT EXISTS idx_m3_skills_proficiency ON m3_skills (proficiency_level);
CREATE INDEX IF NOT EXISTS idx_m3_skills_needs_embedding ON m3_skills (needs_embedding) WHERE needs_embedding = TRUE;
CREATE INDEX IF NOT EXISTS idx_m3_skills_created_at ON m3_skills (created_at);

-- M3 Behaviors Indexes
CREATE INDEX IF NOT EXISTS idx_m3_behaviors_behavior_name ON m3_behaviors (behavior_name);
CREATE INDEX IF NOT EXISTS idx_m3_behaviors_behavior_type ON m3_behaviors (behavior_type);
CREATE INDEX IF NOT EXISTS idx_m3_behaviors_priority ON m3_behaviors (priority);
CREATE INDEX IF NOT EXISTS idx_m3_behaviors_needs_embedding ON m3_behaviors (needs_embedding) WHERE needs_embedding = TRUE;
CREATE INDEX IF NOT EXISTS idx_m3_behaviors_created_at ON m3_behaviors (created_at);

-- JSONB indexes for structured data
CREATE INDEX IF NOT EXISTS idx_m3_procedural_steps_gin ON m3_procedural USING gin (steps);
CREATE INDEX IF NOT EXISTS idx_m3_procedural_conditions_gin ON m3_procedural USING gin (conditions);
CREATE INDEX IF NOT EXISTS idx_m3_skills_parameters_gin ON m3_skills USING gin (parameters);
CREATE INDEX IF NOT EXISTS idx_m3_behaviors_triggers_gin ON m3_behaviors USING gin (triggers);

-- =============================================================================
-- TRIGGERS
-- =============================================================================

-- M3 Procedural updated_at trigger
CREATE OR REPLACE FUNCTION update_m3_procedural_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS m3_procedural_updated_at_trigger ON m3_procedural;
CREATE TRIGGER m3_procedural_updated_at_trigger
    BEFORE UPDATE ON m3_procedural
    FOR EACH ROW
    EXECUTE FUNCTION update_m3_procedural_updated_at();

-- M3 Skills updated_at trigger
CREATE OR REPLACE FUNCTION update_m3_skills_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS m3_skills_updated_at_trigger ON m3_skills;
CREATE TRIGGER m3_skills_updated_at_trigger
    BEFORE UPDATE ON m3_skills
    FOR EACH ROW
    EXECUTE FUNCTION update_m3_skills_updated_at();

-- M3 Behaviors updated_at trigger
CREATE OR REPLACE FUNCTION update_m3_behaviors_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS m3_behaviors_updated_at_trigger ON m3_behaviors;
CREATE TRIGGER m3_behaviors_updated_at_trigger
    BEFORE UPDATE ON m3_behaviors
    FOR EACH ROW
    EXECUTE FUNCTION update_m3_behaviors_updated_at();

-- =============================================================================
-- EMBEDDING NOTIFICATION TRIGGERS
-- =============================================================================

-- M3 Procedural embedding notification function
CREATE OR REPLACE FUNCTION notify_m3_procedural_embedding_needed()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.needs_embedding = TRUE THEN
        PERFORM pg_notify('m3_procedural_embedding_needed', NEW.id::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- M3 Skills embedding notification function
CREATE OR REPLACE FUNCTION notify_m3_skills_embedding_needed()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.needs_embedding = TRUE THEN
        PERFORM pg_notify('m3_skills_embedding_needed', NEW.id::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- M3 Behaviors embedding notification function
CREATE OR REPLACE FUNCTION notify_m3_behaviors_embedding_needed()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.needs_embedding = TRUE THEN
        PERFORM pg_notify('m3_behaviors_embedding_needed', NEW.id::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create embedding triggers for all M3 tables
DROP TRIGGER IF EXISTS m3_procedural_embedding_trigger ON m3_procedural;
CREATE TRIGGER m3_procedural_embedding_trigger
    AFTER INSERT OR UPDATE OF needs_embedding ON m3_procedural
    FOR EACH ROW
    EXECUTE FUNCTION notify_m3_procedural_embedding_needed();

DROP TRIGGER IF EXISTS m3_skills_embedding_trigger ON m3_skills;
CREATE TRIGGER m3_skills_embedding_trigger
    AFTER INSERT OR UPDATE OF needs_embedding ON m3_skills
    FOR EACH ROW
    EXECUTE FUNCTION notify_m3_skills_embedding_needed();

DROP TRIGGER IF EXISTS m3_behaviors_embedding_trigger ON m3_behaviors;
CREATE TRIGGER m3_behaviors_embedding_trigger
    AFTER INSERT OR UPDATE OF needs_embedding ON m3_behaviors
    FOR EACH ROW
    EXECUTE FUNCTION notify_m3_behaviors_embedding_needed();

-- =============================================================================
-- COMMENTS FOR DOCUMENTATION
-- =============================================================================

COMMENT ON TABLE m3_procedural IS 'M3 Procedural Memory Layer - stores procedural patterns and workflows with automatic embedding generation';
COMMENT ON TABLE m3_skills IS 'M3 Skills - stores learned skills and capabilities with proficiency tracking';
COMMENT ON TABLE m3_behaviors IS 'M3 Behaviors - stores adaptive behaviors and their activation patterns';

COMMENT ON COLUMN m3_procedural.pattern_content IS 'The procedural pattern or workflow content';
COMMENT ON COLUMN m3_procedural.steps IS 'JSON array of ordered steps in the procedural pattern';
COMMENT ON COLUMN m3_procedural.success_rate IS 'Historical success rate of this pattern (0.0 to 1.0)';

COMMENT ON COLUMN m3_skills.skill_name IS 'Name of the skill';
COMMENT ON COLUMN m3_skills.proficiency_level IS 'Current proficiency level (0.0 to 1.0)';
COMMENT ON COLUMN m3_skills.learning_rate IS 'Rate at which this skill improves with practice';

COMMENT ON COLUMN m3_behaviors.behavior_name IS 'Name of the behavior';
COMMENT ON COLUMN m3_behaviors.triggers IS 'JSON array of conditions that trigger this behavior';
COMMENT ON COLUMN m3_behaviors.priority IS 'Priority level for behavior activation (0.0 to 1.0)';
