-- MemFuse Multi-Layer Memory System Schema Initialization
-- This script initializes M2, M3, and MSMG layers to complete the memory hierarchy
-- Extends the base M0/M1 schema with semantic, procedural, and graph layers

-- =============================================================================
-- SECTION 1: M2 Semantic Memory Layer
-- =============================================================================

-- Create the M2 semantic memory table with immediate trigger support
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

-- Create M2 lineage table to track fact provenance
CREATE TABLE IF NOT EXISTS m2_lineage (
    id TEXT PRIMARY KEY,
    fact_id TEXT NOT NULL REFERENCES m2_semantic(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,  -- m1_episodic, external, inference, etc.
    source_id TEXT,  -- ID of source record
    confidence FLOAT NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create M2 conflicts table to track and manage conflicting facts
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

-- =============================================================================
-- SECTION 2: M3 Procedural Memory Layer
-- =============================================================================

-- Create the M3 procedural patterns table
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

-- Create the M3 skills table
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

-- Create the M3 behaviors table
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
-- SECTION 3: MSMG Multi-Scale Mental Graph Layer
-- =============================================================================

-- Create the MSMG instances table (Contextual Knowledge Graphs)
CREATE TABLE IF NOT EXISTS msmg_instances (
    -- Primary identification
    id TEXT PRIMARY KEY,
    
    -- Instance metadata
    instance_type TEXT NOT NULL,  -- entity, event, relation, property, etc.
    instance_name TEXT,
    instance_description TEXT,
    
    -- Context information
    context_id TEXT,  -- ID of the context (session, document, etc.)
    context_type TEXT,  -- session, document, workflow, etc.
    
    -- Source tracking
    source_layer TEXT,  -- m0, m1, m2, m3
    source_id TEXT,  -- ID in source layer
    source_confidence FLOAT CHECK (source_confidence >= 0.0 AND source_confidence <= 1.0),
    
    -- Ontology mapping
    ontology_class_id TEXT,  -- Reference to msmg_ontology.id
    
    -- Graph structure
    parent_instances JSONB DEFAULT '[]'::jsonb,  -- Parent instances in the graph
    child_instances JSONB DEFAULT '[]'::jsonb,  -- Child instances in the graph
    related_instances JSONB DEFAULT '[]'::jsonb,  -- Related instances (non-hierarchical)
    
    -- Properties
    properties JSONB DEFAULT '{}'::jsonb,  -- Instance properties
    temporal_info JSONB DEFAULT '{}'::jsonb,  -- Temporal information
    spatial_info JSONB DEFAULT '{}'::jsonb,  -- Spatial information
    
    -- General metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- PgAI embedding infrastructure
    embedding VECTOR(384),  -- 384-dimensional embedding vector
    needs_embedding BOOLEAN DEFAULT TRUE,
    
    -- Audit timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create the MSMG ontology table (Classes and Hierarchy)
CREATE TABLE IF NOT EXISTS msmg_ontology (
    -- Primary identification
    id TEXT PRIMARY KEY,
    
    -- Ontology metadata
    class_name TEXT NOT NULL,
    class_description TEXT,
    class_type TEXT,  -- entity_class, event_class, relation_class, etc.
    
    -- Hierarchy information
    parent_class_id TEXT,  -- Reference to parent class
    root_class BOOLEAN DEFAULT FALSE,  -- Whether this is a root class
    hierarchy_level INTEGER,  -- Level in the hierarchy (0 for root)
    
    -- Class structure
    properties_schema JSONB DEFAULT '{}'::jsonb,  -- Schema for properties
    required_properties JSONB DEFAULT '[]'::jsonb,  -- Required properties
    allowed_relations JSONB DEFAULT '[]'::jsonb,  -- Allowed relation types
    
    -- Constraints
    constraints JSONB DEFAULT '[]'::jsonb,  -- Class constraints
    validation_rules JSONB DEFAULT '[]'::jsonb,  -- Validation rules
    
    -- General metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- PgAI embedding infrastructure
    embedding VECTOR(384),  -- 384-dimensional embedding vector
    needs_embedding BOOLEAN DEFAULT TRUE,
    
    -- Audit timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- SECTION 4: Performance Indexes for All Layers
-- =============================================================================

-- M2 Semantic Memory Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m2_source_id ON m2_semantic (source_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m2_fact_type ON m2_semantic (fact_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m2_confidence ON m2_semantic (confidence);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m2_is_verified ON m2_semantic (is_verified);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m2_conflict_status ON m2_semantic (conflict_status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m2_needs_embedding ON m2_semantic (needs_embedding) WHERE needs_embedding = TRUE;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m2_created_at ON m2_semantic (created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m2_entities_gin ON m2_semantic USING gin (entities);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m2_metadata_gin ON m2_semantic USING gin (metadata);

-- M2 Lineage Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m2_lineage_fact_id ON m2_lineage (fact_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m2_lineage_source_type ON m2_lineage (source_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m2_lineage_source_id ON m2_lineage (source_id);

-- M2 Conflicts Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m2_conflicts_fact_id_1 ON m2_conflicts (fact_id_1);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m2_conflicts_fact_id_2 ON m2_conflicts (fact_id_2);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m2_conflicts_resolution_status ON m2_conflicts (resolution_status);

-- M3 Procedural Patterns Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_procedural_source_id ON m3_procedural (source_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_procedural_pattern_type ON m3_procedural (pattern_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_procedural_confidence ON m3_procedural (confidence);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_procedural_success_rate ON m3_procedural (success_rate);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_procedural_needs_embedding ON m3_procedural (needs_embedding) WHERE needs_embedding = TRUE;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_procedural_created_at ON m3_procedural (created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_procedural_steps_gin ON m3_procedural USING gin (steps);

-- M3 Skills Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_skills_skill_name ON m3_skills (skill_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_skills_skill_type ON m3_skills (skill_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_skills_proficiency ON m3_skills (proficiency_level);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_skills_needs_embedding ON m3_skills (needs_embedding) WHERE needs_embedding = TRUE;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_skills_created_at ON m3_skills (created_at);

-- M3 Behaviors Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_behaviors_behavior_name ON m3_behaviors (behavior_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_behaviors_behavior_type ON m3_behaviors (behavior_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_behaviors_priority ON m3_behaviors (priority);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_behaviors_needs_embedding ON m3_behaviors (needs_embedding) WHERE needs_embedding = TRUE;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_behaviors_created_at ON m3_behaviors (created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_behaviors_triggers_gin ON m3_behaviors USING gin (triggers);

-- MSMG Instances Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_instances_instance_type ON msmg_instances (instance_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_instances_context_id ON msmg_instances (context_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_instances_source_layer ON msmg_instances (source_layer);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_instances_source_id ON msmg_instances (source_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_instances_ontology_class_id ON msmg_instances (ontology_class_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_instances_needs_embedding ON msmg_instances (needs_embedding) WHERE needs_embedding = TRUE;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_instances_created_at ON msmg_instances (created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_instances_properties_gin ON msmg_instances USING gin (properties);

-- MSMG Ontology Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_ontology_class_name ON msmg_ontology (class_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_ontology_class_type ON msmg_ontology (class_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_ontology_parent_class_id ON msmg_ontology (parent_class_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_ontology_root_class ON msmg_ontology (root_class) WHERE root_class = TRUE;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_ontology_hierarchy_level ON msmg_ontology (hierarchy_level);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_ontology_needs_embedding ON msmg_ontology (needs_embedding) WHERE needs_embedding = TRUE;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_ontology_created_at ON msmg_ontology (created_at);

-- MSMG Relations Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_relations_source_id ON msmg_relations (source_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_relations_target_id ON msmg_relations (target_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_relations_relation_type ON msmg_relations (relation_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_relations_created_at ON msmg_relations (created_at);

-- MSMG Meta-cognitive Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_metacognitive_target_id ON msmg_metacognitive (target_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_metacognitive_target_type ON msmg_metacognitive (target_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_metacognitive_certainty_level ON msmg_metacognitive (certainty_level);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_metacognitive_importance_level ON msmg_metacognitive (importance_level);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_metacognitive_last_accessed_at ON msmg_metacognitive (last_accessed_at);

-- =============================================================================
-- SECTION 5: Vector Similarity Search Indexes
-- =============================================================================

-- M2 Semantic Memory Vector Index
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vectorscale') THEN
        BEGIN
            EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m2_semantic_embedding_vectorscale
                     ON m2_semantic USING diskann (embedding vector_cosine_ops)';
            RAISE NOTICE 'Created M2 pgvectorscale diskann index for optimal performance';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'M2 diskann index creation failed, falling back to HNSW: %', SQLERRM;
            EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m2_semantic_embedding_hnsw
                     ON m2_semantic USING hnsw (embedding vector_cosine_ops)';
        END;
    ELSE
        EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m2_semantic_embedding_hnsw
                 ON m2_semantic USING hnsw (embedding vector_cosine_ops)';
        RAISE NOTICE 'Created M2 standard HNSW index for vector similarity search';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'M2 Vector index creation failed, will use sequential scan: %', SQLERRM;
END $$;

-- M3 Procedural Patterns Vector Index
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vectorscale') THEN
        BEGIN
            EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_procedural_embedding_vectorscale
                     ON m3_procedural USING diskann (embedding vector_cosine_ops)';
            RAISE NOTICE 'Created M3 Procedural pgvectorscale diskann index for optimal performance';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'M3 Procedural diskann index creation failed, falling back to HNSW: %', SQLERRM;
            EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_procedural_embedding_hnsw
                     ON m3_procedural USING hnsw (embedding vector_cosine_ops)';
        END;
    ELSE
        EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_procedural_embedding_hnsw
                 ON m3_procedural USING hnsw (embedding vector_cosine_ops)';
        RAISE NOTICE 'Created M3 Procedural standard HNSW index for vector similarity search';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'M3 Procedural Vector index creation failed, will use sequential scan: %', SQLERRM;
END $$;

-- M3 Skills Vector Index
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vectorscale') THEN
        BEGIN
            EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_skills_embedding_vectorscale
                     ON m3_skills USING diskann (embedding vector_cosine_ops)';
            RAISE NOTICE 'Created M3 Skills pgvectorscale diskann index for optimal performance';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'M3 Skills diskann index creation failed, falling back to HNSW: %', SQLERRM;
            EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_skills_embedding_hnsw
                     ON m3_skills USING hnsw (embedding vector_cosine_ops)';
        END;
    ELSE
        EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_skills_embedding_hnsw
                 ON m3_skills USING hnsw (embedding vector_cosine_ops)';
        RAISE NOTICE 'Created M3 Skills standard HNSW index for vector similarity search';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'M3 Skills Vector index creation failed, will use sequential scan: %', SQLERRM;
END $$;

-- M3 Behaviors Vector Index
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vectorscale') THEN
        BEGIN
            EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_behaviors_embedding_vectorscale
                     ON m3_behaviors USING diskann (embedding vector_cosine_ops)';
            RAISE NOTICE 'Created M3 Behaviors pgvectorscale diskann index for optimal performance';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'M3 Behaviors diskann index creation failed, falling back to HNSW: %', SQLERRM;
            EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_behaviors_embedding_hnsw
                     ON m3_behaviors USING hnsw (embedding vector_cosine_ops)';
        END;
    ELSE
        EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_m3_behaviors_embedding_hnsw
                 ON m3_behaviors USING hnsw (embedding vector_cosine_ops)';
        RAISE NOTICE 'Created M3 Behaviors standard HNSW index for vector similarity search';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'M3 Behaviors Vector index creation failed, will use sequential scan: %', SQLERRM;
END $$;

-- MSMG Instances Vector Index
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vectorscale') THEN
        BEGIN
            EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_instances_embedding_vectorscale
                     ON msmg_instances USING diskann (embedding vector_cosine_ops)';
            RAISE NOTICE 'Created MSMG Instances pgvectorscale diskann index for optimal performance';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'MSMG Instances diskann index creation failed, falling back to HNSW: %', SQLERRM;
            EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_instances_embedding_hnsw
                     ON msmg_instances USING hnsw (embedding vector_cosine_ops)';
        END;
    ELSE
        EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_instances_embedding_hnsw
                 ON msmg_instances USING hnsw (embedding vector_cosine_ops)';
        RAISE NOTICE 'Created MSMG Instances standard HNSW index for vector similarity search';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'MSMG Instances Vector index creation failed, will use sequential scan: %', SQLERRM;
END $$;

-- MSMG Ontology Vector Index
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vectorscale') THEN
        BEGIN
            EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_ontology_embedding_vectorscale
                     ON msmg_ontology USING diskann (embedding vector_cosine_ops)';
            RAISE NOTICE 'Created MSMG Ontology pgvectorscale diskann index for optimal performance';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'MSMG Ontology diskann index creation failed, falling back to HNSW: %', SQLERRM;
            EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_ontology_embedding_hnsw
                     ON msmg_ontology USING hnsw (embedding vector_cosine_ops)';
        END;
    ELSE
        EXECUTE 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_msmg_ontology_embedding_hnsw
                 ON msmg_ontology USING hnsw (embedding vector_cosine_ops)';
        RAISE NOTICE 'Created MSMG Ontology standard HNSW index for vector similarity search';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'MSMG Ontology Vector index creation failed, will use sequential scan: %', SQLERRM;
END $$;

-- =============================================================================
-- SECTION 6: Trigger System for All Layers
-- =============================================================================

-- M2 Semantic Memory Triggers
CREATE OR REPLACE FUNCTION update_m2_semantic_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS m2_semantic_updated_at_trigger ON m2_semantic;
CREATE TRIGGER m2_semantic_updated_at_trigger
    BEFORE UPDATE ON m2_semantic
    FOR EACH ROW
    EXECUTE FUNCTION update_m2_semantic_updated_at();

CREATE OR REPLACE FUNCTION notify_m2_embedding_needed()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.needs_embedding = TRUE THEN
        PERFORM pg_notify('m2_embedding_needed', NEW.id::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS m2_semantic_embedding_trigger ON m2_semantic;
CREATE TRIGGER m2_semantic_embedding_trigger
    AFTER INSERT OR UPDATE OF needs_embedding ON m2_semantic
    FOR EACH ROW
    EXECUTE FUNCTION notify_m2_embedding_needed();

-- M3 Procedural Patterns Triggers
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

CREATE OR REPLACE FUNCTION notify_m3_procedural_embedding_needed()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.needs_embedding = TRUE THEN
        PERFORM pg_notify('m3_procedural_embedding_needed', NEW.id::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS m3_procedural_embedding_trigger ON m3_procedural;
CREATE TRIGGER m3_procedural_embedding_trigger
    AFTER INSERT OR UPDATE OF needs_embedding ON m3_procedural
    FOR EACH ROW
    EXECUTE FUNCTION notify_m3_procedural_embedding_needed();

-- M3 Skills Triggers
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

CREATE OR REPLACE FUNCTION notify_m3_skills_embedding_needed()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.needs_embedding = TRUE THEN
        PERFORM pg_notify('m3_skills_embedding_needed', NEW.id::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS m3_skills_embedding_trigger ON m3_skills;
CREATE TRIGGER m3_skills_embedding_trigger
    AFTER INSERT OR UPDATE OF needs_embedding ON m3_skills
    FOR EACH ROW
    EXECUTE FUNCTION notify_m3_skills_embedding_needed();

-- M3 Behaviors Triggers
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

CREATE OR REPLACE FUNCTION notify_m3_behaviors_embedding_needed()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.needs_embedding = TRUE THEN
        PERFORM pg_notify('m3_behaviors_embedding_needed', NEW.id::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS m3_behaviors_embedding_trigger ON m3_behaviors;
CREATE TRIGGER m3_behaviors_embedding_trigger
    AFTER INSERT OR UPDATE OF needs_embedding ON m3_behaviors
    FOR EACH ROW
    EXECUTE FUNCTION notify_m3_behaviors_embedding_needed();

-- MSMG Instances Triggers
CREATE OR REPLACE FUNCTION update_msmg_instances_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS msmg_instances_updated_at_trigger ON msmg_instances;
CREATE TRIGGER msmg_instances_updated_at_trigger
    BEFORE UPDATE ON msmg_instances
    FOR EACH ROW
    EXECUTE FUNCTION update_msmg_instances_updated_at();

CREATE OR REPLACE FUNCTION notify_msmg_instances_embedding_needed()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.needs_embedding = TRUE THEN
        PERFORM pg_notify('msmg_instances_embedding_needed', NEW.id::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS msmg_instances_embedding_trigger ON msmg_instances;
CREATE TRIGGER msmg_instances_embedding_trigger
    AFTER INSERT OR UPDATE OF needs_embedding ON msmg_instances
    FOR EACH ROW
    EXECUTE FUNCTION notify_msmg_instances_embedding_needed();

-- MSMG Ontology Triggers
CREATE OR REPLACE FUNCTION update_msmg_ontology_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS msmg_ontology_updated_at_trigger ON msmg_ontology;
CREATE TRIGGER msmg_ontology_updated_at_trigger
    BEFORE UPDATE ON msmg_ontology
    FOR EACH ROW
    EXECUTE FUNCTION update_msmg_ontology_updated_at();

CREATE OR REPLACE FUNCTION notify_msmg_ontology_embedding_needed()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.needs_embedding = TRUE THEN
        PERFORM pg_notify('msmg_ontology_embedding_needed', NEW.id::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS msmg_ontology_embedding_trigger ON msmg_ontology;
CREATE TRIGGER msmg_ontology_embedding_trigger
    AFTER INSERT OR UPDATE OF needs_embedding ON msmg_ontology
    FOR EACH ROW
    EXECUTE FUNCTION notify_msmg_ontology_embedding_needed();

-- =============================================================================
-- SECTION 7: Foreign Key Constraints and Referential Integrity
-- =============================================================================

-- Add foreign key constraints for referential integrity
ALTER TABLE msmg_instances
ADD CONSTRAINT fk_msmg_instances_ontology_class
FOREIGN KEY (ontology_class_id) REFERENCES msmg_ontology(id) ON DELETE SET NULL;

ALTER TABLE msmg_ontology
ADD CONSTRAINT fk_msmg_ontology_parent_class
FOREIGN KEY (parent_class_id) REFERENCES msmg_ontology(id) ON DELETE CASCADE;

-- =============================================================================
-- SECTION 8: Enhanced Statistics and Monitoring
-- =============================================================================

-- Set up automatic statistics collection for all vector columns
ALTER TABLE m2_semantic ALTER COLUMN embedding SET STATISTICS 1000;
ALTER TABLE m3_procedural ALTER COLUMN embedding SET STATISTICS 1000;
ALTER TABLE m3_skills ALTER COLUMN embedding SET STATISTICS 1000;
ALTER TABLE m3_behaviors ALTER COLUMN embedding SET STATISTICS 1000;
ALTER TABLE msmg_instances ALTER COLUMN embedding SET STATISTICS 1000;
ALTER TABLE msmg_ontology ALTER COLUMN embedding SET STATISTICS 1000;

-- Update the comprehensive vector statistics view
CREATE OR REPLACE VIEW multi_layer_vector_stats AS
WITH layer_stats AS (
    -- M0 Raw Data Layer
    SELECT
        'M0' as memory_layer,
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
        'M1' as memory_layer,
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

    UNION ALL

    -- M2 Semantic Memory Layer
    SELECT
        'M2' as memory_layer,
        'm2_semantic' as table_name,
        COUNT(*) as total_rows,
        COUNT(embedding) as rows_with_embeddings,
        COUNT(*) - COUNT(embedding) as rows_without_embeddings,
        CASE
            WHEN COUNT(*) > 0 THEN ROUND((COUNT(embedding)::float / COUNT(*)::float) * 100, 2)
            ELSE 0
        END as embedding_completion_percentage,
        pg_size_pretty(pg_total_relation_size('m2_semantic')) as table_size
    FROM m2_semantic

    UNION ALL

    -- M3 Procedural Patterns
    SELECT
        'M3' as memory_layer,
        'm3_procedural' as table_name,
        COUNT(*) as total_rows,
        COUNT(embedding) as rows_with_embeddings,
        COUNT(*) - COUNT(embedding) as rows_without_embeddings,
        CASE
            WHEN COUNT(*) > 0 THEN ROUND((COUNT(embedding)::float / COUNT(*)::float) * 100, 2)
            ELSE 0
        END as embedding_completion_percentage,
        pg_size_pretty(pg_total_relation_size('m3_procedural')) as table_size
    FROM m3_procedural

    UNION ALL

    -- M3 Skills
    SELECT
        'M3' as memory_layer,
        'm3_skills' as table_name,
        COUNT(*) as total_rows,
        COUNT(embedding) as rows_with_embeddings,
        COUNT(*) - COUNT(embedding) as rows_without_embeddings,
        CASE
            WHEN COUNT(*) > 0 THEN ROUND((COUNT(embedding)::float / COUNT(*)::float) * 100, 2)
            ELSE 0
        END as embedding_completion_percentage,
        pg_size_pretty(pg_total_relation_size('m3_skills')) as table_size
    FROM m3_skills

    UNION ALL

    -- M3 Behaviors
    SELECT
        'M3' as memory_layer,
        'm3_behaviors' as table_name,
        COUNT(*) as total_rows,
        COUNT(embedding) as rows_with_embeddings,
        COUNT(*) - COUNT(embedding) as rows_without_embeddings,
        CASE
            WHEN COUNT(*) > 0 THEN ROUND((COUNT(embedding)::float / COUNT(*)::float) * 100, 2)
            ELSE 0
        END as embedding_completion_percentage,
        pg_size_pretty(pg_total_relation_size('m3_behaviors')) as table_size
    FROM m3_behaviors

    UNION ALL

    -- MSMG Instances
    SELECT
        'MSMG' as memory_layer,
        'msmg_instances' as table_name,
        COUNT(*) as total_rows,
        COUNT(embedding) as rows_with_embeddings,
        COUNT(*) - COUNT(embedding) as rows_without_embeddings,
        CASE
            WHEN COUNT(*) > 0 THEN ROUND((COUNT(embedding)::float / COUNT(*)::float) * 100, 2)
            ELSE 0
        END as embedding_completion_percentage,
        pg_size_pretty(pg_total_relation_size('msmg_instances')) as table_size
    FROM msmg_instances

    UNION ALL

    -- MSMG Ontology
    SELECT
        'MSMG' as memory_layer,
        'msmg_ontology' as table_name,
        COUNT(*) as total_rows,
        COUNT(embedding) as rows_with_embeddings,
        COUNT(*) - COUNT(embedding) as rows_without_embeddings,
        CASE
            WHEN COUNT(*) > 0 THEN ROUND((COUNT(embedding)::float / COUNT(*)::float) * 100, 2)
            ELSE 0
        END as embedding_completion_percentage,
        pg_size_pretty(pg_total_relation_size('msmg_ontology')) as table_size
    FROM msmg_ontology
)
SELECT * FROM layer_stats ORDER BY memory_layer, table_name;

-- =============================================================================
-- SECTION 9: Maintenance and Utility Functions
-- =============================================================================

-- Enhanced maintenance function for all vector indexes
CREATE OR REPLACE FUNCTION maintain_all_vector_indexes()
RETURNS TEXT AS $$
DECLARE
    result_text TEXT;
BEGIN
    -- Analyze all memory layer tables to update statistics
    ANALYZE m0_raw;
    ANALYZE m1_episodic;
    ANALYZE m2_semantic;
    ANALYZE m3_procedural;
    ANALYZE m3_skills;
    ANALYZE m3_behaviors;
    ANALYZE msmg_instances;
    ANALYZE msmg_ontology;

    result_text := 'Vector index maintenance completed for all memory layers (M0/M1/M2/M3/MSMG). Statistics updated.';

    RETURN result_text;
END;
$$ LANGUAGE plpgsql;

-- Enhanced trigger system status function
CREATE OR REPLACE FUNCTION get_multi_layer_trigger_status()
RETURNS TABLE(
    memory_layer TEXT,
    trigger_name TEXT,
    table_name TEXT,
    trigger_enabled BOOLEAN,
    function_exists BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        CASE
            WHEN c.relname LIKE 'm0_%' THEN 'M0'
            WHEN c.relname LIKE 'm1_%' THEN 'M1'
            WHEN c.relname LIKE 'm2_%' THEN 'M2'
            WHEN c.relname LIKE 'm3_%' THEN 'M3'
            WHEN c.relname LIKE 'msmg_%' THEN 'MSMG'
            ELSE 'Unknown'
        END as memory_layer,
        t.tgname::TEXT as trigger_name,
        c.relname::TEXT as table_name,
        t.tgenabled = 'O' as trigger_enabled,
        EXISTS(SELECT 1 FROM pg_proc p WHERE p.proname LIKE 'notify_%_embedding_needed') as function_exists
    FROM pg_trigger t
    JOIN pg_class c ON t.tgrelid = c.oid
    WHERE c.relname IN ('m0_raw', 'm1_episodic', 'm2_semantic', 'm3_procedural', 'm3_skills', 'm3_behaviors', 'msmg_instances', 'msmg_ontology')
    AND t.tgname LIKE '%_embedding_trigger'
    ORDER BY memory_layer, c.relname, t.tgname;
END;
$$ LANGUAGE plpgsql;

-- Function to get comprehensive memory layer statistics
CREATE OR REPLACE FUNCTION get_memory_layer_summary()
RETURNS TABLE(
    memory_layer TEXT,
    total_tables INTEGER,
    total_records BIGINT,
    total_embeddings BIGINT,
    overall_completion_percentage NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    WITH layer_summary AS (
        SELECT 'M0' as layer, 1 as tables, COUNT(*) as records, COUNT(embedding) as embeddings FROM m0_raw
        UNION ALL
        SELECT 'M1' as layer, 1 as tables, COUNT(*) as records, COUNT(embedding) as embeddings FROM m1_episodic
        UNION ALL
        SELECT 'M2' as layer, 1 as tables, COUNT(*) as records, COUNT(embedding) as embeddings FROM m2_semantic
        UNION ALL
        SELECT 'M3' as layer, 3 as tables,
               (SELECT COUNT(*) FROM m3_procedural) + (SELECT COUNT(*) FROM m3_skills) + (SELECT COUNT(*) FROM m3_behaviors) as records,
               (SELECT COUNT(embedding) FROM m3_procedural) + (SELECT COUNT(embedding) FROM m3_skills) + (SELECT COUNT(embedding) FROM m3_behaviors) as embeddings
        UNION ALL
        SELECT 'MSMG' as layer, 2 as tables,
               (SELECT COUNT(*) FROM msmg_instances) + (SELECT COUNT(*) FROM msmg_ontology) as records,
               (SELECT COUNT(embedding) FROM msmg_instances) + (SELECT COUNT(embedding) FROM msmg_ontology) as embeddings
    )
    SELECT
        layer as memory_layer,
        tables as total_tables,
        records as total_records,
        embeddings as total_embeddings,
        CASE
            WHEN records > 0 THEN ROUND((embeddings::float / records::float) * 100, 2)
            ELSE 0
        END as overall_completion_percentage
    FROM layer_summary
    ORDER BY layer;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- SECTION 10: Initialization Complete
-- =============================================================================

-- Log successful multi-layer initialization
DO $$
DECLARE
    extensions_list TEXT;
    vectorscale_available BOOLEAN;
    table_counts TEXT;
BEGIN
    SELECT string_agg(extname, ', ') INTO extensions_list
    FROM pg_extension
    WHERE extname IN ('timescaledb', 'vector', 'vectorscale');

    SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vectorscale') INTO vectorscale_available;

    -- Count all tables
    SELECT string_agg(
        format('%s: %s',
            CASE
                WHEN tablename LIKE 'm0_%' THEN 'M0'
                WHEN tablename LIKE 'm1_%' THEN 'M1'
                WHEN tablename LIKE 'm2_%' THEN 'M2'
                WHEN tablename LIKE 'm3_%' THEN 'M3'
                WHEN tablename LIKE 'msmg_%' THEN 'MSMG'
                ELSE 'Other'
            END,
            tablename
        ),
        ', '
    ) INTO table_counts
    FROM pg_tables
    WHERE tablename IN ('m0_raw', 'm1_episodic', 'm2_semantic', 'm3_procedural', 'm3_skills', 'm3_behaviors', 'msmg_instances', 'msmg_ontology', 'm2_lineage', 'm2_conflicts', 'msmg_relations', 'msmg_metacognitive');

    RAISE NOTICE '=============================================================================';
    RAISE NOTICE 'MemFuse Complete Multi-Layer Memory System initialization completed successfully';
    RAISE NOTICE 'Extensions enabled: %', extensions_list;
    RAISE NOTICE 'Memory layers initialized:';
    RAISE NOTICE '  - M0 Raw Data Layer: m0_raw (384-dimensional vectors)';
    RAISE NOTICE '  - M1 Episodic Memory Layer: m1_episodic (384-dimensional vectors)';
    RAISE NOTICE '  - M2 Semantic Memory Layer: m2_semantic + lineage + conflicts (384-dimensional vectors)';
    RAISE NOTICE '  - M3 Procedural Memory Layer: m3_procedural + m3_skills + m3_behaviors (384-dimensional vectors)';
    RAISE NOTICE '  - MSMG Multi-Scale Mental Graph: msmg_instances + msmg_ontology + relations + metacognitive';
    RAISE NOTICE 'Immediate trigger system: ENABLED for all layers';
    RAISE NOTICE 'Vector index type: %', CASE WHEN vectorscale_available THEN 'pgvectorscale (diskann)' ELSE 'pgvector (hnsw)' END;
    RAISE NOTICE 'Database optimized for multi-layer vector operations';
    RAISE NOTICE 'Tables created: %', table_counts;
    RAISE NOTICE '=============================================================================';
    RAISE NOTICE 'Use SELECT * FROM multi_layer_vector_stats; to view embedding statistics';
    RAISE NOTICE 'Use SELECT * FROM get_memory_layer_summary(); to view layer summary';
    RAISE NOTICE 'Use SELECT * FROM get_multi_layer_trigger_status(); to view trigger status';
    RAISE NOTICE '=============================================================================';
END $$;

-- Create the MSMG relations table
CREATE TABLE IF NOT EXISTS msmg_relations (
    -- Primary identification
    id TEXT PRIMARY KEY,
    
    -- Relation endpoints
    source_id TEXT NOT NULL,  -- Source instance or class
    source_type TEXT NOT NULL,  -- instance or ontology
    target_id TEXT NOT NULL,  -- Target instance or class
    target_type TEXT NOT NULL,  -- instance or ontology
    
    -- Relation metadata
    relation_type TEXT NOT NULL,  -- is_a, has_part, located_in, etc.
    relation_name TEXT,
    relation_description TEXT,
    bidirectional BOOLEAN DEFAULT FALSE,
    
    -- Relation properties
    weight FLOAT DEFAULT 1.0,  -- Relation strength/weight
    confidence FLOAT CHECK (confidence >= 0.0 AND confidence <= 1.0),
    temporal_info JSONB DEFAULT '{}'::jsonb,  -- Temporal constraints
    
    -- General metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Audit timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create the MSMG meta-cognitive table
CREATE TABLE IF NOT EXISTS msmg_metacognitive (
    -- Primary identification
    id TEXT PRIMARY KEY,
    
    -- Target reference
    target_id TEXT NOT NULL,  -- ID of target (instance, ontology, relation)
    target_type TEXT NOT NULL,  -- instance, ontology, relation
    
    -- Meta-cognitive information
    knowledge_provenance JSONB DEFAULT '{}'::jsonb,  -- Knowledge source and history
    certainty_level FLOAT CHECK (certainty_level >= 0.0 AND certainty_level <= 1.0),
    importance_level FLOAT DEFAULT 0.5,  -- Importance for reasoning
    
    -- Learning metadata
    learning_history JSONB DEFAULT '[]'::jsonb,  -- History of learning events
    reinforcement_count INTEGER DEFAULT 0,  -- Times this knowledge was reinforced
    contradiction_count INTEGER DEFAULT 0,  -- Times this knowledge was contradicted
    
    -- Forgetting mechanism
    last_accessed_at TIMESTAMP,  -- Last time this knowledge was accessed
    access_count INTEGER DEFAULT 0,  -- Number of times accessed
    decay_rate FLOAT DEFAULT 0.1,  -- Rate of memory decay
    forgetting_threshold FLOAT DEFAULT 0.2,  -- Threshold for forgetting
    
    -- General metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Audit timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
