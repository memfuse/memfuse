-- MSMG (Multi-Scale Mental Graph) Schema
-- This schema defines the MSMG tables for storing the multi-scale mental graph
-- with instance layer and ontology layer

-- =============================================================================
-- MSMG INSTANCES TABLE (Contextual Knowledge Graphs)
-- =============================================================================

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

-- =============================================================================
-- MSMG ONTOLOGY TABLE (Classes and Hierarchy)
-- =============================================================================

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
-- MSMG RELATIONS TABLE
-- =============================================================================

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

-- =============================================================================
-- MSMG META-COGNITIVE TABLE
-- =============================================================================

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

-- =============================================================================
-- PERFORMANCE INDEXES
-- =============================================================================

-- MSMG Instances Indexes
CREATE INDEX IF NOT EXISTS idx_msmg_instances_instance_type ON msmg_instances (instance_type);
CREATE INDEX IF NOT EXISTS idx_msmg_instances_context_id ON msmg_instances (context_id);
CREATE INDEX IF NOT EXISTS idx_msmg_instances_source_layer ON msmg_instances (source_layer);
CREATE INDEX IF NOT EXISTS idx_msmg_instances_source_id ON msmg_instances (source_id);
CREATE INDEX IF NOT EXISTS idx_msmg_instances_ontology_class_id ON msmg_instances (ontology_class_id);
CREATE INDEX IF NOT EXISTS idx_msmg_instances_needs_embedding ON msmg_instances (needs_embedding) WHERE needs_embedding = TRUE;
CREATE INDEX IF NOT EXISTS idx_msmg_instances_created_at ON msmg_instances (created_at);

-- MSMG Ontology Indexes
CREATE INDEX IF NOT EXISTS idx_msmg_ontology_class_name ON msmg_ontology (class_name);
CREATE INDEX IF NOT EXISTS idx_msmg_ontology_class_type ON msmg_ontology (class_type);
CREATE INDEX IF NOT EXISTS idx_msmg_ontology_parent_class_id ON msmg_ontology (parent_class_id);
CREATE INDEX IF NOT EXISTS idx_msmg_ontology_root_class ON msmg_ontology (root_class) WHERE root_class = TRUE;
CREATE INDEX IF NOT EXISTS idx_msmg_ontology_hierarchy_level ON msmg_ontology (hierarchy_level);
CREATE INDEX IF NOT EXISTS idx_msmg_ontology_needs_embedding ON msmg_ontology (needs_embedding) WHERE needs_embedding = TRUE;
CREATE INDEX IF NOT EXISTS idx_msmg_ontology_created_at ON msmg_ontology (created_at);

-- MSMG Relations Indexes
CREATE INDEX IF NOT EXISTS idx_msmg_relations_source_id ON msmg_relations (source_id);
CREATE INDEX IF NOT EXISTS idx_msmg_relations_target_id ON msmg_relations (target_id);
CREATE INDEX IF NOT EXISTS idx_msmg_relations_relation_type ON msmg_relations (relation_type);
CREATE INDEX IF NOT EXISTS idx_msmg_relations_created_at ON msmg_relations (created_at);

-- MSMG Meta-cognitive Indexes
CREATE INDEX IF NOT EXISTS idx_msmg_metacognitive_target_id ON msmg_metacognitive (target_id);
CREATE INDEX IF NOT EXISTS idx_msmg_metacognitive_target_type ON msmg_metacognitive (target_type);
CREATE INDEX IF NOT EXISTS idx_msmg_metacognitive_certainty_level ON msmg_metacognitive (certainty_level);
CREATE INDEX IF NOT EXISTS idx_msmg_metacognitive_importance_level ON msmg_metacognitive (importance_level);
CREATE INDEX IF NOT EXISTS idx_msmg_metacognitive_last_accessed_at ON msmg_metacognitive (last_accessed_at);

-- JSONB Indexes
CREATE INDEX IF NOT EXISTS idx_msmg_instances_parent_instances_gin ON msmg_instances USING gin (parent_instances);
CREATE INDEX IF NOT EXISTS idx_msmg_instances_child_instances_gin ON msmg_instances USING gin (child_instances);
CREATE INDEX IF NOT EXISTS idx_msmg_instances_properties_gin ON msmg_instances USING gin (properties);
CREATE INDEX IF NOT EXISTS idx_msmg_ontology_properties_schema_gin ON msmg_ontology USING gin (properties_schema);
CREATE INDEX IF NOT EXISTS idx_msmg_metacognitive_knowledge_provenance_gin ON msmg_metacognitive USING gin (knowledge_provenance);

-- =============================================================================
-- TRIGGERS
-- =============================================================================

-- MSMG Instances updated_at trigger
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

-- MSMG Ontology updated_at trigger
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

-- MSMG embedding notification functions
CREATE OR REPLACE FUNCTION notify_msmg_instances_embedding_needed()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.needs_embedding = TRUE THEN
        PERFORM pg_notify('msmg_instances_embedding_needed', NEW.id::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION notify_msmg_ontology_embedding_needed()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.needs_embedding = TRUE THEN
        PERFORM pg_notify('msmg_ontology_embedding_needed', NEW.id::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- MSMG embedding triggers
DROP TRIGGER IF EXISTS msmg_instances_embedding_trigger ON msmg_instances;
CREATE TRIGGER msmg_instances_embedding_trigger
    AFTER INSERT OR UPDATE OF needs_embedding ON msmg_instances
    FOR EACH ROW
    EXECUTE FUNCTION notify_msmg_instances_embedding_needed();

DROP TRIGGER IF EXISTS msmg_ontology_embedding_trigger ON msmg_ontology;
CREATE TRIGGER msmg_ontology_embedding_trigger
    AFTER INSERT OR UPDATE OF needs_embedding ON msmg_ontology
    FOR EACH ROW
    EXECUTE FUNCTION notify_msmg_ontology_embedding_needed();

-- =============================================================================
-- FOREIGN KEY CONSTRAINTS
-- =============================================================================

-- Add foreign key constraints for referential integrity
ALTER TABLE msmg_instances
ADD CONSTRAINT fk_msmg_instances_ontology_class
FOREIGN KEY (ontology_class_id) REFERENCES msmg_ontology(id) ON DELETE SET NULL;

ALTER TABLE msmg_ontology
ADD CONSTRAINT fk_msmg_ontology_parent_class
FOREIGN KEY (parent_class_id) REFERENCES msmg_ontology(id) ON DELETE CASCADE;

-- =============================================================================
-- COMMENTS FOR DOCUMENTATION
-- =============================================================================

COMMENT ON TABLE msmg_instances IS 'MSMG Instance Layer - stores contextual knowledge graph instances with automatic embedding generation';
COMMENT ON TABLE msmg_ontology IS 'MSMG Ontology Layer - stores classes, hierarchies, and properties for knowledge organization';
COMMENT ON TABLE msmg_relations IS 'MSMG Relations - stores relationships between instances and ontology classes';
COMMENT ON TABLE msmg_metacognitive IS 'MSMG Meta-cognitive Layer - tracks knowledge provenance, certainty, and forgetting mechanisms';

COMMENT ON COLUMN msmg_instances.instance_type IS 'Type of instance (entity, event, relation, property, etc.)';
COMMENT ON COLUMN msmg_instances.context_id IS 'ID of the context this instance belongs to (session, document, etc.)';
COMMENT ON COLUMN msmg_instances.source_layer IS 'Memory layer this instance was derived from (m0, m1, m2, m3)';
COMMENT ON COLUMN msmg_instances.ontology_class_id IS 'Reference to the ontology class this instance belongs to';

COMMENT ON COLUMN msmg_ontology.class_name IS 'Name of the ontology class';
COMMENT ON COLUMN msmg_ontology.parent_class_id IS 'Reference to parent class in the hierarchy';
COMMENT ON COLUMN msmg_ontology.hierarchy_level IS 'Level in the class hierarchy (0 for root classes)';
COMMENT ON COLUMN msmg_ontology.properties_schema IS 'JSON schema defining allowed properties for instances of this class';

COMMENT ON COLUMN msmg_relations.source_id IS 'ID of the source entity (instance or ontology class)';
COMMENT ON COLUMN msmg_relations.target_id IS 'ID of the target entity (instance or ontology class)';
COMMENT ON COLUMN msmg_relations.relation_type IS 'Type of relation (is_a, has_part, located_in, etc.)';

COMMENT ON COLUMN msmg_metacognitive.knowledge_provenance IS 'JSON object tracking the source and history of this knowledge';
COMMENT ON COLUMN msmg_metacognitive.certainty_level IS 'Confidence level in this knowledge (0.0 to 1.0)';
COMMENT ON COLUMN msmg_metacognitive.decay_rate IS 'Rate at which this knowledge decays over time';

-- =============================================================================
-- UTILITY FUNCTIONS
-- =============================================================================

-- Function to get MSMG statistics
CREATE OR REPLACE FUNCTION get_msmg_stats()
RETURNS TABLE(
    layer_name TEXT,
    table_name TEXT,
    total_records INTEGER,
    records_with_embeddings INTEGER,
    embedding_completion_percentage NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        'MSMG' as layer_name,
        'msmg_instances' as table_name,
        COUNT(*)::INTEGER as total_records,
        COUNT(embedding)::INTEGER as records_with_embeddings,
        CASE
            WHEN COUNT(*) > 0 THEN ROUND((COUNT(embedding)::float / COUNT(*)::float) * 100, 2)
            ELSE 0
        END as embedding_completion_percentage
    FROM msmg_instances

    UNION ALL

    SELECT
        'MSMG' as layer_name,
        'msmg_ontology' as table_name,
        COUNT(*)::INTEGER as total_records,
        COUNT(embedding)::INTEGER as records_with_embeddings,
        CASE
            WHEN COUNT(*) > 0 THEN ROUND((COUNT(embedding)::float / COUNT(*)::float) * 100, 2)
            ELSE 0
        END as embedding_completion_percentage
    FROM msmg_ontology;
END;
$$ LANGUAGE plpgsql;
