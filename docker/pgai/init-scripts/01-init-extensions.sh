#!/bin/bash
# MemFuse PgAI Database Initialization Script
# This script initializes the database with required extensions and schema

set -e

echo "🚀 Starting MemFuse PgAI database initialization..."

# Function to execute SQL with error handling
execute_sql() {
    local sql="$1"
    local description="$2"
    
    echo "📝 $description"
    if psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -c "$sql"; then
        echo "✅ $description completed successfully"
    else
        echo "❌ $description failed"
        exit 1
    fi
}

# Create extensions
echo "🔧 Creating PostgreSQL extensions..."

execute_sql "CREATE EXTENSION IF NOT EXISTS vector;" "Creating pgvector extension"
execute_sql "CREATE EXTENSION IF NOT EXISTS pgai;" "Creating pgai extension"

# Verify extensions
echo "🔍 Verifying extensions..."
execute_sql "SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'pgai');" "Listing installed extensions"

# Create MemFuse schema
echo "🏗️ Creating MemFuse database schema..."

# Create m0_raw table (M0 Raw Data Layer)
execute_sql "
CREATE TABLE IF NOT EXISTS m0_raw (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    metadata        JSONB DEFAULT '{}'::jsonb,
    embedding       VECTOR(384),
    needs_embedding BOOLEAN DEFAULT TRUE,
    retry_count     INTEGER DEFAULT 0,
    last_retry_at   TIMESTAMP,
    retry_status    TEXT DEFAULT 'pending',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
" "Creating m0_raw table (M0 Raw Data Layer)"

# Create m1_episodic table (M1 Episodic Memory Layer)
execute_sql "
CREATE TABLE IF NOT EXISTS m1_episodic (
    id                  TEXT PRIMARY KEY,
    source_id           TEXT,
    source_session_id   TEXT,
    source_user_id      TEXT,
    episode_content     TEXT NOT NULL,
    episode_type        TEXT,
    episode_category    JSONB DEFAULT '{}'::jsonb,
    confidence          FLOAT NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    entities            JSONB DEFAULT '[]'::jsonb,
    temporal_info       JSONB DEFAULT '{}'::jsonb,
    source_context      TEXT,
    metadata            JSONB DEFAULT '{}'::jsonb,
    embedding           VECTOR(384),
    needs_embedding     BOOLEAN DEFAULT TRUE,
    retry_count         INTEGER DEFAULT 0,
    last_retry_at       TIMESTAMP,
    retry_status        TEXT DEFAULT 'pending',
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
" "Creating m1_episodic table (M1 Episodic Memory Layer)"

# Create indexes for performance
echo "📊 Creating performance indexes..."

# M0 Raw Data Layer indexes
execute_sql "CREATE INDEX IF NOT EXISTS idx_m0_raw_needs_embedding ON m0_raw (needs_embedding) WHERE needs_embedding = TRUE;" "Creating M0 needs_embedding index"

execute_sql "CREATE INDEX IF NOT EXISTS idx_m0_raw_created ON m0_raw (created_at);" "Creating M0 created_at index"

execute_sql "CREATE INDEX IF NOT EXISTS idx_m0_raw_retry_status ON m0_raw (retry_status);" "Creating M0 retry_status index"

execute_sql "CREATE INDEX IF NOT EXISTS idx_m0_raw_retry_count ON m0_raw (retry_count);" "Creating M0 retry_count index"

# M1 Episodic Memory Layer indexes
execute_sql "CREATE INDEX IF NOT EXISTS idx_m1_episodic_source_id ON m1_episodic (source_id);" "Creating M1 source_id index"

execute_sql "CREATE INDEX IF NOT EXISTS idx_m1_episodic_needs_embedding ON m1_episodic (needs_embedding) WHERE needs_embedding = TRUE;" "Creating M1 needs_embedding index"

execute_sql "CREATE INDEX IF NOT EXISTS idx_m1_episodic_created ON m1_episodic (created_at);" "Creating M1 created_at index"

execute_sql "CREATE INDEX IF NOT EXISTS idx_m1_episodic_retry_status ON m1_episodic (retry_status);" "Creating M1 retry_status index"

execute_sql "CREATE INDEX IF NOT EXISTS idx_m1_episodic_confidence ON m1_episodic (confidence);" "Creating M1 confidence index"

# Create vector indexes (HNSW for fast similarity search)
execute_sql "CREATE INDEX IF NOT EXISTS idx_m0_raw_embedding_hnsw ON m0_raw USING hnsw (embedding vector_cosine_ops);" "Creating M0 HNSW vector index"

execute_sql "CREATE INDEX IF NOT EXISTS idx_m1_episodic_embedding_hnsw ON m1_episodic USING hnsw (embedding vector_cosine_ops);" "Creating M1 HNSW vector index"

# Create updated_at trigger functions
execute_sql "
CREATE OR REPLACE FUNCTION update_m0_raw_updated_at()
RETURNS TRIGGER AS \$\$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
\$\$ LANGUAGE plpgsql;
" "Creating M0 updated_at trigger function"

execute_sql "
CREATE OR REPLACE FUNCTION update_m1_episodic_updated_at()
RETURNS TRIGGER AS \$\$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
\$\$ LANGUAGE plpgsql;
" "Creating M1 updated_at trigger function"

# Create updated_at triggers
execute_sql "
DROP TRIGGER IF EXISTS trigger_update_m0_raw_updated_at ON m0_raw;
CREATE TRIGGER trigger_update_m0_raw_updated_at
    BEFORE UPDATE ON m0_raw
    FOR EACH ROW
    EXECUTE FUNCTION update_m0_raw_updated_at();
" "Creating M0 updated_at trigger"

execute_sql "
DROP TRIGGER IF EXISTS trigger_update_m1_episodic_updated_at ON m1_episodic;
CREATE TRIGGER trigger_update_m1_episodic_updated_at
    BEFORE UPDATE ON m1_episodic
    FOR EACH ROW
    EXECUTE FUNCTION update_m1_episodic_updated_at();
" "Creating M1 updated_at trigger"

echo "🎉 MemFuse Multi-Layer PgAI database initialization completed successfully!"
echo "📋 Summary:"
echo "   ✅ pgvector extension installed"
echo "   ✅ pgai extension installed"
echo "   ✅ M0 Raw Data Layer: m0_raw table created"
echo "   ✅ M1 Episodic Memory Layer: m1_episodic table created"
echo "   ✅ Performance indexes created for both layers"
echo "   ✅ HNSW vector indexes created for both layers"
echo "   ✅ Triggers configured for both layers"
echo ""
echo "🔗 Database is ready for MemFuse multi-layer memory system!"
