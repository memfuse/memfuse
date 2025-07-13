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

# Create m0_episodic table
execute_sql "
CREATE TABLE IF NOT EXISTS m0_episodic (
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
" "Creating m0_episodic table"

# Create indexes for performance
echo "📊 Creating performance indexes..."

execute_sql "CREATE INDEX IF NOT EXISTS idx_m0_episodic_needs_embedding ON m0_episodic (needs_embedding) WHERE needs_embedding = TRUE;" "Creating needs_embedding index"

execute_sql "CREATE INDEX IF NOT EXISTS idx_m0_episodic_created ON m0_episodic (created_at);" "Creating created_at index"

execute_sql "CREATE INDEX IF NOT EXISTS idx_m0_episodic_retry_status ON m0_episodic (retry_status);" "Creating retry_status index"

execute_sql "CREATE INDEX IF NOT EXISTS idx_m0_episodic_retry_count ON m0_episodic (retry_count);" "Creating retry_count index"

# Create vector index (HNSW for fast similarity search)
execute_sql "CREATE INDEX IF NOT EXISTS idx_m0_episodic_embedding_hnsw ON m0_episodic USING hnsw (embedding vector_cosine_ops);" "Creating HNSW vector index"

# Create updated_at trigger function
execute_sql "
CREATE OR REPLACE FUNCTION update_m0_episodic_updated_at()
RETURNS TRIGGER AS \$\$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
\$\$ LANGUAGE plpgsql;
" "Creating updated_at trigger function"

# Create updated_at trigger
execute_sql "
DROP TRIGGER IF EXISTS trigger_update_m0_episodic_updated_at ON m0_episodic;
CREATE TRIGGER trigger_update_m0_episodic_updated_at
    BEFORE UPDATE ON m0_episodic
    FOR EACH ROW
    EXECUTE FUNCTION update_m0_episodic_updated_at();
" "Creating updated_at trigger"

echo "🎉 MemFuse PgAI database initialization completed successfully!"
echo "📋 Summary:"
echo "   ✅ pgvector extension installed"
echo "   ✅ pgai extension installed"
echo "   ✅ m0_episodic table created"
echo "   ✅ Performance indexes created"
echo "   ✅ HNSW vector index created"
echo "   ✅ Triggers configured"
echo ""
echo "🔗 Database is ready for MemFuse connections!"
