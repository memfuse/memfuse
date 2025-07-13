#!/bin/bash
# MemFuse Immediate Trigger System Initialization
# This script sets up the immediate trigger system for real-time embedding generation

set -e

echo "⚡ Setting up MemFuse immediate trigger system..."

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

# Create notification function for immediate embedding trigger
execute_sql "
CREATE OR REPLACE FUNCTION notify_embedding_needed_m0_episodic()
RETURNS TRIGGER AS \$\$
BEGIN
    -- Send notification with record ID for immediate processing
    PERFORM pg_notify('embedding_needed_m0_episodic', NEW.id);
    RETURN NEW;
END;
\$\$ LANGUAGE plpgsql;
" "Creating immediate trigger notification function"

# Create trigger for immediate notification on INSERT
execute_sql "
DROP TRIGGER IF EXISTS trigger_immediate_embedding_m0_episodic ON m0_episodic;
CREATE TRIGGER trigger_immediate_embedding_m0_episodic
    AFTER INSERT ON m0_episodic
    FOR EACH ROW
    WHEN (NEW.needs_embedding = TRUE AND NEW.content IS NOT NULL)
    EXECUTE FUNCTION notify_embedding_needed_m0_episodic();
" "Creating immediate embedding trigger"

# Create trigger for immediate notification on UPDATE (when needs_embedding becomes TRUE)
execute_sql "
DROP TRIGGER IF EXISTS trigger_immediate_embedding_update_m0_episodic ON m0_episodic;
CREATE TRIGGER trigger_immediate_embedding_update_m0_episodic
    AFTER UPDATE ON m0_episodic
    FOR EACH ROW
    WHEN (OLD.needs_embedding = FALSE AND NEW.needs_embedding = TRUE AND NEW.content IS NOT NULL)
    EXECUTE FUNCTION notify_embedding_needed_m0_episodic();
" "Creating immediate embedding update trigger"

# Create helper function to manually trigger embedding for existing records
execute_sql "
CREATE OR REPLACE FUNCTION trigger_embedding_for_record(record_id TEXT)
RETURNS BOOLEAN AS \$\$
DECLARE
    record_exists BOOLEAN;
BEGIN
    -- Check if record exists and needs embedding
    SELECT EXISTS(
        SELECT 1 FROM m0_episodic 
        WHERE id = record_id 
        AND needs_embedding = TRUE 
        AND content IS NOT NULL
    ) INTO record_exists;
    
    IF record_exists THEN
        -- Send notification
        PERFORM pg_notify('embedding_needed_m0_episodic', record_id);
        RETURN TRUE;
    ELSE
        RETURN FALSE;
    END IF;
END;
\$\$ LANGUAGE plpgsql;
" "Creating manual trigger helper function"

# Create function to get trigger system status
execute_sql "
CREATE OR REPLACE FUNCTION get_trigger_system_status()
RETURNS TABLE(
    trigger_name TEXT,
    table_name TEXT,
    trigger_enabled BOOLEAN,
    function_exists BOOLEAN
) AS \$\$
BEGIN
    RETURN QUERY
    SELECT 
        t.tgname::TEXT as trigger_name,
        c.relname::TEXT as table_name,
        t.tgenabled = 'O' as trigger_enabled,
        EXISTS(SELECT 1 FROM pg_proc p WHERE p.proname = 'notify_embedding_needed_m0_episodic') as function_exists
    FROM pg_trigger t
    JOIN pg_class c ON t.tgrelid = c.oid
    WHERE c.relname = 'm0_episodic'
    AND t.tgname LIKE 'trigger_immediate_embedding%';
END;
\$\$ LANGUAGE plpgsql;
" "Creating trigger status function"

# Verify trigger system setup
echo "🔍 Verifying immediate trigger system..."

execute_sql "SELECT * FROM get_trigger_system_status();" "Checking trigger system status"

# Test notification system (this will send a test notification)
execute_sql "
DO \$\$
BEGIN
    -- Send a test notification
    PERFORM pg_notify('embedding_needed_m0_episodic', 'test-notification-' || extract(epoch from now()));
    RAISE NOTICE 'Test notification sent successfully';
END;
\$\$;
" "Testing notification system"

echo "⚡ Immediate trigger system setup completed!"
echo "📋 Summary:"
echo "   ✅ Notification function created"
echo "   ✅ INSERT trigger created"
echo "   ✅ UPDATE trigger created"
echo "   ✅ Helper functions created"
echo "   ✅ System verification completed"
echo ""
echo "🎯 Immediate trigger system is ready for real-time embedding generation!"
