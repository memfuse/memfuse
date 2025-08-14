-- MemFuse Simplified M0/M1 Memory System Schema
-- This script is disabled - only M0/M1 layers are active in the simplified architecture
-- M2, M3, and MSMG layers are not created to maintain clean M0/M1 data flow

-- =============================================================================
-- SIMPLIFIED ARCHITECTURE: M0/M1 ONLY
-- =============================================================================
-- M0 Layer: Raw streaming messages (handled by 00-init-memfuse-pgai.sql)
-- M1 Layer: Intelligent chunks with embeddings (handled by 00-init-memfuse-pgai.sql)
-- M2/M3/MSMG: Disabled for simplified implementation

-- This file intentionally left minimal to prevent creation of unused tables
DO $$ 
BEGIN
    RAISE NOTICE 'M2/M3/MSMG layers disabled - using simplified M0/M1 architecture only';
END $$;
