-- Single entry point for database schema (no migration framework)
-- Execute with: psql -f db/schema.sql <connection args>

\echo 'Applying MemFuse schema (idempotent)'

-- Include order: tables -> functions -> triggers -> views -> seed (optional)
\i db/schema/tables/all.sql
\i db/schema/functions/all.sql
\i db/schema/triggers/all.sql
\i db/schema/views/all.sql
-- Optional seed
-- \i db/schema/seed/minimal.sql

\echo 'Schema apply complete'

