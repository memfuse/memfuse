"""
Unit tests for SchemaManager.

Tests database schema management including creation, migration,
and validation for dual-layer PgAI system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from src.memfuse_core.store.pgai_store.schema_manager import SchemaManager


class TestSchemaManager:
    """Test cases for SchemaManager."""
    
    @pytest.fixture
    def mock_pool(self):
        """Mock database connection pool."""
        pool = AsyncMock()
        
        # Mock connection context manager
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        
        # Setup connection context manager
        pool.connection.return_value.__aenter__.return_value = mock_conn
        pool.connection.return_value.__aexit__.return_value = None
        
        # Setup cursor context manager
        mock_conn.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_conn.cursor.return_value.__aexit__.return_value = None
        
        # Setup execute and fetchall methods
        mock_conn.execute = AsyncMock()
        mock_conn.commit = AsyncMock()
        mock_conn.rollback = AsyncMock()
        
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchall = AsyncMock()
        mock_cursor.fetchone = AsyncMock()
        
        return pool, mock_conn, mock_cursor
    
    def test_initialization(self, mock_pool):
        """Test SchemaManager initialization."""
        pool, _, _ = mock_pool
        
        manager = SchemaManager(pool)
        
        assert manager.pool == pool
        assert manager.current_version == "1.0.0"
        assert "m0" in manager.supported_layers
        assert "m1" in manager.supported_layers
    
    @pytest.mark.asyncio
    async def test_create_schema_version_table(self, mock_pool):
        """Test schema version table creation."""
        pool, mock_conn, _ = mock_pool
        
        manager = SchemaManager(pool)
        
        await manager._create_schema_version_table()
        
        # Verify table creation SQL was executed
        mock_conn.execute.assert_called_once()
        create_sql = mock_conn.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS memfuse_schema_versions" in create_sql
        assert "version TEXT NOT NULL" in create_sql
        assert "layer_name TEXT NOT NULL" in create_sql
        
        mock_conn.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_all_schemas_success(self, mock_pool):
        """Test successful initialization of all schemas."""
        pool, mock_conn, mock_cursor = mock_pool
        
        manager = SchemaManager(pool)
        
        # Mock schema version checks to return False (needs initialization)
        mock_cursor.fetchone.return_value = None
        
        # Mock M1 schema file reading
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value="CREATE TABLE m1_episodic (...);"):
            
            result = await manager.initialize_all_schemas(['m0', 'm1'])
        
        assert result is True
        
        # Verify schema version table was created
        assert mock_conn.execute.call_count >= 1
        
        # Verify commit was called
        assert mock_conn.commit.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_initialize_all_schemas_failure(self, mock_pool):
        """Test schema initialization failure."""
        pool, mock_conn, _ = mock_pool
        
        manager = SchemaManager(pool)
        
        # Mock connection failure
        mock_conn.execute.side_effect = Exception("Database connection failed")
        
        result = await manager.initialize_all_schemas(['m0', 'm1'])
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_is_schema_current_true(self, mock_pool):
        """Test schema currency check when schema is current."""
        pool, mock_conn, mock_cursor = mock_pool
        
        manager = SchemaManager(pool)
        
        # Mock current version found
        mock_cursor.fetchone.return_value = ("1.0.0",)
        
        result = await manager._is_schema_current("m0")
        
        assert result is True
        
        # Verify query was executed
        mock_cursor.execute.assert_called_once()
        query_sql = mock_cursor.execute.call_args[0][0]
        assert "SELECT version FROM memfuse_schema_versions" in query_sql
        assert "WHERE layer_name = %s" in query_sql
    
    @pytest.mark.asyncio
    async def test_is_schema_current_false(self, mock_pool):
        """Test schema currency check when schema is outdated."""
        pool, mock_conn, mock_cursor = mock_pool
        
        manager = SchemaManager(pool)
        
        # Mock old version found
        mock_cursor.fetchone.return_value = ("0.9.0",)
        
        result = await manager._is_schema_current("m0")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_is_schema_current_no_version(self, mock_pool):
        """Test schema currency check when no version exists."""
        pool, mock_conn, mock_cursor = mock_pool
        
        manager = SchemaManager(pool)
        
        # Mock no version found
        mock_cursor.fetchone.return_value = None
        
        result = await manager._is_schema_current("m0")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_initialize_m0_schema_table_exists(self, mock_pool):
        """Test M0 schema initialization when table already exists."""
        pool, mock_conn, mock_cursor = mock_pool
        
        manager = SchemaManager(pool)
        
        # Mock table exists
        mock_cursor.fetchone.return_value = (True,)
        
        result = await manager._initialize_m0_schema()
        
        assert result is True
        
        # Verify existence check was performed
        mock_cursor.execute.assert_called_once()
        query_sql = mock_cursor.execute.call_args[0][0]
        assert "SELECT EXISTS" in query_sql
        assert "table_name = 'm0_raw'" in query_sql
    
    @pytest.mark.asyncio
    async def test_initialize_m0_schema_table_not_exists(self, mock_pool):
        """Test M0 schema initialization when table doesn't exist."""
        pool, mock_conn, mock_cursor = mock_pool
        
        manager = SchemaManager(pool)
        
        # Mock table doesn't exist
        mock_cursor.fetchone.return_value = (False,)
        
        result = await manager._initialize_m0_schema()
        
        assert result is True  # Should still succeed (table will be created by PgaiStore)
    
    @pytest.mark.asyncio
    async def test_initialize_m1_schema_success(self, mock_pool):
        """Test successful M1 schema initialization."""
        pool, mock_conn, _ = mock_pool
        
        manager = SchemaManager(pool)
        
        # Mock schema file exists and has content
        mock_schema_content = """
        CREATE TABLE m1_episodic (
            id TEXT PRIMARY KEY,
            episode_content TEXT NOT NULL
        );
        """
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value=mock_schema_content):
            
            result = await manager._initialize_m1_schema()
        
        assert result is True
        
        # Verify schema SQL was executed
        mock_conn.execute.assert_called_once_with(mock_schema_content)
        mock_conn.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_m1_schema_file_not_found(self, mock_pool):
        """Test M1 schema initialization when schema file is missing."""
        pool, _, _ = mock_pool
        
        manager = SchemaManager(pool)
        
        # Mock schema file doesn't exist
        with patch('pathlib.Path.exists', return_value=False):
            result = await manager._initialize_m1_schema()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_validate_m0_schema_success(self, mock_pool):
        """Test successful M0 schema validation."""
        pool, mock_conn, mock_cursor = mock_pool
        
        manager = SchemaManager(pool)
        
        # Mock all required columns exist
        required_columns = [
            ('id',), ('content',), ('metadata',), ('embedding',), ('needs_embedding',),
            ('retry_count',), ('last_retry_at',), ('retry_status',), 
            ('created_at',), ('updated_at',)
        ]
        mock_cursor.fetchall.return_value = required_columns
        
        result = await manager._validate_m0_schema()
        
        assert result is True
        
        # Verify column query was executed
        mock_cursor.execute.assert_called_once()
        query_sql = mock_cursor.execute.call_args[0][0]
        assert "SELECT column_name" in query_sql
        assert "WHERE table_name = 'm0_raw'" in query_sql
    
    @pytest.mark.asyncio
    async def test_validate_m0_schema_missing_columns(self, mock_pool):
        """Test M0 schema validation with missing columns."""
        pool, mock_conn, mock_cursor = mock_pool
        
        manager = SchemaManager(pool)
        
        # Mock missing some required columns
        partial_columns = [('id',), ('content',), ('metadata',)]
        mock_cursor.fetchall.return_value = partial_columns
        
        result = await manager._validate_m0_schema()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_validate_m1_schema_success(self, mock_pool):
        """Test successful M1 schema validation."""
        pool, mock_conn, mock_cursor = mock_pool
        
        manager = SchemaManager(pool)
        
        # Mock required columns exist
        required_columns = [
            ('id',), ('source_id',), ('fact_content',), ('fact_type',), ('confidence',),
            ('entities',), ('temporal_info',), ('metadata',), ('embedding',), 
            ('needs_embedding',), ('retry_count',), ('last_retry_at',), 
            ('retry_status',), ('created_at',), ('updated_at',)
        ]
        
        # Mock required triggers exist
        required_triggers = [
            ('trigger_update_m1_episodic_updated_at',),
            ('trigger_m1_embedding_notification',)
        ]
        
        # Setup mock to return columns first, then triggers
        mock_cursor.fetchall.side_effect = [required_columns, required_triggers]
        
        result = await manager._validate_m1_schema()
        
        assert result is True
        
        # Verify both column and trigger queries were executed
        assert mock_cursor.execute.call_count == 2
    
    @pytest.mark.asyncio
    async def test_validate_m1_schema_missing_triggers(self, mock_pool):
        """Test M1 schema validation with missing triggers."""
        pool, mock_conn, mock_cursor = mock_pool
        
        manager = SchemaManager(pool)
        
        # Mock columns exist but triggers missing
        required_columns = [
            ('id',), ('source_id',), ('fact_content',), ('fact_type',), ('confidence',),
            ('entities',), ('temporal_info',), ('metadata',), ('embedding',), 
            ('needs_embedding',), ('retry_count',), ('last_retry_at',), 
            ('retry_status',), ('created_at',), ('updated_at',)
        ]
        
        partial_triggers = [('trigger_update_m1_episodic_updated_at',)]  # Missing one trigger
        
        mock_cursor.fetchall.side_effect = [required_columns, partial_triggers]
        
        result = await manager._validate_m1_schema()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_validate_schemas(self, mock_pool):
        """Test validation of multiple schemas."""
        pool, mock_conn, mock_cursor = mock_pool
        
        manager = SchemaManager(pool)
        
        # Mock successful validation for both layers
        # For M0: return required columns
        m0_columns = [
            ('id',), ('content',), ('metadata',), ('embedding',), ('needs_embedding',),
            ('retry_count',), ('last_retry_at',), ('retry_status',), 
            ('created_at',), ('updated_at',)
        ]
        
        # For M1: return required columns and triggers
        m1_columns = [
            ('id',), ('source_id',), ('fact_content',), ('fact_type',), ('confidence',),
            ('entities',), ('temporal_info',), ('metadata',), ('embedding',), 
            ('needs_embedding',), ('retry_count',), ('last_retry_at',), 
            ('retry_status',), ('created_at',), ('updated_at',)
        ]
        
        m1_triggers = [
            ('trigger_update_m1_episodic_updated_at',),
            ('trigger_m1_embedding_notification',)
        ]
        
        # Setup mock to return appropriate results for each query
        mock_cursor.fetchall.side_effect = [m0_columns, m1_columns, m1_triggers]
        
        results = await manager.validate_schemas(['m0', 'm1'])
        
        assert results['m0'] is True
        assert results['m1'] is True
        assert len(results) == 2
    
    @pytest.mark.asyncio
    async def test_get_schema_info(self, mock_pool):
        """Test schema information retrieval."""
        pool, mock_conn, mock_cursor = mock_pool
        
        manager = SchemaManager(pool)
        
        # Mock schema versions
        mock_versions = [
            ('m0', '1.0.0', '2024-01-01 10:00:00', 'M0 layer schema'),
            ('m1', '1.0.0', '2024-01-01 10:01:00', 'M1 layer schema'),
            ('system', '1.0.0', '2024-01-01 10:02:00', 'System schema')
        ]
        
        # Mock table information
        mock_tables = [
            ('m0_raw', 10),
            ('m1_episodic', 15)
        ]
        
        # Setup mock to return versions first, then tables
        mock_cursor.fetchall.side_effect = [mock_versions, mock_tables]
        
        info = await manager.get_schema_info()
        
        assert info['current_version'] == "1.0.0"
        assert info['supported_layers'] == ['m0', 'm1']
        assert len(info['versions']) == 3
        assert len(info['tables']) == 2
        
        # Check version details
        m0_version = next(v for v in info['versions'] if v['layer'] == 'm0')
        assert m0_version['version'] == '1.0.0'
        assert m0_version['description'] == 'M0 layer schema'
        
        # Check table details
        m0_table = next(t for t in info['tables'] if t['name'] == 'm0_raw')
        assert m0_table['columns'] == 10
    
    @pytest.mark.asyncio
    async def test_record_schema_version(self, mock_pool):
        """Test schema version recording."""
        pool, mock_conn, _ = mock_pool
        
        manager = SchemaManager(pool)
        
        await manager._record_schema_version('m0')
        
        # Verify INSERT was executed
        mock_conn.execute.assert_called_once()
        insert_sql = mock_conn.execute.call_args[0][0]
        assert "INSERT INTO memfuse_schema_versions" in insert_sql
        assert "ON CONFLICT (version, layer_name) DO NOTHING" in insert_sql
        
        # Verify parameters
        params = mock_conn.execute.call_args[0][1]
        assert params[0] == "1.0.0"  # version
        assert params[1] == "m0"     # layer_name
        
        mock_conn.commit.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])