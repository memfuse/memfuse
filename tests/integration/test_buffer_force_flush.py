"""
Integration tests for buffer force flush functionality.

This test suite validates:
1. Force flush timeout mechanism
2. Graceful shutdown flush behavior
3. Data consistency between buffer and database
"""

import asyncio
import time
import pytest
import subprocess
import signal
import os
import sys
from typing import Dict, Any, List
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from memfuse_core.services.buffer_service import BufferService
from memfuse_core.services.memory_service import MemoryService
from memfuse_core.config.global_config import GlobalConfig
from memfuse_core.interfaces import MessageList


class TestBufferForceFlush:
    """Test suite for buffer force flush functionality."""

    @pytest.fixture
    async def config(self):
        """Create test configuration with short timeouts for testing."""
        config_dict = {
            "buffer": {
                "enabled": True,
                "performance": {
                    "force_flush_timeout": 5.0,  # 5 seconds for testing
                    "flush_interval": 2.0,       # 2 seconds for testing
                    "enable_auto_flush": True
                },
                "hybrid_buffer": {
                    "max_size": 3,  # Small size to avoid size-based flush
                    "chunk_strategy": "message",
                    "embedding_model": "all-MiniLM-L6-v2"
                },
                "round_buffer": {
                    "max_size": 3,
                    "max_tokens": 800,
                    "token_model": "gpt-4o-mini"
                }
            },
            "database": {
                "url": "postgresql://memfuse:memfuse@localhost:5432/memfuse_test"
            }
        }
        return GlobalConfig(config_dict)

    @pytest.fixture
    async def memory_service(self, config):
        """Create and initialize memory service."""
        service = MemoryService()
        await service.initialize(config.get_raw_config())
        yield service
        await service.shutdown()

    @pytest.fixture
    async def buffer_service(self, config, memory_service):
        """Create and initialize buffer service."""
        service = BufferService(memory_service, "test_user", config.get_raw_config())
        await service.initialize(config.get_raw_config())
        yield service
        await service.shutdown()

    async def test_force_flush_timeout(self, buffer_service):
        """Test that buffer automatically flushes after force_flush_timeout."""
        # Add some test messages
        test_messages = [
            {"role": "user", "content": "Test message 1"},
            {"role": "assistant", "content": "Test response 1"}
        ]
        
        # Add messages to buffer
        result = await buffer_service.add(test_messages, session_id="test_session")
        assert result["status"] == "success"
        
        # Check that data is in buffer
        hybrid_buffer = buffer_service.get_hybrid_buffer()
        assert len(hybrid_buffer.original_rounds) > 0
        
        # Wait for force flush timeout (5 seconds + buffer)
        print("Waiting for force flush timeout...")
        await asyncio.sleep(7.0)
        
        # Check that buffer was flushed
        assert len(hybrid_buffer.original_rounds) == 0, "Buffer should be empty after force flush"
        
        # Verify data is in database
        query_result = await buffer_service.query(
            "Test message", 
            session_id="test_session", 
            top_k=5
        )
        assert query_result["status"] == "success"
        assert len(query_result["data"]["results"]) > 0

    async def test_manual_flush_all(self, buffer_service):
        """Test manual flush_all_buffers method."""
        # Add test messages
        test_messages = [
            {"role": "user", "content": "Manual flush test message"},
            {"role": "assistant", "content": "Manual flush test response"}
        ]
        
        result = await buffer_service.add(test_messages, session_id="test_session_manual")
        assert result["status"] == "success"
        
        # Verify data is in buffer
        hybrid_buffer = buffer_service.get_hybrid_buffer()
        initial_buffer_size = len(hybrid_buffer.original_rounds)
        assert initial_buffer_size > 0
        
        # Manual flush
        flush_result = await buffer_service.flush_all_buffers()
        assert flush_result["status"] == "success"
        
        # Verify buffer is empty
        assert len(hybrid_buffer.original_rounds) == 0
        
        # Verify data is in database
        query_result = await buffer_service.query(
            "Manual flush test", 
            session_id="test_session_manual", 
            top_k=5
        )
        assert query_result["status"] == "success"
        assert len(query_result["data"]["results"]) > 0

    async def test_graceful_shutdown_flush(self, buffer_service):
        """Test that shutdown triggers final flush."""
        # Add test messages
        test_messages = [
            {"role": "user", "content": "Shutdown test message"},
            {"role": "assistant", "content": "Shutdown test response"}
        ]
        
        result = await buffer_service.add(test_messages, session_id="test_session_shutdown")
        assert result["status"] == "success"
        
        # Verify data is in buffer
        hybrid_buffer = buffer_service.get_hybrid_buffer()
        assert len(hybrid_buffer.original_rounds) > 0
        
        # Trigger shutdown (this should flush remaining data)
        await buffer_service.shutdown()
        
        # Create new buffer service to check database
        new_buffer_service = BufferService(
            buffer_service.memory_service, 
            "test_user", 
            buffer_service.config
        )
        await new_buffer_service.initialize()
        
        try:
            # Query database to verify data was flushed
            query_result = await new_buffer_service.query(
                "Shutdown test", 
                session_id="test_session_shutdown", 
                top_k=5
            )
            assert query_result["status"] == "success"
            assert len(query_result["data"]["results"]) > 0
        finally:
            await new_buffer_service.shutdown()

    async def test_configuration_parameters(self, buffer_service):
        """Test that configuration parameters are properly applied."""
        hybrid_buffer = buffer_service.get_hybrid_buffer()
        
        # Check that force_flush_timeout is set correctly
        assert hybrid_buffer.force_flush_timeout == 5.0
        assert hybrid_buffer.auto_flush_interval == 2.0
        assert hybrid_buffer.enable_auto_flush is True

    async def test_buffer_stats_include_force_timeout(self, buffer_service):
        """Test that buffer statistics include force_flush_timeout."""
        hybrid_buffer = buffer_service.get_hybrid_buffer()
        stats = hybrid_buffer.get_stats()
        
        assert "force_flush_timeout" in stats
        assert stats["force_flush_timeout"] == 5.0


class TestBufferForceFlushEndToEnd:
    """End-to-end tests that simulate real server behavior."""

    def test_server_startup_with_force_flush_config(self):
        """Test that server starts with force flush configuration."""
        # This test verifies that the configuration is properly loaded
        config_path = project_root / "config" / "buffer" / "default.yaml"
        assert config_path.exists()
        
        # Read config and verify force_flush_timeout is present
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert "performance" in config
        assert "force_flush_timeout" in config["performance"]
        assert config["performance"]["force_flush_timeout"] == 1800  # 30 minutes


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
