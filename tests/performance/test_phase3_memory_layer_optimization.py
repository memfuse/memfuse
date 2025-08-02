"""
Performance tests for Phase 3 Memory Layer Parallel Processing Optimization.

Tests the elimination of synchronization bottlenecks in M0/M1/M2 memory layers,
specifically focusing on parallel storage operations within each layer.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any, Optional

from src.memfuse_core.hierarchy.layers import M0RawDataLayer, M1EpisodicLayer, M2SemanticLayer
from src.memfuse_core.hierarchy.core import LayerType, LayerConfig, ProcessingResult
from src.memfuse_core.hierarchy.storage import UnifiedStorageManager, StorageType
from src.memfuse_core.rag.chunk.base import ChunkData


class TestPhase3MemoryLayerOptimization:
    """Test Phase 3 optimizations for memory layer parallel processing."""

    @pytest.fixture
    async def mock_storage_manager(self):
        """Create a mock storage manager with controlled latency."""
        manager = AsyncMock(spec=UnifiedStorageManager)
        
        # Simulate storage latency (50ms per operation)
        async def mock_write_to_backend(storage_type: StorageType, chunk: Any, metadata: Optional[Dict] = None) -> str:
            await asyncio.sleep(0.05)  # 50ms latency
            return f"id_{storage_type.value}_{int(time.time() * 1000)}"
        
        async def mock_write_to_all(data: Any, metadata: Optional[Dict] = None) -> Dict[StorageType, Optional[str]]:
            # P3 OPTIMIZATION: This should now be parallel
            await asyncio.sleep(0.05)  # Simulated parallel write time
            return {
                StorageType.VECTOR: f"vector_id_{int(time.time() * 1000)}",
                StorageType.KEYWORD: f"keyword_id_{int(time.time() * 1000)}",
                StorageType.SQL: f"sql_id_{int(time.time() * 1000)}"
            }
        
        manager.write_to_backend.side_effect = mock_write_to_backend
        manager.write_to_all.side_effect = mock_write_to_all
        manager.initialize.return_value = True
        
        return manager

    @pytest.fixture
    def layer_config(self):
        """Create a test layer configuration."""
        return LayerConfig(
            enabled=True,
            storage_backends=["vector", "keyword", "sql"],
            custom_config={
                "episode_formation_enabled": True,
                "fact_extraction_enabled": True,
                "llm_config": {"model": "grok-3-mini"}
            }
        )

    @pytest.fixture
    async def m0_layer(self, mock_storage_manager, layer_config):
        """Create M0 layer with mock storage."""
        layer = M0RawDataLayer(LayerType.M0, layer_config, "test_user", mock_storage_manager)
        await layer.initialize()
        return layer

    @pytest.fixture
    async def m1_layer(self, mock_storage_manager, layer_config):
        """Create M1 layer with mock storage."""
        layer = M1EpisodicLayer(LayerType.M1, layer_config, "test_user", mock_storage_manager)
        await layer.initialize()
        return layer

    @pytest.fixture
    async def m2_layer(self, mock_storage_manager, layer_config):
        """Create M2 layer with mock storage."""
        layer = M2SemanticLayer(LayerType.M2, layer_config, "test_user", mock_storage_manager)
        await layer.initialize()
        return layer

    @pytest.mark.asyncio
    async def test_m0_parallel_storage_performance(self, m0_layer):
        """Test M0 layer parallel storage performance."""
        # Create test data with multiple chunks
        test_data = [
            {"content": f"Test message {i}", "metadata": {"index": i}}
            for i in range(10)
        ]
        
        # Measure processing time
        start_time = time.time()
        result = await m0_layer.process_data(test_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Verify success
        assert result.success
        assert len(result.processed_items) > 0
        
        # P3 OPTIMIZATION: With parallel storage, this should be much faster than sequential
        # Sequential would take ~150ms (3 backends * 50ms), parallel should be ~50ms
        assert processing_time < 0.1, f"M0 parallel storage took {processing_time:.3f}s, expected < 0.1s"
        
        print(f"M0 parallel storage performance: {processing_time:.3f}s for {len(test_data)} items")

    @pytest.mark.asyncio
    async def test_m1_parallel_episode_storage(self, m1_layer):
        """Test M1 layer parallel episode storage performance."""
        # Mock episode formation to return multiple episodes
        episodes = [
            {
                "episode_content": f"Episode {i} content",
                "episode_type": "conversation",
                "timestamp": time.time(),
                "metadata": {"episode_id": i}
            }
            for i in range(8)
        ]
        
        with patch.object(m1_layer, '_form_episodes', return_value=episodes):
            # Measure processing time
            start_time = time.time()
            result = await m1_layer.process_data({"test": "data"})
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Verify success
            assert result.success
            assert len(result.processed_items) == len(episodes)
            
            # P3 OPTIMIZATION: Parallel episode storage should be much faster
            # Sequential would take ~400ms (8 episodes * 50ms), parallel should be ~50ms
            assert processing_time < 0.15, f"M1 parallel episode storage took {processing_time:.3f}s, expected < 0.15s"
            
            print(f"M1 parallel episode storage performance: {processing_time:.3f}s for {len(episodes)} episodes")

    @pytest.mark.asyncio
    async def test_m2_parallel_entity_relationship_storage(self, m2_layer):
        """Test M2 layer parallel entity and relationship storage performance."""
        # Mock entity and relationship extraction
        entities = [
            {"content": f"Entity {i}", "type": "person", "metadata": {"entity_id": i}}
            for i in range(6)
        ]
        relationships = [
            {"content": f"Relationship {i}", "type": "knows", "metadata": {"rel_id": i}}
            for i in range(4)
        ]
        
        with patch.object(m2_layer, '_extract_entities_and_relationships', return_value=(entities, relationships)):
            # Measure processing time
            start_time = time.time()
            result = await m2_layer.process_data({"test": "data"})
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Verify success
            assert result.success
            assert len(result.processed_items) == len(entities) + len(relationships)
            
            # P3 OPTIMIZATION: Parallel storage should be much faster
            # Sequential would take ~500ms (10 items * 50ms), parallel should be ~50ms
            assert processing_time < 0.15, f"M2 parallel storage took {processing_time:.3f}s, expected < 0.15s"
            
            print(f"M2 parallel storage performance: {processing_time:.3f}s for {len(entities)} entities + {len(relationships)} relationships")

    @pytest.mark.asyncio
    async def test_unified_storage_manager_parallel_backends(self):
        """Test UnifiedStorageManager parallel backend writes."""
        # Create mock backends with different latencies
        mock_backends = {}
        for storage_type in [StorageType.VECTOR, StorageType.KEYWORD, StorageType.SQL]:
            backend = AsyncMock()
            backend.write.side_effect = lambda data, metadata=None: asyncio.sleep(0.05) or f"id_{storage_type.value}"
            mock_backends[storage_type] = backend
        
        # Create storage manager with mock backends
        manager = UnifiedStorageManager({}, "test_user")
        manager.backends = mock_backends
        
        # Test parallel write to all backends
        test_data = ChunkData(content="test content", metadata={"test": True})
        
        start_time = time.time()
        results = await manager.write_to_all(test_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Verify all backends were called
        assert len(results) == 3
        assert all(result is not None for result in results.values())
        
        # P3 OPTIMIZATION: Parallel writes should be much faster than sequential
        # Sequential would take ~150ms (3 * 50ms), parallel should be ~50ms
        assert processing_time < 0.1, f"Parallel backend writes took {processing_time:.3f}s, expected < 0.1s"
        
        print(f"UnifiedStorageManager parallel backend performance: {processing_time:.3f}s for 3 backends")

    @pytest.mark.asyncio
    async def test_phase3_performance_regression_detection(self, m0_layer, m1_layer, m2_layer):
        """Test for performance regressions in Phase 3 optimizations."""
        # Performance targets for Phase 3
        M0_TARGET_TIME = 0.1  # 100ms for M0 operations
        M1_TARGET_TIME = 0.15  # 150ms for M1 operations  
        M2_TARGET_TIME = 0.15  # 150ms for M2 operations
        
        # Test M0 performance
        test_data = [{"content": f"Test {i}"} for i in range(5)]
        start_time = time.time()
        m0_result = await m0_layer.process_data(test_data)
        m0_time = time.time() - start_time
        
        # Test M1 performance
        with patch.object(m1_layer, '_form_episodes', return_value=[{"episode_content": f"Episode {i}"} for i in range(5)]):
            start_time = time.time()
            m1_result = await m1_layer.process_data(test_data)
            m1_time = time.time() - start_time
        
        # Test M2 performance
        entities = [{"content": f"Entity {i}"} for i in range(3)]
        relationships = [{"content": f"Rel {i}"} for i in range(2)]
        with patch.object(m2_layer, '_extract_entities_and_relationships', return_value=(entities, relationships)):
            start_time = time.time()
            m2_result = await m2_layer.process_data(test_data)
            m2_time = time.time() - start_time
        
        # Verify performance targets
        assert m0_time < M0_TARGET_TIME, f"M0 performance regression: {m0_time:.3f}s > {M0_TARGET_TIME}s"
        assert m1_time < M1_TARGET_TIME, f"M1 performance regression: {m1_time:.3f}s > {M1_TARGET_TIME}s"
        assert m2_time < M2_TARGET_TIME, f"M2 performance regression: {m2_time:.3f}s > {M2_TARGET_TIME}s"
        
        # Verify all operations succeeded
        assert m0_result.success and m1_result.success and m2_result.success
        
        print(f"Phase 3 performance results:")
        print(f"  M0: {m0_time:.3f}s (target: {M0_TARGET_TIME}s)")
        print(f"  M1: {m1_time:.3f}s (target: {M1_TARGET_TIME}s)")
        print(f"  M2: {m2_time:.3f}s (target: {M2_TARGET_TIME}s)")
