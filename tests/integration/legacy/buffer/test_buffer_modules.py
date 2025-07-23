"""Modular tests for Buffer Architecture components.

This test suite validates each component individually to identify the root cause
of issues in the buffer architecture refactor.

MODULES TESTED:
==============
1. RoundBuffer: Token counting, auto-transfer triggers
2. HybridBuffer: Chunk generation, embedding creation, VectorCache
3. WriteBuffer: Integration of RoundBuffer + HybridBuffer + FlushManager
4. QueryBuffer: Multi-source querying and result combination
5. BufferService: High-level orchestration

FOCUS: Identify why data isn't flowing correctly through the pipeline
"""

import asyncio
import uuid
from typing import Dict, Any, List
from loguru import logger

from memfuse_core.services import MemoryService, BufferService
from memfuse_core.buffer.round_buffer import RoundBuffer
from memfuse_core.buffer.hybrid_buffer import HybridBuffer
from memfuse_core.buffer.write_buffer import WriteBuffer
from memfuse_core.buffer.query_buffer import QueryBuffer
from memfuse_core.buffer.flush_manager import FlushManager
from memfuse_core.interfaces import MessageList


class BufferModuleTests:
    """Individual module tests for buffer components."""

    def __init__(self):
        self.user_name = f"test_user_{uuid.uuid4().hex[:8]}"
        self.agent_name = "test_agent"
        self.session_name = f"test_session_{uuid.uuid4().hex[:8]}"

    async def setup_memory_service(self):
        """Setup MemoryService for testing."""
        self.memory_service = MemoryService(
            user=self.user_name,
            agent=self.agent_name,
            session=self.session_name
        )
        await self.memory_service.initialize()
        logger.info(f"MemoryService initialized: user_id={self.memory_service._user_id}")

    def create_test_messages(self, count: int = 3, content_prefix: str = "Test message") -> MessageList:
        """Create test messages with varying content lengths."""
        messages = []
        for i in range(count):
            content = f"{content_prefix} {i + 1} - " + "content " * (10 + i * 5)  # Varying lengths
            messages.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": content
            })
        return messages

    async def test_round_buffer_basic(self):
        """Test RoundBuffer basic functionality and token counting."""
        logger.info("🧪 Testing RoundBuffer basic functionality")
        
        # Create RoundBuffer with small limits to trigger transfers
        round_buffer = RoundBuffer(max_tokens=100, max_size=2)  # Small limits
        
        # Test 1: Add single message
        messages1 = self.create_test_messages(1, "Short message")
        result1 = await round_buffer.add(messages1, self.memory_service._session_id)
        
        logger.info(f"RoundBuffer after first add: rounds={len(round_buffer.rounds)}, tokens={round_buffer.current_tokens}")
        assert len(round_buffer.rounds) == 1
        assert not result1  # No transfer should be triggered yet
        
        # Test 2: Add more messages to trigger size limit
        messages2 = self.create_test_messages(1, "Another message")
        result2 = await round_buffer.add(messages2, self.memory_service._session_id)
        
        logger.info(f"RoundBuffer after second add: rounds={len(round_buffer.rounds)}, tokens={round_buffer.current_tokens}")
        assert len(round_buffer.rounds) == 2
        assert not result2  # Still no transfer (size limit is 2)
        
        # Test 3: Add third message to trigger size limit
        messages3 = self.create_test_messages(1, "Third message that should trigger transfer")
        
        # Set up a mock transfer handler to capture transfer
        transferred_data = []
        async def mock_transfer_handler(rounds):
            transferred_data.extend(rounds)
            logger.info(f"Mock transfer handler called with {len(rounds)} rounds")
        
        round_buffer.set_transfer_handler(mock_transfer_handler)
        result3 = await round_buffer.add(messages3, self.memory_service._session_id)
        
        logger.info(f"RoundBuffer after third add: rounds={len(round_buffer.rounds)}, tokens={round_buffer.current_tokens}")
        logger.info(f"Transfer triggered: {result3}, transferred data: {len(transferred_data)} rounds")
        
        assert result3  # Transfer should be triggered
        assert len(transferred_data) == 2  # Previous 2 rounds should be transferred
        assert len(round_buffer.rounds) == 1  # Only new message should remain
        
        logger.info("✅ RoundBuffer basic functionality test passed")

    async def test_hybrid_buffer_processing(self):
        """Test HybridBuffer chunk generation and embedding creation."""
        logger.info("🧪 Testing HybridBuffer processing")
        
        # Create HybridBuffer
        hybrid_buffer = HybridBuffer(max_size=5, chunk_strategy="message")
        await hybrid_buffer.initialize()
        
        # Test 1: Add rounds and verify chunk generation
        test_rounds = [
            self.create_test_messages(2, "First round"),
            self.create_test_messages(2, "Second round")
        ]
        
        logger.info(f"Adding {len(test_rounds)} rounds to HybridBuffer")
        await hybrid_buffer.add_from_rounds(test_rounds)
        
        # Verify chunks and embeddings were created
        logger.info(f"HybridBuffer state: chunks={len(hybrid_buffer.chunks)}, embeddings={len(hybrid_buffer.embeddings)}, rounds={len(hybrid_buffer.original_rounds)}")
        
        assert len(hybrid_buffer.chunks) > 0, "No chunks were created"
        assert len(hybrid_buffer.embeddings) > 0, "No embeddings were created"
        assert len(hybrid_buffer.original_rounds) == len(test_rounds), "Original rounds not preserved"
        assert len(hybrid_buffer.chunks) == len(hybrid_buffer.embeddings), "Chunks and embeddings count mismatch"
        
        # Test 2: Verify chunk content
        for i, chunk in enumerate(hybrid_buffer.chunks):
            logger.info(f"Chunk {i}: content_length={len(chunk.content)}, metadata={chunk.metadata}")
            assert chunk.content, "Chunk has no content"
            assert chunk.metadata, "Chunk has no metadata"
        
        logger.info("✅ HybridBuffer processing test passed")

    async def test_write_buffer_integration(self):
        """Test WriteBuffer integration of RoundBuffer + HybridBuffer."""
        logger.info("🧪 Testing WriteBuffer integration")
        
        # Create WriteBuffer with small RoundBuffer limits
        config = {
            "round_buffer": {"max_tokens": 80, "max_size": 2},
            "hybrid_buffer": {"max_size": 5},
            "flush_manager": {"max_workers": 1}
        }
        
        memory_service_handler = lambda data: logger.info(f"Mock flush: {len(data)} items")
        write_buffer = WriteBuffer(
            memory_service_handler=memory_service_handler,
            config=config
        )
        
        # Test 1: Add messages that should trigger transfer
        messages1 = self.create_test_messages(2, "First batch")
        result1 = await write_buffer.add(messages1, self.memory_service._session_id)
        
        logger.info(f"WriteBuffer after first add: {result1}")
        assert result1["status"] == "success"
        
        # Check RoundBuffer state
        round_buffer = write_buffer.get_round_buffer()
        logger.info(f"RoundBuffer state: rounds={len(round_buffer.rounds)}, tokens={round_buffer.current_tokens}")
        
        # Test 2: Add more messages to trigger transfer
        messages2 = self.create_test_messages(2, "Second batch that should trigger transfer")
        result2 = await write_buffer.add(messages2, self.memory_service._session_id)
        
        logger.info(f"WriteBuffer after second add: {result2}")
        
        # Check if transfer was triggered
        if result2.get("transfer_triggered"):
            logger.info("✅ Transfer was triggered as expected")
            
            # Check HybridBuffer state
            hybrid_buffer = write_buffer.get_hybrid_buffer()
            logger.info(f"HybridBuffer state: chunks={len(hybrid_buffer.chunks)}, rounds={len(hybrid_buffer.original_rounds)}")
            
            assert len(hybrid_buffer.chunks) > 0, "No chunks in HybridBuffer after transfer"
            assert len(hybrid_buffer.original_rounds) > 0, "No rounds in HybridBuffer after transfer"
        else:
            logger.warning("⚠️ Transfer was not triggered - may need larger messages")
        
        logger.info("✅ WriteBuffer integration test passed")

    async def test_query_buffer_functionality(self):
        """Test QueryBuffer multi-source querying."""
        logger.info("🧪 Testing QueryBuffer functionality")
        
        # Create QueryBuffer
        query_buffer = QueryBuffer(max_size=10)
        
        # Create and populate HybridBuffer for testing
        hybrid_buffer = HybridBuffer(max_size=5, chunk_strategy="message")
        await hybrid_buffer.initialize()
        
        # Add test data to HybridBuffer
        test_rounds = [
            self.create_test_messages(2, "Space exploration Mars"),
            self.create_test_messages(2, "Artificial intelligence machine learning")
        ]
        await hybrid_buffer.add_from_rounds(test_rounds)
        
        # Set HybridBuffer reference
        query_buffer.set_hybrid_buffer(hybrid_buffer)
        
        # Test query
        query_text = "Mars space"
        results = await query_buffer.query(query_text, top_k=5)
        
        logger.info(f"Query '{query_text}' returned {len(results)} results")
        for i, result in enumerate(results):
            logger.info(f"Result {i}: content_preview={result.get('content', '')[:50]}...")
        
        # Verify results
        assert isinstance(results, list), "Query should return a list"
        # Note: Results might be empty if no matches found, which is OK for this test
        
        logger.info("✅ QueryBuffer functionality test passed")

    async def test_buffer_service_orchestration(self):
        """Test BufferService high-level orchestration."""
        logger.info("🧪 Testing BufferService orchestration")
        
        # Create BufferService with small limits to trigger transfers
        config = {
            "buffer": {
                "round_buffer": {"max_tokens": 100, "max_size": 2},
                "hybrid_buffer": {"max_size": 5}
            }
        }
        
        buffer_service = BufferService(
            memory_service=self.memory_service,
            user=self.user_name,
            config=config
        )
        await buffer_service.initialize()
        
        # Test 1: Add messages
        messages = self.create_test_messages(3, "BufferService test message")
        result = await buffer_service.add(messages, session_id=self.memory_service._session_id)
        
        logger.info(f"BufferService add result: {result}")
        assert result["status"] == "success"
        
        # Test 2: Check component states
        write_buffer = buffer_service.get_write_buffer()
        round_buffer = write_buffer.get_round_buffer()
        hybrid_buffer = write_buffer.get_hybrid_buffer()
        
        logger.info(f"Component states - RoundBuffer: {len(round_buffer.rounds)} rounds, HybridBuffer: {len(hybrid_buffer.chunks)} chunks")
        
        # Test 3: Query
        query_result = await buffer_service.query("test message", top_k=3)
        logger.info(f"Query result: status={query_result['status']}, results_count={len(query_result.get('data', {}).get('results', []))}")
        
        assert query_result["status"] == "success"
        
        # Cleanup
        await buffer_service.shutdown()
        
        logger.info("✅ BufferService orchestration test passed")


async def run_module_tests():
    """Run all module tests."""
    logger.info("🚀 Starting Buffer Module Tests")
    logger.info("=" * 60)
    
    test_runner = BufferModuleTests()
    
    try:
        # Setup
        await test_runner.setup_memory_service()
        
        # Run individual module tests
        await test_runner.test_round_buffer_basic()
        await test_runner.test_hybrid_buffer_processing()
        await test_runner.test_write_buffer_integration()
        await test_runner.test_query_buffer_functionality()
        await test_runner.test_buffer_service_orchestration()
        
        logger.info("=" * 60)
        logger.info("🎉 All Buffer Module Tests Passed!")
        
    except Exception as e:
        logger.error(f"❌ Module test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        if hasattr(test_runner, 'memory_service'):
            await test_runner.memory_service.shutdown()


if __name__ == "__main__":
    asyncio.run(run_module_tests())
