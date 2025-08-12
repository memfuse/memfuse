#!/usr/bin/env python3
"""
PgVectorScale Integration Test

This test validates the complete integration of pgvectorscale with MemFuse,
including M0/M1 data flow, chunking strategies, embedding models, and vector search.
"""

import asyncio
import pytest
import time
from typing import List, Dict, Any
from loguru import logger

from src.memfuse_core.hierarchy.streaming_pipeline import (
    StreamingDataProcessor,
    create_demo_message_generator
)
from src.memfuse_core.hierarchy.pgvectorscale_memory_layer import PgVectorScaleMemoryLayer


class TestPgVectorScaleIntegration:
    """Integration tests for PgVectorScale memory layer."""
    
    @pytest.fixture
    async def db_config(self):
        """Database configuration for testing."""
        return {
            'host': 'localhost',
            'port': 5432,
            'database': 'memfuse',
            'user': 'postgres',
            'password': 'postgres'
        }
    
    @pytest.fixture
    async def streaming_processor(self, db_config):
        """Create a streaming data processor for testing."""
        processor = StreamingDataProcessor(
            user_id="test_user",
            db_config=db_config,
            batch_size=5,
            processing_delay=0.1
        )
        
        # Initialize the processor
        success = await processor.initialize()
        assert success, "Failed to initialize streaming processor"
        
        yield processor
        
        # Cleanup
        await processor.close()
    
    @pytest.fixture
    async def memory_layer(self, db_config):
        """Create a memory layer for testing."""
        layer = PgVectorScaleMemoryLayer(
            user_id="test_user",
            db_config=db_config
        )
        
        # Initialize the layer
        success = await layer.initialize()
        assert success, "Failed to initialize memory layer"
        
        yield layer
        
        # Cleanup
        await layer.close()
    
    @pytest.mark.asyncio
    async def test_memory_layer_initialization(self, memory_layer):
        """Test memory layer initialization."""
        # Check that the layer is initialized
        assert memory_layer.initialized
        
        # Check layer status
        status = await memory_layer.get_layer_status()
        assert "M0" in status
        assert "M1" in status
        
        # Get stats
        stats = await memory_layer.get_stats()
        assert "layer_status" in stats
        assert "initialized" in stats
        assert stats["initialized"] is True
    
    @pytest.mark.asyncio
    async def test_streaming_data_processing(self, streaming_processor):
        """Test streaming data processing through M0 -> M1 pipeline."""
        # Generate demo messages
        messages = []
        async for message in create_demo_message_generator(num_messages=12, delay_between_messages=0):
            messages.append(message)
        
        assert len(messages) == 12, "Should generate 12 demo messages"
        
        # Process messages through the pipeline
        result = await streaming_processor.process_streaming_messages(
            messages=messages,
            session_id="test_session_001"
        )
        
        # Validate processing result
        assert result["success"], f"Processing failed: {result.get('message', 'Unknown error')}"
        assert result["messages_processed"] == 12
        assert result["items_written"] > 0
        assert "processing_time" in result
        
        logger.info(f"Processed {result['messages_processed']} messages in {result['processing_time']:.3f}s")
    
    @pytest.mark.asyncio
    async def test_vector_similarity_search(self, streaming_processor):
        """Test vector similarity search functionality."""
        # First, process some data
        messages = []
        async for message in create_demo_message_generator(num_messages=18, delay_between_messages=0):
            messages.append(message)
        
        # Process messages
        process_result = await streaming_processor.process_streaming_messages(
            messages=messages,
            session_id="test_session_002"
        )
        
        assert process_result["success"], "Failed to process messages for search test"
        
        # Wait a moment for processing to complete
        await asyncio.sleep(1.0)
        
        # Test various queries
        test_queries = [
            "Python machine learning algorithms",
            "deep learning neural networks",
            "data science project workflow",
            "vector databases and embeddings"
        ]
        
        for query in test_queries:
            logger.info(f"Testing query: '{query}'")
            
            query_result = await streaming_processor.query_processed_data(
                query=query,
                top_k=5,
                similarity_threshold=0.1
            )
            
            # Validate query result
            assert query_result["success"], f"Query failed: {query_result.get('error', 'Unknown error')}"
            assert "results" in query_result
            assert "total_count" in query_result
            
            # Check that we got some results
            results = query_result["results"]
            if results:
                logger.info(f"Query '{query}' returned {len(results)} results")
                
                # Validate result structure
                for i, result in enumerate(results[:3]):  # Check first 3 results
                    assert "content" in result
                    assert "similarity_score" in result
                    assert "chunk_id" in result
                    
                    # Check similarity score is in valid range (0-1)
                    similarity = result["similarity_score"]
                    assert 0.0 <= similarity <= 1.0, f"Invalid similarity score: {similarity}"
                    
                    logger.info(f"  Result {i+1}: similarity={similarity:.4f}, content='{result['content'][:100]}...'")
            else:
                logger.warning(f"Query '{query}' returned no results")
    
    @pytest.mark.asyncio
    async def test_data_integrity_and_lineage(self, streaming_processor):
        """Test data integrity and lineage tracking."""
        # Process a batch of messages
        messages = []
        async for message in create_demo_message_generator(num_messages=15, delay_between_messages=0):
            messages.append(message)
        
        # Process with specific session ID for tracking
        session_id = "integrity_test_session"
        result = await streaming_processor.process_streaming_messages(
            messages=messages,
            session_id=session_id
        )
        
        assert result["success"], "Failed to process messages for integrity test"
        
        # Get processing statistics
        stats = await streaming_processor.get_processing_stats()
        
        # Validate statistics
        assert "total_messages_processed" in stats
        assert "total_batches_processed" in stats
        assert "memory_layer_stats" in stats
        
        # Check that messages were processed
        assert stats["total_messages_processed"] >= 15
        assert stats["total_batches_processed"] > 0
        
        # Check memory layer statistics
        memory_stats = stats["memory_layer_stats"]
        if "store_stats" in memory_stats:
            store_stats = memory_stats["store_stats"]
            logger.info(f"Store statistics: {store_stats}")
        
        logger.info(f"Data integrity test passed: {stats['total_messages_processed']} messages processed")
    
    @pytest.mark.asyncio
    async def test_chunking_strategy_integration(self, memory_layer):
        """Test integration with existing chunking strategies."""
        # Create test message batches
        test_messages = [
            [
                {"content": "This is the first message in the batch.", "role": "user"},
                {"content": "This is the second message in the batch.", "role": "assistant"},
                {"content": "This is the third message in the batch.", "role": "user"}
            ],
            [
                {"content": "This is a longer message that should be processed through the chunking strategy. It contains more content to test the token-based chunking logic.", "role": "user"},
                {"content": "This is the response to the longer message, also containing substantial content for chunking.", "role": "assistant"}
            ]
        ]
        
        # Write messages through memory layer
        write_result = await memory_layer.write(
            message_batch_list=test_messages,
            session_id="chunking_test_session",
            metadata={"test_type": "chunking_strategy"}
        )
        
        assert write_result.success, f"Write failed: {write_result.message}"
        assert write_result.items_written > 0
        
        # Query to verify chunking worked
        query_result = await memory_layer.query(
            query="longer message chunking strategy",
            top_k=3
        )
        
        assert len(query_result.results) > 0, "Should find results after chunking"
        
        # Check that results have chunking metadata
        for result in query_result.results:
            metadata = result.get("metadata", {})
            assert "source" in metadata
            
        logger.info(f"Chunking strategy test passed: {len(query_result.results)} results found")
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, streaming_processor):
        """Test performance benchmarks for the integrated system."""
        # Generate a larger dataset for performance testing
        num_messages = 50
        messages = []
        async for message in create_demo_message_generator(num_messages=num_messages, delay_between_messages=0):
            messages.append(message)
        
        # Measure processing time
        start_time = time.time()
        
        result = await streaming_processor.process_streaming_messages(
            messages=messages,
            session_id="performance_test_session"
        )
        
        processing_time = time.time() - start_time
        
        assert result["success"], "Performance test processing failed"
        
        # Calculate performance metrics
        messages_per_second = num_messages / processing_time
        
        logger.info(f"Performance test results:")
        logger.info(f"  Messages processed: {num_messages}")
        logger.info(f"  Total processing time: {processing_time:.3f}s")
        logger.info(f"  Messages per second: {messages_per_second:.2f}")
        logger.info(f"  Items written: {result['items_written']}")
        
        # Performance assertions (adjust thresholds as needed)
        assert processing_time < 30.0, f"Processing took too long: {processing_time:.3f}s"
        assert messages_per_second > 1.0, f"Processing too slow: {messages_per_second:.2f} msg/s"
        
        # Test query performance
        query_start = time.time()
        
        query_result = await streaming_processor.query_processed_data(
            query="machine learning algorithms performance",
            top_k=10
        )
        
        query_time = time.time() - query_start
        
        assert query_result["success"], "Performance test query failed"
        
        logger.info(f"Query performance: {query_time:.3f}s for {len(query_result['results'])} results")
        
        # Query performance assertion
        assert query_time < 5.0, f"Query took too long: {query_time:.3f}s"


# Standalone test runner
async def run_integration_tests():
    """Run integration tests standalone."""
    logger.info("Starting PgVectorScale integration tests...")
    
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'memfuse',
        'user': 'postgres',
        'password': 'postgres'
    }
    
    # Test 1: Basic streaming processing
    logger.info("Test 1: Basic streaming processing")
    processor = StreamingDataProcessor(user_id="standalone_test", db_config=db_config)
    
    try:
        success = await processor.initialize()
        if not success:
            logger.error("Failed to initialize processor")
            return False
        
        # Generate and process demo data
        messages = []
        async for message in create_demo_message_generator(num_messages=20, delay_between_messages=0):
            messages.append(message)
        
        result = await processor.process_streaming_messages(messages, session_id="standalone_test")
        
        if result["success"]:
            logger.info(f"✅ Processed {result['messages_processed']} messages successfully")
        else:
            logger.error(f"❌ Processing failed: {result['message']}")
            return False
        
        # Test 2: Vector search
        logger.info("Test 2: Vector similarity search")
        
        test_queries = [
            "Python machine learning",
            "deep learning frameworks",
            "data science workflow"
        ]
        
        for query in test_queries:
            query_result = await processor.query_processed_data(query, top_k=5)
            
            if query_result["success"]:
                logger.info(f"✅ Query '{query}' returned {len(query_result['results'])} results")
            else:
                logger.error(f"❌ Query '{query}' failed: {query_result.get('error', 'Unknown error')}")
                return False
        
        logger.info("✅ All integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False
    finally:
        await processor.close()


if __name__ == "__main__":
    # Run standalone tests
    asyncio.run(run_integration_tests())
