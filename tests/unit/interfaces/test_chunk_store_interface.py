"""Test cases for the unified chunk store interface.

This module contains tests to verify that the ChunkStoreInterface
is properly implemented across all store types.
"""

import pytest
import asyncio
from typing import List, Optional, Dict, Any

from src.memfuse_core.interfaces.chunk_store import ChunkStoreInterface, StorageError
from src.memfuse_core.rag.chunk.base import ChunkData
from src.memfuse_core.models import Query


class MockChunkStore(ChunkStoreInterface):
    """Mock implementation of ChunkStoreInterface for testing."""
    
    def __init__(self):
        """Initialize mock store."""
        self.chunks = {}  # chunk_id -> ChunkData
        self.call_count = {
            'add': 0,
            'read': 0,
            'update': 0,
            'query': 0,
            'delete': 0,
            'count': 0,
            'clear': 0
        }
    
    async def add(self, chunks: List[ChunkData]) -> List[str]:
        """Add chunks to mock store."""
        self.call_count['add'] += 1
        
        if not chunks:
            return []
        
        chunk_ids = []
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            chunk_ids.append(chunk.chunk_id)
        
        return chunk_ids
    
    async def read(self, chunk_ids: List[str], filters: Optional[Dict[str, Any]] = None) -> List[Optional[ChunkData]]:
        """Read chunks from mock store with optional filters."""
        self.call_count['read'] = self.call_count.get('read', 0) + 1

        result = []
        for chunk_id in chunk_ids:
            chunk = self.chunks.get(chunk_id)

            if chunk is None:
                result.append(None)
                continue

            # Apply filters if provided
            if filters:
                matches = True
                for key, value in filters.items():
                    if chunk.metadata.get(key) != value:
                        matches = False
                        break

                if matches:
                    result.append(chunk)
                else:
                    result.append(None)
            else:
                result.append(chunk)

        return result

    async def update(self, chunk_id: str, chunk: ChunkData) -> bool:
        """Update chunk in mock store."""
        self.call_count['update'] = self.call_count.get('update', 0) + 1

        if chunk_id in self.chunks:
            self.chunks[chunk_id] = chunk
            return True
        else:
            return False
    
    async def query(self, query: Query, top_k: int = 5) -> List[ChunkData]:
        """Query chunks from mock store."""
        self.call_count['query'] += 1
        
        # Simple mock implementation: return all chunks that contain query text
        matching_chunks = []
        for chunk in self.chunks.values():
            if query.text.lower() in chunk.content.lower():
                matching_chunks.append(chunk)
        
        return matching_chunks[:top_k]
    
    async def delete(self, chunk_ids: List[str]) -> List[bool]:
        """Delete chunks from mock store."""
        self.call_count['delete'] += 1
        
        results = []
        for chunk_id in chunk_ids:
            if chunk_id in self.chunks:
                del self.chunks[chunk_id]
                results.append(True)
            else:
                results.append(False)
        
        return results
    
    async def count(self) -> int:
        """Get total number of chunks in mock store."""
        self.call_count['count'] += 1
        return len(self.chunks)
    
    async def clear(self) -> bool:
        """Clear all chunks from mock store."""
        self.call_count['clear'] += 1
        self.chunks.clear()
        return True


@pytest.fixture
def mock_store():
    """Create a mock chunk store for testing."""
    return MockChunkStore()


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        ChunkData(
            content="This is the first test chunk about Python programming.",
            chunk_id="chunk_1",
            metadata={"type": "chunk", "user_id": "test_user", "topic": "python"}
        ),
        ChunkData(
            content="This is the second test chunk about machine learning.",
            chunk_id="chunk_2", 
            metadata={"type": "chunk", "user_id": "test_user", "topic": "ml"}
        ),
        ChunkData(
            content="This is the third test chunk about data science.",
            chunk_id="chunk_3",
            metadata={"type": "chunk", "user_id": "test_user", "topic": "data"}
        )
    ]


@pytest.fixture
def sample_query():
    """Create a sample query for testing."""
    return Query(
        text="Python programming",
        metadata={"user_id": "test_user"}
    )


class TestChunkStoreInterface:
    """Test cases for ChunkStoreInterface."""
    
    @pytest.mark.asyncio
    async def test_add_chunks(self, mock_store, sample_chunks):
        """Test adding chunks to store."""
        # Test adding chunks
        chunk_ids = await mock_store.add(sample_chunks)
        
        # Verify results
        assert len(chunk_ids) == 3
        assert chunk_ids == ["chunk_1", "chunk_2", "chunk_3"]
        assert mock_store.call_count['add'] == 1
        
        # Test adding empty list
        empty_result = await mock_store.add([])
        assert empty_result == []
    
    @pytest.mark.asyncio
    async def test_read_chunks(self, mock_store, sample_chunks):
        """Test reading chunks from store."""
        # Add chunks first
        await mock_store.add(sample_chunks)

        # Test reading existing chunks
        retrieved_chunks = await mock_store.read(["chunk_1", "chunk_3"])

        # Verify results
        assert len(retrieved_chunks) == 2
        assert retrieved_chunks[0].chunk_id == "chunk_1"
        assert retrieved_chunks[1].chunk_id == "chunk_3"
        assert mock_store.call_count['read'] == 1

        # Test reading non-existent chunk
        missing_chunks = await mock_store.read(["chunk_999"])
        assert missing_chunks == [None]

        # Test reading with filters
        filtered_chunks = await mock_store.read(
            ["chunk_1", "chunk_2"],
            filters={"user_id": "test_user", "topic": "python"}
        )
        assert len(filtered_chunks) == 2
        assert filtered_chunks[0].chunk_id == "chunk_1"  # matches filter
        assert filtered_chunks[1] is None  # doesn't match topic filter

    @pytest.mark.asyncio
    async def test_update_chunks(self, mock_store, sample_chunks):
        """Test updating chunks in store."""
        # Add chunks first
        await mock_store.add(sample_chunks)

        # Test updating existing chunk
        updated_chunk = ChunkData(
            content="Updated content for Python programming.",
            chunk_id="chunk_1",
            metadata={"type": "chunk", "user_id": "test_user", "topic": "python", "updated": True}
        )

        success = await mock_store.update("chunk_1", updated_chunk)
        assert success is True
        assert mock_store.call_count['update'] == 1

        # Verify chunk was updated
        retrieved_chunks = await mock_store.read(["chunk_1"])
        assert retrieved_chunks[0].content == "Updated content for Python programming."
        assert retrieved_chunks[0].metadata["updated"] is True

        # Test updating non-existent chunk
        non_existent_success = await mock_store.update("chunk_999", updated_chunk)
        assert non_existent_success is False
    
    @pytest.mark.asyncio
    async def test_query_chunks(self, mock_store, sample_chunks, sample_query):
        """Test querying chunks from store."""
        # Add chunks first
        await mock_store.add(sample_chunks)
        
        # Test query
        results = await mock_store.query(sample_query, top_k=5)
        
        # Verify results
        assert len(results) == 1  # Only chunk_1 contains "Python programming"
        assert results[0].chunk_id == "chunk_1"
        assert mock_store.call_count['query'] == 1
        
        # Test query with no matches
        no_match_query = Query(text="nonexistent topic", metadata={})
        no_results = await mock_store.query(no_match_query)
        assert no_results == []
    
    @pytest.mark.asyncio
    async def test_delete_chunks(self, mock_store, sample_chunks):
        """Test deleting chunks from store."""
        # Add chunks first
        await mock_store.add(sample_chunks)
        
        # Test deleting existing chunks
        delete_results = await mock_store.delete(["chunk_1", "chunk_3"])
        
        # Verify results
        assert delete_results == [True, True]
        assert mock_store.call_count['delete'] == 1
        
        # Verify chunks are actually deleted
        remaining_chunks = await mock_store.read(["chunk_1", "chunk_2", "chunk_3"])
        assert remaining_chunks == [None, sample_chunks[1], None]
        
        # Test deleting non-existent chunk
        delete_missing = await mock_store.delete(["chunk_999"])
        assert delete_missing == [False]
    
    @pytest.mark.asyncio
    async def test_count_chunks(self, mock_store, sample_chunks):
        """Test counting chunks in store."""
        # Test empty store
        initial_count = await mock_store.count()
        assert initial_count == 0
        assert mock_store.call_count['count'] == 1
        
        # Add chunks and test count
        await mock_store.add(sample_chunks)
        final_count = await mock_store.count()
        assert final_count == 3
    
    @pytest.mark.asyncio
    async def test_clear_chunks(self, mock_store, sample_chunks):
        """Test clearing all chunks from store."""
        # Add chunks first
        await mock_store.add(sample_chunks)
        assert await mock_store.count() == 3
        
        # Clear store
        clear_result = await mock_store.clear()
        assert clear_result is True
        assert mock_store.call_count['clear'] == 1
        
        # Verify store is empty
        final_count = await mock_store.count()
        assert final_count == 0
    
    @pytest.mark.asyncio
    async def test_interface_compliance(self, mock_store):
        """Test that mock store implements all required interface methods."""
        # Verify all required CRUD methods exist
        assert hasattr(mock_store, 'add')
        assert hasattr(mock_store, 'read')
        assert hasattr(mock_store, 'update')
        assert hasattr(mock_store, 'delete')

        # Verify query method exists
        assert hasattr(mock_store, 'query')

        # Verify utility methods exist
        assert hasattr(mock_store, 'count')
        assert hasattr(mock_store, 'clear')

        # Verify methods are async
        assert asyncio.iscoroutinefunction(mock_store.add)
        assert asyncio.iscoroutinefunction(mock_store.read)
        assert asyncio.iscoroutinefunction(mock_store.update)
        assert asyncio.iscoroutinefunction(mock_store.delete)
        assert asyncio.iscoroutinefunction(mock_store.query)
        assert asyncio.iscoroutinefunction(mock_store.count)
        assert asyncio.iscoroutinefunction(mock_store.clear)


class TestChunkStoreEdgeCases:
    """Test edge cases and error conditions for ChunkStoreInterface."""

    @pytest.mark.asyncio
    async def test_empty_operations(self, mock_store):
        """Test operations with empty inputs."""
        # Test empty add
        result = await mock_store.add([])
        assert result == []

        # Test empty read
        result = await mock_store.read([])
        assert result == []

        # Test empty delete
        result = await mock_store.delete([])
        assert result == []

    @pytest.mark.asyncio
    async def test_large_batch_operations(self, mock_store):
        """Test operations with large batches."""
        # Create a large batch of chunks
        large_batch = []
        for i in range(100):
            chunk = ChunkData(
                content=f"Test chunk {i} with some content",
                chunk_id=f"chunk_{i:03d}",
                metadata={"type": "chunk", "batch": "large", "index": i}
            )
            large_batch.append(chunk)

        # Test adding large batch
        chunk_ids = await mock_store.add(large_batch)
        assert len(chunk_ids) == 100

        # Test reading large batch
        read_chunks = await mock_store.read(chunk_ids)
        assert len(read_chunks) == 100
        assert all(chunk is not None for chunk in read_chunks)

        # Test deleting large batch
        delete_results = await mock_store.delete(chunk_ids)
        assert len(delete_results) == 100
        assert all(result is True for result in delete_results)

    @pytest.mark.asyncio
    async def test_duplicate_chunk_ids(self, mock_store):
        """Test handling of duplicate chunk IDs."""
        chunk1 = ChunkData(
            content="Original content",
            chunk_id="duplicate_id",
            metadata={"version": 1}
        )

        chunk2 = ChunkData(
            content="Updated content",
            chunk_id="duplicate_id",
            metadata={"version": 2}
        )

        # Add first chunk
        await mock_store.add([chunk1])

        # Add second chunk with same ID (should overwrite)
        await mock_store.add([chunk2])

        # Verify the second chunk overwrote the first
        retrieved = await mock_store.read(["duplicate_id"])
        assert retrieved[0].content == "Updated content"
        assert retrieved[0].metadata["version"] == 2

    @pytest.mark.asyncio
    async def test_complex_metadata_filtering(self, mock_store):
        """Test complex metadata filtering scenarios."""
        chunks = [
            ChunkData(
                content="Content 1",
                chunk_id="chunk_1",
                metadata={"user_id": "user1", "type": "message", "priority": "high", "tags": ["urgent", "important"]}
            ),
            ChunkData(
                content="Content 2",
                chunk_id="chunk_2",
                metadata={"user_id": "user1", "type": "chunk", "priority": "low", "tags": ["normal"]}
            ),
            ChunkData(
                content="Content 3",
                chunk_id="chunk_3",
                metadata={"user_id": "user2", "type": "message", "priority": "high", "tags": ["urgent"]}
            )
        ]

        await mock_store.add(chunks)

        # Test filtering by user_id
        user1_chunks = await mock_store.read(
            ["chunk_1", "chunk_2", "chunk_3"],
            filters={"user_id": "user1"}
        )
        assert user1_chunks[0] is not None  # chunk_1 matches
        assert user1_chunks[1] is not None  # chunk_2 matches
        assert user1_chunks[2] is None      # chunk_3 doesn't match

        # Test filtering by multiple criteria
        high_priority_messages = await mock_store.read(
            ["chunk_1", "chunk_2", "chunk_3"],
            filters={"type": "message", "priority": "high"}
        )
        assert high_priority_messages[0] is not None  # chunk_1 matches
        assert high_priority_messages[1] is None      # chunk_2 doesn't match (wrong type)
        assert high_priority_messages[2] is not None  # chunk_3 matches

    @pytest.mark.asyncio
    async def test_query_with_metadata_filters(self, mock_store):
        """Test query operations with metadata-based filtering."""
        chunks = [
            ChunkData(
                content="Python programming tutorial",
                chunk_id="chunk_1",
                metadata={"language": "python", "difficulty": "beginner"}
            ),
            ChunkData(
                content="Advanced Python concepts",
                chunk_id="chunk_2",
                metadata={"language": "python", "difficulty": "advanced"}
            ),
            ChunkData(
                content="JavaScript programming guide",
                chunk_id="chunk_3",
                metadata={"language": "javascript", "difficulty": "beginner"}
            )
        ]

        await mock_store.add(chunks)

        # Test query that should match multiple chunks
        query = Query(
            text="programming",
            metadata={"language": "python"}
        )

        results = await mock_store.query(query, top_k=5)
        # Note: Our mock implementation doesn't actually filter by metadata in query,
        # but this tests the interface structure
        assert len(results) >= 1
        assert any("programming" in chunk.content.lower() for chunk in results)


class TestStorageError:
    """Test cases for StorageError exception."""

    def test_storage_error_creation(self):
        """Test creating StorageError with different parameters."""
        # Basic error
        error1 = StorageError("Basic error")
        assert str(error1) == "Basic error"

        # Error with store type
        error2 = StorageError("Store error", store_type="vector")
        assert "Store: vector" in str(error2)

        # Error with operation
        error3 = StorageError("Operation error", operation="add")
        assert "Operation: add" in str(error3)

        # Error with all parameters
        error4 = StorageError("Full error", store_type="keyword", operation="query")
        error_str = str(error4)
        assert "Full error" in error_str
        assert "Store: keyword" in error_str
        assert "Operation: query" in error_str


if __name__ == "__main__":
    pytest.main([__file__])
