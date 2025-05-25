"""L0 (Raw Data) layer for the hierarchical memory system."""

from typing import Any, List, Dict, Optional
import asyncio
import time
from .base import MemoryLayer


class L0Manager(MemoryLayer):
    """Manages the L0 (Raw Data) layer.

    The L0 layer stores raw data in its original form using multiple storage components:
    - Vector Store: For semantic similarity search
    - Graph Store: For relationship-based search
    - Keyword Store: For keyword-based search
    - Metadata DB: For raw data storage

    It also coordinates with the L1 layer for fact extraction.
    """

    def __init__(
        self,
        vector_store: Any,
        graph_store: Any,
        keyword_store: Any,
        metadata_db: Any,
        buffer_manager: Optional[Any] = None,
        l1_manager: Optional[Any] = None,
    ):
        """Initialize the L0 manager.

        Args:
            vector_store: Vector store component
            graph_store: Graph store component
            keyword_store: Keyword store component
            metadata_db: Metadata database
            buffer_manager: Buffer manager (optional)
            l1_manager: L1 manager (optional)
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.keyword_store = keyword_store
        self.metadata_db = metadata_db
        self.buffer_manager = buffer_manager
        self.l1_manager = l1_manager

        # Statistics
        self.total_items_added = 0
        self.total_items_retrieved = 0
        self.total_searches = 0
        self.total_search_time = 0

    async def add(self, item: Any) -> str:
        """Add an item to the L0 layer.

        This method adds the item to all storage components.

        Args:
            item: Item to add

        Returns:
            ID of the added item
        """
        # Update statistics
        self.total_items_added += 1

        # If buffer manager is available, use it
        if self.buffer_manager:
            await self.buffer_manager.add(item)
            return item.id

        # Otherwise, add directly to storage components
        return await self._process_item(item)

    async def _process_item(self, item: Any) -> str:
        """Process an item for storage.

        This method is called when the buffer is flushed or when adding directly.

        Args:
            item: Item to process

        Returns:
            ID of the processed item
        """
        # Store in metadata DB
        self.metadata_db.add_item(item)

        # Store in vector store
        await self.vector_store.add(item)

        # Store in graph store
        if hasattr(self.graph_store, 'add_node'):
            # Convert item to node if needed
            if hasattr(self.graph_store, 'Node') and not isinstance(item, self.graph_store.Node):
                node = self.graph_store.Node.from_item(item)
                await self.graph_store.add_node(node)
            else:
                await self.graph_store.add_node(item)
        else:
            await self.graph_store.add(item)

        # Store in keyword store
        await self.keyword_store.add(item)

        # Trigger L1 fact extraction if L1 manager is available
        if self.l1_manager:
            asyncio.create_task(self.l1_manager.process_item(item))

        return item.id

    async def process_batch(self, items: List[Any]) -> List[str]:
        """Process a batch of items.

        This method is called when the buffer is flushed.

        Args:
            items: Items to process

        Returns:
            List of processed item IDs
        """
        # Update statistics
        self.total_items_added += len(items)

        # Process each item
        item_ids = []
        for item in items:
            item_id = await self._process_item(item)
            item_ids.append(item_id)

        return item_ids

    async def get(self, item_id: str) -> Optional[Any]:
        """Get an item from the L0 layer.

        Args:
            item_id: ID of the item to get

        Returns:
            Item if found, None otherwise
        """
        # Update statistics
        self.total_items_retrieved += 1

        # Get from metadata DB
        return self.metadata_db.get_item(item_id)

    async def search(self, query: str, top_k: int = 5, store_type: Optional[str] = None) -> List[Any]:
        """Search for items in the L0 layer.

        Args:
            query: Query string
            top_k: Number of results to return
            store_type: Type of store to search (vector, graph, keyword)

        Returns:
            List of matching items
        """
        start_time = time.time()
        self.total_searches += 1

        try:
            # If buffer manager is available, use it
            if self.buffer_manager and not store_type:
                return await self.buffer_manager.query(query, top_k=top_k)

            # Otherwise, search directly in storage components
            if store_type == "vector" or not store_type:
                return await self.vector_store.search(query, top_k=top_k)
            elif store_type == "graph":
                return await self.graph_store.search(query, top_k=top_k)
            elif store_type == "keyword":
                return await self.keyword_store.search(query, top_k=top_k)
            else:
                # Invalid store type, use vector store
                return await self.vector_store.search(query, top_k=top_k)
        finally:
            # Update statistics
            search_time = time.time() - start_time
            self.total_search_time += search_time

    async def update(self, item_id: str, item: Any) -> bool:
        """Update an item in the L0 layer.

        Args:
            item_id: ID of the item to update
            item: Updated item

        Returns:
            True if the item was updated, False otherwise
        """
        # Update in metadata DB
        self.metadata_db.update_item(item_id, item)

        # Update in vector store
        await self.vector_store.update(item_id, item)

        # Update in graph store
        if hasattr(self.graph_store, 'update_node'):
            # Convert item to node if needed
            if hasattr(self.graph_store, 'Node') and not isinstance(item, self.graph_store.Node):
                node = self.graph_store.Node.from_item(item)
                await self.graph_store.update_node(item_id, node)
            else:
                await self.graph_store.update_node(item_id, item)
        else:
            await self.graph_store.update(item_id, item)

        # Update in keyword store
        await self.keyword_store.update(item_id, item)

        return True

    async def delete(self, item_id: str) -> bool:
        """Delete an item from the L0 layer.

        Args:
            item_id: ID of the item to delete

        Returns:
            True if the item was deleted, False otherwise
        """
        # Delete from metadata DB
        self.metadata_db.delete_item(item_id)

        # Delete from vector store
        await self.vector_store.delete(item_id)

        # Delete from graph store
        if hasattr(self.graph_store, 'delete_node'):
            await self.graph_store.delete_node(item_id)
        else:
            await self.graph_store.delete(item_id)

        # Delete from keyword store
        await self.keyword_store.delete(item_id)

        return True

    async def clear(self) -> bool:
        """Clear all items from the L0 layer.

        Returns:
            True if the L0 layer was cleared, False otherwise
        """
        # Clear metadata DB
        self.metadata_db.clear()

        # Clear vector store
        await self.vector_store.clear()

        # Clear graph store
        await self.graph_store.clear()

        # Clear keyword store
        await self.keyword_store.clear()

        return True

    async def count(self) -> int:
        """Count the number of items in the L0 layer.

        Returns:
            Number of items
        """
        # Count from metadata DB
        return self.metadata_db.count()

    def get_stats(self) -> Dict[str, Any]:
        """Get L0 layer statistics.

        Returns:
            Dictionary of L0 layer statistics
        """
        return {
            "total_items_added": self.total_items_added,
            "total_items_retrieved": self.total_items_retrieved,
            "total_searches": self.total_searches,
            "total_search_time": self.total_search_time,
            "avg_search_time": self.total_search_time / max(1, self.total_searches),
            "item_count": self.metadata_db.count(),
        }
