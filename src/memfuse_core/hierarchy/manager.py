"""Manager for the memory hierarchy system."""

from typing import Any, List, Dict, Optional, Callable, Awaitable, Union
import logging
import asyncio
import time
import os
from .base import MemoryLayer, MemoryItem, Fact, Entity, Relationship
from .l0 import L0Manager
from .l1 import L1Manager, FactsDatabase, LLMService
from .l2 import L2Manager, GraphDatabase
from ..buffer.manager import BufferManager
from ..utils.path_manager import PathManager

logger = logging.getLogger(__name__)


class HierarchyMemoryManager:
    """Manager for the memory hierarchy system.

    The memory hierarchy manager coordinates between the different memory layers:
    - L0 (Raw Data): Stores raw data in its original form
    - L1 (Facts): Extracts and stores facts from raw data
    - L2 (Knowledge Graph): Constructs a knowledge graph from facts

    It provides a unified interface for memory operations and ensures
    data flows correctly between the layers.
    """

    def __init__(
        self,
        user_id: str,
        data_dir: str,
        vector_store: Any,
        graph_store: Any,
        keyword_store: Any,
        metadata_db: Any,
        buffer_manager: Optional[BufferManager] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the memory hierarchy manager.

        Args:
            user_id: User ID
            data_dir: Data directory
            vector_store: Vector store component
            graph_store: Graph store component
            keyword_store: Keyword store component
            metadata_db: Metadata database
            buffer_manager: Buffer manager (optional)
            config: Configuration dictionary
        """
        self.user_id = user_id
        self.data_dir = data_dir
        self.config = config or {}

        # Create user directory
        user_dir = os.path.join(data_dir, user_id)
        PathManager.ensure_directory(user_dir)

        # Create LLM service
        self.llm_service = LLMService()

        # Create buffer manager if not provided
        self.buffer_manager = buffer_manager or BufferManager(
            config=self.config.get("buffer", {})
        )

        # Create L0 manager
        self.l0_manager = L0Manager(
            vector_store=vector_store,
            graph_store=graph_store,
            keyword_store=keyword_store,
            metadata_db=metadata_db,
            buffer_manager=self.buffer_manager
        )

        # Create facts database
        self.facts_db = FactsDatabase(data_dir=user_dir)

        # Create graph database
        self.graph_db = GraphDatabase(data_dir=user_dir)

        # Create L1 manager
        self.l1_manager = L1Manager(
            facts_db=self.facts_db,
            llm_service=self.llm_service,
            buffer_manager=self.buffer_manager,
            extraction_batch_size=self.config.get("l1", {}).get("extraction_batch_size", 5)
        )

        # Create L2 manager
        self.l2_manager = L2Manager(
            graph_db=self.graph_db,
            llm_service=self.llm_service,
            buffer_manager=self.buffer_manager,
            construction_batch_size=self.config.get("l2", {}).get("construction_batch_size", 10)
        )

        # Connect the managers
        self.l0_manager.l1_manager = self.l1_manager
        self.l1_manager.l2_manager = self.l2_manager

        # Register retrieval callbacks
        self.buffer_manager.register_retrieval_callback("vector", self._vector_search)
        self.buffer_manager.register_retrieval_callback("graph", self._graph_search)
        self.buffer_manager.register_retrieval_callback("keyword", self._keyword_search)
        self.buffer_manager.register_retrieval_callback("facts", self._facts_search)
        self.buffer_manager.register_retrieval_callback("knowledge", self._knowledge_search)

        # Set storage callback for writing buffer
        self.buffer_manager.writing_buffer.flush_callback = self._process_items

        # Statistics
        self.total_items_added = 0
        self.total_queries = 0
        self.total_query_time = 0

    async def _process_items(self, items: List[Any]) -> List[str]:
        """Process items from the writing buffer.

        Args:
            items: Items to process

        Returns:
            List of processed item IDs
        """
        return await self.l0_manager.process_batch(items)

    async def _vector_search(self, query: str, top_k: int) -> List[Any]:
        """Search the vector store.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of matching items
        """
        results = await self.l0_manager.search(query, top_k=top_k, store_type="vector")

        # Add retrieval method to metadata
        for result in results:
            if hasattr(result, 'metadata'):
                result.metadata["retrieval"] = {"method": "vector", "layer": "l0"}

        return results

    async def _graph_search(self, query: str, top_k: int) -> List[Any]:
        """Search the graph store.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of matching items
        """
        results = await self.l0_manager.search(query, top_k=top_k, store_type="graph")

        # Add retrieval method to metadata
        for result in results:
            if hasattr(result, 'metadata'):
                result.metadata["retrieval"] = {"method": "graph", "layer": "l0"}

        return results

    async def _keyword_search(self, query: str, top_k: int) -> List[Any]:
        """Search the keyword store.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of matching items
        """
        results = await self.l0_manager.search(query, top_k=top_k, store_type="keyword")

        # Add retrieval method to metadata
        for result in results:
            if hasattr(result, 'metadata'):
                result.metadata["retrieval"] = {"method": "keyword", "layer": "l0"}

        return results

    async def _facts_search(self, query: str, top_k: int) -> List[Any]:
        """Search the facts database.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of matching facts
        """
        results = await self.l1_manager.search(query, top_k=top_k)

        # Add retrieval method to metadata
        for result in results:
            if hasattr(result, 'metadata'):
                result.metadata["retrieval"] = {"method": "facts", "layer": "l1"}

        return results

    async def _knowledge_search(self, query: str, top_k: int) -> List[Any]:
        """Search the knowledge graph.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of matching entities and relationships
        """
        results = await self.l2_manager.search(query, top_k=top_k)

        # Add retrieval method to metadata
        for result in results:
            if hasattr(result, 'metadata'):
                result.metadata["retrieval"] = {"method": "knowledge", "layer": "l2"}

        return results

    async def add(self, item: Any) -> str:
        """Add an item to the memory hierarchy system.

        This method adds the item to the L0 layer, which will trigger
        the memory hierarchy pipeline.

        Args:
            item: Item to add

        Returns:
            ID of the added item
        """
        # Update statistics
        self.total_items_added += 1

        # Add to L0 layer
        return await self.l0_manager.add(item)

    async def add_batch(self, items: List[Any]) -> List[str]:
        """Add multiple items to the memory hierarchy system.

        Args:
            items: Items to add

        Returns:
            List of added item IDs
        """
        # Update statistics
        self.total_items_added += len(items)

        # Add to buffer manager
        item_ids = []
        for item in items:
            item_id = await self.add(item)
            item_ids.append(item_id)

        return item_ids

    async def query(
        self,
        query: str,
        top_k: int = 5,
        layers: Optional[List[str]] = None,
        methods: Optional[List[str]] = None,
    ) -> List[Any]:
        """Query the memory hierarchy system.

        Args:
            query: Query string
            top_k: Number of results to return
            layers: Layers to query (l0, l1, l2)
            methods: Methods to use (vector, graph, keyword, facts, knowledge)

        Returns:
            List of matching items
        """
        start_time = time.time()
        self.total_queries += 1

        try:
            # Filter retrieval callbacks based on layers and methods
            retrieval_callbacks = {}

            if layers is None:
                layers = ["l0", "l1", "l2"]

            if methods is None:
                methods = ["vector", "graph", "keyword", "facts", "knowledge"]

            # Map methods to layers
            layer_methods = {
                "l0": ["vector", "graph", "keyword"],
                "l1": ["facts"],
                "l2": ["knowledge"]
            }

            # Filter callbacks
            for method in methods:
                for layer in layers:
                    if method in layer_methods.get(layer, []):
                        if method in self.buffer_manager.retrieval_callbacks:
                            retrieval_callbacks[method] = self.buffer_manager.retrieval_callbacks[method]

            # Save original callbacks
            original_callbacks = self.buffer_manager.retrieval_callbacks

            try:
                # Set temporary callbacks
                self.buffer_manager.retrieval_callbacks = retrieval_callbacks

                # Process query using query buffer
                results = await self.buffer_manager.query(query, top_k=top_k)

                return results
            finally:
                # Restore original callbacks
                self.buffer_manager.retrieval_callbacks = original_callbacks
        finally:
            # Update statistics
            query_time = time.time() - start_time
            self.total_query_time += query_time

    async def get(self, item_id: str, layer: Optional[str] = None) -> Optional[Any]:
        """Get an item from the memory hierarchy system.

        Args:
            item_id: ID of the item to get
            layer: Layer to get from (l0, l1, l2)

        Returns:
            Item if found, None otherwise
        """
        if layer == "l0" or layer is None:
            item = await self.l0_manager.get(item_id)
            if item:
                return item

        if layer == "l1" or layer is None:
            fact = await self.l1_manager.get(item_id)
            if fact:
                return fact

        if layer == "l2" or layer is None:
            knowledge = await self.l2_manager.get(item_id)
            if knowledge:
                return knowledge

        return None

    async def close(self):
        """Close the memory hierarchy system and clean up resources."""
        if self.buffer_manager:
            await self.buffer_manager.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get memory hierarchy system statistics.

        Returns:
            Dictionary of memory hierarchy system statistics
        """
        return {
            "total_items_added": self.total_items_added,
            "total_queries": self.total_queries,
            "total_query_time": self.total_query_time,
            "avg_query_time": self.total_query_time / max(1, self.total_queries),
            "l0": self.l0_manager.get_stats(),
            "l1": self.l1_manager.get_stats(),
            "l2": self.l2_manager.get_stats(),
            "buffer": self.buffer_manager.get_stats(),
        }
