"""GraphML graph store adapter for the refactored architecture."""

from typing import Dict, List, Optional, Any

from .base import GraphStore
from ...models.core import Item, Query, QueryResult, Node, Edge
from ...rag.encode.base import EncoderBase
from .graphml_store import GraphMLStore as OldGraphMLStore

class GraphMLStore(GraphStore):
    """GraphML graph store adapter for the refactored architecture.

    This class adapts the existing GraphMLStore implementation to the new
    GraphStore interface.
    """

    def __init__(
        self,
        data_dir: str,
        encoder: Optional[EncoderBase] = None,
        model_name: str = "all-MiniLM-L6-v2",
        cache_size: int = 100,
        buffer_size: int = 10,
        **kwargs
    ):
        """Initialize the graph store.

        Args:
            data_dir: Directory to store data
            encoder: Encoder to use
            model_name: Name of the embedding model
            cache_size: Size of the query cache
            buffer_size: Size of the write buffer
            **kwargs: Additional arguments
        """
        super().__init__(data_dir, encoder=encoder, model_name=model_name, cache_size=cache_size, buffer_size=buffer_size, **kwargs)

        # Create the old implementation
        self.store = OldGraphMLStore(
            data_dir=data_dir,
            cache_size=cache_size,
            **kwargs
        )

    async def initialize(self) -> bool:
        """Initialize the graph store.

        Returns:
            True if successful, False otherwise
        """
        return await self.store.initialize()

    async def add_node(self, item: Item) -> str:
        """Add a node to the graph.

        Args:
            item: Item to add as a node

        Returns:
            ID of the added node
        """
        # Convert Item to Node
        node = Node(
            id=item.id,
            content=item.content,
            metadata=item.metadata
        )
        return await self.store.add_node(node)

    async def add_edge(self, source_id: str, target_id: str, edge_type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add an edge to the graph.

        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            edge_type: Type of the edge
            metadata: Edge metadata

        Returns:
            ID of the added edge
        """
        # Create Edge
        edge = Edge(
            id=f"{source_id}_{edge_type}_{target_id}",
            source_id=source_id,
            target_id=target_id,
            relation=edge_type,
            metadata=metadata or {}
        )
        return await self.store.add_edge(edge)

    async def get_node(self, node_id: str) -> Optional[Item]:
        """Get a node by ID.

        Args:
            node_id: ID of the node

        Returns:
            Node if found, None otherwise
        """
        node = await self.store.get_node(node_id)
        if node is None:
            return None

        # Convert Node to Item
        return Item(
            id=node.id,
            content=node.content,
            metadata=node.metadata
        )

    async def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[Item]:
        """Get neighbors of a node.

        Args:
            node_id: ID of the node
            edge_type: Type of edges to follow (optional)

        Returns:
            List of neighbor nodes
        """
        results = await self.store.get_neighbors(node_id, relation=edge_type)

        # Convert QueryResult to Item
        return [
            Item(
                id=result.id,
                content=result.content,
                metadata=result.metadata
            )
            for result in results
        ]

    async def delete_node(self, node_id: str) -> bool:
        """Delete a node.

        Args:
            node_id: ID of the node to delete

        Returns:
            True if successful, False otherwise
        """
        return await self.store.delete_node(node_id)

    async def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge.

        Args:
            edge_id: ID of the edge to delete

        Returns:
            True if successful, False otherwise
        """
        return await self.store.delete_edge(edge_id)

    async def query(self, query: Query, top_k: int = 5) -> List[QueryResult]:
        """Query the store.

        Args:
            query: Query to execute
            top_k: Number of results to return

        Returns:
            List of query results
        """
        return await self.store.query(query, top_k)

    async def clear(self) -> bool:
        """Clear the store.

        Returns:
            True if successful, False otherwise
        """
        return await self.store.clear()
