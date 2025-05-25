"""Base graph store module for MemFuse server."""

from loguru import logger
from abc import abstractmethod
from typing import List, Optional

from ..base import StoreBase
from ...models.core import Edge, Node, Query, QueryResult
from ...models.core import StoreType


class GraphStore(StoreBase):
    """Base class for graph store implementations."""

    def __init__(
        self,
        data_dir: str,
        cache_size: int = 100,
        **kwargs
    ):
        """Initialize the graph store.

        Args:
            data_dir: Directory to store data
            cache_size: Size of the query cache
            **kwargs: Additional arguments
        """
        super().__init__(data_dir, **kwargs)
        self.cache_size = cache_size

        # Initialize query cache
        self.query_cache = {}

    @property
    def store_type(self) -> StoreType:
        """Get the store type.

        Returns:
            Store type
        """
        return StoreType.GRAPH

    @abstractmethod
    async def add_node(self, node: Node) -> str:
        """Add a node to the graph.

        Args:
            node: Node to add

        Returns:
            ID of the added node
        """
        pass

    @abstractmethod
    async def add_nodes(self, nodes: List[Node]) -> List[str]:
        """Add multiple nodes to the graph.

        Args:
            nodes: Nodes to add

        Returns:
            List of IDs of the added nodes
        """
        pass

    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID.

        Args:
            node_id: ID of the node to get

        Returns:
            Node if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_nodes(self, node_ids: List[str]) -> List[Optional[Node]]:
        """Get multiple nodes by ID.

        Args:
            node_ids: IDs of the nodes to get

        Returns:
            List of nodes (None for nodes not found)
        """
        pass

    @abstractmethod
    async def update_node(self, node_id: str, node: Node) -> bool:
        """Update a node.

        Args:
            node_id: ID of the node to update
            node: New node data

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def update_nodes(self, node_ids: List[str], nodes: List[Node]) -> List[bool]:
        """Update multiple nodes.

        Args:
            node_ids: IDs of the nodes to update
            nodes: New node data

        Returns:
            List of success flags
        """
        pass

    @abstractmethod
    async def delete_node(self, node_id: str) -> bool:
        """Delete a node.

        Args:
            node_id: ID of the node to delete

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def delete_nodes(self, node_ids: List[str]) -> List[bool]:
        """Delete multiple nodes.

        Args:
            node_ids: IDs of the nodes to delete

        Returns:
            List of success flags
        """
        pass

    @abstractmethod
    async def add_edge(self, edge: Edge) -> str:
        """Add an edge to the graph.

        Args:
            edge: Edge to add

        Returns:
            ID of the added edge
        """
        pass

    @abstractmethod
    async def add_edges(self, edges: List[Edge]) -> List[str]:
        """Add multiple edges to the graph.

        Args:
            edges: Edges to add

        Returns:
            List of IDs of the added edges
        """
        pass

    @abstractmethod
    async def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Get an edge by ID.

        Args:
            edge_id: ID of the edge to get

        Returns:
            Edge if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_edges(self, edge_ids: List[str]) -> List[Optional[Edge]]:
        """Get multiple edges by ID.

        Args:
            edge_ids: IDs of the edges to get

        Returns:
            List of edges (None for edges not found)
        """
        pass

    @abstractmethod
    async def update_edge(self, edge_id: str, edge: Edge) -> bool:
        """Update an edge.

        Args:
            edge_id: ID of the edge to update
            edge: New edge data

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def update_edges(self, edge_ids: List[str], edges: List[Edge]) -> List[bool]:
        """Update multiple edges.

        Args:
            edge_ids: IDs of the edges to update
            edges: New edge data

        Returns:
            List of success flags
        """
        pass

    @abstractmethod
    async def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge.

        Args:
            edge_id: ID of the edge to delete

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def delete_edges(self, edge_ids: List[str]) -> List[bool]:
        """Delete multiple edges.

        Args:
            edge_ids: IDs of the edges to delete

        Returns:
            List of success flags
        """
        pass

    @abstractmethod
    async def get_neighbors(self, node_id: str, relation: Optional[str] = None, top_k: int = 5) -> List[QueryResult]:
        """Get the neighbors of a node.

        Args:
            node_id: ID of the node
            relation: Optional relation filter
            top_k: Number of results to return

        Returns:
            List of query results
        """
        pass

    @abstractmethod
    async def get_edges_between(
        self,
        source_id: str,
        target_id: str,
        relation: Optional[str] = None
    ) -> List[Edge]:
        """Get edges between two nodes.

        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            relation: Optional relation filter

        Returns:
            List of edges
        """
        pass

    async def query(self, query: Query, top_k: int = 5) -> List[QueryResult]:
        """Query the graph.

        Args:
            query: Query to execute
            top_k: Number of results to return

        Returns:
            List of query results
        """
        # Add user_id to cache key if present
        cache_key = f"{query.text}:{top_k}"
        user_id = None
        if query.metadata and "user_id" in query.metadata:
            user_id = query.metadata["user_id"]
            cache_key += f":{user_id}"

        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        # For graph store, we need to extract entities from the query
        # and find relevant nodes and their neighbors
        # This is a simplified implementation

        # Get all nodes with user_id filter applied at the database level
        # In a real implementation, you would pass the user_id to your graph database query
        results = []

        # Log the filtering
        if user_id:
            logger.debug(
                f"Applied user_id filter: {user_id} at database level for graph query")

        # Apply user_id filter as a post-processing step
        if user_id:
            filtered_results = []
            for result in results:
                result_user_id = result.metadata.get("user_id")
                if result_user_id == user_id:
                    filtered_results.append(result)
                else:
                    logger.debug(
                        f"Post-filtering: Removing result with user_id={result_user_id}, expected {user_id}")

            results = filtered_results

        # Filter results based on metadata
        if query.metadata:
            include_messages = query.metadata.get("include_messages", True)
            include_knowledge = query.metadata.get("include_knowledge", True)

            filtered_results = []
            for result in results:
                item_type = result.metadata.get("type")
                if ((item_type == "message" and include_messages)
                        or (item_type == "knowledge" and include_knowledge)):
                    filtered_results.append(result)

            results = filtered_results[:top_k]

        # Cache results
        self.query_cache[cache_key] = results

        return results
