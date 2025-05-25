"""In-memory graph store implementation."""

import os
import json
from typing import Dict, List, Optional

from ...utils.config import (
    get_top_k,
    get_graph_nodes_file,
    get_graph_edges_file,
)
from ...models.core import Node, Edge, Query, QueryResult
from ...models.core import StoreType
from .base import GraphStore
from ...utils.path_manager import PathManager


class InMemoryGraphStore(GraphStore):
    """In-memory graph store implementation.

    This implementation stores the graph in memory and persists it to disk.
    It provides a simple graph store with minimal dependencies.
    """

    def __init__(
        self,
        data_dir: str,
        model_name: str = "all-MiniLM-L6-v2",
        cache_size: int = 100,
        **kwargs
    ):
        """Initialize the in-memory graph store.

        Args:
            data_dir: Directory to store data
            model_name: Name of the embedding model
            cache_size: Size of the query cache
            **kwargs: Additional arguments
        """
        # Call parent constructor
        super().__init__(
            data_dir=data_dir,
            cache_size=cache_size,
            **kwargs
        )

        self.model_name = model_name

        # Create graph store directory
        self.graph_dir = os.path.join(data_dir, "graph_store")

        # Get file names from configuration
        graph_nodes_file = get_graph_nodes_file()
        graph_edges_file = get_graph_edges_file()

        # Initialize nodes and edges file paths
        self.nodes_file = os.path.join(self.graph_dir, graph_nodes_file)
        self.edges_file = os.path.join(self.graph_dir, graph_edges_file)

        # Initialize nodes and edges dictionaries
        self.nodes = {}
        self.edges = {}

        # Initialize adjacency list
        self.adjacency_list = {}

    async def initialize(self) -> bool:
        """Initialize the graph store.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create graph store directory
            PathManager.ensure_directory(self.graph_dir)

            # Create empty files if they don't exist
            if not os.path.exists(self.nodes_file):
                with open(self.nodes_file, "w") as f:
                    f.write("{}")

            if not os.path.exists(self.edges_file):
                with open(self.edges_file, "w") as f:
                    f.write("{}")

            # Load nodes and edges if they exist
            self.nodes = self._load_nodes()
            self.edges = self._load_edges()

            # Initialize adjacency list
            self.adjacency_list = self._build_adjacency_list()

            self.initialized = True
            return True
        except Exception as e:
            print(f"Error initializing graph store: {e}")
            # Initialize with empty data
            self.nodes = {}
            self.edges = {}
            self.adjacency_list = {}
            self.initialized = True
            return True

    def _load_nodes(self) -> Dict[str, Node]:
        """Load nodes from file.

        Returns:
            Dictionary of nodes
        """
        if os.path.exists(self.nodes_file):
            with open(self.nodes_file, "r") as f:
                nodes_data = json.load(f)
                return {node_id: Node(**node) for node_id, node in nodes_data.items()}
        return {}

    def _save_nodes(self):
        """Save nodes to file."""
        nodes_data = {node_id: node.__dict__ for node_id,
                      node in self.nodes.items()}
        with open(self.nodes_file, "w") as f:
            json.dump(nodes_data, f)

    def _load_edges(self) -> Dict[str, Edge]:
        """Load edges from file.

        Returns:
            Dictionary of edges
        """
        if os.path.exists(self.edges_file):
            with open(self.edges_file, "r") as f:
                edges_data = json.load(f)
                return {edge_id: Edge(**edge) for edge_id, edge in edges_data.items()}
        return {}

    def _save_edges(self):
        """Save edges to file."""
        edges_data = {edge_id: edge.__dict__ for edge_id,
                      edge in self.edges.items()}
        with open(self.edges_file, "w") as f:
            json.dump(edges_data, f)

    def _build_adjacency_list(self) -> Dict[str, Dict[str, List[str]]]:
        """Build adjacency list from edges.

        Returns:
            Adjacency list
        """
        adjacency_list = {}

        for edge_id, edge in self.edges.items():
            # Initialize source node if not exists
            if edge.source_id not in adjacency_list:
                adjacency_list[edge.source_id] = {}

            # Initialize relation if not exists
            if edge.relation not in adjacency_list[edge.source_id]:
                adjacency_list[edge.source_id][edge.relation] = []

            # Add target node to adjacency list
            adjacency_list[edge.source_id][edge.relation].append(edge.target_id)

        return adjacency_list

    async def add_node(self, node: Node) -> str:
        """Add a node to the graph.

        Args:
            node: Node to add

        Returns:
            ID of the added node
        """
        # Add node to dictionary
        self.nodes[node.id] = node

        # Save to disk
        self._save_nodes()

        # Invalidate query cache
        self.query_cache = {}

        return node.id

    async def add_nodes(self, nodes: List[Node]) -> List[str]:
        """Add multiple nodes to the graph.

        Args:
            nodes: Nodes to add

        Returns:
            List of IDs of the added nodes
        """
        if not nodes:
            return []

        # Add nodes to dictionary
        for node in nodes:
            self.nodes[node.id] = node

        # Save to disk
        self._save_nodes()

        # Invalidate query cache
        self.query_cache = {}

        return [node.id for node in nodes]

    async def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID.

        Args:
            node_id: ID of the node to get

        Returns:
            Node if found, None otherwise
        """
        return self.nodes.get(node_id)

    async def get_nodes(self, node_ids: List[str]) -> List[Optional[Node]]:
        """Get multiple nodes by ID.

        Args:
            node_ids: IDs of the nodes to get

        Returns:
            List of nodes (None for nodes not found)
        """
        return [self.nodes.get(node_id) for node_id in node_ids]

    async def update_node(self, node_id: str, node: Node) -> bool:
        """Update a node.

        Args:
            node_id: ID of the node to update
            node: New node data

        Returns:
            True if successful, False otherwise
        """
        # Check if node exists
        if node_id not in self.nodes:
            return False

        # Update node
        self.nodes[node_id] = node

        # Save to disk
        self._save_nodes()

        # Invalidate query cache
        self.query_cache = {}

        return True

    async def update_nodes(self, node_ids: List[str], nodes: List[Node]) -> List[bool]:
        """Update multiple nodes.

        Args:
            node_ids: IDs of the nodes to update
            nodes: New node data

        Returns:
            List of success flags
        """
        results = []
        for node_id, node in zip(node_ids, nodes):
            result = await self.update_node(node_id, node)
            results.append(result)
        return results

    async def delete_node(self, node_id: str) -> bool:
        """Delete a node.

        Args:
            node_id: ID of the node to delete

        Returns:
            True if successful, False otherwise
        """
        # Check if node exists
        if node_id not in self.nodes:
            return False

        # Remove node from dictionary
        del self.nodes[node_id]

        # Remove edges connected to the node
        edges_to_remove = []
        for edge_id, edge in self.edges.items():
            if edge.source_id == node_id or edge.target_id == node_id:
                edges_to_remove.append(edge_id)

        for edge_id in edges_to_remove:
            del self.edges[edge_id]

        # Rebuild adjacency list
        self.adjacency_list = self._build_adjacency_list()

        # Save to disk
        self._save_nodes()
        self._save_edges()

        # Invalidate query cache
        self.query_cache = {}

        return True

    async def delete_nodes(self, node_ids: List[str]) -> List[bool]:
        """Delete multiple nodes.

        Args:
            node_ids: IDs of the nodes to delete

        Returns:
            List of success flags
        """
        results = []
        for node_id in node_ids:
            result = await self.delete_node(node_id)
            results.append(result)
        return results

    async def add_edge(self, edge: Edge) -> str:
        """Add an edge to the graph.

        Args:
            edge: Edge to add

        Returns:
            ID of the added edge
        """
        # Generate edge ID if not provided
        if not edge.id:
            edge.id = str(uuid.uuid4())

        # Check if source and target nodes exist
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            return None

        # Add edge to dictionary
        self.edges[edge.id] = edge

        # Update adjacency list
        if edge.source_id not in self.adjacency_list:
            self.adjacency_list[edge.source_id] = {}

        if edge.relation not in self.adjacency_list[edge.source_id]:
            self.adjacency_list[edge.source_id][edge.relation] = []

        self.adjacency_list[edge.source_id][edge.relation].append(
            edge.target_id)

        # Save to disk
        self._save_edges()

        # Invalidate query cache
        self.query_cache = {}

        return edge.id

    async def add_edges(self, edges: List[Edge]) -> List[str]:
        """Add multiple edges to the graph.

        Args:
            edges: Edges to add

        Returns:
            List of IDs of the added edges
        """
        if not edges:
            return []

        edge_ids = []
        for edge in edges:
            edge_id = await self.add_edge(edge)
            if edge_id:
                edge_ids.append(edge_id)

        return edge_ids

    async def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Get an edge by ID.

        Args:
            edge_id: ID of the edge to get

        Returns:
            Edge if found, None otherwise
        """
        return self.edges.get(edge_id)

    async def get_edges(self, edge_ids: List[str]) -> List[Optional[Edge]]:
        """Get multiple edges by ID.

        Args:
            edge_ids: IDs of the edges to get

        Returns:
            List of edges (None for edges not found)
        """
        return [self.edges.get(edge_id) for edge_id in edge_ids]

    async def update_edge(self, edge_id: str, edge: Edge) -> bool:
        """Update an edge.

        Args:
            edge_id: ID of the edge to update
            edge: New edge data

        Returns:
            True if successful, False otherwise
        """
        # Check if edge exists
        if edge_id not in self.edges:
            return False

        # Get old edge
        old_edge = self.edges[edge_id]

        # Update edge
        self.edges[edge_id] = edge

        # Update adjacency list if source, target, or relation changed
        if (old_edge.source_id != edge.source_id
            or old_edge.target_id != edge.target_id
                or old_edge.relation != edge.relation):

            # Remove old edge from adjacency list
            if (old_edge.source_id in self.adjacency_list
                and old_edge.relation in self.adjacency_list[old_edge.source_id]
                    and old_edge.target_id in self.adjacency_list[old_edge.source_id][old_edge.relation]):

                self.adjacency_list[old_edge.source_id][old_edge.relation].remove(
                    old_edge.target_id)

            # Add new edge to adjacency list
            if edge.source_id not in self.adjacency_list:
                self.adjacency_list[edge.source_id] = {}

            if edge.relation not in self.adjacency_list[edge.source_id]:
                self.adjacency_list[edge.source_id][edge.relation] = []

            self.adjacency_list[edge.source_id][edge.relation].append(
                edge.target_id)

        # Save to disk
        self._save_edges()

        # Invalidate query cache
        self.query_cache = {}

        return True

    async def update_edges(self, edge_ids: List[str], edges: List[Edge]) -> List[bool]:
        """Update multiple edges.

        Args:
            edge_ids: IDs of the edges to update
            edges: New edge data

        Returns:
            List of success flags
        """
        results = []
        for edge_id, edge in zip(edge_ids, edges):
            result = await self.update_edge(edge_id, edge)
            results.append(result)
        return results

    async def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge.

        Args:
            edge_id: ID of the edge to delete

        Returns:
            True if successful, False otherwise
        """
        # Check if edge exists
        if edge_id not in self.edges:
            return False

        # Get edge
        edge = self.edges[edge_id]

        # Remove edge from dictionary
        del self.edges[edge_id]

        # Remove edge from adjacency list
        if (edge.source_id in self.adjacency_list
            and edge.relation in self.adjacency_list[edge.source_id]
                and edge.target_id in self.adjacency_list[edge.source_id][edge.relation]):

            self.adjacency_list[edge.source_id][edge.relation].remove(
                edge.target_id)

        # Save to disk
        self._save_edges()

        # Invalidate query cache
        self.query_cache = {}

        return True

    async def delete_edges(self, edge_ids: List[str]) -> List[bool]:
        """Delete multiple edges.

        Args:
            edge_ids: IDs of the edges to delete

        Returns:
            List of success flags
        """
        results = []
        for edge_id in edge_ids:
            result = await self.delete_edge(edge_id)
            results.append(result)
        return results

    async def get_neighbors(
        self,
        node_id: str,
        relation: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[QueryResult]:
        """Get the neighbors of a node.

        Args:
            node_id: ID of the node
            relation: Optional relation filter
            top_k: Number of results to return (optional)

        Returns:
            List of query results
        """
        # Get top_k from configuration if not provided
        if top_k is None:
            top_k = get_top_k()
        # Check if node exists
        if node_id not in self.nodes:
            return []

        # Check if node has neighbors
        if node_id not in self.adjacency_list:
            return []

        # Get neighbors
        neighbors = set()

        if relation:
            # Get neighbors with specific relation
            if relation in self.adjacency_list[node_id]:
                neighbors.update(self.adjacency_list[node_id][relation])
        else:
            # Get all neighbors
            for rel, targets in self.adjacency_list[node_id].items():
                neighbors.update(targets)

        # Create results
        results = []
        for neighbor_id in neighbors:
            if neighbor_id in self.nodes:
                node = self.nodes[neighbor_id]

                # Find edge with highest weight
                max_weight = 0.0
                for edge in self.edges.values():
                    if edge.source_id == node_id and edge.target_id == neighbor_id:
                        if relation is None or edge.relation == relation:
                            max_weight = max(max_weight, edge.weight)

                results.append(
                    QueryResult(
                        id=node.id,
                        content=node.content,
                        score=max_weight,
                        metadata=node.metadata,
                        store_type=StoreType.GRAPH,
                    )
                )

        # Sort by score and limit to top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def get_edges_between(self, source_id: str, target_id: str, relation: Optional[str] = None) -> List[Edge]:
        """Get edges between two nodes.

        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            relation: Optional relation filter

        Returns:
            List of edges
        """
        # Check if nodes exist
        if source_id not in self.nodes or target_id not in self.nodes:
            return []

        # Find edges
        edges = []
        for edge in self.edges.values():
            if edge.source_id == source_id and edge.target_id == target_id:
                if relation is None or edge.relation == relation:
                    edges.append(edge)

        return edges

    async def query(self, query: Query, top_k: Optional[int] = None) -> List[QueryResult]:
        """Query the graph.

        Args:
            query: Query to execute
            top_k: Number of results to return (optional)

        Returns:
            List of query results
        """
        # Get top_k from configuration if not provided
        if top_k is None:
            top_k = get_top_k()

        # Check cache
        cache_key = (query.text, top_k)
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        # For now, we'll just return the top_k nodes with the highest similarity to the query
        # In a real implementation, you would use a more sophisticated algorithm

        # Generate embedding for query
        query_embedding = await self._generate_embedding(query.text)

        # Calculate similarity for each node
        results = []
        for node_id, node in self.nodes.items():
            # Generate embedding for node
            node_embedding = await self._generate_embedding(node.content)

            # Calculate similarity
            similarity = self._calculate_similarity(
                query_embedding, node_embedding)

            # Filter results based on metadata
            if query.metadata:
                include_messages = query.metadata.get("include_messages", True)
                include_knowledge = query.metadata.get(
                    "include_knowledge", True)

                item_type = node.metadata.get("type")
                if (item_type == "message" and not include_messages) or (item_type == "knowledge" and not include_knowledge):
                    continue

            results.append(
                QueryResult(
                    id=node.id,
                    content=node.content,
                    score=similarity,
                    metadata=node.metadata,
                    store_type=StoreType.GRAPH,
                )
            )

        # Sort by score and limit to top_k
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:top_k]

        # Cache results
        if len(self.query_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]

        self.query_cache[cache_key] = results

        return results

    async def clear(self) -> bool:
        """Clear the graph.

        Returns:
            True if successful, False otherwise
        """
        # Clear dictionaries
        self.nodes = {}
        self.edges = {}

        # Clear adjacency list
        self.adjacency_list = {}

        # Clear cache
        self.query_cache = {}

        # Remove files
        if os.path.exists(self.nodes_file):
            os.remove(self.nodes_file)
        if os.path.exists(self.edges_file):
            os.remove(self.edges_file)

        return True

    async def add(self, item: Node) -> str:
        """Add an item to the store.

        Args:
            item: Item to add

        Returns:
            ID of the added item
        """
        return await self.add_node(item)

    async def add_batch(self, items: List[Node]) -> List[str]:
        """Add multiple items to the store.

        Args:
            items: Items to add

        Returns:
            List of IDs of the added items
        """
        return await self.add_nodes(items)

    async def get(self, item_id: str) -> Optional[Node]:
        """Get an item by ID.

        Args:
            item_id: ID of the item to get

        Returns:
            Item if found, None otherwise
        """
        return await self.get_node(item_id)

    async def get_batch(self, item_ids: List[str]) -> List[Optional[Node]]:
        """Get multiple items by ID.

        Args:
            item_ids: IDs of the items to get

        Returns:
            List of items (None for items not found)
        """
        return await self.get_nodes(item_ids)

    async def update(self, item_id: str, item: Node) -> bool:
        """Update an item.

        Args:
            item_id: ID of the item to update
            item: New item data

        Returns:
            True if successful, False otherwise
        """
        return await self.update_node(item_id, item)

    async def update_batch(self, item_ids: List[str], items: List[Node]) -> List[bool]:
        """Update multiple items.

        Args:
            item_ids: IDs of the items to update
            items: New item data

        Returns:
            List of success flags
        """
        return await self.update_nodes(item_ids, items)

    async def delete(self, item_id: str) -> bool:
        """Delete an item.

        Args:
            item_id: ID of the item to delete

        Returns:
            True if successful, False otherwise
        """
        return await self.delete_node(item_id)

    async def delete_batch(self, item_ids: List[str]) -> List[bool]:
        """Delete multiple items.

        Args:
            item_ids: IDs of the items to delete

        Returns:
            List of success flags
        """
        return await self.delete_nodes(item_ids)

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for a text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        from memfuse_core.common.utils.embeddings import create_embedding

        # Use the embeddings utility to create an embedding
        embedding = await create_embedding(text, self.model_name)
        return embedding

    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score
        """
        from memfuse_core.common.utils.embeddings import calculate_similarity

        # Use the embeddings utility to calculate similarity
        return calculate_similarity(embedding1, embedding2)
