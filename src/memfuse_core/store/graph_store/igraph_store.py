"""IGraph-based graph store implementation."""

import os
import json
from typing import List, Optional
import igraph as ig

from ...models.core import Node, Edge, Query, QueryResult
from ...models.config import DEFAULT_TOP_K
from ...models.core import StoreType
from ...utils.path_manager import PathManager
from .base import GraphStore


class IGraphStore(GraphStore):
    """IGraph-based graph store implementation."""

    def __init__(self, data_dir: str, **kwargs):
        """Initialize the IGraph graph store.

        Args:
            data_dir: Directory to store data
            **kwargs: Additional arguments
        """
        self.data_dir = data_dir

        # Create graph store directory
        self.graph_dir = os.path.join(data_dir, "igraph_store")
        PathManager.ensure_directory(self.graph_dir)

        # GraphML file path
        self.graphml_path = os.path.join(self.graph_dir, "graph.graphml")

        # Initialize graph
        self.graph = ig.Graph(directed=True)

        # Node and edge dictionaries for quick lookup
        self.nodes = {}
        self.edges = {}

        # Load existing graph if available
        self._load_graph()

    async def initialize(self) -> bool:
        """Initialize the graph store."""
        return True

    async def add_node(self, node: Node) -> str:
        """Add a node to the graph.

        Args:
            node: Node to add

        Returns:
            ID of the added node
        """
        # Check if node already exists
        if node.id in self.nodes:
            return node.id

        # Add node to graph
        self.graph.add_vertex(
            name=node.id,
            content=node.content,
            type=node.type,
            metadata=json.dumps(node.metadata)
        )

        # Add node to dictionary
        self.nodes[node.id] = node

        # Save graph
        self._save_graph()

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

        node_ids = []
        for node in nodes:
            node_id = await self.add_node(node)
            node_ids.append(node_id)

        return node_ids

    async def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID.

        Args:
            node_id: ID of the node

        Returns:
            Node if found, None otherwise
        """
        return self.nodes.get(node_id)

    async def get_nodes(self, node_ids: List[str]) -> List[Optional[Node]]:
        """Get multiple nodes by ID.

        Args:
            node_ids: IDs of the nodes

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

        # Get vertex index
        try:
            vertex_idx = self.graph.vs.find(name=node_id).index
        except ValueError:
            return False

        # Update vertex attributes
        self.graph.vs[vertex_idx]["content"] = node.content
        self.graph.vs[vertex_idx]["type"] = node.type
        self.graph.vs[vertex_idx]["metadata"] = json.dumps(node.metadata)

        # Update node in dictionary
        self.nodes[node_id] = node

        # Save graph
        self._save_graph()

        return True

    async def update_nodes(self, node_ids: List[str], nodes: List[Node]) -> List[bool]:
        """Update multiple nodes.

        Args:
            node_ids: IDs of the nodes to update
            nodes: New node data

        Returns:
            List of success flags
        """
        if len(node_ids) != len(nodes):
            raise ValueError("Number of node IDs and nodes must match")

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

        # Get vertex index
        try:
            vertex_idx = self.graph.vs.find(name=node_id).index
        except ValueError:
            return False

        # Delete vertex (this will also delete all connected edges)
        self.graph.delete_vertices(vertex_idx)

        # Remove node from dictionary
        del self.nodes[node_id]

        # Remove any edges connected to this node
        edge_ids_to_remove = []
        for edge_id, edge in self.edges.items():
            if edge.source_id == node_id or edge.target_id == node_id:
                edge_ids_to_remove.append(edge_id)

        for edge_id in edge_ids_to_remove:
            if edge_id in self.edges:
                del self.edges[edge_id]

        # Save graph
        self._save_graph()

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
        # Check if edge already exists
        if edge.id in self.edges:
            return edge.id

        # Check if source and target nodes exist
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            # Add missing nodes with placeholder content
            if edge.source_id not in self.nodes:
                await self.add_node(Node(
                    id=edge.source_id,
                    content="",
                    metadata={"type": "placeholder"}
                ))

            if edge.target_id not in self.nodes:
                await self.add_node(Node(
                    id=edge.target_id,
                    content="",
                    metadata={"type": "placeholder"}
                ))

        # Get vertex indices
        try:
            source_idx = self.graph.vs.find(name=edge.source_id).index
            target_idx = self.graph.vs.find(name=edge.target_id).index
        except ValueError:
            return None

        # Add edge to graph
        self.graph.add_edge(
            source_idx,
            target_idx,
            id=edge.id,
            type=edge.relation,  # Use relation instead of type for Edge
            weight=edge.weight,
            metadata=json.dumps(edge.metadata)
        )

        # Add edge to dictionary
        self.edges[edge.id] = edge

        # Save graph
        self._save_graph()

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
            edge_ids.append(edge_id)

        return edge_ids

    async def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Get an edge by ID.

        Args:
            edge_id: ID of the edge

        Returns:
            Edge if found, None otherwise
        """
        return self.edges.get(edge_id)

    async def get_edges(self, edge_ids: List[str]) -> List[Optional[Edge]]:
        """Get multiple edges by ID.

        Args:
            edge_ids: IDs of the edges

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

        # Find edge in graph
        try:
            edge_idx = None
            for i, e in enumerate(self.graph.es):
                if e["id"] == edge_id:
                    edge_idx = i
                    break

            if edge_idx is None:
                return False
        except (ValueError, KeyError):
            return False

        # Update edge attributes
        # Use relation instead of type for Edge
        self.graph.es[edge_idx]["type"] = edge.relation
        self.graph.es[edge_idx]["weight"] = edge.weight
        self.graph.es[edge_idx]["metadata"] = json.dumps(edge.metadata)

        # Update edge in dictionary
        self.edges[edge_id] = edge

        # Save graph
        self._save_graph()

        return True

    async def update_edges(self, edge_ids: List[str], edges: List[Edge]) -> List[bool]:
        """Update multiple edges.

        Args:
            edge_ids: IDs of the edges to update
            edges: New edge data

        Returns:
            List of success flags
        """
        if len(edge_ids) != len(edges):
            raise ValueError("Number of edge IDs and edges must match")

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

        # Find edge in graph
        try:
            edge_idx = None
            for i, e in enumerate(self.graph.es):
                if e["id"] == edge_id:
                    edge_idx = i
                    break

            if edge_idx is None:
                return False
        except (ValueError, KeyError):
            return False

        # Delete edge
        self.graph.delete_edges(edge_idx)

        # Remove edge from dictionary
        del self.edges[edge_id]

        # Save graph
        self._save_graph()

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
        self, node_id: str, relation: Optional[str] = None, top_k: int = DEFAULT_TOP_K
    ) -> List[QueryResult]:
        """Get the neighbors of a node.

        Args:
            node_id: ID of the node
            relation: Optional relation filter
            top_k: Number of results to return

        Returns:
            List of query results
        """
        # Check if node exists
        if node_id not in self.nodes:
            return []

        # Get vertex index
        try:
            vertex_idx = self.graph.vs.find(name=node_id).index
        except ValueError:
            return []

        # Get neighbor indices
        neighbor_indices = self.graph.neighbors(vertex_idx, mode="out")

        # Get neighbor vertices
        neighbors = [self.graph.vs[idx] for idx in neighbor_indices]

        # Filter by relation if specified
        if relation:
            # Get edges from node to neighbors
            edges = []
            for neighbor_idx in neighbor_indices:
                edge_idx = self.graph.get_eid(
                    vertex_idx, neighbor_idx, error=False)
                if edge_idx != -1:
                    edges.append(self.graph.es[edge_idx])

            # Filter neighbors by relation
            filtered_neighbors = []
            for neighbor, edge in zip(neighbors, edges):
                if edge["type"] == relation:
                    filtered_neighbors.append((neighbor, edge))

            # Sort by edge weight
            filtered_neighbors.sort(key=lambda x: x[1]["weight"], reverse=True)

            # Limit to top_k
            filtered_neighbors = filtered_neighbors[:top_k]

            # Convert to query results
            results = []
            for neighbor, edge in filtered_neighbors:
                metadata = json.loads(
                    neighbor["metadata"]) if "metadata" in neighbor.attributes() else {}
                results.append(
                    QueryResult(
                        id=neighbor["name"],
                        content=neighbor["content"],
                        score=float(edge["weight"]),
                        metadata=metadata,
                        store_type=StoreType.GRAPH
                    )
                )
        else:
            # Sort neighbors by content similarity (placeholder for actual similarity)
            # In a real implementation, you would compute similarity between node and neighbors

            # Convert to query results
            results = []
            for neighbor in neighbors[:top_k]:
                metadata = json.loads(
                    neighbor["metadata"]) if "metadata" in neighbor.attributes() else {}
                results.append(
                    QueryResult(
                        id=neighbor["name"],
                        content=neighbor["content"],
                        score=1.0,  # Placeholder score
                        metadata=metadata,
                        store_type=StoreType.GRAPH
                    )
                )

        return results

    async def get_edges_between(
        self, source_id: str, target_id: str, relation: Optional[str] = None
    ) -> List[Edge]:
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

        # Get vertex indices
        try:
            source_idx = self.graph.vs.find(name=source_id).index
            target_idx = self.graph.vs.find(name=target_id).index
        except ValueError:
            return []

        # Get edge index
        edge_idx = self.graph.get_eid(source_idx, target_idx, error=False)
        if edge_idx == -1:
            return []

        # Get edge
        edge = self.graph.es[edge_idx]

        # Filter by relation if specified
        if relation and edge["type"] != relation:
            return []

        # Get edge ID
        edge_id = edge["id"]

        # Return edge from dictionary
        if edge_id in self.edges:
            return [self.edges[edge_id]]
        else:
            return []

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
            item_id: ID of the item

        Returns:
            Item if found, None otherwise
        """
        return await self.get_node(item_id)

    async def get_batch(self, item_ids: List[str]) -> List[Optional[Node]]:
        """Get multiple items by ID.

        Args:
            item_ids: IDs of the items

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

    async def query(self, query: Query, top_k: int = DEFAULT_TOP_K) -> List[QueryResult]:
        """Query the store.

        Args:
            query: Query
            top_k: Number of results to return

        Returns:
            List of query results
        """
        # For graph store, we don't have a good way to query by text
        # So we return an empty list
        return []

    async def clear(self) -> bool:
        """Clear the store.

        Returns:
            True if successful, False otherwise
        """
        # Clear graph
        self.graph = ig.Graph(directed=True)

        # Clear dictionaries
        self.nodes = {}
        self.edges = {}

        # Save empty graph
        self._save_graph()

        return True

    def _save_graph(self):
        """Save graph to GraphML file."""
        try:
            self.graph.write_graphml(self.graphml_path)
        except Exception as e:
            print(f"Error saving graph: {e}")

    def _load_graph(self):
        """Load graph from GraphML file."""
        if os.path.exists(self.graphml_path):
            try:
                self.graph = ig.Graph.Read_GraphML(self.graphml_path)

                # Rebuild node and edge dictionaries
                for vertex in self.graph.vs:
                    if "name" in vertex.attributes() and "content" in vertex.attributes():
                        node_id = vertex["name"]
                        metadata = json.loads(
                            vertex["metadata"]) if "metadata" in vertex.attributes() else {}
                        # Use the type from the vertex or default to "node"
                        node_type = vertex["type"] if "type" in vertex.attributes(
                        ) else "node"

                        self.nodes[node_id] = Node(
                            id=node_id,
                            content=vertex["content"],
                            type=node_type,
                            metadata=metadata
                        )

                for edge in self.graph.es:
                    if "id" in edge.attributes():
                        edge_id = edge["id"]
                        source_id = self.graph.vs[edge.source]["name"]
                        target_id = self.graph.vs[edge.target]["name"]
                        # Get edge type (relation) from attributes or use default
                        relation = edge["type"] if "type" in edge.attributes(
                        ) else "RELATED_TO"
                        weight = edge["weight"] if "weight" in edge.attributes(
                        ) else 1.0
                        metadata = json.loads(
                            edge["metadata"]) if "metadata" in edge.attributes() else {}

                        self.edges[edge_id] = Edge(
                            id=edge_id,
                            source_id=source_id,
                            target_id=target_id,
                            relation=relation,  # Use relation instead of type
                            weight=weight,
                            metadata=metadata
                        )
            except Exception as e:
                print(f"Error loading graph: {e}")
                # Initialize empty graph
                self.graph = ig.Graph(directed=True)
