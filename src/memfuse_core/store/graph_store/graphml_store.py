"""GraphML graph store implementation for MemFuse server.

This module provides a graph store implementation using GraphML format for storage.
It is designed to be easily migrated to Neo4j in the future.
"""
import os
import json
import asyncio
import time
import numpy as np
from loguru import logger
from typing import List, Optional
import networkx as nx

from ...models.core import Node, Edge, Item, Query, QueryResult
from ...utils.path_manager import PathManager
from .base import GraphStore
from ..adapters.model_adapter import ModelAdapterFactory


class GraphMLStore(GraphStore):
    """Graph store implementation using GraphML format."""

    def __init__(
        self,
        data_dir: str,
        model_name: str = "all-MiniLM-L6-v2",
        cache_size: int = 100,
        buffer_size: int = 10,
        encoder=None,
        **kwargs
    ):
        """Initialize the GraphML store.

        Args:
            data_dir: Directory to store data
            model_name: Name of the embedding model
            cache_size: Size of the query cache
            buffer_size: Size of the write buffer
            encoder: Optional encoder instance to reuse
            **kwargs: Additional arguments
        """
        super().__init__(data_dir, cache_size=cache_size, **kwargs)
        self.graph_dir = os.path.join(data_dir, "graph_store")
        self.model_name = model_name
        self.buffer_size = buffer_size

        # File paths
        self.graph_file = os.path.join(self.graph_dir, "graph.graphml")
        self.nodes_file = os.path.join(self.graph_dir, "nodes.json")
        self.edges_file = os.path.join(self.graph_dir, "edges.json")

        # Initialize graph
        self.graph = nx.DiGraph()

        # Store the model name for reference
        self.model_name = model_name

        # Initialize model adapters for decoupled access
        self.embedding_adapter = ModelAdapterFactory.create_embedding_adapter()
        self.rerank_adapter = ModelAdapterFactory.create_rerank_adapter()

        # Initialize write buffers
        self.node_buffer = []
        self.edge_buffer = []
        self.write_lock = asyncio.Lock()

        # Initialize flag
        self.initialized = False

        # Statistics
        self.stats = {
            "nodes_added": 0,
            "edges_added": 0,
            "nodes_updated": 0,
            "edges_updated": 0,
            "nodes_deleted": 0,
            "edges_deleted": 0,
            "queries": 0,
            "query_time": 0,
        }

    async def initialize(self) -> bool:
        """Initialize the graph store.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create graph directory if it doesn't exist
            PathManager.ensure_directory(self.graph_dir)

            # Load graph if it exists
            if os.path.exists(self.graph_file):
                self.graph = nx.read_graphml(self.graph_file)
            else:
                # Create empty graph
                self.graph = nx.DiGraph()

                # Save empty graph
                nx.write_graphml(self.graph, self.graph_file)

            # Model service is available through ModelRegistry
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing GraphML store: {e}")
            return False

    async def add_node(self, node: Node) -> str:
        """Add a node to the graph.

        Args:
            node: Node to add

        Returns:
            ID of the added node
        """
        async with self.write_lock:
            # Add to buffer
            self.node_buffer.append(("add", node))

            # Flush buffer if it's full
            if len(self.node_buffer) >= self.buffer_size:
                await self._flush_node_buffer()

            self.stats["nodes_added"] += 1
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

        async with self.write_lock:
            # Add all nodes to buffer
            for node in nodes:
                self.node_buffer.append(("add", node))

            # Flush buffer if it's full
            if len(self.node_buffer) >= self.buffer_size:
                await self._flush_node_buffer()

            self.stats["nodes_added"] += len(nodes)
            return [node.id for node in nodes]

    async def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID.

        Args:
            node_id: ID of the node to get

        Returns:
            Node if found, None otherwise
        """
        # Flush buffer to ensure we have the latest data
        await self._flush_node_buffer()

        if node_id in self.graph.nodes:
            node_data = self.graph.nodes[node_id]
            return Node(
                id=node_id,
                content=node_data.get("content", ""),
                metadata=json.loads(node_data.get("metadata", "{}"))
            )

        return None

    async def get_nodes(self, node_ids: List[str]) -> List[Optional[Node]]:
        """Get multiple nodes by ID.

        Args:
            node_ids: IDs of the nodes to get

        Returns:
            List of nodes (None for nodes not found)
        """
        if not node_ids:
            return []

        # Flush buffer to ensure we have the latest data
        await self._flush_node_buffer()

        result = []
        for node_id in node_ids:
            if node_id in self.graph.nodes:
                node_data = self.graph.nodes[node_id]
                result.append(Node(
                    id=node_id,
                    content=node_data.get("content", ""),
                    metadata=json.loads(node_data.get("metadata", "{}"))
                ))
            else:
                result.append(None)

        return result

    async def update_node(self, node_id: str, node: Node) -> bool:
        """Update a node.

        Args:
            node_id: ID of the node to update
            node: New node data

        Returns:
            True if successful, False otherwise
        """
        async with self.write_lock:
            # Add to buffer
            self.node_buffer.append(("update", node_id, node))

            # Flush buffer if it's full
            if len(self.node_buffer) >= self.buffer_size:
                await self._flush_node_buffer()

            self.stats["nodes_updated"] += 1
            return True

    async def update_nodes(self, node_ids: List[str], nodes: List[Node]) -> List[bool]:
        """Update multiple nodes.

        Args:
            node_ids: IDs of the nodes to update
            nodes: New node data

        Returns:
            List of success flags
        """
        if not node_ids or not nodes or len(node_ids) != len(nodes):
            return [False] * len(node_ids)

        async with self.write_lock:
            # Add all nodes to buffer
            for node_id, node in zip(node_ids, nodes):
                self.node_buffer.append(("update", node_id, node))

            # Flush buffer if it's full
            if len(self.node_buffer) >= self.buffer_size:
                await self._flush_node_buffer()

            self.stats["nodes_updated"] += len(node_ids)
            return [True] * len(node_ids)

    async def delete_node(self, node_id: str) -> bool:
        """Delete a node.

        Args:
            node_id: ID of the node to delete

        Returns:
            True if successful, False otherwise
        """
        async with self.write_lock:
            # Add to buffer
            self.node_buffer.append(("delete", node_id))

            # Flush buffer if it's full
            if len(self.node_buffer) >= self.buffer_size:
                await self._flush_node_buffer()

            self.stats["nodes_deleted"] += 1
            return True

    async def delete_nodes(self, node_ids: List[str]) -> List[bool]:
        """Delete multiple nodes.

        Args:
            node_ids: IDs of the nodes to delete

        Returns:
            List of success flags
        """
        if not node_ids:
            return []

        async with self.write_lock:
            # Add all nodes to buffer
            for node_id in node_ids:
                self.node_buffer.append(("delete", node_id))

            # Flush buffer if it's full
            if len(self.node_buffer) >= self.buffer_size:
                await self._flush_node_buffer()

            self.stats["nodes_deleted"] += len(node_ids)
            return [True] * len(node_ids)

    async def add_edge(self, edge: Edge) -> str:
        """Add an edge to the graph.

        Args:
            edge: Edge to add

        Returns:
            ID of the added edge
        """
        async with self.write_lock:
            # Add to buffer
            self.edge_buffer.append(("add", edge))

            # Flush buffer if it's full
            if len(self.edge_buffer) >= self.buffer_size:
                await self._flush_edge_buffer()

            self.stats["edges_added"] += 1
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

        async with self.write_lock:
            # Add all edges to buffer
            for edge in edges:
                self.edge_buffer.append(("add", edge))

            # Flush buffer if it's full
            if len(self.edge_buffer) >= self.buffer_size:
                await self._flush_edge_buffer()

            self.stats["edges_added"] += len(edges)
            return [edge.id for edge in edges]

    async def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Get an edge by ID.

        Args:
            edge_id: ID of the edge to get

        Returns:
            Edge if found, None otherwise
        """
        # Flush buffer to ensure we have the latest data
        await self._flush_edge_buffer()

        # Find the edge in the graph
        for u, v, edge_data in self.graph.edges(data=True):
            if edge_data.get("id") == edge_id:
                return Edge(
                    id=edge_id,
                    source=u,
                    target=v,
                    relation=edge_data.get("relation", ""),
                    weight=float(edge_data.get("weight", 1.0)),
                    metadata=json.loads(edge_data.get("metadata", "{}"))
                )

        return None

    async def get_edges(self, edge_ids: List[str]) -> List[Optional[Edge]]:
        """Get multiple edges by ID.

        Args:
            edge_ids: IDs of the edges to get

        Returns:
            List of edges (None for edges not found)
        """
        if not edge_ids:
            return []

        # Flush buffer to ensure we have the latest data
        await self._flush_edge_buffer()

        # Create a map of edge_id to edge
        edge_map = {}
        for u, v, edge_data in self.graph.edges(data=True):
            edge_id = edge_data.get("id")
            if edge_id in edge_ids:
                edge_map[edge_id] = Edge(
                    id=edge_id,
                    source=u,
                    target=v,
                    relation=edge_data.get("relation", ""),
                    weight=float(edge_data.get("weight", 1.0)),
                    metadata=json.loads(edge_data.get("metadata", "{}"))
                )

        # Return edges in the same order as edge_ids
        return [edge_map.get(edge_id) for edge_id in edge_ids]

    async def update_edge(self, edge_id: str, edge: Edge) -> bool:
        """Update an edge.

        Args:
            edge_id: ID of the edge to update
            edge: New edge data

        Returns:
            True if successful, False otherwise
        """
        async with self.write_lock:
            # Add to buffer
            self.edge_buffer.append(("update", edge_id, edge))

            # Flush buffer if it's full
            if len(self.edge_buffer) >= self.buffer_size:
                await self._flush_edge_buffer()

            self.stats["edges_updated"] += 1
            return True

    async def update_edges(self, edge_ids: List[str], edges: List[Edge]) -> List[bool]:
        """Update multiple edges.

        Args:
            edge_ids: IDs of the edges to update
            edges: New edge data

        Returns:
            List of success flags
        """
        if not edge_ids or not edges or len(edge_ids) != len(edges):
            return [False] * len(edge_ids)

        async with self.write_lock:
            # Add all edges to buffer
            for edge_id, edge in zip(edge_ids, edges):
                self.edge_buffer.append(("update", edge_id, edge))

            # Flush buffer if it's full
            if len(self.edge_buffer) >= self.buffer_size:
                await self._flush_edge_buffer()

            self.stats["edges_updated"] += len(edge_ids)
            return [True] * len(edge_ids)

    async def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge.

        Args:
            edge_id: ID of the edge to delete

        Returns:
            True if successful, False otherwise
        """
        async with self.write_lock:
            # Add to buffer
            self.edge_buffer.append(("delete", edge_id))

            # Flush buffer if it's full
            if len(self.edge_buffer) >= self.buffer_size:
                await self._flush_edge_buffer()

            self.stats["edges_deleted"] += 1
            return True

    async def delete_edges(self, edge_ids: List[str]) -> List[bool]:
        """Delete multiple edges.

        Args:
            edge_ids: IDs of the edges to delete

        Returns:
            List of success flags
        """
        if not edge_ids:
            return []

        async with self.write_lock:
            # Add all edges to buffer
            for edge_id in edge_ids:
                self.edge_buffer.append(("delete", edge_id))

            # Flush buffer if it's full
            if len(self.edge_buffer) >= self.buffer_size:
                await self._flush_edge_buffer()

            self.stats["edges_deleted"] += len(edge_ids)
            return [True] * len(edge_ids)

    async def get_neighbors(
        self,
        node_id: str,
        relation: Optional[str] = None,
        top_k: int = 5
    ) -> List[QueryResult]:
        """Get the neighbors of a node.

        Args:
            node_id: ID of the node
            relation: Optional relation filter
            top_k: Number of results to return

        Returns:
            List of query results
        """
        # Flush buffers to ensure we have the latest data
        await self._flush_node_buffer()
        await self._flush_edge_buffer()

        if node_id not in self.graph.nodes:
            return []

        neighbors = []

        # Get outgoing neighbors
        for _, neighbor_id, edge_data in self.graph.out_edges(node_id, data=True):
            if relation and edge_data.get("relation") != relation:
                continue

            if neighbor_id in self.graph.nodes:
                neighbor_data = self.graph.nodes[neighbor_id]
                weight = float(edge_data.get("weight", 1.0))

                neighbors.append((
                    neighbor_id,
                    neighbor_data.get("content", ""),
                    json.loads(neighbor_data.get("metadata", "{}")),
                    weight
                ))

        # Get incoming neighbors
        for neighbor_id, _, edge_data in self.graph.in_edges(node_id, data=True):
            if relation and edge_data.get("relation") != relation:
                continue

            if neighbor_id in self.graph.nodes:
                neighbor_data = self.graph.nodes[neighbor_id]
                weight = float(edge_data.get("weight", 1.0))

                neighbors.append((
                    neighbor_id,
                    neighbor_data.get("content", ""),
                    json.loads(neighbor_data.get("metadata", "{}")),
                    weight
                ))

        # Sort by weight (descending) and limit to top_k
        neighbors.sort(key=lambda x: x[3], reverse=True)
        neighbors = neighbors[:top_k]

        # Convert to QueryResult objects
        results = []
        for neighbor_id, content, metadata, weight in neighbors:
            # Add retrieval method to metadata
            if "retrieval" not in metadata:
                metadata["retrieval"] = {}
            metadata["retrieval"]["method"] = "graph"

            results.append(QueryResult(
                id=neighbor_id,
                content=content,
                score=weight,
                metadata=metadata
            ))

        return results

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
        # Flush buffer to ensure we have the latest data
        await self._flush_edge_buffer()

        edges = []

        # Check if there are edges between the nodes
        if self.graph.has_edge(source_id, target_id):
            edge_data_list = self.graph.get_edge_data(source_id, target_id)

            # In MultiDiGraph, edge_data_list is a dict of edge keys to edge data
            # In DiGraph, edge_data_list is just the edge data
            if isinstance(edge_data_list, dict) and not isinstance(next(iter(edge_data_list.values()), None), dict):
                edge_data_list = {0: edge_data_list}

            for key, edge_data in edge_data_list.items():
                if relation and edge_data.get("relation") != relation:
                    continue

                edges.append(Edge(
                    id=edge_data.get("id", f"{source_id}_{target_id}_{key}"),
                    source=source_id,
                    target=target_id,
                    relation=edge_data.get("relation", ""),
                    weight=float(edge_data.get("weight", 1.0)),
                    metadata=json.loads(edge_data.get("metadata", "{}"))
                ))

        return edges

    async def query(self, query: Query, top_k: int = 5) -> List[QueryResult]:
        """Query the graph.

        Args:
            query: Query to execute
            top_k: Number of results to return

        Returns:
            List of query results
        """
        # Check cache
        cache_key = f"{query.text}:{top_k}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        start_time = time.time()

        # Generate embedding for the query
        query_embedding = await self.model_service.get_embedding(query.text)

        # Flush buffers to ensure we have the latest data
        await self._flush_node_buffer()
        await self._flush_edge_buffer()

        # Get all nodes
        node_scores = []
        for node_id, node_data in self.graph.nodes(data=True):
            # Skip nodes without content
            if "content" not in node_data:
                continue

            # Get node embedding
            node_content = node_data.get("content", "")
            node_embedding = await self.model_service.get_embedding(node_content)

            # Calculate similarity
            similarity = self._calculate_similarity(query_embedding, node_embedding)

            # Get node metadata
            metadata = json.loads(node_data.get("metadata", "{}"))

            # Add to results
            node_scores.append((
                node_id,
                node_content,
                metadata,
                similarity
            ))

        # Sort by similarity (descending)
        node_scores.sort(key=lambda x: x[3], reverse=True)

        # Filter results based on metadata
        if query.metadata:
            include_messages = query.metadata.get("include_messages", True)
            include_knowledge = query.metadata.get("include_knowledge", True)

            filtered_scores = []
            for node_id, content, metadata, score in node_scores:
                item_type = metadata.get("type")
                if ((item_type == "message" and include_messages)
                        or (item_type == "knowledge" and include_knowledge)):
                    filtered_scores.append((node_id, content, metadata, score))

            node_scores = filtered_scores

        # Limit to top_k
        node_scores = node_scores[:top_k]

        # Convert to QueryResult objects
        results = []
        for node_id, content, metadata, score in node_scores:
            # Add retrieval method to metadata
            if "retrieval" not in metadata:
                metadata["retrieval"] = {}
            metadata["retrieval"]["method"] = "graph"

            results.append(QueryResult(
                id=node_id,
                content=content,
                score=score,
                metadata=metadata
            ))

        # Update statistics
        self.stats["queries"] += 1
        self.stats["query_time"] += time.time() - start_time

        # Cache results
        self.query_cache[cache_key] = results

        return results

    async def _flush_node_buffer(self):
        """Flush the node buffer to the graph."""
        if not self.node_buffer:
            return

        try:
            # Process all operations in the buffer
            for operation in self.node_buffer:
                if operation[0] == "add":
                    node = operation[1]

                    # Add node to graph
                    self.graph.add_node(
                        node.id,
                        content=node.content,
                        metadata=json.dumps(node.metadata)
                    )

                elif operation[0] == "update":
                    node_id, node = operation[1], operation[2]

                    # Update node in graph
                    if node_id in self.graph.nodes:
                        self.graph.nodes[node_id]["content"] = node.content
                        self.graph.nodes[node_id]["metadata"] = json.dumps(
                            node.metadata)

                elif operation[0] == "delete":
                    node_id = operation[1]

                    # Delete node from graph
                    if node_id in self.graph.nodes:
                        self.graph.remove_node(node_id)

            # Save graph to file
            nx.write_graphml(self.graph, self.graph_file)

            # Clear buffer
            self.node_buffer.clear()

        except Exception as e:
            logger.error(f"Error flushing node buffer: {e}")
            raise

    def _calculate_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score
        """
        # Normalize the embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate cosine similarity
        return np.dot(embedding1, embedding2) / (norm1 * norm2)

    async def _flush_edge_buffer(self):
        """Flush the edge buffer to the graph."""
        if not self.edge_buffer:
            return

        try:
            # Process all operations in the buffer
            for operation in self.edge_buffer:
                if operation[0] == "add":
                    edge = operation[1]

                    # Add edge to graph
                    self.graph.add_edge(
                        edge.source,
                        edge.target,
                        id=edge.id,
                        relation=edge.relation,
                        weight=edge.weight,
                        metadata=json.dumps(edge.metadata)
                    )

                elif operation[0] == "update":
                    edge_id, edge = operation[1], operation[2]

                    # Find and update edge in graph
                    for u, v, edge_data in list(self.graph.edges(data=True)):
                        if edge_data.get("id") == edge_id:
                            # Remove old edge
                            self.graph.remove_edge(u, v)

                            # Add new edge
                            self.graph.add_edge(
                                edge.source,
                                edge.target,
                                id=edge.id,
                                relation=edge.relation,
                                weight=edge.weight,
                                metadata=json.dumps(edge.metadata)
                            )

                            break

                elif operation[0] == "delete":
                    edge_id = operation[1]

                    # Find and delete edge from graph
                    for u, v, edge_data in list(self.graph.edges(data=True)):
                        if edge_data.get("id") == edge_id:
                            self.graph.remove_edge(u, v)
                            break

            # Save graph to file
            nx.write_graphml(self.graph, self.graph_file)

            # Clear buffer
            self.edge_buffer.clear()

        except Exception as e:
            logger.error(f"Error flushing edge buffer: {e}")
            raise

    async def add(self, item: Item) -> str:
        """Add an item to the store.

        Args:
            item: Item to add

        Returns:
            ID of the added item
        """
        # Convert Item to Node
        node = Node(
            id=item.id,
            content=item.content,
            metadata=item.metadata
        )

        # Add node to graph
        return await self.add_node(node)

    async def add_batch(self, items: List[Item]) -> List[str]:
        """Add multiple items to the store.

        Args:
            items: Items to add

        Returns:
            List of IDs of the added items
        """
        # Convert Items to Nodes
        nodes = [
            Node(
                id=item.id,
                content=item.content,
                metadata=item.metadata
            )
            for item in items
        ]

        # Add nodes to graph
        return await self.add_nodes(nodes)

    async def get(self, item_id: str) -> Optional[Item]:
        """Get an item by ID.

        Args:
            item_id: ID of the item to get

        Returns:
            Item if found, None otherwise
        """
        # Get node from graph
        node = await self.get_node(item_id)

        # Convert Node to Item
        if node:
            return Item(
                id=node.id,
                content=node.content,
                metadata=node.metadata
            )

        return None

    async def get_batch(self, item_ids: List[str]) -> List[Optional[Item]]:
        """Get multiple items by ID.

        Args:
            item_ids: IDs of the items to get

        Returns:
            List of items (None for items not found)
        """
        # Get nodes from graph
        nodes = await self.get_nodes(item_ids)

        # Convert Nodes to Items
        return [
            Item(
                id=node.id,
                content=node.content,
                metadata=node.metadata
            ) if node else None
            for node in nodes
        ]

    async def update(self, item_id: str, item: Item) -> bool:
        """Update an item.

        Args:
            item_id: ID of the item to update
            item: New item data

        Returns:
            True if successful, False otherwise
        """
        # Convert Item to Node
        node = Node(
            id=item.id,
            content=item.content,
            metadata=item.metadata
        )

        # Update node in graph
        return await self.update_node(item_id, node)

    async def update_batch(self, item_ids: List[str], items: List[Item]) -> List[bool]:
        """Update multiple items.

        Args:
            item_ids: IDs of the items to update
            items: New item data

        Returns:
            List of success flags
        """
        # Convert Items to Nodes
        nodes = [
            Node(
                id=item.id,
                content=item.content,
                metadata=item.metadata
            )
            for item in items
        ]

        # Update nodes in graph
        return await self.update_nodes(item_ids, nodes)

    async def delete(self, item_id: str) -> bool:
        """Delete an item.

        Args:
            item_id: ID of the item to delete

        Returns:
            True if successful, False otherwise
        """
        # Delete node from graph
        return await self.delete_node(item_id)

    async def delete_batch(self, item_ids: List[str]) -> List[bool]:
        """Delete multiple items.

        Args:
            item_ids: IDs of the items to delete

        Returns:
            List of success flags
        """
        # Delete nodes from graph
        return await self.delete_nodes(item_ids)

    async def clear(self) -> bool:
        """Clear the store.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create empty graph
            self.graph = nx.DiGraph()

            # Save empty graph
            nx.write_graphml(self.graph, self.graph_file)

            # Clear buffers
            self.node_buffer.clear()
            self.edge_buffer.clear()

            # Clear cache
            self.query_cache.clear()

            return True
        except Exception as e:
            logger.error(f"Error clearing graph store: {e}")
            return False

    async def close(self):
        """Close the graph store."""
        # Flush any remaining items in the buffers
        await self._flush_node_buffer()
        await self._flush_edge_buffer()

        # Save graph to file
        nx.write_graphml(self.graph, self.graph_file)

        # Print statistics
        logger.info(f"GraphMLStore statistics: {self.stats}")
