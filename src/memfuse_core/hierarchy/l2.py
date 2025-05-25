"""L2 (Knowledge Graph) layer for the hierarchical memory system."""

from typing import Any, List, Dict, Optional, Tuple
import time
import uuid
from .base import MemoryLayer, Entity, Relationship, Fact
from ..buffer.base import BufferBase


class GraphDatabase:
    """Database for storing knowledge graph."""

    def __init__(self, data_dir: str):
        """Initialize the graph database.

        Args:
            data_dir: Data directory
        """
        self.data_dir = data_dir
        self.entities = {}  # In-memory storage for demo purposes
        self.relationships = {}  # In-memory storage for demo purposes

    def add_entity(self, content: str, entity_type: str, source_fact_ids: List[str], metadata: Dict[str, Any]) -> str:
        """Add an entity to the database.

        Args:
            content: Entity content
            entity_type: Type of entity
            source_fact_ids: IDs of source facts
            metadata: Entity metadata

        Returns:
            ID of the added entity
        """
        entity_id = str(uuid.uuid4())
        self.entities[entity_id] = {
            "id": entity_id,
            "content": content,
            "entity_type": entity_type,
            "source_fact_ids": source_fact_ids,
            "metadata": metadata
        }
        return entity_id

    def add_relationship(
        self,
        content: str,
        source_id: str,
        target_id: str,
        relationship_type: str,
        source_fact_ids: List[str],
        metadata: Dict[str, Any]
    ) -> str:
        """Add a relationship to the database.

        Args:
            content: Relationship content
            source_id: ID of the source entity
            target_id: ID of the target entity
            relationship_type: Type of relationship
            source_fact_ids: IDs of source facts
            metadata: Relationship metadata

        Returns:
            ID of the added relationship
        """
        relationship_id = str(uuid.uuid4())
        self.relationships[relationship_id] = {
            "id": relationship_id,
            "content": content,
            "source_id": source_id,
            "target_id": target_id,
            "relationship_type": relationship_type,
            "source_fact_ids": source_fact_ids,
            "metadata": metadata
        }
        return relationship_id

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get an entity from the database.

        Args:
            entity_id: ID of the entity to get

        Returns:
            Entity if found, None otherwise
        """
        return self.entities.get(entity_id)

    def get_relationship(self, relationship_id: str) -> Optional[Dict[str, Any]]:
        """Get a relationship from the database.

        Args:
            relationship_id: ID of the relationship to get

        Returns:
            Relationship if found, None otherwise
        """
        return self.relationships.get(relationship_id)

    def search_graph(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for entities and relationships in the database.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of matching entities and relationships
        """
        # Simple search implementation for demo purposes
        results = []

        # Search entities
        for entity in self.entities.values():
            # Simple word overlap score
            query_words = set(query.lower().split())
            content_words = set(entity["content"].lower().split())

            if not query_words:
                continue

            overlap = len(query_words.intersection(content_words))
            score = overlap / len(query_words)

            if score > 0:
                entity_copy = entity.copy()
                entity_copy["score"] = score
                entity_copy["element_type"] = "entity"
                results.append(entity_copy)

        # Search relationships
        for relationship in self.relationships.values():
            # Simple word overlap score
            query_words = set(query.lower().split())
            content_words = set(relationship["content"].lower().split())

            if not query_words:
                continue

            overlap = len(query_words.intersection(content_words))
            score = overlap / len(query_words)

            if score > 0:
                relationship_copy = relationship.copy()
                relationship_copy["score"] = score
                relationship_copy["element_type"] = "relationship"
                results.append(relationship_copy)

        # Sort by score and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_related_entities(self, entity_id: str) -> List[Tuple[str, str, str]]:
        """Get entities related to the given entity.

        Args:
            entity_id: ID of the entity

        Returns:
            List of (relationship_id, related_entity_id, relationship_type) tuples
        """
        related = []

        # Find relationships where entity is source
        for rel_id, rel in self.relationships.items():
            if rel["source_id"] == entity_id:
                related.append(
                    (rel_id, rel["target_id"], rel["relationship_type"]))
            elif rel["target_id"] == entity_id:
                related.append(
                    (rel_id, rel["source_id"], rel["relationship_type"]))

        return related

    def update_entity(
        self,
        entity_id: str,
        content: str,
        entity_type: str,
        source_fact_ids: List[str],
        metadata: Dict[str, Any]
    ) -> bool:
        """Update an entity in the database.

        Args:
            entity_id: ID of the entity to update
            content: Updated entity content
            entity_type: Updated entity type
            source_fact_ids: Updated IDs of source facts
            metadata: Updated entity metadata

        Returns:
            True if the entity was updated, False otherwise
        """
        if entity_id not in self.entities:
            return False

        self.entities[entity_id] = {
            "id": entity_id,
            "content": content,
            "entity_type": entity_type,
            "source_fact_ids": source_fact_ids,
            "metadata": metadata
        }
        return True

    def update_relationship(
        self,
        relationship_id: str,
        content: str,
        source_id: str,
        target_id: str,
        relationship_type: str,
        source_fact_ids: List[str],
        metadata: Dict[str, Any]
    ) -> bool:
        """Update a relationship in the database.

        Args:
            relationship_id: ID of the relationship to update
            content: Updated relationship content
            source_id: Updated ID of the source entity
            target_id: Updated ID of the target entity
            relationship_type: Updated relationship type
            source_fact_ids: Updated IDs of source facts
            metadata: Updated relationship metadata

        Returns:
            True if the relationship was updated, False otherwise
        """
        if relationship_id not in self.relationships:
            return False

        self.relationships[relationship_id] = {
            "id": relationship_id,
            "content": content,
            "source_id": source_id,
            "target_id": target_id,
            "relationship_type": relationship_type,
            "source_fact_ids": source_fact_ids,
            "metadata": metadata
        }
        return True

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity from the database.

        Args:
            entity_id: ID of the entity to delete

        Returns:
            True if the entity was deleted, False otherwise
        """
        if entity_id not in self.entities:
            return False

        # Delete the entity
        del self.entities[entity_id]

        # Delete relationships involving the entity
        rel_ids_to_delete = []
        for rel_id, rel in self.relationships.items():
            if rel["source_id"] == entity_id or rel["target_id"] == entity_id:
                rel_ids_to_delete.append(rel_id)

        for rel_id in rel_ids_to_delete:
            del self.relationships[rel_id]

        return True

    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship from the database.

        Args:
            relationship_id: ID of the relationship to delete

        Returns:
            True if the relationship was deleted, False otherwise
        """
        if relationship_id not in self.relationships:
            return False

        del self.relationships[relationship_id]
        return True

    def clear(self) -> bool:
        """Clear all entities and relationships from the database.

        Returns:
            True if the database was cleared, False otherwise
        """
        self.entities.clear()
        self.relationships.clear()
        return True

    def count_entities(self) -> int:
        """Count the number of entities in the database.

        Returns:
            Number of entities
        """
        return len(self.entities)

    def count_relationships(self) -> int:
        """Count the number of relationships in the database.

        Returns:
            Number of relationships
        """
        return len(self.relationships)


class L2Manager(MemoryLayer):
    """Manages the L2 (Knowledge Graph) layer.

    The L2 layer constructs a knowledge graph from facts:
    - Entity Extraction: Identifies entities from facts
    - Relationship Extraction: Identifies relationships between entities
    - Graph Construction: Builds a knowledge graph
    - Graph Traversal: Navigates the knowledge graph
    """

    def __init__(
        self,
        graph_db: GraphDatabase,
        llm_service: Any,
        buffer_manager: Optional[Any] = None,
        construction_batch_size: int = 10,
    ):
        """Initialize the L2 manager.

        Args:
            graph_db: Graph database
            llm_service: LLM service for entity and relationship extraction
            buffer_manager: Buffer manager (optional)
            construction_batch_size: Batch size for graph construction
        """
        self.graph_db = graph_db
        self.llm_service = llm_service
        self.buffer_manager = buffer_manager
        self.construction_batch_size = construction_batch_size

        # Set up a dedicated buffer for graph construction
        self.construction_buffer = BufferBase[Fact](
            max_size=construction_batch_size,
            flush_callback=self._construct_graph_batch
        )

        # Statistics
        self.total_facts_processed = 0
        self.total_entities_extracted = 0
        self.total_relationships_extracted = 0
        self.total_construction_time = 0
        self.total_searches = 0
        self.total_search_time = 0

    async def process_fact(self, fact: Fact) -> None:
        """Process a fact for graph construction.

        Args:
            fact: Fact to process
        """
        # Update statistics
        self.total_facts_processed += 1

        # Add fact to construction buffer
        await self.construction_buffer.add(fact)

    async def _construct_graph_batch(self, facts: List[Fact]) -> List[Any]:
        """Construct graph from a batch of facts.

        Args:
            facts: Batch of facts to construct graph from

        Returns:
            List of extracted entity and relationship IDs
        """
        start_time = time.time()
        result_ids = []

        # This is a placeholder for the actual LLM-based graph construction
        # In a real implementation, this would call an LLM API to extract
        # entities and relationships from facts

        # For demo purposes, use a simple approach
        for fact in facts:
            # Extract entities (simple approach: assume each word is a potential entity)
            words = fact.content.split()
            entities = set()
            for word in words:
                if len(word) > 5 and word[0].isupper():  # Simple heuristic
                    entities.add(word)

            # Add entities to graph
            entity_ids = {}
            for entity_content in entities:
                entity_id = self.graph_db.add_entity(
                    content=entity_content,
                    entity_type="generic",
                    source_fact_ids=[fact.id],
                    metadata={
                        "user_id": fact.metadata.get("user_id"),
                        "extraction_time": time.time(),
                    }
                )
                entity_ids[entity_content] = entity_id
                result_ids.append(entity_id)
                self.total_entities_extracted += 1

            # Add relationships between entities (simple approach: connect all entities)
            entity_list = list(entity_ids.items())
            for i in range(len(entity_list)):
                for j in range(i + 1, len(entity_list)):
                    source_content, source_id = entity_list[i]
                    target_content, target_id = entity_list[j]

                    relationship_id = self.graph_db.add_relationship(
                        content=f"{source_content} is related to {target_content}",
                        source_id=source_id,
                        target_id=target_id,
                        relationship_type="related_to",
                        source_fact_ids=[fact.id],
                        metadata={
                            "user_id": fact.metadata.get("user_id"),
                            "extraction_time": time.time(),
                        }
                    )
                    result_ids.append(relationship_id)
                    self.total_relationships_extracted += 1

        # Update statistics
        construction_time = time.time() - start_time
        self.total_construction_time += construction_time

        return result_ids

    async def add(self, item: Any) -> str:
        """Add an item to the L2 layer.

        This method processes the item for graph construction.

        Args:
            item: Item to add (should be a Fact)

        Returns:
            ID of the added item
        """
        if isinstance(item, Fact):
            await self.process_fact(item)
            return item.id
        else:
            # Convert to Fact if possible
            if hasattr(item, 'id') and hasattr(item, 'content') and hasattr(item, 'metadata'):
                fact = Fact(
                    id=item.id,
                    content=item.content,
                    metadata=item.metadata,
                    source_ids=item.source_ids if hasattr(
                        item, 'source_ids') else []
                )
                await self.process_fact(fact)
                return item.id
            else:
                raise ValueError(
                    "Item must be a Fact or have id, content, and metadata attributes")

    async def get(self, item_id: str) -> Optional[Any]:
        """Get an item from the L2 layer.

        Args:
            item_id: ID of the item to get

        Returns:
            Entity or Relationship if found, None otherwise
        """
        # Try to get as entity
        entity_data = self.graph_db.get_entity(item_id)
        if entity_data:
            return Entity(
                id=entity_data["id"],
                content=entity_data["content"],
                entity_type=entity_data["entity_type"],
                source_ids=entity_data["source_fact_ids"],
                metadata=entity_data["metadata"]
            )

        # Try to get as relationship
        relationship_data = self.graph_db.get_relationship(item_id)
        if relationship_data:
            return Relationship(
                id=relationship_data["id"],
                content=relationship_data["content"],
                source_id=relationship_data["source_id"],
                target_id=relationship_data["target_id"],
                relationship_type=relationship_data["relationship_type"],
                source_ids=relationship_data["source_fact_ids"],
                metadata=relationship_data["metadata"]
            )

        return None

    async def search(self, query: str, top_k: int = 5) -> List[Any]:
        """Search for items in the L2 layer.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of matching entities and relationships
        """
        start_time = time.time()
        self.total_searches += 1

        try:
            # Search graph database
            graph_data_list = self.graph_db.search_graph(query, top_k=top_k)

            # Convert to Entity and Relationship objects
            results = []
            for data in graph_data_list:
                if data.get("element_type") == "entity":
                    entity = Entity(
                        id=data["id"],
                        content=data["content"],
                        entity_type=data["entity_type"],
                        source_ids=data["source_fact_ids"],
                        metadata=data["metadata"]
                    )
                    entity.score = data.get("score", 0.0)
                    results.append(entity)
                elif data.get("element_type") == "relationship":
                    relationship = Relationship(
                        id=data["id"],
                        content=data["content"],
                        source_id=data["source_id"],
                        target_id=data["target_id"],
                        relationship_type=data["relationship_type"],
                        source_ids=data["source_fact_ids"],
                        metadata=data["metadata"]
                    )
                    relationship.score = data.get("score", 0.0)
                    results.append(relationship)

            return results
        finally:
            # Update statistics
            search_time = time.time() - start_time
            self.total_search_time += search_time

    async def update(self, item_id: str, item: Any) -> bool:
        """Update an item in the L2 layer.

        Args:
            item_id: ID of the item to update
            item: Updated item

        Returns:
            True if the item was updated, False otherwise
        """
        if isinstance(item, Entity):
            return self.graph_db.update_entity(
                entity_id=item_id,
                content=item.content,
                entity_type=item.entity_type,
                source_fact_ids=item.source_ids,
                metadata=item.metadata
            )
        elif isinstance(item, Relationship):
            return self.graph_db.update_relationship(
                relationship_id=item_id,
                content=item.content,
                source_id=item.source_id,
                target_id=item.target_id,
                relationship_type=item.relationship_type,
                source_fact_ids=item.source_ids,
                metadata=item.metadata
            )
        else:
            return False

    async def delete(self, item_id: str) -> bool:
        """Delete an item from the L2 layer.

        Args:
            item_id: ID of the item to delete

        Returns:
            True if the item was deleted, False otherwise
        """
        # Try to delete as entity
        if self.graph_db.delete_entity(item_id):
            return True

        # Try to delete as relationship
        if self.graph_db.delete_relationship(item_id):
            return True

        return False

    async def clear(self) -> bool:
        """Clear all items from the L2 layer.

        Returns:
            True if the L2 layer was cleared, False otherwise
        """
        return self.graph_db.clear()

    async def count(self) -> int:
        """Count the number of items in the L2 layer.

        Returns:
            Number of items
        """
        return self.graph_db.count_entities() + self.graph_db.count_relationships()

    def get_stats(self) -> Dict[str, Any]:
        """Get L2 layer statistics.

        Returns:
            Dictionary of L2 layer statistics
        """
        return {
            "total_facts_processed": self.total_facts_processed,
            "total_entities_extracted": self.total_entities_extracted,
            "total_relationships_extracted": self.total_relationships_extracted,
            "total_construction_time": self.total_construction_time,
            "avg_construction_time": self.total_construction_time / max(1, self.total_facts_processed),
            "total_searches": self.total_searches,
            "total_search_time": self.total_search_time,
            "avg_search_time": self.total_search_time / max(1, self.total_searches),
            "entity_count": self.graph_db.count_entities(),
            "relationship_count": self.graph_db.count_relationships(),
        }
