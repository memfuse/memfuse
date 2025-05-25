"""L1 (Facts) layer for the hierarchical memory system."""

from typing import Any, List, Dict, Optional
import asyncio
import time
import uuid
from .base import MemoryLayer, Fact
from ..buffer.base import BufferBase


class FactsDatabase:
    """Database for storing facts."""

    def __init__(self, data_dir: str):
        """Initialize the facts database.

        Args:
            data_dir: Data directory
        """
        self.data_dir = data_dir
        self.facts = {}  # In-memory storage for demo purposes

    def add_fact(self, content: str, source_chunk_ids: List[str], metadata: Dict[str, Any]) -> str:
        """Add a fact to the database.

        Args:
            content: Fact content
            source_chunk_ids: IDs of source chunks
            metadata: Fact metadata

        Returns:
            ID of the added fact
        """
        fact_id = str(uuid.uuid4())
        self.facts[fact_id] = {
            "id": fact_id,
            "content": content,
            "source_chunk_ids": source_chunk_ids,
            "metadata": metadata
        }
        return fact_id

    def get_fact(self, fact_id: str) -> Optional[Dict[str, Any]]:
        """Get a fact from the database.

        Args:
            fact_id: ID of the fact to get

        Returns:
            Fact if found, None otherwise
        """
        return self.facts.get(fact_id)

    def search_facts(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for facts in the database.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of matching facts
        """
        # Simple search implementation for demo purposes
        results = []
        for fact in self.facts.values():
            # Simple word overlap score
            query_words = set(query.lower().split())
            content_words = set(fact["content"].lower().split())

            if not query_words:
                continue

            overlap = len(query_words.intersection(content_words))
            score = overlap / len(query_words)

            if score > 0:
                fact_copy = fact.copy()
                fact_copy["score"] = score
                results.append(fact_copy)

        # Sort by score and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def update_fact(self, fact_id: str, content: str, source_chunk_ids: List[str], metadata: Dict[str, Any]) -> bool:
        """Update a fact in the database.

        Args:
            fact_id: ID of the fact to update
            content: Updated fact content
            source_chunk_ids: Updated IDs of source chunks
            metadata: Updated fact metadata

        Returns:
            True if the fact was updated, False otherwise
        """
        if fact_id not in self.facts:
            return False

        self.facts[fact_id] = {
            "id": fact_id,
            "content": content,
            "source_chunk_ids": source_chunk_ids,
            "metadata": metadata
        }
        return True

    def delete_fact(self, fact_id: str) -> bool:
        """Delete a fact from the database.

        Args:
            fact_id: ID of the fact to delete

        Returns:
            True if the fact was deleted, False otherwise
        """
        if fact_id not in self.facts:
            return False

        del self.facts[fact_id]
        return True

    def clear(self) -> bool:
        """Clear all facts from the database.

        Returns:
            True if the database was cleared, False otherwise
        """
        self.facts.clear()
        return True

    def count(self) -> int:
        """Count the number of facts in the database.

        Returns:
            Number of facts
        """
        return len(self.facts)


class LLMService:
    """Service for LLM-based operations."""

    async def extract_facts(self, context: str) -> List[str]:
        """Extract facts from context.

        Args:
            context: Context to extract facts from

        Returns:
            List of extracted facts
        """
        # This is a placeholder for the actual LLM-based fact extraction
        # In a real implementation, this would call an LLM API

        # For demo purposes, just split the context into sentences
        sentences = context.split(". ")
        facts = [s.strip() + "." for s in sentences if s.strip()]
        return facts


class L1Manager(MemoryLayer):
    """Manages the L1 (Facts) layer.

    The L1 layer extracts and stores facts from raw data:
    - Fact Extraction: Extracts facts from raw data
    - Fact Storage: Stores facts with links to source data
    - Fact Retrieval: Retrieves facts based on relevance

    It also coordinates with the L2 layer for knowledge graph construction.
    """

    def __init__(
        self,
        facts_db: FactsDatabase,
        llm_service: LLMService,
        buffer_manager: Optional[Any] = None,
        l2_manager: Optional[Any] = None,
        extraction_batch_size: int = 5,
    ):
        """Initialize the L1 manager.

        Args:
            facts_db: Facts database
            llm_service: LLM service for fact extraction
            buffer_manager: Buffer manager (optional)
            l2_manager: L2 manager (optional)
            extraction_batch_size: Batch size for fact extraction
        """
        self.facts_db = facts_db
        self.llm_service = llm_service
        self.buffer_manager = buffer_manager
        self.l2_manager = l2_manager
        self.extraction_batch_size = extraction_batch_size

        # Set up a dedicated buffer for facts extraction
        self.extraction_buffer = BufferBase[List[Any]](
            max_size=extraction_batch_size,
            flush_callback=self._extract_facts_batch
        )

        # Statistics
        self.total_items_processed = 0
        self.total_facts_extracted = 0
        self.total_extraction_time = 0
        self.total_searches = 0
        self.total_search_time = 0

    async def process_item(self, item: Any) -> None:
        """Process an item for fact extraction.

        Args:
            item: Item to process
        """
        # Update statistics
        self.total_items_processed += 1

        # Add item to extraction buffer
        await self.extraction_buffer.add([item])

    async def _extract_facts_batch(self, batch_of_items: List[List[Any]]) -> List[Any]:
        """Extract facts from a batch of items.

        Args:
            batch_of_items: Batch of items to extract facts from

        Returns:
            List of extracted fact IDs
        """
        start_time = time.time()
        fact_ids = []

        for items in batch_of_items:
            # Flatten items into a single context
            context = "\n\n".join([item.content for item in items])

            # Extract facts using LLM
            facts = await self.llm_service.extract_facts(context)

            # Store facts in database with links to source items
            for fact_content in facts:
                fact_id = self.facts_db.add_fact(
                    content=fact_content,
                    source_chunk_ids=[item.id for item in items],
                    metadata={
                        "user_id": items[0].metadata.get("user_id") if items else None,
                        "extraction_time": time.time(),
                    }
                )
                fact_ids.append(fact_id)
                self.total_facts_extracted += 1

                # Create fact object
                fact = Fact(
                    id=fact_id,
                    content=fact_content,
                    source_ids=[item.id for item in items],
                    metadata={
                        "user_id": items[0].metadata.get("user_id") if items else None,
                        "extraction_time": time.time(),
                    }
                )

                # Trigger L2 graph construction if L2 manager is available
                if self.l2_manager:
                    asyncio.create_task(self.l2_manager.process_fact(fact))

        # Update statistics
        extraction_time = time.time() - start_time
        self.total_extraction_time += extraction_time

        return fact_ids

    async def add(self, item: Any) -> str:
        """Add an item to the L1 layer.

        This method processes the item for fact extraction.

        Args:
            item: Item to add

        Returns:
            ID of the added item
        """
        await self.process_item(item)
        return item.id

    async def get(self, fact_id: str) -> Optional[Fact]:
        """Get a fact from the L1 layer.

        Args:
            fact_id: ID of the fact to get

        Returns:
            Fact if found, None otherwise
        """
        fact_data = self.facts_db.get_fact(fact_id)
        if not fact_data:
            return None

        return Fact(
            id=fact_data["id"],
            content=fact_data["content"],
            source_ids=fact_data["source_chunk_ids"],
            metadata=fact_data["metadata"]
        )

    async def search(self, query: str, top_k: int = 5) -> List[Fact]:
        """Search for facts in the L1 layer.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of matching facts
        """
        start_time = time.time()
        self.total_searches += 1

        try:
            # Search facts database
            fact_data_list = self.facts_db.search_facts(query, top_k=top_k)

            # Convert to Fact objects
            facts = []
            for fact_data in fact_data_list:
                fact = Fact(
                    id=fact_data["id"],
                    content=fact_data["content"],
                    source_ids=fact_data["source_chunk_ids"],
                    metadata=fact_data["metadata"]
                )
                fact.score = fact_data.get("score", 0.0)
                facts.append(fact)

            return facts
        finally:
            # Update statistics
            search_time = time.time() - start_time
            self.total_search_time += search_time

    async def update(self, fact_id: str, fact: Fact) -> bool:
        """Update a fact in the L1 layer.

        Args:
            fact_id: ID of the fact to update
            fact: Updated fact

        Returns:
            True if the fact was updated, False otherwise
        """
        return self.facts_db.update_fact(
            fact_id=fact_id,
            content=fact.content,
            source_chunk_ids=fact.source_ids,
            metadata=fact.metadata
        )

    async def delete(self, fact_id: str) -> bool:
        """Delete a fact from the L1 layer.

        Args:
            fact_id: ID of the fact to delete

        Returns:
            True if the fact was deleted, False otherwise
        """
        return self.facts_db.delete_fact(fact_id)

    async def clear(self) -> bool:
        """Clear all facts from the L1 layer.

        Returns:
            True if the L1 layer was cleared, False otherwise
        """
        return self.facts_db.clear()

    async def count(self) -> int:
        """Count the number of facts in the L1 layer.

        Returns:
            Number of facts
        """
        return self.facts_db.count()

    def get_stats(self) -> Dict[str, Any]:
        """Get L1 layer statistics.

        Returns:
            Dictionary of L1 layer statistics
        """
        return {
            "total_items_processed": self.total_items_processed,
            "total_facts_extracted": self.total_facts_extracted,
            "total_extraction_time": self.total_extraction_time,
            "avg_extraction_time": self.total_extraction_time / max(1, self.total_items_processed),
            "total_searches": self.total_searches,
            "total_search_time": self.total_search_time,
            "avg_search_time": self.total_search_time / max(1, self.total_searches),
            "fact_count": self.facts_db.count(),
        }
