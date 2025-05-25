"""Storage handler interfaces for MemFuse buffer system."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..services.memory_service import MemoryService


class StorageHandler(ABC):
    """Interface for storage handlers used by the buffer system.

    Storage handlers are responsible for persisting items from the buffer
    to the underlying storage system. They are used by the WriteBuffer
    to handle batch write operations.
    """

    @abstractmethod
    async def handle_batch(self, items: List[Any]) -> List[str]:
        """Handle a batch of items to be stored.

        Args:
            items: List of items to store

        Returns:
            List of item IDs from the storage operation
        """
        pass


class MemoryServiceStorageHandler(StorageHandler):
    """Storage handler that delegates to a MemoryService instance.

    This handler is used to store items in the memory service, which
    is the default storage backend for the buffer system.
    """

    def __init__(
        self,
        memory_service: "MemoryService",
        user: Optional[str] = None,
        agent: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Initialize the storage handler.

        Args:
            memory_service: MemoryService instance to delegate to
            user: User ID (optional)
            agent: Agent ID (optional)
            session_id: Session ID (optional)
        """
        self.memory_service = memory_service
        self.user = user
        self.agent = agent
        self.session_id = session_id

    async def handle_batch(self, items: List[Any]) -> List[str]:
        """Handle a batch of items to be stored in the memory service.

        Args:
            items: List of items to store

        Returns:
            List of item IDs from the storage operation
        """
        from loguru import logger

        if not self.memory_service:
            logger.error(
                "MemoryServiceStorageHandler: No memory service available")
            return []

        # If no items, return early
        if not items:
            logger.warning("MemoryServiceStorageHandler: No items to process")
            return []

        try:
            # Process items to ensure metadata is complete
            processed_items = self._process_items(items)

            # If no processed items, return early
            if not processed_items:
                logger.warning(
                    "MemoryServiceStorageHandler: No processed items to add")
                return []

            # Use memory_service.add_batch for batch writing
            logger.info(
                f"MemoryServiceStorageHandler: Calling memory_service.add_batch with "
                f"{len(processed_items)} items"
            )
            result = await self.memory_service.add_batch(processed_items)
            logger.info(
                f"MemoryServiceStorageHandler: MemoryService.add_batch result: {result}"
            )

            # Extract item IDs from result
            item_ids = []
            if result and result.get("status") == "success" and result.get("data"):
                item_ids = result["data"].get("message_ids", [])
                logger.info(
                    f"MemoryServiceStorageHandler: Successfully added {len(item_ids)} "
                    f"items via MemoryService"
                )
            else:
                logger.error(
                    f"MemoryServiceStorageHandler: Failed to add items via MemoryService: "
                    f"{result}"
                )

            return item_ids
        except Exception as e:
            from loguru import logger
            logger.error(
                f"MemoryServiceStorageHandler: Error using MemoryService: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _process_items(self, items: List[Any]) -> List[Any]:
        """Process items to ensure metadata is complete.

        Args:
            items: List of items to process

        Returns:
            List of processed items
        """
        processed_items = []
        from loguru import logger

        for item in items:
            if not isinstance(item, dict):
                logger.warning(
                    f"MemoryServiceStorageHandler: Skipping non-dict item: {type(item)}"
                )
                continue

            # Create a copy of the item to avoid modifying the original
            processed_item = item.copy()

            # Ensure metadata exists
            if 'metadata' not in processed_item:
                processed_item['metadata'] = {}

            # Add user, agent, and session_id if available
            if self.user and 'user_id' not in processed_item['metadata']:
                processed_item['metadata']['user_id'] = self.user
            if self.agent and 'agent_id' not in processed_item['metadata']:
                processed_item['metadata']['agent_id'] = self.agent
            if self.session_id and 'session_id' not in processed_item['metadata']:
                processed_item['metadata']['session_id'] = self.session_id

            processed_items.append(processed_item)

        return processed_items
