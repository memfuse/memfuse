"""Base retrieval interface for MemFuse server."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseRetrieval(ABC):
    """Base class for retrieval implementations."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant items based on the query.

        Args:
            query: Query string
            user_id: User ID (optional)
            session_id: Session ID (optional)
            top_k: Number of results to return
            **kwargs: Additional arguments

        Returns:
            List of retrieved items
        """
        pass
