"""Message processing interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


# Type aliases for better readability
MessageList = List[Dict[str, Any]]  # List of Messages
MessageBatchList = List[MessageList]  # List of lists of Messages


class MessageInterface(ABC):
    """Interface for message processing.

    Provides consistent add() and add_batch() methods where:
    - add() takes MessageList and internally calls add_batch()
    - add_batch() takes MessageBatchList and is the core processing method
    """
    
    async def add(self, messages: MessageList, **kwargs) -> Dict[str, Any]:
        """Add a single list of messages.
        
        This method wraps the MessageList in a MessageBatchList and calls add_batch().
        
        Args:
            messages: List of message dictionaries (MessageList)
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary with status, data, and message information
        """
        return await self.add_batch([messages], **kwargs)
    
    @abstractmethod
    async def add_batch(self, message_batch_list: MessageBatchList, **kwargs) -> Dict[str, Any]:
        """Add a batch of message lists.
        
        This is the core processing method that handles MessageBatchList.
        
        Args:
            message_batch_list: List of lists of messages (MessageBatchList)
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary with status, data, and message information
        """
        pass
    
    def _success_response(self, message_ids: List[str], message: str, **extra_data) -> Dict[str, Any]:
        """Create a success response.
        
        Args:
            message_ids: List of message IDs
            message: Success message
            **extra_data: Additional data to include
            
        Returns:
            Success response dictionary
        """
        data = {"message_ids": message_ids}
        data.update(extra_data)
        
        return {
            "status": "success",
            "code": 200,
            "data": data,
            "message": message,
            "errors": None,
        }
    
    def _error_response(self, message: str, code: int = 500, errors: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Create an error response.
        
        Args:
            message: Error message
            code: HTTP status code
            errors: List of error details
            
        Returns:
            Error response dictionary
        """
        return {
            "status": "error",
            "code": code,
            "data": None,
            "message": message,
            "errors": errors or [{"field": "general", "message": message}],
        }
