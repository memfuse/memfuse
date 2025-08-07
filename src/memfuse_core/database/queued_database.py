"""
LEGACY - Queued Database Operations Wrapper.

This module contains legacy database queue management code that is no longer used.
The current implementation uses direct database connections with GlobalConnectionManager.

This file is kept for reference but should not be used in new code.
See docs/connection_pool_optimization.md for details.
"""

from typing import Dict, Any, Optional, List
from loguru import logger

from .base import Database
# NOTE: database_queue_manager module was removed as it's no longer needed
# from ..services.database_queue_manager import (
#     get_db_queue_manager, 
#     execute_db_operation, 
#     OperationType
# )


class QueuedDatabase:
    """
    LEGACY DATABASE WRAPPER - NOT USED
    
    This class is part of the old multi-layer connection pool approach that was
    replaced with direct connection management through GlobalConnectionManager.
    
    Current approach: API → DatabaseService → GlobalConnectionManager → PostgresDB
    Old approach: API → QueuedDatabase → DatabaseQueueManager → GlobalConnectionManager → PostgresDB
    
    This file should not be used in new code.
    See docs/connection_pool_optimization.md for the current approach.
    """
    
    def __init__(self, database: Database):
        self.database = database
        self.queue_manager = get_db_queue_manager()
    
    async def ensure_queue_started(self):
        """Ensure the queue manager is started."""
        if not self.queue_manager._running:
            await self.queue_manager.start()
    
    # User operations
    async def create_user(self, name: str, user_id: Optional[str] = None, description: Optional[str] = None) -> str:
        """Create a new user through the queue."""
        await self.ensure_queue_started()
        # Filter out None description to match base database interface
        kwargs = {"name": name}
        if user_id is not None:
            kwargs["user_id"] = user_id
        if description is not None and hasattr(self.database, '_supports_user_description'):
            kwargs["description"] = description
        
        return await execute_db_operation(
            self.database.create_user,
            OperationType.WRITE,
            priority=3,
            **kwargs
        )
    
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user through the queue."""
        await self.ensure_queue_started()
        return await execute_db_operation(
            self.database.get_user,
            OperationType.READ,
            priority=1,
            user_id=user_id
        )
    
    async def get_user_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a user by name through the queue."""
        await self.ensure_queue_started()
        return await execute_db_operation(
            self.database.get_user_by_name,
            OperationType.READ,
            priority=1,
            name=name
        )
    
    async def list_users(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """List users through the queue."""
        await self.ensure_queue_started()
        return await execute_db_operation(
            self.database.list_users,
            OperationType.READ,
            priority=2,
            limit=limit
        )
    
    # Agent operations
    async def create_agent(self, name: str, description: Optional[str] = None, agent_id: Optional[str] = None) -> str:
        """Create a new agent through the queue."""
        await self.ensure_queue_started()
        return await execute_db_operation(
            self.database.create_agent,
            OperationType.WRITE,
            priority=3,
            name=name,
            description=description,
            agent_id=agent_id
        )
    
    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get an agent through the queue."""
        await self.ensure_queue_started()
        return await execute_db_operation(
            self.database.get_agent,
            OperationType.READ,
            priority=1,
            agent_id=agent_id
        )
    
    # Session operations
    async def create_session(self, user_id: str, agent_id: str, name: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """Create a new session through the queue."""
        await self.ensure_queue_started()
        return await execute_db_operation(
            self.database.create_session,
            OperationType.WRITE,
            priority=3,
            user_id=user_id,
            agent_id=agent_id,
            name=name,
            session_id=session_id
        )
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session through the queue."""
        await self.ensure_queue_started()
        return await execute_db_operation(
            self.database.get_session,
            OperationType.READ,
            priority=1,
            session_id=session_id
        )
    
    # Round operations
    async def create_round(self, session_id: str, round_id: Optional[str] = None) -> str:
        """Create a new round through the queue."""
        await self.ensure_queue_started()
        return await execute_db_operation(
            self.database.create_round,
            OperationType.WRITE,
            priority=4,
            session_id=session_id,
            round_id=round_id
        )
    
    async def get_round(self, round_id: str) -> Optional[Dict[str, Any]]:
        """Get a round through the queue."""
        await self.ensure_queue_started()
        return await execute_db_operation(
            self.database.get_round,
            OperationType.READ,
            priority=1,
            round_id=round_id
        )
    
    # Message operations (high priority for user experience)
    async def add_message(self, round_id: str, role: str, content: str, 
                         message_id: Optional[str] = None,
                         created_at: Optional[str] = None, 
                         updated_at: Optional[str] = None) -> str:
        """Add a message through the queue."""
        await self.ensure_queue_started()
        return await execute_db_operation(
            self.database.add_message,
            OperationType.WRITE,
            priority=2,  # High priority for user messages
            timeout=30.0,
            round_id=round_id,
            role=role,
            content=content,
            message_id=message_id,
            created_at=created_at,
            updated_at=updated_at
        )
    
    async def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get a message through the queue."""
        await self.ensure_queue_started()
        return await execute_db_operation(
            self.database.get_message,
            OperationType.READ,
            priority=1,
            message_id=message_id
        )
    
    async def get_messages_by_session(self, session_id: str, limit: Optional[int] = None,
                                    sort_by: str = 'timestamp', order: str = 'desc',
                                    buffer_only: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Get messages by session through the queue."""
        await self.ensure_queue_started()
        return await execute_db_operation(
            self.database.get_messages_by_session,
            OperationType.READ,
            priority=1,
            session_id=session_id,
            limit=limit,
            sort_by=sort_by,
            order=order,
            buffer_only=buffer_only
        )
    
    # Batch operations (lower priority, longer timeout)
    async def add_messages_batch(self, messages_data: List[Dict[str, Any]]) -> List[str]:
        """Add multiple messages through the queue."""
        await self.ensure_queue_started()
        
        # For batch operations, we can either:
        # 1. Execute as a single batch operation (better for performance)
        # 2. Execute individually (better for flow control)
        
        # Option 1: Single batch operation
        if len(messages_data) > 10:  # Use batch for large operations
            return await execute_db_operation(
                self._add_messages_batch_internal,
                OperationType.BATCH_WRITE,
                priority=6,  # Lower priority for batch operations
                timeout=120.0,  # Longer timeout for batch operations
                messages_data=messages_data
            )
        else:
            # Option 2: Individual operations for better flow control
            message_ids = []
            for msg_data in messages_data:
                message_id = await self.add_message(
                    round_id=msg_data['round_id'],
                    role=msg_data['role'],
                    content=msg_data['content'],
                    message_id=msg_data.get('message_id'),
                    created_at=msg_data.get('created_at'),
                    updated_at=msg_data.get('updated_at')
                )
                message_ids.append(message_id)
            return message_ids
    
    async def _add_messages_batch_internal(self, messages_data: List[Dict[str, Any]]) -> List[str]:
        """Internal method for batch message addition."""
        message_ids = []
        for msg_data in messages_data:
            message_id = await self.database.add_message(
                round_id=msg_data['round_id'],
                role=msg_data['role'],
                content=msg_data['content'],
                message_id=msg_data.get('message_id'),
                created_at=msg_data.get('created_at'),
                updated_at=msg_data.get('updated_at')
            )
            message_ids.append(message_id)
        return message_ids
    
    # Health check operations
    async def health_check(self) -> bool:
        """Perform a health check through the queue."""
        await self.ensure_queue_started()
        try:
            return await execute_db_operation(
                self._health_check_internal,
                OperationType.HEALTH_CHECK,
                priority=0,  # Highest priority
                timeout=5.0
            )
        except Exception as e:
            logger.error(f"QueuedDatabase: Health check failed: {e}")
            return False
    
    async def _health_check_internal(self) -> bool:
        """Internal health check method."""
        try:
            # Try to execute a simple query
            users = await self.database.list_users(limit=1)
            return True
        except Exception:
            return False
    
    # Statistics and monitoring
    def get_queue_stats(self):
        """Get queue statistics."""
        return self.queue_manager.get_stats()
    
    def log_queue_stats(self):
        """Log queue statistics."""
        self.queue_manager.log_stats()
    
    async def shutdown(self):
        """Shutdown the queued database."""
        await self.queue_manager.stop()
        logger.info("QueuedDatabase: Shutdown complete")
    
    async def close(self):
        """Close the queued database (alias for shutdown)."""
        await self.shutdown()
        # Also close the underlying database if it has a close method
        if hasattr(self.database, 'close'):
            await self.database.close()
    
    # Additional user methods
    async def get_or_create_user_by_name(self, name: str, description: Optional[str] = None) -> str:
        """Get or create user by name through the queue."""
        await self.ensure_queue_started()
        return await execute_db_operation(
            self.database.get_or_create_user_by_name,
            OperationType.WRITE,
            priority=3,
            name=name,
            description=description
        )
    
    async def get_or_create_agent_by_name(self, name: str, description: Optional[str] = None) -> str:
        """Get or create agent by name through the queue."""
        await self.ensure_queue_started()
        return await execute_db_operation(
            self.database.get_or_create_agent_by_name,
            OperationType.WRITE,
            priority=3,
            name=name,
            description=description
        )
    
    async def get_session_by_name(self, name: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get session by name through the queue."""
        await self.ensure_queue_started()
        return await execute_db_operation(
            self.database.get_session_by_name,
            OperationType.READ,
            priority=1,
            name=name,
            user_id=user_id
        )
    
    async def create_session_with_name(self, user_id: str, agent_id: str, name: str) -> str:
        """Create session with name through the queue."""
        await self.ensure_queue_started()
        return await execute_db_operation(
            self.database.create_session_with_name,
            OperationType.WRITE,
            priority=3,
            user_id=user_id,
            agent_id=agent_id,
            name=name
        )
    
    async def get_or_create_session_by_name(self, user_id: str, agent_id: str, name: str) -> str:
        """Get or create session by name through the queue."""
        await self.ensure_queue_started()
        return await execute_db_operation(
            self.database.get_or_create_session_by_name,
            OperationType.WRITE,
            priority=3,
            user_id=user_id,
            agent_id=agent_id,
            name=name
        )
    
    # Knowledge operations
    async def create_knowledge(self, user_id: str, content: str, knowledge_id: Optional[str] = None) -> str:
        """Create knowledge through the queue."""
        await self.ensure_queue_started()
        return await execute_db_operation(
            self.database.create_knowledge,
            OperationType.WRITE,
            priority=4,
            user_id=user_id,
            content=content,
            knowledge_id=knowledge_id
        )
    
    async def get_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """Get knowledge through the queue."""
        await self.ensure_queue_started()
        return await execute_db_operation(
            self.database.get_knowledge,
            OperationType.READ,
            priority=1,
            knowledge_id=knowledge_id
        )
    
    # Initialize method
    async def initialize(self):
        """Initialize the database through the queue."""
        await self.ensure_queue_started()
        return await execute_db_operation(
            self.database.initialize,
            OperationType.WRITE,
            priority=0,  # Highest priority for initialization
            timeout=120.0
        )
    
    # Delegate other methods to the underlying database
    def __getattr__(self, name):
        """Delegate unknown methods to the underlying database."""
        attr = getattr(self.database, name)
        if callable(attr):
            # For methods not explicitly wrapped, execute through queue with default settings
            async def queued_method(*args, **kwargs):
                await self.ensure_queue_started()
                
                # Determine operation type based on method name
                if name.startswith(('get_', 'list_', 'select')):
                    op_type = OperationType.READ
                    priority = 1
                elif name.startswith(('create_', 'add_', 'insert')):
                    op_type = OperationType.WRITE
                    priority = 3
                elif name.startswith(('update_', 'modify_')):
                    op_type = OperationType.WRITE
                    priority = 4
                elif name.startswith(('delete_', 'remove_')):
                    op_type = OperationType.WRITE
                    priority = 4
                else:
                    op_type = OperationType.WRITE
                    priority = 5
                
                return await execute_db_operation(
                    attr,
                    op_type,
                    priority,
                    None,  # timeout
                    *args,
                    **kwargs
                )
            return queued_method
        return attr