"""Memory service proxy implementation.

This module provides a proxy for the MemoryService class that allows
for user/agent/session-specific operations while sharing the underlying
storage components.
"""

from typing import Dict, Any, Optional, List
from loguru import logger

from .memory_service import MemoryService
from .database_service import DatabaseService


class MemoryServiceProxy:
    """A proxy for the MemoryService class.

    This class wraps a MemoryService instance and provides a way to
    use it with different user, agent, and session parameters without
    reinitializing the storage components.

    Instead of inheriting from MemoryService, this class uses composition
    to delegate operations to the underlying MemoryService instance.

    The proxy layer is responsible for:
    1. Managing user/agent/session context
    2. Validation and error handling
    3. Pre-processing requests and post-processing responses
    4. Providing a consistent API interface

    The core MemoryService is responsible for:
    1. Managing storage components (vector store, graph store, etc.)
    2. Implementing core memory operations (add, query, etc.)
    3. Optimizing storage and retrieval performance
    """

    # Class variable to track instances and avoid duplicate initialization logs
    _instances = {}

    def __init__(
        self,
        memory_service: MemoryService,
        user: str = "user_default",
        agent: Optional[str] = None,
        session: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Initialize the MemoryService proxy.

        P1 OPTIMIZATION: For user-level proxies, agent/session parameters are ignored
        and should be passed to individual method calls instead.

        Args:
            memory_service: The global MemoryService instance to proxy
            user: User name (default: "user_default")
            agent: Agent name (optional, ignored for user-level proxies)
            session: Session name (optional, ignored for user-level proxies)
            session_id: Session ID (optional, ignored for user-level proxies)
        """
        # Store the global memory service instance
        self._memory_service = memory_service

        # Use the global database instance
        self.db = DatabaseService.get_instance()

        # Ensure user exists and get user_id
        self._user_id = self.db.get_or_create_user_by_name(user)

        # P1 OPTIMIZATION: Store user context only, session context passed as parameters
        self.user = user

        # For backward compatibility, handle session context if provided during initialization
        if agent is not None or session is not None or session_id is not None:
            logger.debug(f"P1: Session context provided during init, will be handled per-method")
            # Store for backward compatibility but prefer parameter-based approach
            self._default_agent = agent
            self._default_session = session
            self._default_session_id = session_id
        else:
            self._default_agent = None
            self._default_session = None
            self._default_session_id = None

        # Create a unique key for this user-level instance
        instance_key = f"user_level:{user}"

        # Only log initialization if this is a new instance
        if instance_key not in MemoryServiceProxy._instances:
            MemoryServiceProxy._instances[instance_key] = True
            logger.info(f"MemoryServiceProxy: Initialized user-level proxy for user {user}")

    def _resolve_session_context(
        self,
        agent: Optional[str] = None,
        session: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Resolve session context for a method call.

        P1 OPTIMIZATION: This method handles session context resolution
        for individual method calls rather than storing it in the instance.

        Args:
            agent: Agent name (optional)
            session: Session name (optional)
            session_id: Session ID (optional, takes precedence if provided)

        Returns:
            Tuple of (resolved_agent_id, resolved_session_id)
        """
        # Use provided parameters or fall back to defaults
        effective_agent = agent or self._default_agent or "agent_default"
        effective_session = session or self._default_session
        effective_session_id = session_id or self._default_session_id

        # Get or create agent_id
        agent_id = self.db.get_or_create_agent_by_name(effective_agent)

        # Resolve session_id
        resolved_session_id = None
        if effective_session_id is not None:
            # If session_id is provided, use it directly
            session_data = self.db.get_session(effective_session_id)
            if session_data is None:
                # Session not found, create a new one
                resolved_session_id = self.db.create_session(
                    self._user_id, agent_id, effective_session)
            else:
                resolved_session_id = effective_session_id
        elif effective_session is not None:
            # If session name is provided, check if it already exists for this user
            session_data = self.db.get_session_by_name(effective_session, user_id=self._user_id)
            if session_data is not None:
                # Session exists for this user
                resolved_session_id = session_data['id']
            else:
                # Session not found, create a new one
                resolved_session_id = self.db.create_session_with_name(
                    self._user_id, agent_id, effective_session)

        return agent_id, resolved_session_id

    async def initialize(self):
        """Initialize the memory service proxy.

        This method ensures that the underlying MemoryService is properly
        initialized before use.

        Returns:
            Self for method chaining
        """
        # Ensure the underlying MemoryService is initialized
        if self._memory_service is not None:
            # Check if the MemoryService has been initialized
            has_retrieval = hasattr(self._memory_service, 'multi_path_retrieval')
            retrieval_is_none = (has_retrieval and
                                 self._memory_service.multi_path_retrieval is None)
            if not has_retrieval or retrieval_is_none:
                await self._memory_service.initialize()

        return self

    async def add(
        self,
        messages: List[Dict[str, Any]],
        agent: Optional[str] = None,
        session: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add messages to memory.

        This method adds validation and error handling before delegating
        to the underlying memory service.

        P1 OPTIMIZATION: Session context can be passed as parameters.

        Args:
            messages: List of message dictionaries
            agent: Agent name (optional, overrides default)
            session: Session name (optional, overrides default)
            session_id: Session ID (optional, overrides default)

        Returns:
            Dictionary with status, code, and message IDs
        """
        # Validate input parameters
        if not messages:
            return {
                "status": "error",
                "code": 400,
                "data": None,
                "message": "No messages provided",
                "errors": [
                    {"field": "messages", "message": "No messages provided"}
                ],
            }

        # P1 OPTIMIZATION: Resolve session context for this method call
        agent_id, resolved_session_id = self._resolve_session_context(
            agent=agent, session=session, session_id=session_id
        )

        # Validate each message
        for i, message in enumerate(messages):
            if 'metadata' not in message:
                message['metadata'] = {}

            # Check required fields
            if 'content' not in message:
                return {
                    "status": "error",
                    "code": 400,
                    "data": None,
                    "message": f"Message at index {i} is missing 'content' field",
                    "errors": [
                        {"field": f"messages[{i}].content",
                         "message": "Content field is required"}
                    ],
                }

            # Add user_id if available and not already set
            if self._user_id and 'user_id' not in message:
                message['user_id'] = self._user_id

            # P1: Add resolved session_id to both message and metadata
            if resolved_session_id:
                if 'session_id' not in message:
                    message['session_id'] = resolved_session_id
                if 'session_id' not in message.get('metadata', {}):
                    message['metadata']['session_id'] = resolved_session_id

        try:
            # Delegate to the underlying memory service
            result = await self._memory_service.add(messages)
            return result

        except Exception as e:
            # Handle exceptions
            logger.error(f"Error adding messages: {str(e)}")
            return {
                "status": "error",
                "code": 500,
                "data": None,
                "message": f"Error adding messages: {str(e)}",
                "errors": [
                    {"field": "general", "message": str(e)}
                ],
            }

    async def add_batch(self, messages: List[Dict[str, Any]], agent=None, session=None, session_id=None) -> Dict[str, Any]:
        """Add a batch of messages to memory.

        This method adds validation and error handling before delegating
        to the underlying memory service.

        Args:
            messages: List of message dictionaries
            agent: Agent name (optional, for session context resolution)
            session: Session name (optional, for session context resolution)
            session_id: Session ID (optional, for session context resolution)

        Returns:
            Dictionary with status, code, and message IDs
        """
        # P1 OPTIMIZATION: Resolve session context from parameters
        agent_id, resolved_session_id = self._resolve_session_context(agent, session, session_id)

        # Validate input parameters
        if not messages:
            return {
                "status": "error",
                "code": 400,
                "data": None,
                "message": "No messages provided",
                "errors": [
                    {"field": "messages", "message": "No messages provided"}
                ],
            }

        # Validate each message
        for i, message in enumerate(messages):
            if 'metadata' not in message:
                message['metadata'] = {}

            # Check required fields
            if 'content' not in message:
                return {
                    "status": "error",
                    "code": 400,
                    "data": None,
                    "message": f"Message at index {i} is missing 'content' field",
                    "errors": [
                        {"field": f"messages[{i}].content",
                         "message": "Content field is required"}
                    ],
                }

            # Add user_id if available and not already set
            if self._user_id and 'user_id' not in message:
                message['user_id'] = self._user_id

            # P1: Add resolved session_id to both message and metadata
            if resolved_session_id:
                if 'session_id' not in message:
                    message['session_id'] = resolved_session_id
                if 'session_id' not in message.get('metadata', {}):
                    message['metadata']['session_id'] = resolved_session_id

        try:
            # Delegate to the underlying memory service
            result = await self._memory_service.add_batch(messages)
            return result

        except Exception as e:
            # Handle exceptions
            logger.error(f"Error adding batch messages: {str(e)}")
            return {
                "status": "error",
                "code": 500,
                "data": None,
                "message": f"Error adding batch messages: {str(e)}",
                "errors": [
                    {"field": "general", "message": str(e)}
                ],
            }

    async def read(self, message_ids: List[str]) -> Dict[str, Any]:
        """Read messages from memory.

        This method adds validation and error handling before delegating
        to the underlying memory service.

        Args:
            message_ids: List of message IDs

        Returns:
            Dictionary with status, code, and messages
        """
        # Validate input parameters
        if not message_ids:
            return {
                "status": "error",
                "code": 400,
                "data": None,
                "message": "No message IDs provided",
                "errors": [
                    {"field": "message_ids", "message": "No message IDs provided"}
                ],
            }

        try:
            # Check if messages exist before reading
            not_found_ids = []
            for message_id in message_ids:
                if not self.db.get_message(message_id):
                    not_found_ids.append(message_id)

            if not_found_ids:
                return {
                    "status": "error",
                    "code": 404,
                    "data": {"not_found_ids": not_found_ids},
                    "message": f"Some message IDs were not found",
                    "errors": [
                        {"field": "message_ids",
                         "message": f"Message IDs not found: {', '.join(not_found_ids)}"}
                    ],
                }

            # Delegate to the underlying memory service
            result = await self._memory_service.read(message_ids)
            return result

        except Exception as e:
            # Handle exceptions
            logger.error(f"Error reading messages: {str(e)}")
            return {
                "status": "error",
                "code": 500,
                "data": None,
                "message": f"Error reading messages: {str(e)}",
                "errors": [
                    {"field": "general", "message": str(e)}
                ],
            }

    async def update(self, message_ids: List[str], new_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update messages in memory.

        This method adds validation and error handling before delegating
        to the underlying memory service.

        Args:
            message_ids: List of message IDs
            new_messages: List of new message dictionaries

        Returns:
            Dictionary with status, code, and updated message IDs
        """
        # Validate input parameters
        if not message_ids:
            return {
                "status": "error",
                "code": 400,
                "data": None,
                "message": "No message IDs provided",
                "errors": [
                    {"field": "message_ids", "message": "No message IDs provided"}
                ],
            }

        if not new_messages:
            return {
                "status": "error",
                "code": 400,
                "data": None,
                "message": "No new messages provided",
                "errors": [
                    {"field": "new_messages", "message": "No new messages provided"}
                ],
            }

        if len(message_ids) != len(new_messages):
            return {
                "status": "error",
                "code": 400,
                "data": None,
                "message": "Number of message IDs must match number of new messages",
                "errors": [
                    {"field": "general",
                     "message": "Number of message IDs must match number of new messages"}
                ],
            }

        try:
            # Check if messages exist before updating
            not_found_ids = []
            for message_id in message_ids:
                if not self.db.get_message(message_id):
                    not_found_ids.append(message_id)

            if not_found_ids:
                return {
                    "status": "error",
                    "code": 404,
                    "data": {"not_found_ids": not_found_ids},
                    "message": f"Some message IDs were not found",
                    "errors": [
                        {"field": "message_ids",
                         "message": f"Message IDs not found: {', '.join(not_found_ids)}"}
                    ],
                }

            # Validate each new message
            for i, message in enumerate(new_messages):
                # Check required fields
                if 'content' not in message:
                    return {
                        "status": "error",
                        "code": 400,
                        "data": None,
                        "message": f"New message at index {i} is missing 'content' field",
                        "errors": [
                            {"field": f"new_messages[{i}].content",
                             "message": "Content field is required"}
                        ],
                    }

                # Add user_id if available and not already set
                if hasattr(self, '_user_id') and self._user_id and 'user_id' not in message:
                    message['user_id'] = self._user_id

                # Add session_id if available and not already set
                if hasattr(self, '_session_id') and self._session_id and 'session_id' not in message:
                    message['session_id'] = self._session_id

            # Delegate to the underlying memory service
            result = await self._memory_service.update(message_ids, new_messages)
            return result

        except Exception as e:
            # Handle exceptions
            logger.error(f"Error updating messages: {str(e)}")
            return {
                "status": "error",
                "code": 500,
                "data": None,
                "message": f"Error updating messages: {str(e)}",
                "errors": [
                    {"field": "general", "message": str(e)}
                ],
            }

    async def delete(self, message_ids: List[str]) -> Dict[str, Any]:
        """Delete messages from memory.

        This method adds validation and error handling before delegating
        to the underlying memory service.

        Args:
            message_ids: List of message IDs

        Returns:
            Dictionary with status, code, and deleted message IDs
        """
        # Validate input parameters
        if not message_ids:
            return {
                "status": "error",
                "code": 400,
                "data": None,
                "message": "No message IDs provided",
                "errors": [
                    {"field": "message_ids", "message": "No message IDs provided"}
                ],
            }

        deleted_ids = []
        not_found_ids = []

        for message_id in message_ids:
            # Check if message exists before attempting to delete
            message = self.db.get_message(message_id)
            if not message:
                not_found_ids.append(message_id)
                continue

            # Delete message using the core service method
            success = await self._memory_service.delete(message_id)
            if success:
                deleted_ids.append(message_id)

        # If any messages were not found, return an error
        if not_found_ids:
            return {
                "status": "error",
                "code": 404,
                "data": {
                    "deleted_ids": deleted_ids,
                    "not_found_ids": not_found_ids
                },
                "message": f"Some message IDs were not found: {', '.join(not_found_ids)}",
                "errors": [{"field": "message_ids", "message": f"Message IDs not found: {', '.join(not_found_ids)}"}],
            }

        # All messages were successfully deleted
        return {
            "status": "success",
            "code": 200,
            "data": {"message_ids": deleted_ids},
            "message": f"Deleted {len(deleted_ids)} messages",
            "errors": None,
        }

    async def query(self, query: str, top_k: int = 10, **kwargs) -> Dict[str, Any]:
        """Query memory for relevant information.

        This method adds validation and error handling before delegating
        to the underlying memory service.

        Args:
            query: The content to query for
            top_k: Maximum number of results to return
            **kwargs: Additional parameters to pass to the underlying query method

        Returns:
            Dictionary with status, code, and query results
        """
        # Validate input parameters
        if not query or not isinstance(query, str):
            return {
                "status": "error",
                "code": 400,
                "data": None,
                "message": "Query content must be a non-empty string",
                "errors": [
                    {"field": "query", "message": "Query content must be a non-empty string"}
                ],
            }

        if top_k <= 0:
            return {
                "status": "error",
                "code": 400,
                "data": None,
                "message": "top_k must be a positive integer",
                "errors": [
                    {"field": "top_k", "message": "top_k must be a positive integer"}
                ],
            }

        # Get filter_dict from kwargs or create a new one
        filter_dict = kwargs.get('filter_dict', {})
        if filter_dict is None:
            filter_dict = {}
            kwargs['filter_dict'] = filter_dict

        # Add user_id to filter_dict if available
        if hasattr(self, '_user_id') and self._user_id and 'user_id' not in filter_dict:
            filter_dict['user_id'] = self._user_id

        # Add session_id to filter_dict if available
        if hasattr(self, '_session_id') and self._session_id and 'session_id' not in filter_dict:
            filter_dict['session_id'] = self._session_id

        try:
            # Delegate to the underlying memory service
            result = await self._memory_service.query(query=query, top_k=top_k, **kwargs)
            return result

        except Exception as e:
            # Handle exceptions
            logger.error(f"Error in query: {str(e)}")
            return {
                "status": "error",
                "code": 500,
                "data": None,
                "message": f"Error querying memory: {str(e)}",
                "errors": [
                    {"field": "general", "message": str(e)}
                ],
            }

    async def delete_knowledge(self, knowledge_ids: List[str]) -> Dict[str, Any]:
        """Delete knowledge items from memory.

        This method adds validation and error handling before delegating
        to the underlying memory service.

        Args:
            knowledge_ids: List of knowledge IDs

        Returns:
            Dictionary with status, code, and deleted knowledge IDs
        """
        # Validate input parameters
        if not knowledge_ids:
            return {
                "status": "error",
                "code": 400,
                "data": None,
                "message": "No knowledge IDs provided",
                "errors": [
                    {"field": "knowledge_ids", "message": "No knowledge IDs provided"}
                ],
            }

        deleted_ids = []
        not_found_ids = []

        for knowledge_id in knowledge_ids:
            # Check if knowledge exists before attempting to delete
            knowledge = self.db.get_knowledge(knowledge_id)
            if not knowledge:
                not_found_ids.append(knowledge_id)
                continue

            # Delete knowledge using the core service method
            success = await self._memory_service.delete_knowledge(knowledge_id)
            if success:
                deleted_ids.append(knowledge_id)

        # If any knowledge items were not found, return an error
        if not_found_ids:
            return {
                "status": "error",
                "code": 404,
                "data": {
                    "deleted_ids": deleted_ids,
                    "not_found_ids": not_found_ids
                },
                "message": f"Some knowledge IDs were not found: {', '.join(not_found_ids)}",
                "errors": [{"field": "knowledge_ids", "message": f"Knowledge IDs not found: {', '.join(not_found_ids)}"}],
            }

        # All knowledge items were successfully deleted
        return {
            "status": "success",
            "code": 200,
            "data": {"knowledge_ids": deleted_ids},
            "message": f"Deleted {len(deleted_ids)} knowledge items",
            "errors": None,
        }
