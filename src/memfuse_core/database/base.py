"""Abstract database interface for MemFuse server."""

import abc
from typing import Dict, List, Any, Optional
from loguru import logger


class DBBase(abc.ABC):
    """Abstract base class for database backends."""

    @abc.abstractmethod
    def execute(self, query: str, params: tuple = ()):
        """Execute a SQL query.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Database cursor
        """
        pass

    @abc.abstractmethod
    def commit(self):
        """Commit changes to the database."""
        pass

    @abc.abstractmethod
    def close(self):
        """Close the database connection."""
        pass

    @abc.abstractmethod
    def create_tables(self):
        """Create database tables if they don't exist."""
        pass

    @abc.abstractmethod
    def insert(self, table: str, data: Dict[str, Any]) -> str:
        """Insert data into a table.

        Args:
            table: Table name
            data: Data to insert

        Returns:
            ID of the inserted row
        """
        pass

    @abc.abstractmethod
    def select(self, table: str, conditions: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Select data from a table.

        Args:
            table: Table name
            conditions: Selection conditions (optional)

        Returns:
            List of selected rows
        """
        pass

    @abc.abstractmethod
    def select_one(self, table: str, conditions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Select a single row from a table.

        Args:
            table: Table name
            conditions: Selection conditions

        Returns:
            Selected row or None if not found
        """
        pass

    @abc.abstractmethod
    def update(self, table: str, data: Dict[str, Any], conditions: Dict[str, Any]) -> int:
        """Update data in a table.

        Args:
            table: Table name
            data: Data to update
            conditions: Update conditions

        Returns:
            Number of rows updated
        """
        pass

    @abc.abstractmethod
    def delete(self, table: str, conditions: Dict[str, Any]) -> int:
        """Delete data from a table.

        Args:
            table: Table name
            conditions: Delete conditions

        Returns:
            Number of rows deleted
        """
        pass


class Database:
    """Database class for MemFuse server.

    This class provides a unified interface to the database backend.
    """

    def __init__(self, backend: DBBase):
        """Initialize the database.

        Args:
            backend: Database backend
        """
        self.backend = backend
        self.backend.create_tables()

    def close(self):
        """Close the database connection."""
        self.backend.close()

    def __del__(self):
        """Close the database connection when the object is deleted."""
        self.close()

    # User methods

    def get_or_create_user_by_name(self, name: str, description: Optional[str] = None) -> str:
        """Get a user by name or create if it doesn't exist.

        Args:
            name: User name
            description: User description (optional)

        Returns:
            User ID

        Raises:
            ValueError: If name is None or empty
            RuntimeError: If user creation fails due to database constraints
        """
        if name is None or name.strip() == "":
            raise ValueError("User name cannot be None or empty")

        # Normalize the name (strip whitespace)
        name = name.strip()

        # Check if user exists
        user = self.get_user_by_name(name)
        if user is not None:
            logger.debug(f"Found existing user: {name} with ID: {user['id']}")
            return user["id"]

        # Create new user
        import uuid
        from datetime import datetime
        user_id = str(uuid.uuid4())

        # Create the user with the backend
        data = {
            'id': user_id,
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }

        try:
            self.backend.insert('users', data)
            logger.info(f"Created new user: {name} with ID: {user_id}")
            return user_id
        except Exception as e:
            # Check if this is a uniqueness constraint violation
            error_msg = str(e).lower()
            if 'unique' in error_msg or 'constraint' in error_msg:
                # Race condition: another process created the user between our check and insert
                # Try to get the user again
                user = self.get_user_by_name(name)
                if user is not None:
                    logger.warning(f"User {name} was created by another process, using existing ID: {user['id']}")
                    return user["id"]
                else:
                    raise RuntimeError(f"Failed to create user '{name}' due to uniqueness constraint, but user not found on retry") from e
            else:
                # Some other database error
                raise RuntimeError(f"Failed to create user '{name}': {e}") from e

    def get_or_create_agent_by_name(self, name: str, description: Optional[str] = None) -> str:
        """Get an agent by name or create if it doesn't exist.

        Args:
            name: Agent name
            description: Agent description (optional)

        Returns:
            Agent ID

        Raises:
            ValueError: If name is None or empty
            RuntimeError: If agent creation fails due to database constraints
        """
        if name is None or name.strip() == "":
            raise ValueError("Agent name cannot be None or empty")

        # Normalize the name (strip whitespace)
        name = name.strip()

        # Check if agent exists
        agent = self.get_agent_by_name(name)
        if agent is not None:
            logger.debug(f"Found existing agent: {name} with ID: {agent['id']}")
            return agent["id"]

        # Create new agent
        import uuid
        from datetime import datetime
        agent_id = str(uuid.uuid4())

        # Create the agent with the backend
        data = {
            'id': agent_id,
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        self.backend.insert('agents', data)
        logger.info(f"Created new agent: {name} with ID: {agent_id}")
        return agent_id
    
    def create_user(self, user_id: Optional[str] = None, name: Optional[str] = None, description: Optional[str] = None) -> str:
        """Create a new user.

        Args:
            user_id: User ID (optional, will be auto-generated if not provided)
            name: User name (optional)
            description: User description (optional)

        Returns:
            User ID
        """
        import uuid
        from datetime import datetime

        if user_id is None:
            user_id = str(uuid.uuid4())

        now = datetime.now().isoformat()

        data = {
            'id': user_id,
            'name': name,
            'description': description,
            'created_at': now,
            'updated_at': now
        }

        return self.backend.insert('users', data)

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user by ID.

        Args:
            user_id: User ID

        Returns:
            User data or None if not found
        """
        return self.backend.select_one('users', {'id': user_id})

    def get_user_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a user by name.

        Args:
            name: User name

        Returns:
            User data or None if not found
        """
        return self.backend.select_one('users', {'name': name})

    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users.

        Returns:
            List of user data
        """
        return self.backend.select('users')

    def update_user(self, user_id: str, name: Optional[str] = None,
                    description: Optional[str] = None) -> bool:
        """Update a user.

        Args:
            user_id: User ID
            name: New user name (optional)
            description: New user description (optional)

        Returns:
            True if successful, False otherwise
        """
        # Get current user data
        user = self.get_user(user_id)
        if not user:
            return False

        # Use current values if new values are not provided
        name = name if name is not None else user.get("name")
        description = description if description is not None else user.get("description")

        from datetime import datetime
        now = datetime.now().isoformat()

        data = {
            'name': name,
            'description': description,
            'updated_at': now
        }

        rows_updated = self.backend.update('users', data, {'id': user_id})
        return rows_updated > 0

    def delete_user(self, user_id: str) -> bool:
        """Delete a user.

        Args:
            user_id: User ID

        Returns:
            True if successful, False otherwise
        """
        rows_deleted = self.backend.delete('users', {'id': user_id})
        return rows_deleted > 0

    def get_user_name(self, user_id: str) -> Optional[str]:
        """Get a user's name by ID.

        Args:
            user_id: User ID

        Returns:
            User name or None if not found
        """
        user = self.get_user(user_id)
        if user is None:
            return None
        return user.get("name")

    # Agent methods (additional)

    def get_all_agents(self) -> List[Dict[str, Any]]:
        """Get all agents.

        Returns:
            List of agent data
        """
        return self.backend.select('agents')

    def update_agent(self, agent_id: str, name: Optional[str] = None,
                     description: Optional[str] = None) -> bool:
        """Update an agent.

        Args:
            agent_id: Agent ID
            name: New agent name (optional)
            description: New agent description (optional)

        Returns:
            True if successful, False otherwise
        """
        # Get current agent data
        agent = self.get_agent(agent_id)
        if not agent:
            return False

        # Use current values if new values are not provided
        name = name if name is not None else agent.get("name")
        description = description if description is not None else agent.get("description")

        from datetime import datetime
        now = datetime.now().isoformat()

        data = {
            'name': name,
            'description': description,
            'updated_at': now
        }

        rows_updated = self.backend.update('agents', data, {'id': agent_id})
        return rows_updated > 0

    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent.

        Args:
            agent_id: Agent ID

        Returns:
            True if successful, False otherwise
        """
        rows_deleted = self.backend.delete('agents', {'id': agent_id})
        return rows_deleted > 0

    def get_agent_name(self, agent_id: str) -> Optional[str]:
        """Get an agent's name by ID.

        Args:
            agent_id: Agent ID

        Returns:
            Agent name or None if not found
        """
        agent = self.get_agent(agent_id)
        if agent is None:
            return None
        return agent.get("name")

    # Agent methods

    def create_agent(self, agent_id: Optional[str] = None, name: Optional[str] = None, description: Optional[str] = None) -> str:
        """Create a new agent.

        Args:
            agent_id: Agent ID (optional, will be auto-generated if not provided)
            name: Agent name (optional)
            description: Agent description (optional)

        Returns:
            Agent ID
        """
        import uuid
        from datetime import datetime

        if agent_id is None:
            agent_id = str(uuid.uuid4())

        now = datetime.now().isoformat()

        data = {
            'id': agent_id,
            'name': name,
            'description': description,
            'created_at': now,
            'updated_at': now
        }

        return self.backend.insert('agents', data)

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get an agent by ID.

        Args:
            agent_id: Agent ID

        Returns:
            Agent data or None if not found
        """
        return self.backend.select_one('agents', {'id': agent_id})

    def get_agent_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get an agent by name.

        Args:
            name: Agent name

        Returns:
            Agent data or None if not found
        """
        return self.backend.select_one('agents', {'name': name})

    # Session methods

    def create_session(self, user_id: str, agent_id: str, name: Optional[str] = None,
                       session_id: Optional[str] = None) -> str:
        """Create a new session.

        Args:
            user_id: User ID
            agent_id: Agent ID
            name: Session name (optional)
            session_id: Session ID (optional, will be auto-generated if not provided)

        Returns:
            Session ID
        """
        import uuid
        from datetime import datetime

        if session_id is None:
            session_id = str(uuid.uuid4())

        now = datetime.now().isoformat()

        data = {
            'id': session_id,
            'user_id': user_id,
            'agent_id': agent_id,
            'name': name,
            'created_at': now,
            'updated_at': now
        }

        return self.backend.insert('sessions', data)

    def get_session_by_name(self, name: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get a session by name.

        Args:
            name: Session name
            user_id: User ID to filter by (optional, for user-specific session lookup)
                    When None, performs global lookup for backward compatibility

        Returns:
            Session data or None if not found
        """
        if user_id is not None:
            # User-scoped lookup: Filter by both name and user_id for proper data isolation
            return self.backend.select_one('sessions', {'name': name, 'user_id': user_id})
        else:
            # Global lookup (backward compatibility): Use with caution in multi-user scenarios
            return self.backend.select_one('sessions', {'name': name})

    def create_session_with_name(self, user_id: str, agent_id: str, name: str) -> str:
        """Create a new session with a specific name.

        Args:
            user_id: User ID
            agent_id: Agent ID
            name: Session name

        Returns:
            Session ID

        Raises:
            ValueError: If a session with this name already exists for this user
        """
        import uuid
        from datetime import datetime

        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        data = {
            'id': session_id,
            'user_id': user_id,
            'agent_id': agent_id,
            'name': name,
            'created_at': now,
            'updated_at': now
        }

        try:
            return self.backend.insert('sessions', data)
        except Exception as e:
            # Check if this is a uniqueness constraint violation
            error_msg = str(e).lower()
            if 'unique' in error_msg or 'constraint' in error_msg:
                raise ValueError(
                    f"Session with name '{name}' already exists for this user. "
                    f"Session names must be unique within each user's scope."
                ) from e
            else:
                # Some other database error
                raise RuntimeError(f"Failed to create session '{name}': {e}") from e

    def get_or_create_session_by_name(self, user_id: str, agent_id: str, name: str) -> str:
        """Get a session by name or create if it doesn't exist.

        Args:
            user_id: User ID (used for scoped session lookup)
            agent_id: Agent ID
            name: Session name

        Returns:
            Session ID

        Raises:
            ValueError: If name is None
        """
        if name is None:
            raise ValueError("Session name cannot be None")

        # Check if session exists for this specific user (FIXED: added user_id parameter)
        session = self.get_session_by_name(name, user_id=user_id)
        if session is not None:
            return session["id"]

        # Create new session
        return self.create_session_with_name(user_id, agent_id, name)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID.

        Args:
            session_id: Session ID

        Returns:
            Session data or None if not found
        """
        return self.backend.select_one('sessions', {'id': session_id})

    def update_session(self, session_id: str, name: Optional[str] = None) -> bool:
        """Update a session.

        Args:
            session_id: Session ID
            name: New session name (optional)

        Returns:
            True if successful, False otherwise
        """
        # Get current session data
        session = self.get_session(session_id)
        if not session:
            return False

        # Use current values if new values are not provided
        name = name if name is not None else session.get("name")

        from datetime import datetime
        now = datetime.now().isoformat()

        data = {
            'name': name,
            'updated_at': now
        }

        rows_updated = self.backend.update('sessions', data, {'id': session_id})
        return rows_updated > 0

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session ID

        Returns:
            True if successful, False otherwise
        """
        rows_deleted = self.backend.delete('sessions', {'id': session_id})
        return rows_deleted > 0

    def get_sessions(self, user_id: Optional[str] = None, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get sessions, optionally filtered by user and/or agent.

        Args:
            user_id: User ID (optional)
            agent_id: Agent ID (optional)

        Returns:
            List of session data
        """
        conditions = {}

        if user_id:
            conditions['user_id'] = user_id

        if agent_id:
            conditions['agent_id'] = agent_id

        if conditions:
            return self.backend.select('sessions', conditions)
        else:
            return self.backend.select('sessions')

    # Round methods

    def create_round(self, session_id: str, round_id: Optional[str] = None) -> str:
        """Create a new round.

        Args:
            session_id: Session ID
            round_id: Round ID (optional, will be auto-generated if not provided)

        Returns:
            Round ID
        """
        import uuid
        from datetime import datetime

        if round_id is None:
            round_id = str(uuid.uuid4())

        now = datetime.now().isoformat()

        data = {
            'id': round_id,
            'session_id': session_id,
            'created_at': now,
            'updated_at': now
        }

        return self.backend.insert('rounds', data)

    def get_round(self, round_id: str) -> Optional[Dict[str, Any]]:
        """Get a round by ID.

        Args:
            round_id: Round ID

        Returns:
            Round data or None if not found
        """
        return self.backend.select_one('rounds', {'id': round_id})

    # Message methods

    def add_message(self, round_id: str, role: str, content: str, message_id: Optional[str] = None) -> str:
        """Add a message to a round.

        Args:
            round_id: Round ID
            role: Message role (user, assistant, system, memfuse)
            content: Message content
            message_id: Message ID (optional, will be auto-generated if not provided)

        Returns:
            Message ID
        """
        import uuid
        from datetime import datetime

        if message_id is None:
            message_id = str(uuid.uuid4())

        now = datetime.now().isoformat()

        data = {
            'id': message_id,
            'round_id': round_id,
            'role': role,
            'content': content,
            'created_at': now,
            'updated_at': now
        }

        return self.backend.insert('messages', data)

    def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get a message by ID.

        Args:
            message_id: Message ID

        Returns:
            Message data or None if not found
        """
        return self.backend.select_one('messages', {'id': message_id})

    def get_messages_by_round(self, round_id: str) -> List[Dict[str, Any]]:
        """Get all messages in a round.

        Args:
            round_id: Round ID

        Returns:
            List of message data
        """
        return self.backend.select('messages', {'round_id': round_id})

    def update_message(self, message_id: str, content: str) -> bool:
        """Update a message.

        Args:
            message_id: Message ID
            content: New message content

        Returns:
            True if successful, False otherwise
        """
        from datetime import datetime
        now = datetime.now().isoformat()

        data = {
            'content': content,
            'updated_at': now
        }

        rows_updated = self.backend.update('messages', data, {'id': message_id})
        return rows_updated > 0

    def delete_message(self, message_id: str) -> bool:
        """Delete a message.

        Args:
            message_id: Message ID

        Returns:
            True if successful, False otherwise
        """
        rows_deleted = self.backend.delete('messages', {'id': message_id})
        return rows_deleted > 0

    def get_messages_by_session(self, session_id: str, limit: Optional[int] = None,
                               sort_by: str = 'timestamp', order: str = 'desc') -> List[Dict[str, Any]]:
        """Get messages for a session with optional limit and sorting.

        Args:
            session_id: Session ID
            limit: Maximum number of messages to return (optional)
            sort_by: Field to sort by, either 'timestamp' or 'id' (default: 'timestamp')
            order: Sort order, either 'asc' or 'desc' (default: 'desc')

        Returns:
            List of message data
        """
        # First, get all rounds for the session
        rounds = self.backend.select('rounds', {'session_id': session_id})

        if not rounds:
            return []

        # Then, get all messages for each round
        messages = []
        for round_data in rounds:
            round_messages = self.get_messages_by_round(round_data['id'])
            messages.extend(round_messages)

        # Sort messages based on the specified field and order
        if sort_by == 'timestamp':
            # Sort by created_at timestamp
            messages.sort(key=lambda x: x.get('created_at', ''), reverse=(order == 'desc'))
        elif sort_by == 'id':
            # Sort by message ID
            messages.sort(key=lambda x: x.get('id', ''), reverse=(order == 'desc'))

        # Apply limit if specified
        if limit is not None and limit > 0:
            messages = messages[:limit]

        return messages

    # API key methods

    def create_api_key(
        self,
        user_id: str,
        key: Optional[str] = None,
        name: Optional[str] = None,
        permissions: Optional[str] = None,
        expires_at: Optional[str] = None,
        api_key_id: Optional[str] = None
    ) -> str:
        """Create a new API key.

        Args:
            user_id: User ID
            key: API key (optional, will be auto-generated if not provided)
            name: API key name (optional)
            permissions: API key permissions (optional)
            expires_at: API key expiration date (optional)
            api_key_id: API key ID (optional, will be auto-generated if not provided)

        Returns:
            API key ID
        """
        import uuid
        from datetime import datetime

        if api_key_id is None:
            api_key_id = str(uuid.uuid4())

        if key is None:
            # Generate a random API key
            key = str(uuid.uuid4()) + str(uuid.uuid4())

        now = datetime.now().isoformat()

        data = {
            'id': api_key_id,
            'user_id': user_id,
            'key': key,
            'name': name,
            'permissions': permissions,
            'created_at': now,
            'expires_at': expires_at
        }

        return self.backend.insert('api_keys', data)

    def get_api_key(self, api_key_id: str) -> Optional[Dict[str, Any]]:
        """Get an API key by ID.

        Args:
            api_key_id: API key ID

        Returns:
            API key data or None if not found
        """
        return self.backend.select_one('api_keys', {'id': api_key_id})

    def get_api_key_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Get an API key by key.

        Args:
            key: API key

        Returns:
            API key data or None if not found
        """
        return self.backend.select_one('api_keys', {'key': key})

    def get_api_keys_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all API keys for a user.

        Args:
            user_id: User ID

        Returns:
            List of API key data
        """
        return self.backend.select('api_keys', {'user_id': user_id})

    def delete_api_key(self, api_key_id: str) -> bool:
        """Delete an API key.

        Args:
            api_key_id: API key ID

        Returns:
            True if successful, False otherwise
        """
        rows_deleted = self.backend.delete('api_keys', {'id': api_key_id})
        return rows_deleted > 0

    def validate_api_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key.

        Args:
            key: API key

        Returns:
            API key data if valid, None otherwise
        """
        api_key = self.get_api_key_by_key(key)

        if api_key is None:
            return None

        # Check if the API key has expired
        if api_key.get("expires_at") is not None:
            from datetime import datetime
            expires_at = datetime.fromisoformat(api_key["expires_at"])
            if expires_at < datetime.now():
                return None

        return api_key

    # Knowledge methods

    def add_knowledge(self, user_id: str, content: str, knowledge_id: Optional[str] = None) -> str:
        """Add knowledge for a user.

        Args:
            user_id: User ID
            content: Knowledge content
            knowledge_id: Knowledge ID (optional, will be auto-generated if not provided)

        Returns:
            Knowledge ID
        """
        import uuid
        from datetime import datetime

        if knowledge_id is None:
            knowledge_id = str(uuid.uuid4())

        now = datetime.now().isoformat()

        data = {
            'id': knowledge_id,
            'user_id': user_id,
            'content': content,
            'created_at': now,
            'updated_at': now
        }

        return self.backend.insert('knowledge', data)

    def get_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """Get knowledge by ID.

        Args:
            knowledge_id: Knowledge ID

        Returns:
            Knowledge data or None if not found
        """
        return self.backend.select_one('knowledge', {'id': knowledge_id})

    def get_knowledge_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all knowledge for a user.

        Args:
            user_id: User ID

        Returns:
            List of knowledge data
        """
        return self.backend.select('knowledge', {'user_id': user_id})

    def update_knowledge(self, knowledge_id: str, content: str) -> bool:
        """Update knowledge.

        Args:
            knowledge_id: Knowledge ID
            content: New knowledge content

        Returns:
            True if successful, False otherwise
        """
        from datetime import datetime
        now = datetime.now().isoformat()

        data = {
            'content': content,
            'updated_at': now
        }

        rows_updated = self.backend.update('knowledge', data, {'id': knowledge_id})
        return rows_updated > 0

    def delete_knowledge(self, knowledge_id: str) -> bool:
        """Delete knowledge.

        Args:
            knowledge_id: Knowledge ID

        Returns:
            True if successful, False otherwise
        """
        rows_deleted = self.backend.delete('knowledge', {'id': knowledge_id})
        return rows_deleted > 0
