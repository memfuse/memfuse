"""
Configuration and fixtures for contract tests.
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

# Add src to path so we can import the modules
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

@pytest.fixture(autouse=True, scope="session")
def mock_database_service():
    """Mock the database service to avoid threading issues and focus on contract validation."""
    
    # In-memory storage for the mock
    mock_users = {}
    user_counter = 0
    mock_agents = {}
    agent_counter = 0
    mock_sessions = {}
    session_counter = 0
    
    async def create_user(name, description=None):
        nonlocal user_counter
        # Check for duplicate names
        for user in mock_users.values():
            if user["name"] == name:
                raise ValueError(f"User with name '{name}' already exists")

        user_counter += 1
        user_id = f"user-{user_counter}"
        mock_users[user_id] = {
            "id": user_id,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        return user_id
    
    async def get_user(user_id):
        return mock_users.get(user_id)

    async def get_all_users():
        return list(mock_users.values())

    async def get_user_by_name(name):
        for user in mock_users.values():
            if user["name"] == name:
                return user
        return None
    
    async def update_user(user_id, name=None, description=None):
        if user_id not in mock_users:
            return False
        
        # Check for duplicate names if name is being changed
        if name is not None:
            for uid, user in mock_users.items():
                if uid != user_id and user["name"] == name:
                    raise ValueError(f"User with name '{name}' already exists")
        
        if name is not None:
            mock_users[user_id]["name"] = name
        if description is not None:
            mock_users[user_id]["description"] = description
        mock_users[user_id]["updated_at"] = datetime.now().isoformat()
        return True
    
    async def delete_user(user_id):
        if user_id in mock_users:
            del mock_users[user_id]
            return True
        return False
    
    async def create_agent(name, description=None):
        nonlocal agent_counter
        # Check for duplicate names
        for agent in mock_agents.values():
            if agent["name"] == name:
                raise ValueError(f"Agent with name '{name}' already exists")

        agent_counter += 1
        agent_id = f"agent-{agent_counter}"
        mock_agents[agent_id] = {
            "id": agent_id,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        return agent_id

    async def get_agent(agent_id):
        return mock_agents.get(agent_id)

    async def get_all_agents():
        return list(mock_agents.values())

    async def get_agent_by_name(name):
        for agent in mock_agents.values():
            if agent["name"] == name:
                return agent
        return None

    async def update_agent(agent_id, name=None, description=None):
        if agent_id not in mock_agents:
            return False

        # Check for duplicate names if name is being changed
        if name is not None:
            for aid, agent in mock_agents.items():
                if aid != agent_id and agent["name"] == name:
                    raise ValueError(f"Agent with name '{name}' already exists")

        if name is not None:
            mock_agents[agent_id]["name"] = name
        if description is not None:
            mock_agents[agent_id]["description"] = description
        mock_agents[agent_id]["updated_at"] = datetime.now().isoformat()
        return True

    async def delete_agent(agent_id):
        if agent_id in mock_agents:
            del mock_agents[agent_id]
            return True
        return False
    
    async def create_session(user_id, agent_id, name=None):
        nonlocal session_counter
        # Validate user and agent exist
        if user_id not in mock_users:
            raise ValueError(f"User with ID '{user_id}' not found")
        if agent_id not in mock_agents:
            raise ValueError(f"Agent with ID '{agent_id}' not found")
        
        session_counter += 1
        session_id = f"session-{session_counter}"
        mock_sessions[session_id] = {
            "id": session_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "name": name,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        return session_id
    
    async def get_session(session_id):
        return mock_sessions.get(session_id)
    
    async def get_sessions(user_id=None, agent_id=None):
        sessions = list(mock_sessions.values())
        
        # Filter by user_id if provided
        if user_id:
            sessions = [s for s in sessions if s["user_id"] == user_id]
        
        # Filter by agent_id if provided
        if agent_id:
            sessions = [s for s in sessions if s["agent_id"] == agent_id]
        
        return sessions
    
    async def get_session_by_name(name, user_id=None):
        for session in mock_sessions.values():
            if session["name"] == name:
                if user_id is None or session["user_id"] == user_id:
                    return session
        return None
    
    async def update_session(session_id, name=None):
        if session_id not in mock_sessions:
            return False
        
        if name is not None:
            mock_sessions[session_id]["name"] = name
        mock_sessions[session_id]["updated_at"] = datetime.now().isoformat()
        return True
    
    async def delete_session(session_id):
        if session_id in mock_sessions:
            del mock_sessions[session_id]
            return True
        return False
    
    async def create_session_with_name(user_id, agent_id, name):
        """Create a session with a specific name."""
        # Validate user and agent exist
        if user_id not in mock_users:
            raise ValueError(f"User with ID '{user_id}' not found")
        if agent_id not in mock_agents:
            raise ValueError(f"Agent with ID '{agent_id}' not found")
        
        # Check if session with this name already exists for this user
        for session in mock_sessions.values():
            if session["name"] == name and session["user_id"] == user_id:
                raise ValueError(f"Session with name '{name}' already exists for this user")
        
        # Create the session
        return create_session(user_id, agent_id, name)
    
    async def get_messages_by_session(session_id, limit=20, sort_by="timestamp", order="desc"):
        """Mock method for getting messages by session."""
        return []
    
    async def get_message(message_id):
        """Mock method for getting a single message."""
        return None
    
    # Knowledge methods
    mock_knowledge = {}
    knowledge_counter = 0
    
    async def add_knowledge(user_id, content, knowledge_id=None):
        nonlocal knowledge_counter
        # Validate user exists
        if user_id not in mock_users:
            raise ValueError(f"User with ID '{user_id}' not found")
        
        if knowledge_id is None:
            knowledge_counter += 1
            knowledge_id = f"knowledge-{knowledge_counter}"
        
        mock_knowledge[knowledge_id] = {
            "id": knowledge_id,
            "user_id": user_id,
            "content": content,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        return knowledge_id
    
    async def get_knowledge(knowledge_id):
        return mock_knowledge.get(knowledge_id)
    
    async def get_knowledge_by_user(user_id):
        return [k for k in mock_knowledge.values() if k["user_id"] == user_id]
    
    async def update_knowledge(knowledge_id, content):
        if knowledge_id not in mock_knowledge:
            return False
        
        mock_knowledge[knowledge_id]["content"] = content
        mock_knowledge[knowledge_id]["updated_at"] = datetime.now().isoformat()
        return True
    
    async def delete_knowledge(knowledge_id):
        if knowledge_id in mock_knowledge:
            del mock_knowledge[knowledge_id]
            return True
        return False
    
    async def get_or_create_user_by_name(name, description=None):
        """Get or create a user by name - returns user_id."""
        # Check if user exists
        for user in mock_users.values():
            if user["name"] == name:
                return user["id"]
        
        # Create new user if not found
        return await create_user(name, description)
    
    async def get_or_create_agent_by_name(name, description=None):
        """Get or create an agent by name - returns agent_id."""
        # Check if agent exists
        for agent in mock_agents.values():
            if agent["name"] == name:
                return agent["id"]
        
        # Create new agent if not found
        return await create_agent(name, description)
    
    # Create a mock instance with async methods
    mock_instance = AsyncMock()
    mock_instance.create_user = AsyncMock(side_effect=create_user)
    mock_instance.get_user = AsyncMock(side_effect=get_user)
    mock_instance.get_all_users = AsyncMock(side_effect=get_all_users)
    mock_instance.get_user_by_name = AsyncMock(side_effect=get_user_by_name)
    mock_instance.get_or_create_user_by_name = AsyncMock(side_effect=get_or_create_user_by_name)
    mock_instance.update_user = AsyncMock(side_effect=update_user)
    mock_instance.delete_user = AsyncMock(side_effect=delete_user)
    mock_instance.create_agent = AsyncMock(side_effect=create_agent)
    mock_instance.get_agent = AsyncMock(side_effect=get_agent)
    mock_instance.get_all_agents = AsyncMock(side_effect=get_all_agents)
    mock_instance.get_agent_by_name = AsyncMock(side_effect=get_agent_by_name)
    mock_instance.get_or_create_agent_by_name = AsyncMock(side_effect=get_or_create_agent_by_name)
    mock_instance.update_agent = AsyncMock(side_effect=update_agent)
    mock_instance.delete_agent = AsyncMock(side_effect=delete_agent)
    mock_instance.create_session = AsyncMock(side_effect=create_session)
    mock_instance.get_session = AsyncMock(side_effect=get_session)
    mock_instance.get_sessions = AsyncMock(side_effect=get_sessions)
    mock_instance.get_session_by_name = AsyncMock(side_effect=get_session_by_name)
    mock_instance.create_session_with_name = AsyncMock(side_effect=create_session_with_name)
    mock_instance.update_session = AsyncMock(side_effect=update_session)
    mock_instance.delete_session = AsyncMock(side_effect=delete_session)
    mock_instance.get_messages_by_session = AsyncMock(side_effect=get_messages_by_session)
    mock_instance.get_message = AsyncMock(side_effect=get_message)
    mock_instance.add_knowledge = AsyncMock(side_effect=add_knowledge)
    mock_instance.get_knowledge = AsyncMock(side_effect=get_knowledge)
    mock_instance.get_knowledge_by_user = AsyncMock(side_effect=get_knowledge_by_user)
    mock_instance.update_knowledge = AsyncMock(side_effect=update_knowledge)
    mock_instance.delete_knowledge = AsyncMock(side_effect=delete_knowledge)
    
    # Mock additional services for messages API
    mock_service = MagicMock()
    
    # Mock async methods
    async def mock_add(messages, **kwargs):
        return {
            "status": "success",
            "data": {"message_ids": ["msg-1", "msg-2"]}
        }
    
    async def mock_get_messages_by_session(*args, **kwargs):
        return []
    
    async def mock_read(message_ids):
        # Return mock messages with all required fields including session_id
        mock_messages = []
        for i, msg_id in enumerate(message_ids):
            mock_messages.append({
                "id": msg_id,
                "session_id": "session-1",  # Use the test session ID
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Mock message content {i + 1}",
                "created_at": "2023-01-01T12:00:00Z",
                "updated_at": "2023-01-01T12:00:00Z"
            })



        return {
            "status": "success",
            "data": {"messages": mock_messages}
        }
    
    async def mock_update(message_ids, new_messages):
        return {
            "status": "success",
            "data": {"message_ids": message_ids}
        }
    
    async def mock_delete(message_ids):
        """Mock delete that returns 404 for non-existent messages."""
        if message_ids and message_ids[0] == "nonexistent-message-id":
            return {
                "status": "error",
                "code": 404,
                "message": "Some message IDs were not found",
                "errors": [{"field": "message_ids", "message": "Message not found"}]
            }
        return {
            "status": "success",
            "data": {"message_ids": message_ids}
        }
    
    mock_service.add = mock_add
    mock_service.get_messages_by_session = mock_get_messages_by_session
    mock_service.read = mock_read
    mock_service.update = mock_update
    mock_service.delete = mock_delete
    
    # Patch DatabaseService in all the places it's imported
    patches = [
        patch('memfuse_core.services.database_service.DatabaseService'),
        patch('memfuse_core.api.users.DatabaseService'),
        patch('memfuse_core.api.agents.DatabaseService'),
        patch('memfuse_core.api.sessions.DatabaseService'),
        patch('memfuse_core.api.messages.DatabaseService'),
        patch('memfuse_core.api.knowledge.DatabaseService'),
        patch('memfuse_core.api.api_keys.DatabaseService'),
        patch('memfuse_core.api.chunks.DatabaseService'),
    ]

    with patches[0] as mock_class, \
         patches[1] as mock_users_class, \
         patches[2] as mock_agents_class, \
         patches[3] as mock_sessions_class, \
         patches[4] as mock_messages_class, \
         patches[5] as mock_knowledge_class, \
         patches[6] as mock_api_keys_class, \
         patches[7] as mock_chunks_class:

        # Configure all mock classes to return the same mock instance
        for mock_cls in [mock_class, mock_users_class, mock_agents_class,
                        mock_sessions_class, mock_messages_class, mock_knowledge_class,
                        mock_api_keys_class, mock_chunks_class]:
            mock_cls.get_instance = AsyncMock(return_value=mock_instance)
        
        # Mock service factory - make async methods return awaitable
        with patch('memfuse_core.services.service_factory.ServiceFactory') as mock_factory, \
             patch('memfuse_core.api.messages.ServiceFactory', mock_factory):
            async def mock_get_buffer_service(*args, **kwargs):
                return mock_service
            
            async def mock_get_memory_service(*args, **kwargs):
                return mock_service
            
            mock_factory.get_buffer_service = mock_get_buffer_service
            mock_factory.get_memory_service = mock_get_memory_service
            
            yield mock_instance

# Contract tests focus on validating API behavior, JSON schemas, and HTTP status codes
# without testing business logic or relying on actual database operations. 