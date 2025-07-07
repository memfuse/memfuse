"""
Configuration and fixtures for contract tests.
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add src to path so we can import the modules
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

@pytest.fixture(autouse=True)
def mock_database_service():
    """Mock the database service to avoid threading issues and focus on contract validation."""
    
    # In-memory storage for the mock
    mock_users = {}
    user_counter = 0
    mock_agents = {}
    agent_counter = 0
    
    def create_user(name, description=None):
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
    
    def get_user(user_id):
        return mock_users.get(user_id)
    
    def get_all_users():
        return list(mock_users.values())
    
    def get_user_by_name(name):
        for user in mock_users.values():
            if user["name"] == name:
                return user
        return None
    
    def update_user(user_id, name=None, description=None):
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
    
    def delete_user(user_id):
        if user_id in mock_users:
            del mock_users[user_id]
            return True
        return False
    
    def create_agent(name, description=None):
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

    def get_agent(agent_id):
        return mock_agents.get(agent_id)

    def get_all_agents():
        return list(mock_agents.values())

    def get_agent_by_name(name):
        for agent in mock_agents.values():
            if agent["name"] == name:
                return agent
        return None

    def update_agent(agent_id, name=None, description=None):
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

    def delete_agent(agent_id):
        if agent_id in mock_agents:
            del mock_agents[agent_id]
            return True
        return False
    
    # Create a mock instance
    mock_instance = MagicMock()
    mock_instance.create_user.side_effect = create_user
    mock_instance.get_user.side_effect = get_user
    mock_instance.get_all_users.side_effect = get_all_users
    mock_instance.get_user_by_name.side_effect = get_user_by_name
    mock_instance.update_user.side_effect = update_user
    mock_instance.delete_user.side_effect = delete_user
    mock_instance.create_agent.side_effect = create_agent
    mock_instance.get_agent.side_effect = get_agent
    mock_instance.get_all_agents.side_effect = get_all_agents
    mock_instance.get_agent_by_name.side_effect = get_agent_by_name
    mock_instance.update_agent.side_effect = update_agent
    mock_instance.delete_agent.side_effect = delete_agent
    
    with patch('memfuse_core.services.database_service.DatabaseService') as mock_class:
        mock_class.get_instance.return_value = mock_instance
        yield mock_instance

# Contract tests focus on validating API behavior, JSON schemas, and HTTP status codes
# without testing business logic or relying on actual database operations. 