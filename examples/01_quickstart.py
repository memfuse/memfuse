"""
MemFuse Quickstart API Example

This example demonstrates how to use MemFuse with OpenAI for memory-enhanced conversations.
It showcases both direct API usage and best practices for production applications.

Features:
- Efficient session management
- Optimized memory retrieval
- Error handling and retry logic
- Clean separation of concerns
- Production-ready patterns
"""

import os
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv(override=True)

# Configuration
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 10
MAX_RETRIES = 3
RETRY_DELAY = 1.0


@dataclass
class MemFuseConfig:
    """Configuration for MemFuse client."""
    base_url: str = DEFAULT_BASE_URL
    api_key: Optional[str] = None
    timeout: int = DEFAULT_TIMEOUT
    max_retries: int = MAX_RETRIES
    retry_delay: float = RETRY_DELAY


@dataclass
class SessionContext:
    """Context information for a MemFuse session."""
    user_id: str
    agent_id: str
    session_id: str
    user_name: str
    agent_name: str
    session_name: str


class MemFuseError(Exception):
    """Base exception for MemFuse operations."""
    pass


class ConnectionError(MemFuseError):
    """Raised when unable to connect to MemFuse server."""
    pass


class APIError(MemFuseError):
    """Raised when API request fails."""
    pass


class MemFuseClient:
    """
    MemFuse client with improved error handling, retry logic, and performance.

    This client provides a clean interface to the MemFuse API with:
    - Automatic retry with exponential backoff
    - Connection pooling for better performance
    - Comprehensive error handling
    - Efficient session management
    """

    def __init__(self, config: Optional[MemFuseConfig] = None):
        """Initialize the MemFuse client.

        Args:
            config: Client configuration. If None, uses defaults from environment.
        """
        self.config = config or MemFuseConfig(
            api_key=os.environ.get("MEMFUSE_API_KEY")
        )

        # Initialize HTTP session with connection pooling
        self.session = requests.Session()
        self.session.timeout = self.config.timeout

        # Set authentication header if API key is provided
        if self.config.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.config.api_key}"

        # Add common headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "MemFuse-Client/1.0"
        })

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.session.close()

    def _check_server_health(self) -> bool:
        """Check if the MemFuse server is running and healthy."""
        try:
            response = self.session.get(
                f"{self.config.base_url}/api/v1/health",
                timeout=5  # Shorter timeout for health check
            )
            return response.status_code == 200
        except Exception:
            return False

    def _make_request_with_retry(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic and error handling."""
        if not self._check_server_health():
            raise ConnectionError(
                f"Cannot connect to MemFuse server at {self.config.base_url}. "
                "Please ensure the server is running with: poetry run memfuse-core"
            )

        url = f"{self.config.base_url}{endpoint}"
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                response = getattr(self.session, method.lower())(url, json=data)

                if response.status_code < 400:
                    return response.json()

                # Handle HTTP errors
                try:
                    error_data = response.json()
                    error_message = error_data.get(
                        "message", f"HTTP {response.status_code}")
                except (ValueError, requests.exceptions.JSONDecodeError):
                    error_message = f"HTTP {response.status_code} error"

                if response.status_code >= 500 and attempt < self.config.max_retries - 1:
                    # Retry on server errors
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                    continue

                raise APIError(f"API request failed: {error_message}")

            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                    continue
                break

        raise ConnectionError(
            f"Failed to connect after {self.config.max_retries} attempts: {last_exception}")

    def get_or_create_user(self, name: str, description: Optional[str] = None) -> str:
        """Get existing user or create new one. Returns user_id."""
        # Try to find existing user
        try:
            response = self._make_request_with_retry(
                "GET", f"/api/v1/users?name={name}")
            if response.get("status") == "success":
                users = response.get("data", {}).get("users", [])
                if users:
                    user_id = users[0]["id"]
                    print(f"✓ Found existing user: {name} (ID: {user_id})")
                    return user_id
        except APIError:
            pass  # User doesn't exist, will create below

        # Create new user
        user_data = {"name": name}
        if description:
            user_data["description"] = description
        else:
            user_data["description"] = "User created by MemFuse client"

        response = self._make_request_with_retry(
            "POST", "/api/v1/users", user_data)
        user_id = response["data"]["user"]["id"]
        print(f"✓ Created new user: {name} (ID: {user_id})")
        return user_id

    def get_or_create_agent(self, name: str, description: Optional[str] = None) -> str:
        """Get existing agent or create new one. Returns agent_id."""
        # Try to find existing agent
        try:
            response = self._make_request_with_retry(
                "GET", f"/api/v1/agents?name={name}")
            if response.get("status") == "success":
                agents = response.get("data", {}).get("agents", [])
                if agents:
                    agent_id = agents[0]["id"]
                    print(f"✓ Found existing agent: {name} (ID: {agent_id})")
                    return agent_id
        except APIError:
            pass  # Agent doesn't exist, will create below

        # Create new agent
        agent_data = {"name": name}
        if description:
            agent_data["description"] = description
        else:
            agent_data["description"] = "AI agent for memory-enhanced conversations"

        response = self._make_request_with_retry(
            "POST", "/api/v1/agents", agent_data)
        agent_id = response["data"]["agent"]["id"]
        print(f"✓ Created new agent: {name} (ID: {agent_id})")
        return agent_id

    def create_session(
        self,
        user_id: str,
        agent_id: str,
        name: Optional[str] = None
    ) -> str:
        """Create a new conversation session. Returns session_id."""
        import uuid

        # If name is provided, try to find existing session first
        if name:
            try:
                response = self._make_request_with_retry(
                    "GET", f"/api/v1/sessions?name={name}&user_id={user_id}")
                if response.get("status") == "success":
                    sessions = response.get("data", {}).get("sessions", [])
                    if sessions:
                        session = sessions[0]
                        print(f"✓ Found existing session: {session['name']} (ID: {session['id']})")
                        return session["id"]
            except Exception:
                pass  # Continue to create new session

        # Create new session with unique name if needed
        session_name = name
        if not session_name:
            session_name = f"session-{str(uuid.uuid4())[:8]}"
        else:
            # Add unique suffix to avoid conflicts
            session_name = f"{name}-{str(uuid.uuid4())[:8]}"

        session_data = {
            "user_id": user_id,
            "agent_id": agent_id,
            "name": session_name
        }

        response = self._make_request_with_retry(
            "POST", "/api/v1/sessions", session_data)

        if not response:
            raise APIError("Session creation returned None response")

        if response.get("status") != "success":
            error_msg = response.get("message", "Unknown error")
            raise APIError(f"Session creation failed: {error_msg}")

        if "data" not in response or not response["data"]:
            raise APIError(f"Session response missing 'data' field: {response}")

        if "session" not in response["data"]:
            raise APIError(f"Session response missing 'session' field: {response['data']}")

        session_id = response["data"]["session"]["id"]
        session_name = response["data"]["session"]["name"]
        print(f"✓ Created session: {session_name} (ID: {session_id})")
        return session_id

    def initialize_session(
        self,
        user_name: str,
        agent_name: str = "assistant",
        session_name: Optional[str] = None
    ) -> SessionContext:
        """Initialize a complete session context with user, agent, and session."""
        print(f"🚀 Initializing MemFuse session for user '{user_name}'...")

        # Create or get user and agent
        user_id = self.get_or_create_user(user_name)
        agent_id = self.get_or_create_agent(agent_name)

        # Create session
        session_id = self.create_session(user_id, agent_id, session_name)

        return SessionContext(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            user_name=user_name,
            agent_name=agent_name,
            session_name=session_name or f"{user_name}-{agent_name}-session"
        )

    def get_chat_history(
        self,
        session_id: str,
        limit: int = 10,
        include_buffer: bool = True
    ) -> List[Dict[str, str]]:
        """
        Get recent chat history for a session with optimized retrieval.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve
            include_buffer: Whether to include buffer messages (faster)

        Returns:
            List of messages in chronological order
        """
        all_messages = []

        if include_buffer:
            # Get from buffer first (faster)
            try:
                buffer_response = self._make_request_with_retry(
                    "GET",
                    f"/api/v1/sessions/{session_id}/messages"
                    f"?limit={limit}&sort_by=timestamp&order=desc&buffer_only=true"
                )
                buffer_messages = buffer_response.get(
                    "data", {}).get("messages", [])
                all_messages.extend(reversed(buffer_messages))
            except APIError:
                pass  # Fall back to database if buffer fails

        # Get additional messages from database if needed
        if len(all_messages) < limit:
            remaining = limit - len(all_messages)
            try:
                db_response = self._make_request_with_retry(
                    "GET",
                    f"/api/v1/sessions/{session_id}/messages"
                    f"?limit={remaining}&sort_by=timestamp&order=desc&buffer_only=false"
                )
                db_messages = db_response.get("data", {}).get("messages", [])
                # Prepend database messages (older messages first)
                all_messages = list(reversed(db_messages)) + all_messages
            except APIError:
                pass  # Continue with what we have

        # Convert to OpenAI format and limit to requested count
        formatted_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in all_messages[-limit:]
        ]

        return formatted_messages

    def query_memories(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        query: str,
        top_k: int = 5,
        include_messages: bool = True,
        include_knowledge: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query for relevant memories with enhanced parameters.

        Args:
            user_id: User identifier
            session_id: Session identifier
            agent_id: Agent identifier
            query: Search query
            top_k: Maximum number of results
            include_messages: Whether to include message memories
            include_knowledge: Whether to include knowledge memories

        Returns:
            List of relevant memory items
        """
        query_data = {
            "query": query,
            "session_id": session_id,
            "agent_id": agent_id,
            "top_k": top_k,
            "include_messages": include_messages,
            "include_knowledge": include_knowledge
        }

        response = self._make_request_with_retry(
            "POST",
            f"/api/v1/users/{user_id}/query",
            query_data
        )
        return response.get("data", {}).get("results", [])

    def add_messages(
        self,
        session_id: str,
        messages: List[Dict[str, str]]
    ) -> List[str]:
        """
        Add messages to a session with batch processing.

        Args:
            session_id: Session identifier
            messages: List of messages to add

        Returns:
            List of created message IDs
        """
        if not messages:
            return []

        response = self._make_request_with_retry(
            "POST",
            f"/api/v1/sessions/{session_id}/messages",
            {"messages": messages}
        )
        return response.get("data", {}).get("message_ids", [])


class PromptFormatter:
    """Helper class for formatting prompts and queries."""

    @staticmethod
    def messages_to_query(messages: List[Dict[str, str]]) -> str:
        """Convert messages to a search query string."""
        query_parts = []
        for message in messages:
            role = message.get("role", "unknown").capitalize()
            content = message.get("content", "")
            if content.strip():
                query_parts.append(f"{role}: {content}")
        return "\n".join(query_parts)

    @staticmethod
    def format_memory_item(item: Dict[str, Any]) -> str:
        """Format a memory item for display."""
        content = item.get("content", "N/A")
        mem_type = item.get("type", "unknown").upper()
        role = item.get("role")

        prefix = f"[{mem_type}"
        if role and mem_type == "MESSAGE":
            prefix += f" from {role.upper()}"
        prefix += "]"

        return f"{prefix}: {content}"


class PromptContext:
    """Context manager for composing prompts with memory integration."""

    def __init__(
        self,
        query_messages: List[Dict[str, str]],
        retrieved_memories: Optional[List[Dict[str, Any]]] = None,
        retrieved_chat_history: Optional[List[Dict[str, str]]] = None,
        max_chat_history: int = 10,
    ):
        self.query_messages = query_messages
        self.retrieved_memories = retrieved_memories or []
        self.retrieved_chat_history = retrieved_chat_history or []
        self.max_chat_history = max_chat_history

    @property
    def system_instruction(self) -> Dict[str, str]:
        """Get the system instruction from query messages."""
        if self.query_messages and self.query_messages[0].get("role") == "system":
            return self.query_messages[0]
        return {"role": "system", "content": "You are a helpful assistant with access to conversation memory."}

    @property
    def user_query(self) -> List[Dict[str, str]]:
        """Get user query messages (excluding system message if present)."""
        if self.query_messages and self.query_messages[0].get("role") == "system":
            return self.query_messages[1:]
        return self.query_messages

    @property
    def long_term_memory(self) -> List[Dict[str, Any]]:
        """Get long-term memory items (cross-session scope)."""
        return [
            item for item in self.retrieved_memories
            if item.get("metadata", {}).get("scope") == "cross_session"
        ]

    @property
    def short_term_memory(self) -> List[Dict[str, Any]]:
        """Get short-term memory items (in-session scope)."""
        return [
            item for item in self.retrieved_memories
            if item.get("metadata", {}).get("scope") == "in_session"
        ]

    @property
    def chat_history(self) -> List[Dict[str, str]]:
        """Get recent chat history, limited by max_chat_history."""
        if not self.retrieved_chat_history:
            return []
        return self.retrieved_chat_history[-self.max_chat_history:]

    def compose_for_openai(self) -> List[Dict[str, str]]:
        """Compose the final message list for OpenAI API with memory context."""
        messages: List[Dict[str, str]] = []

        # 1. System instruction
        messages.append(self.system_instruction)

        # 2. Long-term memory context
        if self.long_term_memory:
            lt_snippets = [
                PromptFormatter.format_memory_item(item)
                for item in self.long_term_memory
            ]

            lt_content = (
                "You have access to the following long-term memory from previous conversations. "
                "Use these insights to provide more personalized and contextual responses:\n\n"
                + "\n- ".join([""] + lt_snippets)
            )
            messages.append({"role": "system", "content": lt_content})

        # 3. Short-term memory context
        if self.short_term_memory:
            st_snippets = [
                PromptFormatter.format_memory_item(item)
                for item in self.short_term_memory
            ]

            st_content = (
                "Here are relevant notes from the current conversation context:\n\n"
                + "\n- ".join([""] + st_snippets)
            )
            messages.append({"role": "system", "content": st_content})

        # 4. Recent chat history
        if self.chat_history:
            messages.extend(self.chat_history)

        # 5. Current user query
        messages.extend(self.user_query)

        return messages


class MemoryEnhancedOpenAI:
    """
    OpenAI client wrapper that adds MemFuse memory functionality.

    This class wraps the OpenAI client to automatically:
    - Retrieve relevant memories before each request
    - Include chat history in the context
    - Store new conversations in MemFuse
    - Provide seamless memory-enhanced conversations
    """

    def __init__(
        self,
        memfuse_client: MemFuseClient,
        session_context: SessionContext,
        max_chat_history: int = 10,
        **openai_kwargs
    ):
        """
        Initialize the memory-enhanced OpenAI client.

        Args:
            memfuse_client: MemFuse client instance
            session_context: Session context with user, agent, and session info
            max_chat_history: Maximum number of chat history messages to include
            **openai_kwargs: Arguments passed to OpenAI client
        """
        self.memfuse_client = memfuse_client
        self.session_context = session_context
        self.max_chat_history = max_chat_history

        # Create the OpenAI client
        self.openai_client = OpenAI(**openai_kwargs)

        # Wrap the chat completions create method
        self._original_create = self.openai_client.chat.completions.create
        self.openai_client.chat.completions.create = self._memory_enhanced_create

    def _memory_enhanced_create(self, **kwargs):
        """Enhanced version of chat.completions.create with memory integration."""
        query_messages = kwargs.get("messages", [])

        try:
            # 1. Get recent chat history
            chat_history = self.memfuse_client.get_chat_history(
                self.session_context.session_id,
                limit=self.max_chat_history
            )

            # 2. Query for relevant memories
            query_string = PromptFormatter.messages_to_query(
                chat_history + query_messages)
            retrieved_memories = self.memfuse_client.query_memories(
                self.session_context.user_id,
                self.session_context.session_id,
                self.session_context.agent_id,
                query_string,
                top_k=5
            )

            # 3. Compose enhanced prompt with memory context
            prompt_context = PromptContext(
                query_messages=query_messages,
                retrieved_memories=retrieved_memories,
                retrieved_chat_history=chat_history,
                max_chat_history=self.max_chat_history,
            )

            enhanced_messages = prompt_context.compose_for_openai()

            # Log memory integration info
            print(f"💭 Enhanced prompt with {len(retrieved_memories)} memories "
                  f"and {len(chat_history)} history messages")

            # 4. Update messages with memory context
            kwargs["messages"] = enhanced_messages

            # 5. Call OpenAI API
            response = self._original_create(**kwargs)

            # 6. Store conversation in MemFuse
            self._store_conversation(query_messages, response)

            return response

        except Exception as e:
            print(f"⚠️  Memory integration failed: {e}")
            print("🔄 Falling back to direct OpenAI call...")
            # Fallback to original call without memory
            return self._original_create(**kwargs)

    def _store_conversation(self, query_messages: List[Dict[str, str]], response) -> None:
        """Store the conversation in MemFuse."""
        try:
            messages_to_store = list(query_messages)

            # Add assistant response if available
            if response and response.choices and response.choices[0].message:
                assistant_message = response.choices[0].message
                messages_to_store.append({
                    "role": assistant_message.role,
                    "content": assistant_message.content
                })

            if messages_to_store:
                message_ids = self.memfuse_client.add_messages(
                    self.session_context.session_id,
                    messages_to_store
                )
                print(
                    f"💾 Stored {len(messages_to_store)} messages (IDs: {len(message_ids)})")

        except Exception as e:
            print(f"⚠️  Failed to store conversation: {e}")

    @property
    def chat(self):
        """Expose the chat completions interface."""
        return self.openai_client.chat


def demonstrate_basic_usage():
    """Demonstrate basic MemFuse usage with optimized patterns."""
    print("🚀 MemFuse Quickstart - API Version")
    print("=" * 50)

    # Initialize MemFuse client with context manager for proper cleanup
    with MemFuseClient() as memfuse_client:
        # Initialize session context
        session_context = memfuse_client.initialize_session(
            user_name="alice",
            agent_name="assistant",
            session_name="space-exploration-chat"
        )

        print("\n📋 Session Details:")
        print(
            f"   User: {session_context.user_name} ({session_context.user_id})")
        print(
            f"   Agent: {session_context.agent_name} ({session_context.agent_id})")
        print(
            f"   Session: {session_context.session_name} ({session_context.session_id})")

        # Create memory-enhanced OpenAI client
        client = MemoryEnhancedOpenAI(
            memfuse_client=memfuse_client,
            session_context=session_context,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )

        # First conversation
        print("\n🤖 First Question:")
        response1 = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": "I'm working on a project about space exploration. Can you tell me something interesting about Mars?"
            }],
        )
        print(f"Assistant: {response1.choices[0].message.content}")

        # Follow-up question to test memory
        print("\n🤖 Follow-up Question (testing memory):")
        response2 = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": "What would be the biggest challenges for humans living on that planet?"
            }],
        )
        print(f"Assistant: {response2.choices[0].message.content}")

        print("\n✅ MemFuse Quickstart Complete!")
        print(
            "   The assistant remembered we were discussing Mars from the previous message.")
        print("   All conversations are stored in MemFuse for future reference.")


def main():
    """Main entry point for the quickstart API example."""
    try:
        demonstrate_basic_usage()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure the MemFuse server is running: poetry run memfuse-core")


if __name__ == "__main__":
    main()
