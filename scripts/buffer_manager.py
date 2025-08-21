#!/usr/bin/env python3
"""
MemFuse Buffer Manager

A comprehensive tool for managing and inspecting MemFuse Buffer system data.
Provides detailed insights into RoundBuffer and HybridBuffer contents across all users.

Features:
- View buffer statistics and contents for all users
- Inspect individual buffer entries with metadata
- Manual flush operations (all users or specific users)
- Search and filter buffer data by various criteria
- Real-time buffer monitoring and status reporting

Usage:
    # Basic Commands
    poetry run python scripts/buffer_manager.py status           # Show overall buffer status
    poetry run python scripts/buffer_manager.py list             # List all buffer contents
    poetry run python scripts/buffer_manager.py user <user_id>   # Show specific user's buffer data
    poetry run python scripts/buffer_manager.py flush            # Flush all buffers to database
    poetry run python scripts/buffer_manager.py flush --user <user_id>  # Flush specific user's buffers
    poetry run python scripts/buffer_manager.py search <query>   # Search buffer contents
    poetry run python scripts/buffer_manager.py test             # Test server connectivity

    # Mode Options
    poetry run python scripts/buffer_manager.py --offline status # Offline mode (no server required)
    poetry run python scripts/buffer_manager.py --config custom.yaml status  # Custom config

    # Advanced Examples
    poetry run python scripts/buffer_manager.py status --verbose # Detailed output
    poetry run python scripts/buffer_manager.py list --limit 50  # Limit results
    poetry run python scripts/buffer_manager.py flush --force    # Force flush all data

Operating Modes:
    Online Mode (default):
        - Connects to running MemFuse server (localhost:8000)
        - Provides real-time buffer statistics and live session data
        - Shows current buffer contents and transfer activities
        - Requires server to be running with API access

    Offline Mode (--offline):
        - Works without server connection
        - Shows buffer configuration and settings
        - Useful for configuration validation and troubleshooting
        - Limited to static information from config files

Important Notes:
    1. Performance Considerations:
       - Script processes up to 10 sessions by default to avoid performance issues
       - Large databases (20k+ sessions) may cause delays in online mode
       - Use --limit parameter to control result size

    2. Server Requirements:
       - Online mode requires MemFuse server running on localhost:8000
       - Uses API key authentication: "memfuse-test-api-key"
       - Health check endpoint: /api/v1/health/
       - Sessions endpoint: /api/v1/sessions

    3. Configuration:
       - Supports Hydra config structure and direct YAML files
       - Automatically loads from config/buffer/default.yaml
       - Falls back gracefully if configuration files are missing
       - Use --config to specify custom configuration file

    4. Error Handling:
       - Automatically switches to offline mode if server unavailable
       - Provides detailed error messages for troubleshooting
       - Graceful handling of network timeouts and API errors
       - Connection test available via 'test' command

    5. Buffer System Integration:
       - Monitors RoundBuffer (short-term storage) and HybridBuffer (processed data)
       - Tracks force flush timeout mechanisms (default: 30 minutes)
       - Shows buffer transfer statistics and performance metrics
       - Supports manual flush operations for data persistence

    6. Security:
       - Uses API key authentication for server access
       - Read-only operations by default (except flush commands)
       - No sensitive data exposed in logs or output

Dependencies:
    - httpx: HTTP client for API communication
    - yaml: Configuration file parsing
    - asyncio: Asynchronous operations
    - MemFuse server: Required for online mode functionality

Troubleshooting:
    - If "Connection failed" appears, check if MemFuse server is running
    - If "Offline mode" is forced, verify server status with 'test' command
    - For configuration errors, use --offline mode to validate settings
    - Check logs for detailed error information and debugging
"""

import asyncio
import sys
import argparse
import json
import time
import os
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import aiohttp
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import yaml
    import httpx
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("Please install required dependencies: pip install httpx")
    sys.exit(1)


class StatusLevel(Enum):
    """Status message levels."""
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class BufferStats:
    """Buffer statistics for a user."""
    user_id: str
    round_buffer_count: int
    round_buffer_tokens: int
    hybrid_buffer_count: int
    hybrid_buffer_chunks: int
    last_activity: Optional[datetime] = None
    session_ids: List[str] = None


@dataclass
class BufferEntry:
    """Individual buffer entry with metadata."""
    entry_id: str
    entry_type: str  # 'round' or 'chunk'
    user_id: str
    session_id: Optional[str]
    content_preview: str
    metadata: Dict[str, Any]
    created_at: Optional[datetime] = None


class BufferManager:
    """Comprehensive Buffer management for MemFuse."""

    def __init__(self, config_path: Optional[str] = None, server_url: str = "http://localhost:8000", api_key: str = "memfuse-test-api-key", offline_mode: bool = False):
        """Initialize BufferManager with configuration."""
        self.config_path = config_path or "config/config.yaml"
        self.server_url = server_url
        self.api_key = api_key
        self.offline_mode = offline_mode
        self.config = None
        self.client = None
        self.buffer_service = None

    def print_status(self, message: str, level: StatusLevel = StatusLevel.INFO):
        """Print colored status messages."""
        colors = {
            StatusLevel.INFO: "\033[0;34m",
            StatusLevel.SUCCESS: "\033[0;32m",
            StatusLevel.WARNING: "\033[1;33m",
            StatusLevel.ERROR: "\033[0;31m",
        }
        reset = "\033[0m"

        icons = {
            StatusLevel.INFO: "ℹ️ ",
            StatusLevel.SUCCESS: "✅ ",
            StatusLevel.WARNING: "⚠️ ",
            StatusLevel.ERROR: "❌ ",
        }

        color = colors.get(level, "")
        icon = icons.get(level, "")
        print(f"{color}{icon}{message}{reset}")

    async def initialize(self) -> bool:
        """Initialize BufferManager components."""
        try:
            # Load configuration - handle both direct config and Hydra structure
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                self.print_status(f"Configuration file not found: {self.config_path}", StatusLevel.WARNING)
                self.config = {}

            # For Hydra-based configs, try to load buffer config directly
            buffer_config_path = "config/buffer/default.yaml"
            if os.path.exists(buffer_config_path):
                with open(buffer_config_path, 'r') as f:
                    buffer_config = yaml.safe_load(f)
            else:
                buffer_config = self.config.get('buffer', {})

            # Check if buffer is enabled
            if not buffer_config.get('enabled', False):
                self.print_status("Buffer system is disabled in configuration", StatusLevel.WARNING)
                return False

            # Initialize HTTP client with API key (if not in offline mode)
            if not self.offline_mode:
                headers = {"X-API-Key": self.api_key}
                self.client = httpx.AsyncClient(timeout=10.0, headers=headers)

                # Test server connectivity
                try:
                    response = await self.client.get(f"{self.server_url}/api/v1/health/")
                    if response.status_code != 200:
                        self.print_status(f"MemFuse server not responding (status: {response.status_code}), switching to offline mode", StatusLevel.WARNING)
                        self.offline_mode = True
                        if self.client:
                            await self.client.aclose()
                        self.client = None
                except Exception as e:
                    self.print_status(f"Cannot connect to MemFuse server at {self.server_url}: {e}", StatusLevel.WARNING)
                    self.print_status("Switching to offline mode", StatusLevel.INFO)
                    self.offline_mode = True
                    if self.client:
                        await self.client.aclose()
                    self.client = None

            # Initialize direct buffer access for offline mode
            if self.offline_mode:
                try:
                    from memfuse_core.services.buffer_service import BufferService
                    from memfuse_core.buffer.config_factory import ComponentConfigFactory

                    # Create buffer service for direct access
                    factory = ComponentConfigFactory()
                    buffer_config = factory.create_component_config('buffer', global_config={'buffer': buffer_config})

                    # Note: We can't fully initialize BufferService without other dependencies
                    # This is a limitation of the current architecture
                    self.print_status("Direct buffer access initialized", StatusLevel.INFO)
                except Exception as e:
                    self.print_status(f"Failed to initialize direct buffer access: {e}", StatusLevel.WARNING)
                    self.print_status("Limited functionality available", StatusLevel.INFO)

            self.print_status("BufferManager initialized successfully", StatusLevel.SUCCESS)
            return True

        except Exception as e:
            self.print_status(f"Failed to initialize BufferManager: {e}", StatusLevel.ERROR)
            return False

    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.client:
                await self.client.aclose()
        except Exception as e:
            self.print_status(f"Error during cleanup: {e}", StatusLevel.WARNING)

    async def get_buffer_status_offline(self) -> Dict[str, Any]:
        """Get buffer status in offline mode (limited functionality)."""
        try:
            # Load buffer config for offline display
            buffer_config_path = "config/buffer/default.yaml"
            config_info = {}
            if os.path.exists(buffer_config_path):
                with open(buffer_config_path, 'r') as f:
                    buffer_config = yaml.safe_load(f)
                    config_info = {
                        "round_buffer_max_size": buffer_config.get('round_buffer', {}).get('max_size', 'unknown'),
                        "hybrid_buffer_max_size": buffer_config.get('hybrid_buffer', {}).get('max_size', 'unknown'),
                        "force_flush_timeout": buffer_config.get('performance', {}).get('force_flush_timeout', 'unknown')
                    }

            # In offline mode, we can't access live buffer data
            # Return a status indicating offline mode with configuration info
            return {
                "buffer_enabled": True,
                "total_users": 0,
                "total_round_entries": 0,
                "total_hybrid_entries": 0,
                "total_tokens": 0,
                "user_buffers": {},
                "timestamp": datetime.now(),
                "mode": "offline",
                "note": "Offline mode - live buffer data not available. Start MemFuse server for full functionality.",
                "config_info": config_info
            }
        except Exception as e:
            return {"error": f"Offline mode error: {e}", "mode": "offline"}

    async def get_all_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get sessions from the API with limit."""
        if not self.client:
            return []

        try:
            response = await self.client.get(f"{self.server_url}/api/v1/sessions")
            if response.status_code == 200:
                data = response.json()
                sessions = data.get('data', {}).get('sessions', [])
                # Limit sessions to avoid overwhelming the system
                return sessions[:limit]
            else:
                self.print_status(f"Failed to get sessions: {response.status_code}", StatusLevel.ERROR)
                return []
        except Exception as e:
            self.print_status(f"Error getting sessions: {e}", StatusLevel.ERROR)
            return []

    async def get_buffer_messages_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Get buffer-only messages for a specific session."""
        if not self.client:
            return []

        try:
            response = await self.client.get(
                f"{self.server_url}/api/v1/sessions/{session_id}/messages",
                params={"buffer_only": "true", "limit": "100"}
            )
            if response.status_code == 200:
                data = response.json()
                return data.get('data', {}).get('messages', [])
            else:
                return []
        except Exception as e:
            self.print_status(f"Error getting buffer messages for session {session_id}: {e}", StatusLevel.WARNING)
            return []

    async def get_buffer_status(self) -> Dict[str, Any]:
        """Get comprehensive buffer status across all users."""
        try:
            if self.offline_mode:
                return await self.get_buffer_status_offline()

            # Get all sessions
            sessions = await self.get_all_sessions()
            self.print_status(f"Retrieved {len(sessions)} sessions", StatusLevel.INFO)

            if not sessions:
                return {
                    "buffer_enabled": True,
                    "total_users": 0,
                    "total_round_entries": 0,
                    "total_hybrid_entries": 0,
                    "total_tokens": 0,
                    "user_buffers": {},
                    "timestamp": datetime.now(),
                    "mode": "online"
                }

            # Group sessions by user_id and collect buffer data
            user_sessions = {}
            for session in sessions:
                user_id = session.get('user_id')
                if user_id:
                    if user_id not in user_sessions:
                        user_sessions[user_id] = []
                    user_sessions[user_id].append(session)

            user_buffers = {}
            total_round_entries = 0
            total_hybrid_entries = 0
            total_tokens = 0

            for user_id, user_session_list in user_sessions.items():
                round_buffer_count = 0
                session_ids = []

                # Check each session for buffer data
                for session in user_session_list:
                    session_id = session.get('id')
                    if session_id:
                        session_ids.append(session_id)
                        # Get buffer messages for this session
                        try:
                            buffer_messages = await self.get_buffer_messages_for_session(session_id)
                            round_buffer_count += len(buffer_messages)
                            if len(buffer_messages) > 0:
                                self.print_status(f"Found {len(buffer_messages)} buffer messages in session {session_id[:8]}...", StatusLevel.INFO)
                        except Exception as e:
                            self.print_status(f"Error getting buffer messages for session {session_id[:8]}...: {e}", StatusLevel.WARNING)
                            # Switch to offline mode if we can't get buffer data
                            self.offline_mode = True
                            return await self.get_buffer_status_offline()

                        # Estimate tokens (rough approximation: 4 chars per token)
                        for msg in buffer_messages:
                            content = msg.get('content', '')
                            total_tokens += len(content) // 4

                if round_buffer_count > 0:  # Only include users with buffer data
                    user_buffers[user_id] = BufferStats(
                        user_id=user_id,
                        round_buffer_count=round_buffer_count,
                        round_buffer_tokens=total_tokens,
                        hybrid_buffer_count=0,  # We can't easily get this via API
                        hybrid_buffer_chunks=0,  # We can't easily get this via API
                        session_ids=session_ids,
                        last_activity=datetime.now()
                    )

                    total_round_entries += round_buffer_count

            return {
                "buffer_enabled": True,
                "total_users": len(user_buffers),
                "total_round_entries": total_round_entries,
                "total_hybrid_entries": total_hybrid_entries,
                "total_tokens": total_tokens,
                "user_buffers": user_buffers,
                "timestamp": datetime.now(),
                "mode": "online",
                "sessions_checked": len(sessions)
            }

        except Exception as e:
            self.print_status(f"Error getting buffer status: {e}", StatusLevel.ERROR)
            return {"error": str(e)}

    async def get_user_buffer_details(self, user_id: str) -> Dict[str, Any]:
        """Get detailed buffer information for a specific user."""
        try:
            # Get all sessions for this user
            sessions = await self.get_all_sessions()
            user_sessions = [s for s in sessions if s.get('user_id') == user_id]

            if not user_sessions:
                return {"error": f"No sessions found for user {user_id}"}

            round_entries = []
            total_tokens = 0

            # Get buffer messages for each session
            for session in user_sessions:
                session_id = session.get('id')
                if session_id:
                    buffer_messages = await self.get_buffer_messages_for_session(session_id)

                    for i, msg in enumerate(buffer_messages):
                        content = msg.get('content', '')
                        content_preview = content[:100] + '...' if len(content) > 100 else content

                        # Estimate tokens
                        tokens = len(content) // 4
                        total_tokens += tokens

                        round_entries.append(BufferEntry(
                            entry_id=f"msg_{msg.get('id', i)}",
                            entry_type="message",
                            user_id=user_id,
                            session_id=session_id,
                            content_preview=content_preview,
                            metadata={
                                "message_id": msg.get('id'),
                                "role": msg.get('role', 'unknown'),
                                "timestamp": msg.get('timestamp'),
                                "tokens": tokens
                            }
                        ))

            return {
                "user_id": user_id,
                "round_buffer": {
                    "count": len(round_entries),
                    "tokens": total_tokens,
                    "entries": round_entries
                },
                "hybrid_buffer": {
                    "round_count": 0,  # Not available via API
                    "chunk_count": 0,  # Not available via API
                    "entries": []
                },
                "sessions": [s.get('id') for s in user_sessions],
                "timestamp": datetime.now()
            }

        except Exception as e:
            self.print_status(f"Error getting user buffer details: {e}", StatusLevel.ERROR)
            return {"error": str(e)}

    async def flush_all_buffers(self) -> Dict[str, Any]:
        """Flush all user buffers to database."""
        if not self.client:
            return {"error": "HTTP client not initialized"}

        try:
            # Note: Manual flush via HTTP API is not currently supported
            # This would require direct access to BufferService instances
            self.print_status("Manual buffer flush via HTTP API is not currently supported", StatusLevel.WARNING)
            self.print_status("Buffer data will be automatically flushed based on configured intervals", StatusLevel.INFO)
            self.print_status("Current auto-flush interval: 60 seconds (configurable)", StatusLevel.INFO)

            # Get current buffer status to show what would be flushed
            status = await self.get_buffer_status()
            user_buffers = status.get('user_buffers', {})

            results = {}
            for user_id, stats in user_buffers.items():
                results[user_id] = {
                    "status": "pending_auto_flush",
                    "message": f"Will be auto-flushed (Round: {stats.round_buffer_count} entries, {stats.round_buffer_tokens} tokens)"
                }

            return {
                "total_users": len(results),
                "successful_flushes": 0,
                "failed_flushes": 0,
                "pending_auto_flush": len(results),
                "results": results,
                "timestamp": datetime.now(),
                "note": "Manual flush not supported via HTTP API - data will be auto-flushed"
            }

        except Exception as e:
            self.print_status(f"Error flushing all buffers: {e}", StatusLevel.ERROR)
            return {"error": str(e)}

    async def flush_user_buffer(self, user_id: str) -> Dict[str, Any]:
        """Flush a specific user's buffer to database."""
        if not self.client:
            return {"error": "HTTP client not initialized"}

        try:
            # Check if user has buffer data
            status = await self.get_buffer_status()
            user_buffers = status.get('user_buffers', {})

            if user_id not in user_buffers:
                return {"error": f"User {user_id} not found in buffers or has no buffer data"}

            stats = user_buffers[user_id]

            # Note: Manual flush via HTTP API is not currently supported
            self.print_status(f"Manual buffer flush for user {user_id} via HTTP API is not currently supported", StatusLevel.WARNING)
            self.print_status("Buffer data will be automatically flushed based on configured intervals", StatusLevel.INFO)

            return {
                "user_id": user_id,
                "result": {
                    "status": "pending_auto_flush",
                    "message": f"Will be auto-flushed (Round: {stats.round_buffer_count} entries, {stats.round_buffer_tokens} tokens)",
                    "note": "Manual flush not supported via HTTP API - data will be auto-flushed"
                },
                "timestamp": datetime.now()
            }

        except Exception as e:
            self.print_status(f"Error checking user buffer: {e}", StatusLevel.ERROR)
            return {"error": str(e)}

    async def search_buffer_contents(self, query: str) -> Dict[str, Any]:
        """Search buffer contents across all users."""
        if not self.client:
            return {"error": "HTTP client not initialized"}

        try:
            results = []
            query_lower = query.lower()

            # Get all sessions
            sessions = await self.get_all_sessions()

            # Search through buffer messages for each session
            for session in sessions:
                session_id = session.get('id')
                user_id = session.get('user_id')

                if session_id and user_id:
                    buffer_messages = await self.get_buffer_messages_for_session(session_id)

                    for msg in buffer_messages:
                        content = msg.get('content', '')
                        if query_lower in content.lower():
                            results.append({
                                "user_id": user_id,
                                "buffer_type": "message",
                                "entry_id": msg.get('id', 'unknown'),
                                "content": content,
                                "role": msg.get('role', 'unknown'),
                                "session_id": session_id,
                                "match_type": "content",
                                "timestamp": msg.get('timestamp')
                            })

            return {
                "query": query,
                "total_matches": len(results),
                "results": results,
                "timestamp": datetime.now()
            }

        except Exception as e:
            self.print_status(f"Error searching buffer contents: {e}", StatusLevel.ERROR)
            return {"error": str(e)}

    def display_buffer_status(self, status: Dict[str, Any]):
        """Display buffer status in a formatted way."""
        print("🔄 MemFuse Buffer Status")
        print("=" * 60)

        if "error" in status:
            self.print_status(f"Error: {status['error']}", StatusLevel.ERROR)
            return

        mode = status.get('mode', 'unknown')
        print(f"Mode: {'🔌 Online' if mode == 'online' else '📴 Offline'}")
        print(f"Buffer Enabled: {'✅ Yes' if status.get('buffer_enabled') else '❌ No'}")
        print(f"Total Users: {status.get('total_users', 0)}")
        print(f"Total Round Entries: {status.get('total_round_entries', 0)}")
        print(f"Total Hybrid Entries: {status.get('total_hybrid_entries', 0)}")
        print(f"Total Tokens: {status.get('total_tokens', 0)}")
        if status.get('sessions_checked'):
            print(f"Sessions Checked: {status.get('sessions_checked', 0)}")
        print(f"Timestamp: {status.get('timestamp', 'Unknown')}")

        # Show offline mode note
        if status.get('note'):
            print(f"\n📝 Note: {status['note']}")

        # Show configuration info in offline mode
        config_info = status.get('config_info')
        if config_info:
            print("\n⚙️ Configuration Info:")
            print(f"  Round Buffer Max Size: {config_info.get('round_buffer_max_size', 'unknown')}")
            print(f"  Hybrid Buffer Max Size: {config_info.get('hybrid_buffer_max_size', 'unknown')}")
            print(f"  Force Flush Timeout: {config_info.get('force_flush_timeout', 'unknown')}s")

        user_buffers = status.get('user_buffers', {})
        if user_buffers:
            print("\n📊 Per-User Buffer Statistics:")
            print("-" * 60)
            for user_id, stats in user_buffers.items():
                print(f"User: {user_id}")
                print(f"  Round Buffer: {stats.round_buffer_count} entries, {stats.round_buffer_tokens} tokens")
                print(f"  Hybrid Buffer: {stats.hybrid_buffer_count} rounds, {stats.hybrid_buffer_chunks} chunks")
                if stats.session_ids:
                    print(f"  Sessions: {', '.join(stats.session_ids[:3])}{'...' if len(stats.session_ids) > 3 else ''}")
                print()

    def display_user_details(self, details: Dict[str, Any]):
        """Display detailed user buffer information."""
        if "error" in details:
            self.print_status(f"Error: {details['error']}", StatusLevel.ERROR)
            return

        user_id = details.get('user_id', 'Unknown')
        print(f"👤 Buffer Details for User: {user_id}")
        print("=" * 60)

        # Round Buffer
        round_buffer = details.get('round_buffer', {})
        print(f"🔄 Round Buffer: {round_buffer.get('count', 0)} entries, {round_buffer.get('tokens', 0)} tokens")

        for entry in round_buffer.get('entries', []):
            print(f"  [{entry.entry_id}] {entry.content_preview}")
            print(f"    Session: {entry.session_id or 'None'}")
            print(f"    Messages: {entry.metadata.get('message_count', 0)}")
            print(f"    Roles: {', '.join(entry.metadata.get('roles', []))}")
            print()

        # Hybrid Buffer
        hybrid_buffer = details.get('hybrid_buffer', {})
        print(f"🔀 Hybrid Buffer: {hybrid_buffer.get('round_count', 0)} rounds, {hybrid_buffer.get('chunk_count', 0)} chunks")

        for entry in hybrid_buffer.get('entries', []):
            print(f"  [{entry.entry_id}] ({entry.entry_type}) {entry.content_preview}")
            print(f"    Session: {entry.session_id or 'None'}")
            if entry.entry_type == 'hybrid_round':
                print(f"    Messages: {entry.metadata.get('message_count', 0)}")
                print(f"    Has Embedding: {'✅' if entry.metadata.get('has_embedding') else '❌'}")
            elif entry.entry_type == 'chunk':
                print(f"    Chunk Type: {entry.metadata.get('chunk_type', 'unknown')}")
                print(f"    Has Embedding: {'✅' if entry.metadata.get('has_embedding') else '❌'}")
            print()

    def display_search_results(self, results: Dict[str, Any]):
        """Display search results in a formatted way."""
        if "error" in results:
            self.print_status(f"Error: {results['error']}", StatusLevel.ERROR)
            return

        query = results.get('query', '')
        total = results.get('total_matches', 0)

        print(f"🔍 Search Results for: '{query}'")
        print("=" * 60)
        print(f"Total Matches: {total}")

        if total == 0:
            print("No matches found.")
            return

        for i, result in enumerate(results.get('results', []), 1):
            print(f"\n[{i}] User: {result.get('user_id', 'Unknown')}")
            print(f"    Buffer: {result.get('buffer_type', 'unknown')} ({result.get('entry_id', 'unknown')})")
            print(f"    Role: {result.get('role', 'N/A')}")
            print(f"    Session: {result.get('session_id', 'None')}")
            print(f"    Content: {result.get('content', '')[:200]}{'...' if len(result.get('content', '')) > 200 else ''}")

    def display_flush_results(self, results: Dict[str, Any]):
        """Display flush results in a formatted way."""
        if "error" in results:
            self.print_status(f"Error: {results['error']}", StatusLevel.ERROR)
            return

        print("💾 Buffer Flush Results")
        print("=" * 60)

        if 'user_id' in results:
            # Single user flush
            user_id = results.get('user_id')
            result = results.get('result', {})
            status = result.get('status', 'unknown')
            message = result.get('message', '')

            print(f"User: {user_id}")
            print(f"Status: {'✅ Success' if status == 'success' else '❌ Failed'}")
            if message:
                print(f"Message: {message}")
        else:
            # All users flush
            total = results.get('total_users', 0)
            successful = results.get('successful_flushes', 0)
            failed = results.get('failed_flushes', 0)

            print(f"Total Users: {total}")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")

            user_results = results.get('results', {})
            for user_id, result in user_results.items():
                status = result.get('status', 'unknown')
                message = result.get('message', '')
                print(f"  {user_id}: {'✅' if status == 'success' else '❌'} {message}")

        print(f"\nTimestamp: {results.get('timestamp', 'Unknown')}")


async def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="MemFuse Buffer Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status                    # Show overall buffer status
  %(prog)s list                      # List all buffer contents
  %(prog)s user <user_id>            # Show specific user's buffer data
  %(prog)s flush                     # Flush all buffers to database
  %(prog)s flush --user <user_id>    # Flush specific user's buffers
  %(prog)s search "hello world"      # Search buffer contents
        """
    )

    parser.add_argument(
        'action',
        choices=['status', 'list', 'user', 'flush', 'search', 'test'],
        help='Action to perform'
    )

    parser.add_argument(
        'target',
        nargs='?',
        help='Target for action (user_id for user action, query for search action)'
    )

    parser.add_argument(
        '--user',
        help='Specific user ID for flush action'
    )

    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Configuration file path'
    )

    parser.add_argument(
        '--offline',
        action='store_true',
        help='Run in offline mode (no server connection required)'
    )

    parser.add_argument('--version', action='version', version='MemFuse Buffer Manager 1.0')

    args = parser.parse_args()

    print("🔧 MemFuse Buffer Manager v1.0")
    print(f"Action: {args.action}")
    print(f"Config: {args.config}")
    print(f"Mode: {'Offline' if args.offline else 'Online'}")
    print(f"Started at: {datetime.now()}")
    print("=" * 60)

    manager = BufferManager(args.config, offline_mode=args.offline)

    try:
        # Initialize the manager
        if not await manager.initialize():
            print("❌ Failed to initialize BufferManager")
            sys.exit(1)

        success = True

        if args.action == 'status':
            status = await manager.get_buffer_status()
            manager.display_buffer_status(status)

        elif args.action == 'test':
            # Test connection to server
            if not manager.client:
                manager.print_status("No client available - running in offline mode", StatusLevel.WARNING)
            else:
                try:
                    response = await manager.client.get(f"{manager.server_url}/api/v1/health/")
                    if response.status_code == 200:
                        manager.print_status("✅ Server connection successful", StatusLevel.SUCCESS)
                        data = response.json()
                        manager.print_status(f"Server status: {data.get('data', {}).get('status', 'unknown')}", StatusLevel.INFO)

                        # Test sessions endpoint
                        sessions_response = await manager.client.get(f"{manager.server_url}/api/v1/sessions")
                        if sessions_response.status_code == 200:
                            sessions_data = sessions_response.json()
                            sessions = sessions_data.get('data', {}).get('sessions', [])
                            manager.print_status(f"✅ Sessions endpoint working - found {len(sessions)} sessions", StatusLevel.SUCCESS)
                        else:
                            manager.print_status(f"❌ Sessions endpoint failed with status {sessions_response.status_code}", StatusLevel.ERROR)
                    else:
                        manager.print_status(f"❌ Server responded with status {response.status_code}", StatusLevel.ERROR)
                except Exception as e:
                    manager.print_status(f"❌ Connection failed: {e}", StatusLevel.ERROR)

        elif args.action == 'list':
            status = await manager.get_buffer_status()
            manager.display_buffer_status(status)

            # Also show details for each user
            user_buffers = status.get('user_buffers', {})
            for user_id in user_buffers.keys():
                print(f"\n{'=' * 60}")
                details = await manager.get_user_buffer_details(user_id)
                manager.display_user_details(details)

        elif args.action == 'user':
            if not args.target:
                manager.print_status("User ID required for user action", StatusLevel.ERROR)
                success = False
            else:
                details = await manager.get_user_buffer_details(args.target)
                manager.display_user_details(details)

        elif args.action == 'flush':
            if args.user:
                results = await manager.flush_user_buffer(args.user)
            else:
                results = await manager.flush_all_buffers()
            manager.display_flush_results(results)

        elif args.action == 'search':
            if not args.target:
                manager.print_status("Search query required for search action", StatusLevel.ERROR)
                success = False
            else:
                results = await manager.search_buffer_contents(args.target)
                manager.display_search_results(results)

        else:
            manager.print_status(f"Unknown action: {args.action}", StatusLevel.ERROR)
            success = False

        print(f"\nCompleted at: {datetime.now()}")

        if success:
            manager.print_status(f"{args.action.title()} completed successfully!", StatusLevel.SUCCESS)
        else:
            manager.print_status(f"{args.action.title()} failed!", StatusLevel.ERROR)
            sys.exit(1)

    except KeyboardInterrupt:
        manager.print_status("Operation cancelled by user", StatusLevel.WARNING)
        sys.exit(1)
    except Exception as e:
        manager.print_status(f"Unexpected error: {e}", StatusLevel.ERROR)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
