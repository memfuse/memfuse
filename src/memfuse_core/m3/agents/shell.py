"""ShellCommandAgent implementation for M3."""

from __future__ import annotations

import asyncio
import os
import shutil
from typing import Any, Dict

from loguru import logger


class ShellCommandAgent:
    """Agent that executes limited, safe shell commands."""
    
    def __init__(self) -> None:
        # Check if agent is enabled via environment variable
        self.enabled = str(os.getenv("ALLOW_SHELL_AGENT", "false")).lower() in ("1", "true", "yes")
        
        # Define allowed commands
        self.allowed_commands = {
            "rg": self._execute_ripgrep,
            "echo": self._execute_echo,
        }
    
    async def _execute_ripgrep(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ripgrep (rg) command safely."""
        if not shutil.which("rg"):
            return {"error": "ripgrep (rg) not installed"}
        
        pattern = str(payload.get("pattern", payload.get("query", ""))).strip()
        if not pattern:
            return {"error": "pattern required for ripgrep"}
        
        path = str(payload.get("path", ".")).strip()
        max_count = min(int(payload.get("max", payload.get("max_count", 200))), 1000)  # Cap at 1000
        
        # Build ripgrep command
        cmd = [
            "rg",
            "-n",              # Show line numbers
            "--no-heading",    # Don't show file headers
            "-S",              # Smart case
            "-m", str(max_count),  # Max matches
            pattern,
            path
        ]
        
        try:
            # Execute with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=30.0  # 30 second timeout
            )
            
            return {
                "command": " ".join(cmd),
                "exit_code": process.returncode,
                "output": stdout.decode('utf-8', errors='replace'),
                "error": stderr.decode('utf-8', errors='replace') if stderr else None,
                "pattern": pattern,
                "path": path
            }
            
        except asyncio.TimeoutError:
            return {"error": "Command timed out after 30 seconds", "command": " ".join(cmd)}
        except Exception as e:
            return {"error": f"Command execution failed: {str(e)}", "command": " ".join(cmd)}
    
    async def _execute_echo(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute echo command safely."""
        text = str(payload.get("text", payload.get("message", ""))).strip()
        
        if not text:
            return {"error": "text or message required for echo"}
        
        # Limit text length for safety
        if len(text) > 1000:
            text = text[:1000] + "..."
        
        try:
            process = await asyncio.create_subprocess_exec(
                "echo", text,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "command": f"echo {text}",
                "exit_code": process.returncode,
                "output": stdout.decode('utf-8', errors='replace'),
                "error": stderr.decode('utf-8', errors='replace') if stderr else None
            }
            
        except Exception as e:
            return {"error": f"Echo command failed: {str(e)}", "command": f"echo {text}"}
    
    async def execute(self, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a safe shell command."""
        if not self.enabled:
            return {"error": "Shell agent is disabled. Set ALLOW_SHELL_AGENT=true to enable."}
        
        cmd = str(payload.get("cmd", payload.get("command", ""))).strip()
        if not cmd:
            return {"error": "cmd or command parameter required"}
        
        # Extract base command
        base_cmd = cmd.split()[0] if cmd else ""
        
        if base_cmd not in self.allowed_commands:
            allowed = ", ".join(self.allowed_commands.keys())
            return {
                "error": f"Command '{base_cmd}' not allowed. Allowed commands: {allowed}",
                "allowed_commands": list(self.allowed_commands.keys())
            }
        
        logger.info(f"ShellCommandAgent executing: {cmd}")
        
        try:
            # Execute the specific command handler
            result = await self.allowed_commands[base_cmd](payload)
            result["session_id"] = session_id
            return result
            
        except Exception as e:
            logger.error(f"ShellCommandAgent execution failed: {e}")
            return {
                "error": f"Command execution failed: {str(e)}",
                "command": cmd,
                "session_id": session_id
            }