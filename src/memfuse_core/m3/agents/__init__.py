"""M3 Agents module."""

from .websearch import WebSearchAgent
from .database import DatabaseQueryAgent
from .shell import ShellCommandAgent

__all__ = [
    "WebSearchAgent",
    "DatabaseQueryAgent", 
    "ShellCommandAgent",
]