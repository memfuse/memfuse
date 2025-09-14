"""M3 type definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class PlanStep:
    """Represents a single step in an M3 workflow plan."""
    agent: str
    input: Dict[str, Any]