from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class PlanStep:
    agent: str
    input: Dict[str, Any]

