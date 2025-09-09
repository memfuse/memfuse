"""StorePort protocol for persistence adapters.

Directory is named 'persistence'; class names retain 'Store' suffix.
"""
from typing import Protocol
from ..models.core import Item, Query, QueryResult


class StorePort(Protocol):
    async def initialize(self) -> bool: ...
    async def add(self, item: Item) -> str: ...
    async def query(self, query: Query) -> QueryResult: ...
    async def delete(self, item_id: str) -> bool: ...

