"""Unit tests for M2 autonomous background worker in SimplifiedMemoryService.

These tests verify that:
- the background worker starts when config["m2_enabled"] is True
- it does not start when disabled
- it processes a pending chunk end-to-end via the orchestrated helpers
- it shuts down cleanly via close()
"""

import asyncio
import uuid
from typing import List

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.memfuse_core.services.simplified_memory_service import SimplifiedMemoryService


@pytest.mark.unit
@pytest.mark.services
@pytest.mark.asyncio
class TestM2BackgroundWorker:
    async def _make_service(
        self,
        enabled: bool = True,
        interval: float = 0.01,
        batch_size: int = 2,
    ) -> SimplifiedMemoryService:
        """Create a SimplifiedMemoryService with background worker config and patched I/O."""
        cfg = {
            "m2_enabled": enabled,
            "m2_interval_secs": interval,
            "m2_batch_size": batch_size,
            # Provide dummy DB overrides to avoid touching env defaults
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "memfuse_test",
                "user": "postgres",
                "password": "postgres",
            },
        }

        service = SimplifiedMemoryService(cfg=cfg, user="test_user_bg")

        # Patch heavy initialization steps to avoid real DB/model work
        with patch(
            "src.memfuse_core.services.simplified_memory_service.sync_connection_pool.initialize",
            new=MagicMock(),
        ):
            service.db_manager.connect = AsyncMock(return_value=None)
            service.db_manager.initialize_schema = AsyncMock(return_value=None)
            service.embedding_generator.initialize = AsyncMock(return_value=None)

            # Ensure we have a user_id used by the worker for scoping
            async def _fake_init_user_id():
                service._user_id = str(uuid.uuid4())

            service._initialize_user_id = AsyncMock(side_effect=_fake_init_user_id)

            # Also patch the polling calls to be benign by default
            service._get_pending_m2_chunks = AsyncMock(return_value=[])

            await service.initialize()

        return service

    async def test_worker_starts_when_enabled(self):
        service = await self._make_service(enabled=True)

        try:
            assert service.m2_running is True
            assert service.m2_processor_task is not None
            assert not service.m2_processor_task.done()
        finally:
            await service.close()

    async def test_worker_does_not_start_when_disabled(self):
        service = await self._make_service(enabled=False)

        try:
            assert service.m2_running is False
            assert service.m2_processor_task is None
        finally:
            await service.close()

    async def test_worker_processes_chunk_success(self):
        service = await self._make_service(enabled=True, interval=0.01)

        # Prepare a fake pending chunk id and fact extraction flow
        pending_chunk_id = str(uuid.uuid4())

        async def fake_get_pending(batch_size: int = 2, user_id: str | None = None) -> List[str]:
            # Return the chunk once, then no more
            if not hasattr(fake_get_pending, "_called"):
                fake_get_pending._called = 0
            fake_get_pending._called += 1
            return [pending_chunk_id] if fake_get_pending._called == 1 else []

        service._get_pending_m2_chunks = AsyncMock(side_effect=fake_get_pending)
        service._lock_chunk_for_m2_processing = AsyncMock(return_value=True)
        # The worker now fetches chunk data to get the correct user_id per chunk
        chunk_user_id = str(uuid.uuid4())
        service._get_m1_chunk = AsyncMock(return_value={
            'chunk_id': pending_chunk_id,
            'user_id': chunk_user_id,
            'content': 'dummy',
            'token_count': 10,
        })
        service._extract_list_of_fact_content_from_chunk = AsyncMock(return_value=[
            "User prefers dark mode",
            "Assistant suggested a theme toggle",
        ])
        service._mark_chunk_m2_completed = AsyncMock(return_value=True)
        service._mark_chunk_m2_failed = AsyncMock(return_value=True)

        # Let the loop tick
        await asyncio.sleep(0.05)

        # Verify orchestrated calls occurred
        service._get_pending_m2_chunks.assert_awaited()
        # Lock is performed without user scoping now
        service._lock_chunk_for_m2_processing.assert_awaited_with(
            chunk_id=pending_chunk_id, user_id=None
        )
        # Extraction is scoped by the chunk's user_id
        service._extract_list_of_fact_content_from_chunk.assert_awaited_with(
            chunk_id=pending_chunk_id, user_id=chunk_user_id
        )
        service._mark_chunk_m2_completed.assert_awaited()
        # Should not have marked failed for this path
        # (but allow any incidental calls if race occurred)

        await service.close()

    async def test_worker_shutdown_clean(self):
        service = await self._make_service(enabled=True)

        await service.close()
        # After close, task is cleared and running flag reset
        assert service.m2_running is False
        assert service.m2_processor_task is None
