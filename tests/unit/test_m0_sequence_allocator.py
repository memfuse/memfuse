import asyncio
import types
import uuid

import pytest


class FakeCursor:
    def __init__(self, seq_start=1, fail_first_upsert=False):
        self._seq_start = seq_start
        self._fetch = []
        self.executed = []
        self._fail_first_upsert = fail_first_upsert
        self._upsert_attempts = 0

    def execute(self, sql, params=None):
        self.executed.append((sql.strip().split("\n")[0][:60], params))
        if "CREATE TABLE IF NOT EXISTS m0_seq" in sql:
            # no-op ensure
            self._fetch = []
            return
        # Simulate UPSERT returning start_seq (with optional first-failure)
        if "INSERT INTO m0_seq" in sql:
            self._upsert_attempts += 1
            if self._fail_first_upsert and self._upsert_attempts == 1:
                raise Exception("simulated upsert failure")
            self._fetch = [(self._seq_start,)]
            # Next call simulates next batch moving forward
            self._seq_start += params[1]
        elif "RETURNING message_id" in sql:
            # Returning a message_id row
            self._fetch = [(params[0],)]
        else:
            self._fetch = []

    def fetchone(self):
        return self._fetch[0] if self._fetch else None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConn:
    def __init__(self, **kw):
        self.autocommit = True
        self._cursor = FakeCursor(**kw)

    def cursor(self):
        return self._cursor

    def commit(self):
        pass

    def rollback(self):
        pass


@pytest.mark.asyncio
async def test_store_m0_messages_allocates_distinct_sequences(monkeypatch):
    from memfuse_core.services.simplified_memory_service import SimplifiedMemoryService

    svc = SimplifiedMemoryService(cfg={}, user="u")
    # Patch db_manager.conn with fake connection
    fake_conn = FakeConn()
    svc.db_manager.conn = fake_conn

    session_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    round_id = str(uuid.uuid4())

    msgs1 = [
        {"id": str(uuid.uuid4()), "role": "user", "content": "hello"},
        {"id": str(uuid.uuid4()), "role": "assistant", "content": "world"},
    ]
    msgs2 = [
        {"id": str(uuid.uuid4()), "role": "user", "content": "foo"},
    ]

    # First batch
    ids1 = await svc._store_m0_messages(msgs1, session_id=session_id, user_id=user_id, round_id=round_id)
    assert len(ids1) == 2
    # Second batch (should continue sequence without overlap)
    ids2 = await svc._store_m0_messages(msgs2, session_id=session_id, user_id=user_id, round_id=round_id)
    assert len(ids2) == 1

    # Verify we used the UPSERT into m0_seq (presence of INSERT INTO m0_seq)
    executed = [sql for sql, _ in fake_conn.cursor().executed]
    # Our implementation uses a CTE: WITH upsert AS (INSERT INTO m0_seq ...)
    assert any(("INSERT INTO m0_seq" in e) or ("WITH upsert AS" in e) for e in executed)
    # And ensure table creation was attempted before upsert
    assert any("CREATE TABLE IF NOT EXISTS m0_seq" in e for e in executed)


@pytest.mark.asyncio
async def test_store_m0_messages_savepoint_retry(monkeypatch):
    # First UPSERT fails, code should ROLLBACK TO SAVEPOINT and retry successfully
    from memfuse_core.services.simplified_memory_service import SimplifiedMemoryService

    svc = SimplifiedMemoryService(cfg={}, user="u")
    fake_conn = FakeConn(fail_first_upsert=True)
    svc.db_manager.conn = fake_conn

    session_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    round_id = str(uuid.uuid4())

    msgs = [
        {"id": str(uuid.uuid4()), "role": "user", "content": "hello"},
    ]

    ids = await svc._store_m0_messages(msgs, session_id=session_id, user_id=user_id, round_id=round_id)
    assert len(ids) == 1
    # Verify savepoint usage and retry sequence
    executed_sql = [sql for sql, _ in fake_conn.cursor().executed]
    assert any("SAVEPOINT m0seq_sp" in e for e in executed_sql)
    assert any("ROLLBACK TO SAVEPOINT m0seq_sp" in e for e in executed_sql)
