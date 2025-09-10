import pytest


@pytest.mark.asyncio
async def test_m3_query_route_registered():
    from memfuse_core.server import create_app_async

    app = await create_app_async()
    paths = [getattr(r, "path", "") for r in app.router.routes]
    # Ensure the M3 query route exists
    assert any(p.startswith("/api/v1/users/") and p.endswith("/query") for p in paths), paths

