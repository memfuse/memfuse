import os
import pytest

@pytest.mark.integration
class TestAgentsIdempotency:
    def test_post_agents_is_idempotent_on_duplicate(self, client, headers, integration_helper):
        # pick a fixed name (no unique suffix) to force duplicate
        name = "idempotency_test_agent"
        payload = {"name": name, "description": "idempotency check"}

        # First create
        resp1 = client.post("/api/v1/agents", json=payload, headers=headers)
        assert resp1.status_code in (201, 200)
        data1 = resp1.json()
        assert data1["status"] == "success"
        agent1 = data1["data"]["agent"]
        assert agent1["name"] == name

        # Second create (duplicate) should NOT be 500, and should return existing agent
        resp2 = client.post("/api/v1/agents", json=payload, headers=headers)
        assert resp2.status_code in (200, 400)
        data2 = resp2.json()

        if resp2.status_code == 200:
            assert data2["status"] == "success"
            agent2 = data2["data"]["agent"]
            assert agent2["id"] == agent1["id"]
        else:
            # 400 path is still acceptable contract-wise; must not be server error
            assert data2["status"] == "error"
            assert "already exists" in data2["message"].lower()
