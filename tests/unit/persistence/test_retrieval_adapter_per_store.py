import pytest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

from src.memfuse_core.persistence.adapters import RetrievalAdapter
from src.memfuse_core.utils.global_config_manager import get_global_config_manager


class FakeStoreA:
    """Mock store class A for per-store config testing."""
    
    async def query_topk(self, query_text: str, max_results: int) -> List[Dict[str, Any]]:
        return [{"id": "a1", "content": "result from A", "score": 0.9}]


class FakeStoreB:
    """Mock store class B for per-store config testing."""
    
    async def query_topk(self, query_text: str, max_results: int) -> List[Dict[str, Any]]:
        return [{"id": "b1", "content": "result from B", "score": 0.8}]


@pytest.mark.asyncio
async def test_per_store_timeout_override():
    """Test that per-store timeout overrides global timeout."""
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "buffer": {
            "retrieval_timeout_seconds": 1.0,  # global default
            "retrieval_per_store": {
                "FakeStoreA": {"timeout_seconds": 0.5},  # override for A
                "FakeStoreB": {"timeout_seconds": 2.0},  # override for B
            }
        }
    })
    
    store_a = FakeStoreA()
    store_b = FakeStoreB()
    
    adapter_a = RetrievalAdapter(store_a)
    adapter_b = RetrievalAdapter(store_b)
    
    # Check that per-store overrides are applied
    assert adapter_a.timeout_seconds == 0.5
    assert adapter_b.timeout_seconds == 2.0
    assert adapter_a.store_class_name == "FakeStoreA"
    assert adapter_b.store_class_name == "FakeStoreB"


@pytest.mark.asyncio
async def test_per_store_retry_override():
    """Test that per-store retry config overrides global retry config."""
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "buffer": {
            "retrieval_retry": {
                "enabled": True,
                "max_attempts": 2,
                "backoff_ms": 100
            },
            "retrieval_per_store": {
                "FakeStoreA": {
                    "retry": {
                        "enabled": False,  # disable retry for A
                        "max_attempts": 1,
                        "backoff_ms": 0
                    }
                },
                "FakeStoreB": {
                    "retry": {
                        "enabled": True,
                        "max_attempts": 5,  # more attempts for B
                        "backoff_ms": 200
                    }
                }
            }
        }
    })
    
    store_a = FakeStoreA()
    store_b = FakeStoreB()
    
    adapter_a = RetrievalAdapter(store_a)
    adapter_b = RetrievalAdapter(store_b)
    
    # Check that per-store retry overrides are applied
    assert adapter_a.retry_enabled is False
    assert adapter_a.retry_attempts == 1
    assert adapter_a.retry_backoff_ms == 0
    
    assert adapter_b.retry_enabled is True
    assert adapter_b.retry_attempts == 5
    assert adapter_b.retry_backoff_ms == 200


@pytest.mark.asyncio
async def test_per_store_partial_override():
    """Test that per-store config can partially override global config."""
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "buffer": {
            "retrieval_timeout_seconds": 1.0,
            "retrieval_retry": {
                "enabled": True,
                "max_attempts": 3,
                "backoff_ms": 150
            },
            "retrieval_per_store": {
                "FakeStoreA": {
                    "timeout_seconds": 0.3,  # only override timeout
                    # retry config inherits from global
                }
            }
        }
    })
    
    store_a = FakeStoreA()
    adapter_a = RetrievalAdapter(store_a)
    
    # Timeout overridden, retry config inherited from global
    assert adapter_a.timeout_seconds == 0.3
    assert adapter_a.retry_enabled is True
    assert adapter_a.retry_attempts == 3
    assert adapter_a.retry_backoff_ms == 150


@pytest.mark.asyncio
async def test_no_per_store_config_uses_global():
    """Test that stores without per-store config use global defaults."""
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "buffer": {
            "retrieval_timeout_seconds": 0.8,
            "retrieval_retry": {
                "enabled": True,
                "max_attempts": 2,
                "backoff_ms": 50
            },
            "retrieval_per_store": {
                "SomeOtherStore": {"timeout_seconds": 999}  # not our store
            }
        }
    })
    
    store_a = FakeStoreA()
    adapter_a = RetrievalAdapter(store_a)
    
    # Should use global config since no per-store config for FakeStoreA
    assert adapter_a.timeout_seconds == 0.8
    assert adapter_a.retry_enabled is True
    assert adapter_a.retry_attempts == 2
    assert adapter_a.retry_backoff_ms == 50
