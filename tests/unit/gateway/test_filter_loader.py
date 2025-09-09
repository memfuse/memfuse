"""Tests for gateway filter pipeline loader."""

from memfuse_core.gateway.filters import build_filters_from_config, NoOpInboundFilter, NoOpOutboundFilter


def test_build_filters_from_config_noop():
    cfg = {
        "pipeline": {
            "inbound": [{"name": "noop_inbound", "enabled": True}],
            "outbound": [{"name": "noop_outbound", "enabled": True}],
        }
    }
    inbound, outbound = build_filters_from_config(cfg)
    assert len(inbound) == 1 and isinstance(inbound[0], NoOpInboundFilter)
    assert len(outbound) == 1 and isinstance(outbound[0], NoOpOutboundFilter)


def test_build_filters_from_config_disabled():
    cfg = {
        "pipeline": {
            "inbound": [{"name": "noop_inbound", "enabled": False}],
            "outbound": [{"name": "noop_outbound", "enabled": False}],
        }
    }
    inbound, outbound = build_filters_from_config(cfg)
    assert inbound == []
    assert outbound == []

