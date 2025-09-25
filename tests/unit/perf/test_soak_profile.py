import os
import pytest

from tests.performance.api.profile_loader import load_profile


pytestmark = pytest.mark.unit


def test_load_soak_profile(monkeypatch):
    monkeypatch.setenv("PROFILE", "soak")
    p = load_profile()
    assert p["name"] == "soak"
    tt = p.get("think_time_ms", {})
    assert int(tt.get("min", 0)) >= 120000
    assert int(tt.get("max", 0)) >= 120000


def test_weights_nonzero_for_tick(monkeypatch):
    monkeypatch.setenv("PROFILE", "soak")
    p = load_profile()
    w = p.get("weights", {})
    assert w.get("user_tick", 0) > 0
