from pathlib import Path


def test_m3_enabled_in_config_present():
    cfg_path = Path("config/memory/default.yaml")
    assert cfg_path.exists(), "config/memory/default.yaml is missing"
    text = cfg_path.read_text(encoding="utf-8")
    # Simple string checks to avoid external dependencies
    assert "layers:" in text and "m3:" in text, "layers.m3 not found in memory config"
    # We expect an 'enabled' key under m3 block
    assert "m3:" in text and "enabled:" in text, "layers.m3.enabled missing in memory config"


def test_store_has_m3_target_schema():
    # Ensure the repository already contains the target M3 schema we plan to align with in Phase B
    schema_path = Path("src/memfuse_core/store/pgai_store/schemas/m3_procedural.sql")
    assert schema_path.exists(), "m3_procedural.sql not found (Phase B target schema missing)"
