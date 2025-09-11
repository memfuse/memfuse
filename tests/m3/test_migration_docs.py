import os
from pathlib import Path


def test_migration_readme_exists():
    assert Path("migration/README.md").exists(), "migration/README.md is missing"


def test_m3_plan_exists_and_has_sections():
    p = Path("migration/m3_migration_plan.md")
    assert p.exists(), "migration/m3_migration_plan.md is missing"
    text = p.read_text(encoding="utf-8")
    # Spot check a few critical sections
    for phrase in [
        "Phase A",
        "Schemas (temporary, non-destructive)",
        "LLM Unification",
        "RAG Placement",
        "Acceptance Criteria",
    ]:
        assert phrase in text, f"Missing section: {phrase}"


def test_api_integration_notes_exist():
    p = Path("migration/m3_api_integration.md")
    assert p.exists(), "migration/m3_api_integration.md is missing"
    text = p.read_text(encoding="utf-8")
    assert "metadata" in text and "message_workflows" in text


def test_phaseA_schema_doc_exists_and_defines_tables():
    p = Path("docs/m3/schema_phaseA.md")
    assert p.exists(), "docs/m3/schema_phaseA.md is missing"
    text = p.read_text(encoding="utf-8").lower()
    for tbl in ["message_workflows", "procedural_memory", "procedural_lessons"]:
        assert tbl in text, f"Expected table name not found in schema doc: {tbl}"


def test_docs_overview_and_api_exist():
    assert Path("docs/m3/overview.md").exists(), "docs/m3/overview.md missing"
    assert Path("docs/m3/api.md").exists(), "docs/m3/api.md missing"
    assert Path("docs/m3/getting_started.md").exists(), "docs/m3/getting_started.md missing"
