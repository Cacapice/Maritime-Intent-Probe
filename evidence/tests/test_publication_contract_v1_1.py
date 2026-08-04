import json
from pathlib import Path

def _repo_root() -> Path:
    """Locate the repository root by marker rather than by counting parents."""
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[2]


from evidence.adapter import scientific_result_json_schema, evidence_graph_json_schema, verify_bundle

def test_contract_schema_is_v1_1():
    schema=scientific_result_json_schema()
    assert schema["properties"]["schema_version"]["const"] == "1.1"
    assert "research_phase" in schema["required"]

def test_committed_schemas_exist():
    root=_repo_root()
    assert (root/"schemas/scientific-result-1.1.schema.json").is_file()
    assert (root/"schemas/evidence-graph-1.1.schema.json").is_file()

def test_maritime_declares_phase_1_in_readme():
    text=(_repo_root()/"README.md").read_text()
    assert "Phase 1 construct-validity diagnostic" in text
    assert "deployable monitor remain" in text
