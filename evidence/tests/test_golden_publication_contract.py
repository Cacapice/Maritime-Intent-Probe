import json
from pathlib import Path

def _repo_root() -> Path:
    """Locate the repository root by marker rather than by counting parents."""
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[2]


import pytest
from evidence.adapter import ScientificResult, Qualification, qualification_is_monotone

def test_golden_contract_identity():
    root = _repo_root()
    payload=json.loads((root/"golden/publication_contract.json").read_text())
    assert payload["schema_version"] == "1.1"
    assert payload["research_phase"] == "phase_1_construct_validity_diagnostic"
    assert payload["contract_invariants"]["qualification_may_silently_strengthen"] is False

def test_qualification_cannot_silently_strengthen():
    q=Qualification(kind="censoring", effect="weaken")
    assert qualification_is_monotone("publishable", "qualified", q)
    assert not qualification_is_monotone("qualified", "publishable", q)

def test_phase_1_blocks_deployment_claims():
    result=ScientificResult(repository="Maritime Intent Probe", estimand="BC1", estimate=False, estimate_kind="identifiability_gate", publication_status="qualified", research_phase="phase_1_construct_validity_diagnostic", supported_conclusions=("This is a deployable monitor",))
    with pytest.raises(ValueError, match="Phase 1"):
        result.strict_publish()
