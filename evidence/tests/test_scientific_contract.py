from pathlib import Path
import json
import pytest

from evidence.adapter import ScientificResult, Uncertainty, EvidenceNode, EvidenceEdge, EvidenceGraph, publish_bundle, from_bc1_report, MARITIME_PHASE1_ADAPTER_POLICY, validate_adapter_result


def test_scientific_result_round_trip_and_bundle(tmp_path):
    graph=EvidenceGraph(nodes=(EvidenceNode("a","observation","A"), EvidenceNode("b","inference","B")), edges=(EvidenceEdge("a","b","supports"),))
    result=ScientificResult(repository="test", estimand="quantity", estimate=1.0, estimate_kind="point_estimate", publication_status="qualified", data_classification="unrestricted", uncertainty=Uncertainty(kind="interval", level=.95, low=.8, high=1.2, n=20), evidence_graph=graph)
    payload=result.strict_publish()
    assert payload["publishable"] is True
    files=publish_bundle(result,tmp_path,seed=7)
    assert {"scientific_result","evidence_graph","provenance","manifest","scientific_result_schema","evidence_graph_schema"} <= set(files)
    assert json.loads(Path(files["scientific_result"]).read_text())["schema_version"]=="1.1"


def test_blocked_result_cannot_strict_publish():
    result=ScientificResult(repository="test", estimand="claim", estimate=False, estimate_kind="identifiability_gate", publication_status="blocked")
    with pytest.raises(ValueError):
        result.strict_publish()


def test_native_adapter_maritime():
    from evidence_platform.epistemic import maritime_counterexample_report
    result=from_bc1_report(maritime_counterexample_report())
    assert result.publication_status=="blocked"
    assert result.estimate is False


def test_bundle_requires_a_declared_classification(tmp_path):
    """Emission is gated even for a result this repository would consider routine."""
    result = ScientificResult(repository="test", estimand="q", estimate=1.0,
                              estimate_kind="point_estimate", publication_status="qualified")
    with pytest.raises(ValueError, match="data_classification is undeclared"):
        publish_bundle(result, tmp_path)


def test_adapter_policy_rejects_underqualification_and_phase_bypass():
    from dataclasses import replace
    from evidence_platform.epistemic import maritime_counterexample_report
    result=from_bc1_report(maritime_counterexample_report())
    under=replace(result, qualifications=())
    report=validate_adapter_result(under, MARITIME_PHASE1_ADAPTER_POLICY)
    assert not report.conformant
    assert any("phase_1_scope" in v or "crossed_design_incomplete" in v for v in report.violations)
    wrong_phase=replace(result, research_phase="mature_method")
    assert not validate_adapter_result(wrong_phase, MARITIME_PHASE1_ADAPTER_POLICY).conformant
