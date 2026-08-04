"""Maritime Intent Probe adapter for the shared publication contract.

The dependency-light contract runtime is grouped under `evidence/runtime`. This
module re-exports it and adds only the adapter that translates this repository's native result object into `ScientificResult`. Research code remains isolated
under `science/`; publication enforcement remains infrastructure.
"""
from __future__ import annotations

from typing import Any

from evidence.runtime import *  # noqa: F401,F403  (re-export the contract)
from evidence.runtime import (  # explicit names used by the adapter below
    AdapterPolicy,
    QualificationRequirement,
    assert_adapter_result,
    EstimateKind,
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
    PublicationStatus,
    Qualification,
    ScientificResult,
    Uncertainty,
)


MARITIME_PHASE1_ADAPTER_POLICY = AdapterPolicy(
    repository="Maritime Intent Probe",
    adapter_name="from_bc1_report",
    required_native_fields=("semantic_interpretation_supported", "construct_identifiable", "crossed_design_completed"),
    requirements=(
        QualificationRequirement("phase_1_scope", lambda p: True, "all current results are Phase 1 construct-validity diagnostics", "qualified"),
        QualificationRequirement("crossed_design_incomplete", lambda p: not bool(p.get("crossed_design_completed")), "the crossed design is not complete and operational claims remain prohibited", "qualified"),
        QualificationRequirement("bc1_identifiability_failure", lambda p: not bool(p.get("semantic_interpretation_supported")), "BC1 failure must block semantic interpretation", "blocked"),
    ),
    minimum_qualification_count=2,
    allowed_research_phases=("phase_1_construct_validity_diagnostic",),
    prohibited_supported_claim_fragments=("deployable monitor", "validated intent detector", "operational intent monitor", "completed validation"),
)

def from_bc1_report(report: Any) -> ScientificResult:
    native=report.to_dict()
    native["crossed_design_completed"] = bool(native.get("crossed_design_completed", False))
    supported=bool(native.get("semantic_interpretation_supported"))
    qualifications=[Qualification(kind="phase_1_scope", effect="weaken", resulting_publication_status="qualified", rationale="Phase 1 establishes a construct-validity diagnostic, not an operational monitor")]
    if not native["crossed_design_completed"]:
        qualifications.append(Qualification(kind="crossed_design_incomplete", effect="preserve", rationale="the crossed 2x2 intent/action-space design has not been completed"))
    if not supported:
        qualifications.append(Qualification(kind="bc1_identifiability_failure", effect="block", resulting_publication_status="blocked", rationale=native.get("rationale", "construct is not identified")))
    graph=EvidenceGraph(
        nodes=(
            EvidenceNode("support", "design", "Observed experimental support", {"support": native.get("observed_support")}),
            EvidenceNode("witness", "alternative_explanation", "Model-blind witness", {"match": native.get("model_blind_match")}),
            EvidenceNode("bc1", "qualification", "BC1 identifiability gate", {"identifiable": native.get("construct_identifiable")}),
            EvidenceNode("interpretation", "inference", "Semantic interpretation", {"supported": supported}),
        ),
        edges=(EvidenceEdge("support","bc1","evaluated_by"), EvidenceEdge("witness","bc1","challenges"), EvidenceEdge("bc1","interpretation","licenses_or_blocks")),
    )
    adapted = ScientificResult(
        repository="Maritime Intent Probe", estimand="permission for semantic interpretation of a neural probe",
        research_phase="phase_1_construct_validity_diagnostic",
        estimate=supported, estimate_kind="identifiability_gate",
        publication_status="publishable", base_publication_status="publishable",
        assumptions=("the stated observed-cell support and witness result are correct",),
        qualifications=tuple(qualifications),
        supported_conclusions=(("Semantic interpretation passes BC1.",) if supported else ("Probe validity may hold while semantic interpretation remains blocked.",)),
        unsupported_conclusions=(("BC1 alone does not establish causal mechanism.",) if supported else ("Held-out probe accuracy does not identify adversarial intent.", "Phase 1 does not validate a deployable intent monitor.")),
        evidence_graph=graph, native_payload=native,
    )
    return assert_adapter_result(adapted, MARITIME_PHASE1_ADAPTER_POLICY)
