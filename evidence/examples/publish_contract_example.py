"""Example publication bundle, showing the emission gate.

There is no ungated variant of this script. An earlier revision shipped 2
examples side by side, 1 routed through the privacy plane and 1 not, with
nothing to indicate which was correct. That is how an enforced rule becomes an
optional one.

Two legitimate paths exist, and both require a declaration:

  unrestricted   the result derives from no owner-held data. Declare it and
                 emit. The declaration is recorded in the bundle manifest.
  owner_data     the result derives from owner-held data. Build a sovereignty
                 envelope, evaluate it for the intended purpose, and pass the
                 resulting decision to publish_bundle.
"""
from pathlib import Path

from evidence.adapter import (
    ApprovalRecord, DataSovereigntyPolicy, EvidenceEdge, EvidenceGraph,
    EvidenceNode, PrivacyAssessment, ScientificResult, SovereignEvidenceEnvelope,
    publish_bundle,
)

GRAPH = EvidenceGraph(
    nodes=(EvidenceNode("observation", "observation", "Observed input"),
           EvidenceNode("inference", "inference", "Qualified output")),
    edges=(EvidenceEdge("observation", "inference", "supports"),),
)


def unrestricted_example(out: Path) -> dict:
    """A result derived from no owner-held data."""
    result = ScientificResult(
        repository="Maritime Intent Probe", estimand="example publication contract",
        estimate=True, estimate_kind="point_estimate",
        publication_status="qualified", qualifications=("example only",),
        evidence_graph=GRAPH, data_classification="unrestricted",
    )
    return publish_bundle(result, out, strict=True)


def owner_data_example(out: Path) -> dict:
    """A result derived from owner-held data, routed through the privacy plane."""
    result = ScientificResult(
        repository="Maritime Intent Probe", estimand="example publication contract",
        estimate=True, estimate_kind="point_estimate",
        publication_status="qualified", qualifications=("example only",),
        evidence_graph=GRAPH, data_classification="owner_data",
    )
    envelope = SovereignEvidenceEnvelope(
        scientific_result=result.to_dict(),
        sovereignty_policy=DataSovereigntyPolicy(
            data_owner="example owner", custodian="example custodian",
            jurisdiction="example", permitted_purposes=("research",),
            consent_basis="documented consent"),
        privacy_assessment=PrivacyAssessment(data_minimized=True, aggregate_only=True),
        approvals=(
            ApprovalRecord(role="data_owner", actor="example owner", approved=True, scope="research"),
            ApprovalRecord(role="publication_authority", actor="example authority", approved=True, scope="research"),
        ),
    )
    decision = envelope.evaluate("research", publication=True)
    return publish_bundle(result, out, strict=True, trust_decision=decision)


if __name__ == "__main__":
    base = Path("results")
    for label, fn in (("unrestricted", unrestricted_example), ("owner_data", owner_data_example)):
        files = fn(base / f"contract_example_{label}")
        print(f"[{label}]")
        for key, value in files.items():
            print(f"  {key}: {value}")
