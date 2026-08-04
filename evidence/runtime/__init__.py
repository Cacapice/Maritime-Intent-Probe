"""Qualification Before Interpretation: the shared publication contract.

One implementation, depended on by every repository in the portfolio. Each
repository contributes only an adapter that translates its own native result
object into `ScientificResult`.

The doctrine the package exists to enforce: a qualification may never increase
publication strength without explicit new evidence, and a result that cannot
carry its own warrant is refused rather than degraded.
"""

from .contract import (
    AdapterConformanceReport,
    AdapterPolicy,
    QualificationRequirement,
    EPISTEMIC_STRENGTH,
    EstimateKind,
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
    PublicationStatus,
    Qualification,
    ScientificResult,
    Uncertainty,
    assert_adapter_result,
    capture_provenance,
    derive_publication_status,
    evidence_graph_json_schema,
    publish_bundle,
    qualification_is_monotone,
    scientific_result_json_schema,
    validate_adapter_result,
    validate_evidence_layers,
    verify_bundle,
)
from .sovereign import (
    ApprovalRecord,
    DataClassificationDeclaration,
    DataSovereigntyPolicy,
    PrivacyAssessment,
    PolicyMigrationRecord,
    create_policy_migration,
    SovereignEvidenceEnvelope,
    TrustDecision,
    publish_sovereign_bundle,
    sovereign_evidence_json_schema,
    verify_sovereign_bundle,
)

__version__ = "1.3.0"
SCIENTIFIC_RESULT_SCHEMA = "1.1"
SOVEREIGN_SCHEMA = "sovereign-evidence-1.2"

__all__ = [
    "AdapterConformanceReport", "AdapterPolicy", "QualificationRequirement",
    "EPISTEMIC_STRENGTH", "EstimateKind", "EvidenceEdge", "EvidenceGraph",
    "EvidenceNode", "PublicationStatus", "Qualification", "ScientificResult",
    "Uncertainty", "assert_adapter_result", "capture_provenance", "derive_publication_status", "evidence_graph_json_schema",
    "publish_bundle", "qualification_is_monotone", "scientific_result_json_schema",
    "validate_adapter_result", "validate_evidence_layers", "verify_bundle",
    "ApprovalRecord", "DataClassificationDeclaration", "DataSovereigntyPolicy", "PrivacyAssessment",
    "PolicyMigrationRecord", "create_policy_migration",
    "SovereignEvidenceEnvelope", "TrustDecision", "publish_sovereign_bundle",
    "sovereign_evidence_json_schema", "verify_sovereign_bundle",
    "SCIENTIFIC_RESULT_SCHEMA", "SOVEREIGN_SCHEMA", "__version__",
]
