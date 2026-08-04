"""Generate an illustrative sovereign evidence bundle with synthetic metadata only."""
from pathlib import Path
from evidence.sovereign import (
    ApprovalRecord, DataSovereigntyPolicy, PrivacyAssessment, SovereignEvidenceEnvelope,
    publish_sovereign_bundle, verify_sovereign_bundle,
)

result = {"publishable": True, "publication_status": "qualified", "supported_conclusions": ["qualified evidence may inform authorized human review"], "unsupported_conclusions": ["raw data may be centralized"]}
envelope = SovereignEvidenceEnvelope(
    scientific_result=result,
    sovereignty_policy=DataSovereigntyPolicy(
        data_owner="example-owner", custodian="example-custodian", jurisdiction="owner-defined",
        permitted_purposes=("scientific_publication",), consent_basis="example governance charter"),
    privacy_assessment=PrivacyAssessment(data_minimized=True, aggregate_only=True, privacy_mechanisms=("local computation",)),
    approvals=(ApprovalRecord("data_owner", "owner-delegate", True, "scientific_publication"), ApprovalRecord("publication_authority", "reviewer", True, "scientific_publication")),
    research_phase="phase_1_construct_validity_diagnostic",
)
publish_sovereign_bundle(
    envelope, Path("sovereign_bundle"), purpose="scientific_publication",
    signing_key="replace-with-managed-enterprise-key",
)
