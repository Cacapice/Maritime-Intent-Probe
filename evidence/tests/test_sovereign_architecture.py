from pathlib import Path
import json
import pytest

from evidence.sovereign import (
    ApprovalRecord, DataSovereigntyPolicy, PrivacyAssessment, SovereignEvidenceEnvelope,
    publish_sovereign_bundle, verify_sovereign_bundle,
)


def _scientific(*, publishable=True, conclusions=("qualified evidence may inform review",)):
    return {
        "publishable": publishable,
        "publication_status": "qualified" if publishable else "blocked",
        "supported_conclusions": list(conclusions),
        "unsupported_conclusions": ["raw records may be centralized without owner approval"],
    }


def _envelope(*, phase="phase_1_construct_validity_diagnostic", publishable=True, conclusions=("qualified evidence may inform review",)):
    return SovereignEvidenceEnvelope(
        scientific_result=_scientific(publishable=publishable, conclusions=conclusions),
        sovereignty_policy=DataSovereigntyPolicy(
            data_owner="owner-community",
            custodian="designated-custodian",
            jurisdiction="owner-defined",
            permitted_purposes=("scientific_publication", "authorized_decision_support"),
            consent_basis="owner-approved governance charter",
        ),
        privacy_assessment=PrivacyAssessment(
            data_minimized=True,
            aggregate_only=True,
            privacy_mechanisms=("local computation", "aggregate-only release"),
        ),
        approvals=(
            ApprovalRecord("data_owner", "owner-delegate", True, "scientific_publication"),
            ApprovalRecord("publication_authority", "publication-reviewer", True, "scientific_publication"),
        ),
        research_phase=phase,
    )


def test_sovereign_release_allows_authorized_minimal_publication(tmp_path):
    env = _envelope()
    assert env.evaluate("scientific_publication", publication=True).allowed
    publish_sovereign_bundle(env, tmp_path, purpose="scientific_publication", signing_key="managed-test-key")
    assert verify_sovereign_bundle(tmp_path, signing_key="managed-test-key")
    payload = json.loads((tmp_path / "sovereign_evidence.json").read_text())
    assert payload["sovereignty_policy"]["raw_data_movement"] == "prohibited"
    assert "raw_records" not in payload


def test_sovereign_release_blocks_unapproved_purpose():
    decision = _envelope().evaluate("commercial_resale", publication=True)
    assert decision.status == "block"
    assert "outside" in " ".join(decision.reasons)


def test_sovereign_release_blocks_missing_owner_approval():
    env = _envelope()
    env = SovereignEvidenceEnvelope(
        scientific_result=env.scientific_result,
        sovereignty_policy=env.sovereignty_policy,
        privacy_assessment=env.privacy_assessment,
        approvals=(),
        research_phase=env.research_phase,
    )
    with pytest.raises(ValueError, match="missing publication approvals"):
        env.strict_release("scientific_publication")


def test_release_envelope_rejects_raw_records():
    with pytest.raises(ValueError, match="raw"):
        SovereignEvidenceEnvelope(
            scientific_result={"publishable": True, "raw_records": [{"id": 1}]},
            sovereignty_policy=_envelope().sovereignty_policy,
            privacy_assessment=_envelope().privacy_assessment,
        )


def test_phase_one_blocks_operational_intent_claims():
    env = _envelope(phase="phase_1_construct_validity_diagnostic", conclusions=("deployable monitor is validated",))
    assert env.evaluate("scientific_publication", publication=True).status == "block"



def test_controlled_release_requires_signature_by_default(tmp_path):
    with pytest.raises(ValueError, match="managed signing key"):
        publish_sovereign_bundle(_envelope(), tmp_path, purpose="scientific_publication")


def test_controlled_release_cannot_explicitly_disable_signature(tmp_path):
    with pytest.raises(ValueError, match="cannot opt out"):
        publish_sovereign_bundle(
            _envelope(), tmp_path, purpose="scientific_publication",
            require_signature=False,
        )
