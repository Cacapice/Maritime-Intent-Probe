"""Sovereign-data and high-trust privacy contract for evidence release.

Authorization is bound to the exact canonical scientific payload, declared
purpose, disclosure class, policy version, and (for sovereign envelopes) the
exact envelope. Hashes provide integrity; optional HMAC signatures provide
manifest authenticity in environments with a managed shared signing secret.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest_payload(payload: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _policy_digest(policy: "DataSovereigntyPolicy") -> str:
    return _digest_payload(asdict(policy))


def _controlled_release(declaration: "DataClassificationDeclaration") -> bool:
    return (
        declaration.data_provenance in {"owner_held", "derived_from_owner_data"}
        or declaration.disclosure_class != "unrestricted"
    )


@dataclass(frozen=True)
class DataClassificationDeclaration:
    """Auditable declaration separating provenance from disclosure class."""
    data_provenance: str
    disclosure_class: str
    basis: str
    declared_by: str
    policy_version: str = "SEA-CLASS-1.0"
    declared_at_utc: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if self.data_provenance not in {"public", "synthetic", "owner_held", "derived_from_owner_data", "unknown"}:
            raise ValueError("invalid data_provenance")
        if self.disclosure_class not in {"unrestricted", "internal", "controlled", "aggregate_release"}:
            raise ValueError("invalid disclosure_class")
        if not self.basis.strip() or not self.declared_by.strip():
            raise ValueError("classification basis and authority are required")


@dataclass(frozen=True)
class DataSovereigntyPolicy:
    data_owner: str
    custodian: str
    jurisdiction: str
    permitted_purposes: Sequence[str]
    consent_basis: str
    computation_mode: str = "federated_local"
    raw_data_movement: str = "prohibited"
    retention_policy: str = "owner_controlled"
    disclosure_level: str = "qualified_aggregate_only"
    policy_version: str = "SEA-POLICY-1.2"
    supersedes_policy_version: str | None = None

    def __post_init__(self) -> None:
        if not self.data_owner.strip() or not self.custodian.strip():
            raise ValueError("data_owner and custodian must be declared")
        if not self.jurisdiction.strip() or not self.permitted_purposes:
            raise ValueError("jurisdiction and at least one permitted purpose are required")
        if self.raw_data_movement not in {"prohibited", "restricted", "owner_approved"}:
            raise ValueError("invalid raw_data_movement policy")
        if self.computation_mode not in {"federated_local", "trusted_enclave", "owner_hosted", "centralized_approved"}:
            raise ValueError("invalid computation_mode")


@dataclass(frozen=True)
class PrivacyAssessment:
    data_minimized: bool
    aggregate_only: bool
    contains_direct_identifiers: bool = False
    disclosure_risk: str = "low"
    privacy_mechanisms: Sequence[str] = ()
    residual_risks: Sequence[str] = ()

    def __post_init__(self) -> None:
        if self.disclosure_risk not in {"low", "moderate", "high", "unknown"}:
            raise ValueError("invalid disclosure_risk")


@dataclass(frozen=True)
class ApprovalRecord:
    role: str
    actor: str
    approved: bool
    scope: str
    timestamp_utc: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyMigrationRecord:
    """Auditable re-evaluation warrant for a policy-version transition.

    A policy bump invalidates decisions issued under the previous policy.  This
    record does not grandfather an old decision; it proves that the transition
    was acknowledged and that the evidence must be evaluated again.
    """
    from_policy_version: str
    to_policy_version: str
    from_policy_digest: str
    to_policy_digest: str
    previous_decision_id: str
    approved_by: str
    rationale: str
    re_evaluated_at_utc: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if self.from_policy_version == self.to_policy_version:
            raise ValueError("policy migration requires a version change")
        if not self.approved_by.strip() or not self.rationale.strip():
            raise ValueError("policy migration approver and rationale are required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_policy_migration(
    previous_policy: DataSovereigntyPolicy,
    new_policy: DataSovereigntyPolicy,
    previous_decision: "TrustDecision",
    *,
    approved_by: str,
    rationale: str,
) -> PolicyMigrationRecord:
    """Create a migration record; the new policy still requires re-evaluation."""
    if previous_decision.policy_version != previous_policy.policy_version:
        raise ValueError("previous decision was not issued under previous_policy")
    if new_policy.supersedes_policy_version != previous_policy.policy_version:
        raise ValueError("new policy must explicitly declare the version it supersedes")
    return PolicyMigrationRecord(
        from_policy_version=previous_policy.policy_version,
        to_policy_version=new_policy.policy_version,
        from_policy_digest=_policy_digest(previous_policy),
        to_policy_digest=_policy_digest(new_policy),
        previous_decision_id=previous_decision.decision_id,
        approved_by=approved_by,
        rationale=rationale,
    )


@dataclass(frozen=True)
class TrustDecision:
    """Immutable authorization bound to one payload and one release purpose."""
    status: str
    purpose: str
    reasons: Sequence[str]
    result_digest: str
    envelope_digest: str
    data_provenance: str
    disclosure_class: str
    policy_version: str
    policy_digest: str
    classification_policy_version: str
    issuer: str
    required_actions: Sequence[str] = ()
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    issued_at_utc: str = field(default_factory=_now)
    expires_at_utc: str | None = None
    revoked: bool = False

    @property
    def allowed(self) -> bool:
        return self.status in {"allow", "qualified_allow"} and not self.revoked

    def validate_for(
        self,
        *,
        result_payload: Mapping[str, Any],
        purpose: str,
        declaration: DataClassificationDeclaration,
        policy: DataSovereigntyPolicy,
        envelope_payload: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.allowed:
            raise ValueError("trust decision does not permit release")
        if self.purpose != purpose:
            raise ValueError("trust decision purpose does not match release purpose")
        if self.result_digest != _digest_payload(result_payload):
            raise ValueError("trust decision is not bound to this scientific result")
        if self.data_provenance != declaration.data_provenance or self.disclosure_class != declaration.disclosure_class:
            raise ValueError("trust decision classification does not match release declaration")
        if self.policy_version != policy.policy_version:
            raise ValueError("trust decision policy version does not match the current policy")
        if self.policy_digest != _policy_digest(policy):
            raise ValueError("trust decision is not bound to the current policy bytes")
        if self.classification_policy_version != declaration.policy_version:
            raise ValueError("trust decision classification-policy version does not match")
        if envelope_payload is not None and self.envelope_digest != _digest_payload(envelope_payload):
            raise ValueError("trust decision is not bound to this sovereign envelope")
        if self.expires_at_utc and datetime.fromisoformat(self.expires_at_utc) <= datetime.now(timezone.utc):
            raise ValueError("trust decision has expired")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "allowed": self.allowed}


@dataclass(frozen=True)
class SovereignEvidenceEnvelope:
    scientific_result: Mapping[str, Any]
    sovereignty_policy: DataSovereigntyPolicy
    privacy_assessment: PrivacyAssessment
    classification_declaration: DataClassificationDeclaration | None = None
    approvals: Sequence[ApprovalRecord] = ()
    data_provenance: Mapping[str, Any] = field(default_factory=dict)
    identity_provenance: Mapping[str, Any] = field(default_factory=dict)
    qualification_provenance: Mapping[str, Any] = field(default_factory=dict)
    policy_migrations: Sequence[PolicyMigrationRecord] = ()
    research_phase: str = "exploratory"
    schema_version: str = "sovereign-evidence-1.2"

    def __post_init__(self) -> None:
        forbidden = {"raw_records", "raw_data", "direct_identifiers", "row_level_data"}
        if forbidden.intersection(self.scientific_result):
            raise ValueError("release envelopes cannot contain raw or directly identifying records")
        if self.classification_declaration is None:
            legacy = self.scientific_result.get("data_classification")
            mapping = {
                "unrestricted": ("unknown", "unrestricted"),
                "owner_data": ("owner_held", "controlled"),
                "derived": ("derived_from_owner_data", "aggregate_release"),
            }
            if legacy not in mapping:
                # A SovereignEvidenceEnvelope is itself an owner-controlled
                # boundary. Legacy callers therefore fail closed to controlled
                # owner-held data rather than being silently marked unrestricted.
                provenance, disclosure = ("owner_held", "controlled")
            else:
                provenance, disclosure = mapping[legacy]
            object.__setattr__(self, "classification_declaration", DataClassificationDeclaration(
                data_provenance=provenance, disclosure_class=disclosure,
                basis="legacy scientific-result declaration", declared_by=self.sovereignty_policy.custodian,
            ))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "research_phase": self.research_phase,
            "scientific_result": dict(self.scientific_result),
            "sovereignty_policy": asdict(self.sovereignty_policy),
            "privacy_assessment": asdict(self.privacy_assessment),
            "classification_declaration": asdict(self.classification_declaration),
            "approvals": [a.to_dict() for a in self.approvals],
            "data_provenance": dict(self.data_provenance),
            "identity_provenance": dict(self.identity_provenance),
            "qualification_provenance": dict(self.qualification_provenance),
            "policy_migrations": [m.to_dict() for m in self.policy_migrations],
        }

    def evaluate(self, purpose: str, *, publication: bool = False) -> TrustDecision:
        reasons: list[str] = []
        actions: list[str] = []
        blocked = False
        qualified = False
        if purpose not in set(self.sovereignty_policy.permitted_purposes):
            blocked = True; reasons.append("requested purpose is outside the owner-declared purpose boundary")
        if self.privacy_assessment.contains_direct_identifiers:
            blocked = True; reasons.append("direct identifiers are present")
        if not self.privacy_assessment.data_minimized:
            qualified = True; actions.append("minimize the release payload before use")
        if not self.privacy_assessment.aggregate_only:
            qualified = True; actions.append("apply aggregate-only disclosure or obtain explicit owner approval")
        if self.privacy_assessment.disclosure_risk in {"high", "unknown"}:
            blocked = True; reasons.append("disclosure risk is not acceptable for release")
        elif self.privacy_assessment.disclosure_risk == "moderate":
            qualified = True; actions.append("record and approve residual disclosure risk")
        if publication and not bool(self.scientific_result.get("publishable", False)):
            blocked = True; reasons.append("the underlying scientific result is not publishable")
        if publication:
            approved_roles = {a.role for a in self.approvals if a.approved}
            missing = sorted({"data_owner", "publication_authority"} - approved_roles)
            if missing:
                blocked = True; reasons.append("missing publication approvals: " + ", ".join(missing))
        if self.sovereignty_policy.supersedes_policy_version:
            matches = [
                m for m in self.policy_migrations
                if m.from_policy_version == self.sovereignty_policy.supersedes_policy_version
                and m.to_policy_version == self.sovereignty_policy.policy_version
                and m.to_policy_digest == _policy_digest(self.sovereignty_policy)
            ]
            if not matches:
                blocked = True
                reasons.append(
                    "policy version changed without an explicit migration and re-evaluation record"
                )
        claims = " ".join(self.scientific_result.get("supported_conclusions", ())).lower()
        if self.research_phase.startswith("phase_1"):
            prohibited = ("deployable monitor", "production monitor", "validated intent detector", "operational intent monitor", "completed validation")
            if any(term in claims for term in prohibited):
                blocked = True; reasons.append("Phase 1 evidence cannot license operational or completed-validation claims")
        status = "block" if blocked else "qualified_allow" if qualified else "allow"
        if not reasons and not blocked:
            reasons.append("purpose, privacy, scientific, and approval checks passed")
        envelope_payload = self.to_dict()
        return TrustDecision(
            status=status,
            purpose=purpose,
            reasons=tuple(reasons),
            required_actions=tuple(actions),
            result_digest=_digest_payload(dict(self.scientific_result)),
            envelope_digest=_digest_payload(envelope_payload),
            data_provenance=self.classification_declaration.data_provenance,
            disclosure_class=self.classification_declaration.disclosure_class,
            policy_version=self.sovereignty_policy.policy_version,
            policy_digest=_policy_digest(self.sovereignty_policy),
            classification_policy_version=self.classification_declaration.policy_version,
            issuer=self.sovereignty_policy.custodian,
        )

    def strict_release(self, purpose: str) -> dict[str, Any]:
        decision = self.evaluate(purpose, publication=True)
        if not decision.allowed:
            raise ValueError("sovereign release blocked: " + "; ".join(decision.reasons))
        decision.validate_for(
            result_payload=dict(self.scientific_result), purpose=purpose,
            declaration=self.classification_declaration, policy=self.sovereignty_policy, envelope_payload=self.to_dict(),
        )
        return self.to_dict()


def sovereign_evidence_json_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://sovereign-evidence-architecture.org/schema/sovereign-evidence-1.2.json",
        "title": "SovereignEvidenceEnvelope", "type": "object",
        "required": ["schema_version", "research_phase", "scientific_result", "sovereignty_policy", "privacy_assessment", "classification_declaration", "approvals", "data_provenance", "identity_provenance", "qualification_provenance", "policy_migrations"],
        "properties": {k: {"type": "object"} for k in ["scientific_result", "sovereignty_policy", "privacy_assessment", "classification_declaration", "data_provenance", "identity_provenance", "qualification_provenance", "policy_migrations"]} | {
            "schema_version": {"const": "sovereign-evidence-1.2"}, "research_phase": {"type": "string", "minLength": 1}, "approvals": {"type": "array"}, "policy_migrations": {"type": "array"}
        },
        "additionalProperties": False,
    }


def _write_signed_manifest(out: Path, manifest: Mapping[str, Any], signing_key: bytes | str | None) -> None:
    raw = _canonical_bytes(manifest) + b"\n"
    (out / "manifest.json").write_bytes(raw)
    if signing_key is not None:
        key = signing_key.encode() if isinstance(signing_key, str) else signing_key
        sig = hmac.new(key, raw, hashlib.sha256).hexdigest()
        (out / "manifest.sig").write_text("hmac-sha256:" + sig + "\n", encoding="utf-8")


def publish_sovereign_bundle(
    envelope: SovereignEvidenceEnvelope,
    directory: str | Path,
    *,
    purpose: str,
    signing_key: bytes | str | None = None,
    require_signature: bool | None = None,
) -> dict[str, Any]:
    """Authorize canonical bytes, then atomically emit those exact bytes.

    Controlled releases are signed by default.  A caller may request a
    signature for unrestricted data as well, but may not opt controlled data
    out of authenticity protection.
    """
    controlled = _controlled_release(envelope.classification_declaration)
    effective_signature_required = controlled if require_signature is None else require_signature
    if controlled and effective_signature_required is False:
        raise ValueError("controlled releases cannot opt out of manifest authenticity")
    if effective_signature_required and signing_key is None:
        raise ValueError("controlled or signature-required publication needs a managed signing key")
    out = Path(directory); out.mkdir(parents=True, exist_ok=True)
    envelope_payload = envelope.to_dict()
    result_payload = dict(envelope.scientific_result)
    decision = envelope.evaluate(purpose, publication=True)
    if not decision.allowed:
        raise ValueError("sovereign release blocked: " + "; ".join(decision.reasons))
    decision.validate_for(result_payload=result_payload, purpose=purpose, declaration=envelope.classification_declaration, policy=envelope.sovereignty_policy, envelope_payload=envelope_payload)
    payloads = {
        "sovereign_evidence.json": envelope_payload,
        "trust_decision.json": decision.to_dict(),
        "sovereign_evidence.schema.json": sovereign_evidence_json_schema(),
    }
    hashes: dict[str, str] = {}
    for name, payload in payloads.items():
        raw = _canonical_bytes(payload) + b"\n"
        (out / name).write_bytes(raw)
        hashes[name] = "sha256:" + hashlib.sha256(raw).hexdigest()
    manifest = {
        "manifest_version": "1.2", "architecture": "Sovereign Evidence Architecture",
        "scientific_doctrine": "Qualification Before Interpretation", "purpose": purpose,
        "decision_id": decision.decision_id, "result_digest": decision.result_digest,
        "envelope_digest": decision.envelope_digest,
        "classification_declaration": asdict(envelope.classification_declaration),
        "signature_required": effective_signature_required, "policy_version": envelope.sovereignty_policy.policy_version, "policy_digest": decision.policy_digest, "files": hashes,
    }
    _write_signed_manifest(out, manifest, signing_key)
    return manifest


def verify_sovereign_bundle(directory: str | Path, *, signing_key: bytes | str | None = None, require_signature: bool | None = None) -> bool:
    root = Path(directory)
    manifest_raw = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    for name, expected in manifest["files"].items():
        actual = "sha256:" + hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            return False
    sig_path = root / "manifest.sig"
    effective_signature_required = manifest.get("signature_required", False) if require_signature is None else require_signature
    if effective_signature_required and not sig_path.is_file():
        return False
    if sig_path.is_file():
        if signing_key is None:
            return False
        key = signing_key.encode() if isinstance(signing_key, str) else signing_key
        expected = "hmac-sha256:" + hmac.new(key, manifest_raw, hashlib.sha256).hexdigest()
        if sig_path.read_text(encoding="utf-8").strip() != expected:
            return False
    return True
