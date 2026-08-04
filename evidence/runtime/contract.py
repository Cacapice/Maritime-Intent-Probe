"""Shared scientific-output contract for the Qualification Before Interpretation program.

This is the canonical implementation. It was previously vendored into each of
the 4 repositories, which drifted. Repositories now depend on this package and
hold only their own adapter functions.

This module is intentionally dependency-light.  It represents an estimate, its
uncertainty and qualifications, the evidence path that supports it, and the
conclusions that remain unsupported.  Repository-specific adapters translate
native result objects into this common publication schema.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import json
import hashlib
import hmac
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence


class EstimateKind(str, Enum):
    POINT_ESTIMATE = "point_estimate"
    MONTE_CARLO_MEAN = "monte_carlo_mean"
    BINOMIAL_PROPORTION = "binomial_proportion"
    EMPIRICAL_QUANTILE = "empirical_quantile"
    CERTIFIED_FLOOR = "certified_floor"
    LOWER_BOUND = "lower_bound"
    EXACT = "exact"
    INDETERMINATE = "indeterminate"
    EVIDENCE_PROFILE = "evidence_profile"
    IDENTIFIABILITY_GATE = "identifiability_gate"


class PublicationStatus(str, Enum):
    PUBLISHABLE = "publishable"
    QUALIFIED = "qualified"
    BLOCKED = "blocked"
    EXPLORATORY = "exploratory"


@dataclass(frozen=True)
class Uncertainty:
    kind: str
    level: float | None = None
    low: float | None = None
    high: float | None = None
    standard_error: float | None = None
    n: int | None = None
    method: str | None = None

    def __post_init__(self) -> None:
        if self.level is not None and not 0 < self.level < 1:
            raise ValueError("uncertainty level must lie in (0, 1)")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError("uncertainty low cannot exceed high")
        if self.n is not None and self.n <= 0:
            raise ValueError("uncertainty n must be positive")


@dataclass(frozen=True)
class EvidenceNode:
    id: str
    kind: str
    label: str
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceEdge:
    source: str
    target: str
    relation: str


@dataclass(frozen=True)
class EvidenceGraph:
    nodes: Sequence[EvidenceNode]
    edges: Sequence[EvidenceEdge]

    def __post_init__(self) -> None:
        ids=[n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence-node ids must be unique")
        known=set(ids)
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError("evidence edge references an unknown node")

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": [asdict(n) for n in self.nodes], "edges": [asdict(e) for e in self.edges]}


@dataclass(frozen=True)
class ScientificResult:
    repository: str
    estimand: str
    estimate: Any
    estimate_kind: str
    publication_status: str
    base_publication_status: str | None = None
    assumptions: Sequence[str] = ()
    qualifications: Sequence["Qualification | str"] = ()
    supported_conclusions: Sequence[str] = ()
    unsupported_conclusions: Sequence[str] = ()
    uncertainty: Uncertainty | None = None
    evidence_graph: EvidenceGraph | None = None
    native_payload: Mapping[str, Any] = field(default_factory=dict)
    # Default to the weakest phase. A default that grants "mature_method"
    # silently awards the strongest claim to any caller who does not think
    # about it, which is the wrong direction for a schema whose doctrine is
    # qualification before interpretation.
    research_phase: str = "exploratory"
    #: Whether this result derives from owner-held data. There is no safe
    #: default: "unrestricted" would silently exempt owner data from the
    #: privacy plane, and "owner_data" would silently claim a provenance the
    #: result may not have. Emission refuses until the caller declares one.
    data_classification: str | None = None
    schema_version: str = "1.1"

    def __post_init__(self) -> None:
        if not self.repository.strip() or not self.estimand.strip():
            raise ValueError("repository and estimand must be non-empty")
        valid_kinds={item.value for item in EstimateKind}
        if self.estimate_kind not in valid_kinds:
            raise ValueError(f"unknown estimate_kind: {self.estimate_kind!r}")
        valid_status={item.value for item in PublicationStatus}
        if self.publication_status not in valid_status:
            raise ValueError(f"unknown publication_status: {self.publication_status!r}")
        base = self.base_publication_status or self.publication_status
        if base not in valid_status:
            raise ValueError(f"unknown base_publication_status: {base!r}")
        if self.data_classification is not None and self.data_classification not in DATA_CLASSIFICATIONS:
            raise ValueError(
                f"unknown data_classification: {self.data_classification!r}; "
                f"expected one of {sorted(DATA_CLASSIFICATIONS)} or None"
            )
        # Bare strings are retained as descriptive, status-preserving metadata.
        # Only structured Qualifications may change publication strength.
        coerced=tuple(
            q if isinstance(q, Qualification)
            else Qualification(kind=str(q), effect="weaken", rationale="legacy free-text qualification")
            for q in self.qualifications
        )
        object.__setattr__(self, "qualifications", coerced)
        object.__setattr__(self, "base_publication_status", base)
        # Status is derived, never asserted independently. Adapters provide the
        # base status and qualifications; the algebra alone computes the result.
        object.__setattr__(self, "publication_status", derive_publication_status(base, coerced))

    @property
    def publishable(self) -> bool:
        return self.publication_status in {
            PublicationStatus.PUBLISHABLE.value,
            PublicationStatus.QUALIFIED.value,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repository": self.repository,
            "estimand": self.estimand,
            "estimate": self.estimate,
            "estimate_kind": self.estimate_kind,
            "base_publication_status": self.base_publication_status,
            "publication_status": self.publication_status,
            "publishable": self.publishable,
            "research_phase": self.research_phase,
            "data_classification": self.data_classification,
            "uncertainty": None if self.uncertainty is None else asdict(self.uncertainty),
            "assumptions": list(self.assumptions),
            "qualifications": [asdict(q) if isinstance(q, Qualification) else {"kind": str(q), "effect": "weaken", "target": "estimate", "resulting_estimate_kind": None, "rationale": ""} for q in self.qualifications],
            "supported_conclusions": list(self.supported_conclusions),
            "unsupported_conclusions": list(self.unsupported_conclusions),
            "evidence_graph": None if self.evidence_graph is None else self.evidence_graph.to_dict(),
            "native_payload": dict(self.native_payload),
        }

    def apply(self, qualification: "Qualification", new_status: str | None = None) -> "ScientificResult":
        """Apply one qualification and derive the new status algebraically.

        ``new_status`` is retained for compatibility, but is interpreted as the
        qualification's declared transition target and must agree with the
        algebra. Production adapters should normally omit it.
        """
        q = qualification
        if new_status is not None:
            if new_status not in EPISTEMIC_STRENGTH:
                raise ValueError(f"unknown publication_status: {new_status!r}")
            if q.resulting_publication_status not in {None, new_status}:
                raise ValueError("new_status conflicts with qualification transition target")
            q = replace(q, resulting_publication_status=new_status)
        qualifications = tuple(self.qualifications) + (q,)
        derived = derive_publication_status(self.base_publication_status or self.publication_status, qualifications)
        if new_status is not None and derived != new_status:
            raise ValueError(
                f"non-monotone or inconsistent transition: algebra derived {derived!r}, "
                f"caller requested {new_status!r}"
            )
        return replace(self, qualifications=qualifications, publication_status=derived)

    def strict_publish(self) -> dict[str, Any]:
        if not self.publishable:
            raise ValueError(
                "scientific result is not publishable: "
                f"publication_status={self.publication_status!r}"
            )
        if self.research_phase.startswith("phase_1"):
            prohibited=("deployable monitor", "validated intent detector", "completed validation", "production intent monitor")
            claims=" ".join(self.supported_conclusions).lower()
            if any(term in claims for term in prohibited):
                raise ValueError("Phase 1 results cannot publish later-phase deployment or completed-validation claims")
        return self.to_dict()



@dataclass(frozen=True)
class Qualification:
    """A transition on epistemic strength, not an annotation.

    The qualification algebra is a monotone transition system over publication
    status. ``preserve``, ``weaken``, and ``block`` may never increase strength.
    ``new_evidence`` is intentionally isolated because strengthening requires
    additional evidence rather than reinterpretation of existing evidence.
    """
    kind: str
    effect: str
    target: str = "estimate"
    resulting_estimate_kind: str | None = None
    rationale: str = ""
    resulting_publication_status: str | None = None
    evidence_reference: str | None = None

    def __post_init__(self) -> None:
        if self.effect not in {"preserve", "weaken", "block", "new_evidence"}:
            raise ValueError("qualification effect must be preserve, weaken, block, or new_evidence")
        if self.resulting_publication_status is not None and self.resulting_publication_status not in EPISTEMIC_STRENGTH:
            raise ValueError("unknown resulting_publication_status")


def derive_publication_status(base_status: str, qualifications: Sequence[Qualification]) -> str:
    """Derive publication status from a base status and ordered qualifications.

    Load-bearing invariant, for every qualification ``q``:

        strength(apply(q, s)) <= strength(s)

    unless ``q.effect == 'new_evidence'``. A blocked state is absorbing for all
    non-evidentiary operations. Adapters no longer decide final publication
    status; they provide evidence, a base status, and qualifications.
    """
    if base_status not in EPISTEMIC_STRENGTH:
        raise ValueError(f"unknown publication status: {base_status!r}")
    ordered = ["blocked", "exploratory", "qualified", "publishable"]
    current = base_status
    for q in qualifications:
        before = current
        if q.effect == "preserve":
            after = before
        elif q.effect == "block":
            after = "blocked"
        elif q.effect == "weaken":
            if before == "blocked":
                after = "blocked"
            elif q.resulting_publication_status is not None:
                after = q.resulting_publication_status
            else:
                after = ordered[max(0, ordered.index(before) - 1)]
        else:  # new_evidence
            if not q.evidence_reference:
                raise ValueError("new_evidence requires an evidence_reference")
            if q.resulting_publication_status is None:
                raise ValueError("new_evidence must declare resulting_publication_status")
            after = q.resulting_publication_status
        if not qualification_is_monotone(before, after, q):
            raise ValueError(
                f"non-monotone qualification transition: {before!r} -> {after!r} "
                f"under effect={q.effect!r}"
            )
        current = after
    return current


#: Whether a result carries data subject to the privacy plane.
#:   unrestricted   derived from no owner-held data; emission needs no decision
#:   owner_data     derived from owner-held data; emission requires a TrustDecision
#:   derived        aggregated from owner data; still requires a decision
DATA_CLASSIFICATIONS = {"unrestricted", "owner_data", "derived"}

#: Classifications that may be emitted without a sovereignty decision.
UNGATED_CLASSIFICATIONS = {"unrestricted"}


EPISTEMIC_STRENGTH = {
    "blocked": 0,
    "exploratory": 1,
    "qualified": 2,
    "publishable": 3,
}


def qualification_is_monotone(before: str, after: str, qualification: Qualification) -> bool:
    """Check the algebra's no-silent-strengthening invariant."""
    if before not in EPISTEMIC_STRENGTH or after not in EPISTEMIC_STRENGTH:
        return False
    if qualification.effect == "new_evidence":
        return bool(qualification.evidence_reference)
    if before == "blocked" and after != "blocked":
        return False
    return EPISTEMIC_STRENGTH[after] <= EPISTEMIC_STRENGTH[before]



@dataclass(frozen=True)
class QualificationRequirement:
    """One domain obligation that an adapter must express structurally.

    The shared algebra cannot infer whether a domain result is under-qualified.
    Repositories therefore declare predicates over their native payload and the
    qualification kind/status ceiling required when each predicate is true.
    """
    qualification_kind: str
    predicate: Callable[[Mapping[str, Any]], bool]
    description: str
    maximum_publication_status: str | None = None

    def __post_init__(self) -> None:
        if not self.qualification_kind.strip() or not self.description.strip():
            raise ValueError("qualification requirement kind and description are required")
        if self.maximum_publication_status is not None and self.maximum_publication_status not in EPISTEMIC_STRENGTH:
            raise ValueError("unknown maximum_publication_status")


@dataclass(frozen=True)
class AdapterPolicy:
    """Repository-owned conformance policy for one native-result adapter.

    Adapters are responsible for translating domain evidence into honest
    qualifications.  This policy makes that responsibility executable rather
    than relying on the shared algebra to guess domain semantics.
    """
    repository: str
    adapter_name: str
    required_native_fields: Sequence[str] = ()
    requirements: Sequence[QualificationRequirement] = ()
    minimum_qualification_count: int = 1
    allowed_research_phases: Sequence[str] = ()
    prohibited_supported_claim_fragments: Sequence[str] = ()
    require_evidence_graph: bool = True

    def __post_init__(self) -> None:
        if not self.repository.strip() or not self.adapter_name.strip():
            raise ValueError("adapter policy repository and name are required")
        if self.minimum_qualification_count < 0:
            raise ValueError("minimum_qualification_count cannot be negative")


@dataclass(frozen=True)
class AdapterConformanceReport:
    adapter_name: str
    conformant: bool
    violations: Sequence[str]
    active_requirements: Sequence[str]
    emitted_qualification_kinds: Sequence[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_adapter_result(result: ScientificResult, policy: AdapterPolicy) -> AdapterConformanceReport:
    """Validate that a domain adapter emitted every qualification it owes.

    This is deliberately repository-specific.  Contract conformance prevents
    local redefinition of the algebra; adapter conformance prevents a thin
    adapter from inflating claims by omitting known domain qualifications.
    """
    violations: list[str] = []
    payload = result.native_payload
    if result.repository != policy.repository:
        violations.append(f"repository mismatch: expected {policy.repository!r}")
    missing = [name for name in policy.required_native_fields if name not in payload]
    if missing:
        violations.append("missing native fields: " + ", ".join(sorted(missing)))
    emitted = tuple(q.kind for q in result.qualifications if isinstance(q, Qualification))
    if len(emitted) < policy.minimum_qualification_count:
        violations.append(
            f"adapter emitted {len(emitted)} structured qualifications; "
            f"minimum is {policy.minimum_qualification_count}"
        )
    active: list[str] = []
    for requirement in policy.requirements:
        try:
            applies = bool(requirement.predicate(payload))
        except Exception as exc:
            violations.append(
                f"qualification predicate failed for {requirement.qualification_kind!r}: {exc}"
            )
            continue
        if not applies:
            continue
        active.append(requirement.qualification_kind)
        if requirement.qualification_kind not in emitted:
            violations.append(
                f"missing qualification {requirement.qualification_kind!r}: "
                f"{requirement.description}"
            )
        if requirement.maximum_publication_status is not None:
            if EPISTEMIC_STRENGTH[result.publication_status] > EPISTEMIC_STRENGTH[requirement.maximum_publication_status]:
                violations.append(
                    f"publication_status={result.publication_status!r} exceeds "
                    f"{requirement.maximum_publication_status!r} under "
                    f"{requirement.qualification_kind!r}"
                )
    if policy.allowed_research_phases and result.research_phase not in set(policy.allowed_research_phases):
        violations.append(
            f"research_phase={result.research_phase!r} is outside the adapter policy"
        )
    claims = " ".join(result.supported_conclusions).lower()
    for fragment in policy.prohibited_supported_claim_fragments:
        if fragment.lower() in claims:
            violations.append(f"prohibited supported claim fragment: {fragment!r}")
    if policy.require_evidence_graph and result.evidence_graph is None:
        violations.append("adapter did not emit an evidence graph")
    return AdapterConformanceReport(
        adapter_name=policy.adapter_name,
        conformant=not violations,
        violations=tuple(violations),
        active_requirements=tuple(active),
        emitted_qualification_kinds=emitted,
    )


def assert_adapter_result(result: ScientificResult, policy: AdapterPolicy) -> ScientificResult:
    """Fail closed when a native adapter under-qualifies its evidence."""
    report = validate_adapter_result(result, policy)
    if not report.conformant:
        raise ValueError(
            f"adapter conformance failed for {policy.adapter_name}: "
            + "; ".join(report.violations)
        )
    return result

def scientific_result_json_schema() -> dict[str, Any]:
    """Language-neutral schema for publication payloads (schema v1.1)."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://qualification-before-interpretation.org/schema/scientific-result-1.1.json",
        "title": "ScientificResult",
        "type": "object",
        "required": ["schema_version", "repository", "estimand", "estimate_kind", "base_publication_status", "publication_status", "publishable", "research_phase"],
        "properties": {
            "schema_version": {"const": "1.1"},
            "repository": {"type": "string", "minLength": 1},
            "estimand": {"type": "string", "minLength": 1},
            "estimate": {},
            "estimate_kind": {"type": "string"},
            "base_publication_status": {"enum": [item.value for item in PublicationStatus]},
            "publication_status": {"enum": [item.value for item in PublicationStatus]},
            "publishable": {"type": "boolean"},
            "research_phase": {"type": "string", "minLength": 1},
            "uncertainty": {"type": ["object", "null"]},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "qualifications": {"type": "array"},
            "supported_conclusions": {"type": "array", "items": {"type": "string"}},
            "unsupported_conclusions": {"type": "array", "items": {"type": "string"}},
            "evidence_graph": {"type": ["object", "null"]},
            "native_payload": {"type": "object"},
        },
        "additionalProperties": True,
    }


def evidence_graph_json_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://qualification-before-interpretation.org/schema/evidence-graph-1.1.json",
        "type": "object",
        "required": ["nodes", "edges"],
        "properties": {
            "nodes": {"type": "array"},
            "edges": {"type": "array"},
        },
        "additionalProperties": False,
    }


def validate_evidence_layers(graph: EvidenceGraph) -> None:
    """Prevent unsupported jumps from observations directly to decisions."""
    kinds={node.id: node.kind for node in graph.nodes}
    for edge in graph.edges:
        if kinds[edge.source] in {"observation", "data"} and kinds[edge.target] in {"decision", "policy"}:
            raise ValueError("evidence graph may not jump directly from observation/data to decision/policy")


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n").encode("utf-8")


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def capture_provenance(*, seed: int | None = None, command: str | None = None, input_files: Sequence[str] = ()) -> dict[str, Any]:
    """Capture source, computational, data, and randomness provenance."""
    try:
        commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False, timeout=2).stdout.strip() or None
        dirty=bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False, timeout=2).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty=None, None
    inputs={}
    for raw in input_files:
        path=Path(raw)
        if path.is_file():
            inputs[str(path)] = _sha256(path)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"git_commit": commit, "dirty_tree": dirty},
        "computational": {"python": sys.version.split()[0], "platform": platform.platform(), "command": command},
        "data": {"input_sha256": inputs},
        "randomness": {"seed": seed, "reproducibility": "bitwise_when_backends_are_deterministic" if seed is not None else "not_guaranteed"},
    }


def _legacy_classification_declaration(result: ScientificResult):
    """Create an auditable declaration from the legacy result field.

    This preserves the v1.1 API while ensuring the manifest carries a basis,
    authority, provenance class, and disclosure class. New code should pass an
    explicit ``DataClassificationDeclaration``.
    """
    from .sovereign import DataClassificationDeclaration
    cls = result.data_classification
    if cls is None:
        raise ValueError("cannot emit a bundle: data_classification is undeclared")
    mapping = {
        "unrestricted": ("unknown", "unrestricted"),
        "owner_data": ("owner_held", "controlled"),
        "derived": ("derived_from_owner_data", "aggregate_release"),
    }
    provenance, disclosure = mapping[cls]
    return DataClassificationDeclaration(
        data_provenance=provenance,
        disclosure_class=disclosure,
        basis="legacy declaration carried on ScientificResult",
        declared_by=result.repository,
    )


def _authorise_emission(
    result: ScientificResult,
    result_payload: Mapping[str, Any],
    *,
    release_purpose: str,
    classification_declaration: Any,
    trust_decision: Any,
    sovereignty_policy: Any,
) -> None:
    """Verify a concrete decision against this exact payload and purpose."""
    from .sovereign import DataClassificationDeclaration, DataSovereigntyPolicy, TrustDecision
    if not isinstance(classification_declaration, DataClassificationDeclaration):
        raise TypeError("classification_declaration must be DataClassificationDeclaration")
    if classification_declaration.disclosure_class == "unrestricted":
        if result.data_classification not in {"unrestricted"}:
            raise ValueError("unrestricted declaration conflicts with result data_classification")
        return
    if not isinstance(trust_decision, TrustDecision):
        raise ValueError("controlled emission requires a TrustDecision bound to this result and purpose")
    if not isinstance(sovereignty_policy, DataSovereigntyPolicy):
        raise ValueError("controlled emission requires the current DataSovereigntyPolicy")
    trust_decision.validate_for(
        result_payload=result_payload,
        purpose=release_purpose,
        declaration=classification_declaration,
        policy=sovereignty_policy,
    )


def publish_bundle(
    result: ScientificResult,
    output_dir: str | os.PathLike[str],
    *,
    seed: int | None = None,
    strict: bool = True,
    command: str | None = None,
    input_files: Sequence[str] = (),
    trust_decision: Any = None,
    classification_declaration: Any = None,
    sovereignty_policy: Any = None,
    release_purpose: str = "general_scientific_release",
    signing_key: bytes | str | None = None,
    require_signature: bool | None = None,
) -> dict[str, str]:
    """Canonicalize, authorize, and emit a signed-capable publication bundle.

    The exact serialized scientific payload is hashed before authorization.
    Controlled decisions must match that payload, the release purpose, and the
    classification declaration. Enterprise mode may require an HMAC-authenticated
    manifest in addition to file hashes.
    """
    import hmac
    payload=result.strict_publish() if strict else result.to_dict()
    declaration = classification_declaration or _legacy_classification_declaration(result)
    _authorise_emission(
        result, payload, release_purpose=release_purpose,
        classification_declaration=declaration, trust_decision=trust_decision,
        sovereignty_policy=sovereignty_policy,
    )
    controlled = declaration.data_provenance in {"owner_held", "derived_from_owner_data"} or declaration.disclosure_class != "unrestricted"
    effective_signature_required = controlled if require_signature is None else require_signature
    if controlled and effective_signature_required is False:
        raise ValueError("controlled releases cannot opt out of manifest authenticity")
    if effective_signature_required and signing_key is None:
        raise ValueError("controlled or signature-required publication needs a managed signing key")
    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    validate_evidence_layers(result.evidence_graph) if result.evidence_graph is not None else None
    paths={
        "scientific_result": out/"scientific_result.json",
        "evidence_graph": out/"evidence_graph.json",
        "provenance": out/"provenance.json",
        "scientific_result_schema": out/"scientific_result.schema.json",
        "evidence_graph_schema": out/"evidence_graph.schema.json",
    }
    content={
        "scientific_result": payload,
        "evidence_graph": payload.get("evidence_graph") or {"nodes": [], "edges": []},
        "provenance": capture_provenance(seed=seed, command=command, input_files=input_files),
        "scientific_result_schema": scientific_result_json_schema(),
        "evidence_graph_schema": evidence_graph_json_schema(),
    }
    for key,path in paths.items(): path.write_bytes(_json_bytes(content[key]))
    result_digest="sha256:" + hashlib.sha256(_json_bytes(payload)).hexdigest()
    manifest={
        "bundle_schema_version":"1.1",
        "scientific_schema_version": payload["schema_version"],
        "repository": payload["repository"],
        "research_phase": payload["research_phase"],
        "release_purpose": release_purpose,
        "result_digest": result_digest,
        "data_classification": payload["data_classification"],
        "classification_declaration": asdict(declaration),
        "trust_decision": None if trust_decision is None else trust_decision.to_dict(),
        "signature_required": effective_signature_required,
        "policy_version": None if sovereignty_policy is None else sovereignty_policy.policy_version,
        "policy_digest": None if trust_decision is None else trust_decision.policy_digest,
        "files": {path.name:_sha256(path) for path in paths.values()},
    }
    manifest_path=out/"manifest.json"; manifest_raw=_json_bytes(manifest); manifest_path.write_bytes(manifest_raw); paths["manifest"]=manifest_path
    if signing_key is not None:
        key=signing_key.encode() if isinstance(signing_key,str) else signing_key
        sig="hmac-sha256:" + hmac.new(key, manifest_raw, hashlib.sha256).hexdigest() + "\n"
        sig_path=out/"manifest.sig"; sig_path.write_text(sig,encoding="utf-8"); paths["manifest_signature"]=sig_path
    return {name:str(path) for name,path in paths.items()}


def verify_bundle(output_dir: str | os.PathLike[str], *, signing_key: bytes | str | None = None, require_signature: bool | None = None) -> dict[str, Any]:
    """Verify hashes, schema compatibility, and optional manifest authenticity."""
    import hmac
    out=Path(output_dir); manifest_raw=(out/"manifest.json").read_bytes(); manifest=json.loads(manifest_raw)
    errors=[]
    for name,expected in manifest["files"].items():
        path=out/name
        if not path.is_file(): errors.append(f"missing:{name}")
        elif _sha256(path) != expected: errors.append(f"hash_mismatch:{name}")
    payload=json.loads((out/"scientific_result.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.1": errors.append("unsupported_scientific_schema")
    if payload.get("research_phase") != manifest.get("research_phase"): errors.append("phase_mismatch")
    sig_path=out/"manifest.sig"
    effective_signature_required = manifest.get("signature_required", False) if require_signature is None else require_signature
    if effective_signature_required and not sig_path.is_file(): errors.append("missing_manifest_signature")
    if sig_path.is_file():
        if signing_key is None:
            errors.append("signature_key_required")
        else:
            key=signing_key.encode() if isinstance(signing_key,str) else signing_key
            expected="hmac-sha256:"+hmac.new(key, manifest_raw, hashlib.sha256).hexdigest()
            if sig_path.read_text(encoding="utf-8").strip()!=expected: errors.append("invalid_manifest_signature")
    return {"valid": not errors, "errors": errors, "manifest": manifest}
