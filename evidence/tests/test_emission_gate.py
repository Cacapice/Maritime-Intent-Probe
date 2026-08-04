"""The emission gate binds authorization to exact bytes, purpose and class."""
import ast
import json
import tempfile
from pathlib import Path

def _repo_root() -> Path:
    """Locate the repository root by marker rather than by counting parents."""
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[2]


import pytest
from evidence.runtime import (
    ApprovalRecord, DataClassificationDeclaration, DataSovereigntyPolicy,
    PrivacyAssessment, ScientificResult, SovereignEvidenceEnvelope,
    publish_bundle, verify_bundle,
)

REPO = _repo_root() if "tests" in Path(__file__).parts else _repo_root()
BUNDLE_FILENAMES = {"scientific_result.json", "sovereign_evidence.json"}


def _result(**kw):
    base = dict(repository="r", estimand="e", estimate=1.0, estimate_kind="exact",
                publication_status="qualified")
    base.update(kw)
    return ScientificResult(**base)


def _controlled_decision(result, purpose="research"):
    declaration = DataClassificationDeclaration(
        data_provenance="owner_held", disclosure_class="controlled",
        basis="owner-controlled analytical source", declared_by="owner")
    env = SovereignEvidenceEnvelope(
        scientific_result=result.to_dict(),
        sovereignty_policy=DataSovereigntyPolicy(
            data_owner="owner", custodian="custodian", jurisdiction="US",
            permitted_purposes=(purpose,), consent_basis="owner authorization"),
        privacy_assessment=PrivacyAssessment(data_minimized=True, aggregate_only=True),
        classification_declaration=declaration,
        approvals=(ApprovalRecord("data_owner", "owner", True, purpose),
                   ApprovalRecord("publication_authority", "authority", True, purpose)),
    )
    return declaration, env.sovereignty_policy, env.evaluate(purpose, publication=True)


def test_undeclared_classification_cannot_be_emitted():
    with pytest.raises(ValueError, match="data_classification is undeclared"):
        publish_bundle(_result(), tempfile.mkdtemp())


def test_controlled_data_requires_concrete_bound_decision():
    with pytest.raises(ValueError, match="TrustDecision"):
        publish_bundle(_result(data_classification="owner_data"), tempfile.mkdtemp())


def test_decision_is_bound_to_exact_result_and_purpose():
    result = _result(data_classification="owner_data")
    declaration, policy, decision = _controlled_decision(result)
    with pytest.raises(ValueError, match="purpose"):
        publish_bundle(result, tempfile.mkdtemp(), trust_decision=decision,
                       classification_declaration=declaration, sovereignty_policy=policy,
                       release_purpose="public_release")
    changed = _result(data_classification="owner_data", estimate=2.0)
    with pytest.raises(ValueError, match="not bound"):
        publish_bundle(changed, tempfile.mkdtemp(), trust_decision=decision,
                       classification_declaration=declaration, sovereignty_policy=policy,
                       release_purpose="research")


def test_controlled_release_records_warrant_and_verifies_signature():
    result = _result(data_classification="owner_data")
    declaration, policy, decision = _controlled_decision(result)
    out = tempfile.mkdtemp()
    publish_bundle(result, out, trust_decision=decision,
                   classification_declaration=declaration, sovereignty_policy=policy,
                   release_purpose="research", signing_key="secret",
                   require_signature=True)
    manifest = json.loads((Path(out) / "manifest.json").read_text())
    assert manifest["trust_decision"]["decision_id"] == decision.decision_id
    assert manifest["result_digest"]
    assert manifest["classification_declaration"]["basis"]
    assert verify_bundle(out, signing_key="secret", require_signature=True)["valid"]
    assert not verify_bundle(out, signing_key="wrong", require_signature=True)["valid"]


def test_unrestricted_emits_with_auditable_declaration():
    out = tempfile.mkdtemp()
    publish_bundle(_result(data_classification="unrestricted"), out,
                   release_purpose="public_research_artifact")
    manifest = json.loads((Path(out) / "manifest.json").read_text())
    assert manifest["classification_declaration"]["disclosure_class"] == "unrestricted"
    assert manifest["release_purpose"] == "public_research_artifact"


def _python_files():
    for p in REPO.rglob("*.py"):
        if set(p.parts) & {"__pycache__", ".git", "build", "dist"} or p.name.startswith("test_"):
            continue
        yield p


def test_no_module_writes_a_publication_payload_outside_the_gate():
    offenders=[]
    for path in _python_files():
        text=path.read_text(encoding="utf-8", errors="ignore")
        for name in BUNDLE_FILENAMES:
            if name in text and "publish_bundle" not in text and "publish_sovereign_bundle" not in text:
                offenders.append(str(path.relative_to(REPO)))
    assert not offenders, offenders


def test_no_module_calls_to_dict_straight_into_a_file_write():
    offenders=[]
    for path in _python_files():
        # Figure sidecars serialize FigureResult metadata, not scientific
        # publication payloads, and are governed by the figure contract.
        if path.name == "figures.py":
            continue
        try: tree=ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError: continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call): continue
            name=getattr(node.func,"attr",None) or getattr(node.func,"id",None)
            if name not in {"dump","write_text","write_bytes"}: continue
            if any(isinstance(sub, ast.Call) and getattr(sub.func,"attr",None)=="to_dict" for arg in node.args for sub in ast.walk(arg)):
                offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert not offenders, offenders
