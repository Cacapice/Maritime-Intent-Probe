import pytest
from evidence.runtime import (
    ScientificResult, Qualification, qualification_is_monotone,
    derive_publication_status, EPISTEMIC_STRENGTH,
)


def _r(status="publishable", quals=()):
    return ScientificResult(repository="r", estimand="e", estimate=1.0,
                            estimate_kind="exact", publication_status=status,
                            qualifications=quals)


def test_qualifications_are_structured_not_free_text():
    q = _r(quals=["a floor"]).to_dict()["qualifications"][0]
    assert q["kind"] == "a floor" and q["effect"] == "weaken"


def test_status_is_derived_not_asserted():
    q = Qualification(kind="identifiability", effect="block")
    r = _r("publishable", [q])
    assert r.base_publication_status == "publishable"
    assert r.publication_status == "blocked"
    assert not r.publishable


def test_apply_weakens_and_records_same_object():
    q = Qualification(kind="censoring", effect="weaken",
                      resulting_publication_status="qualified")
    out = _r("publishable").apply(q)
    assert out.publication_status == "qualified"
    assert out.qualifications[-1] is q


def test_apply_refuses_silent_strengthening():
    q = Qualification(kind="restatement", effect="preserve")
    with pytest.raises(ValueError, match="non-monotone|inconsistent"):
        _r("qualified").apply(q, "publishable")


def test_new_evidence_is_separate_and_requires_reference():
    q = Qualification(kind="replication", effect="new_evidence",
                      resulting_publication_status="publishable",
                      evidence_reference="doi:example/replication")
    assert _r("blocked").apply(q).publication_status == "publishable"
    with pytest.raises(ValueError, match="evidence_reference"):
        derive_publication_status("blocked", [Qualification(
            kind="replication", effect="new_evidence",
            resulting_publication_status="publishable")])


def test_blocked_is_absorbing_without_new_evidence():
    q = Qualification(kind="restatement", effect="preserve",
                      resulting_publication_status="qualified")
    assert derive_publication_status("blocked", [q]) == "blocked"
    assert not qualification_is_monotone("blocked", "qualified", q)


def test_strength_ladder_is_total_and_ordered():
    assert EPISTEMIC_STRENGTH["blocked"] < EPISTEMIC_STRENGTH["exploratory"] \
        < EPISTEMIC_STRENGTH["qualified"] < EPISTEMIC_STRENGTH["publishable"]
