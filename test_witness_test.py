import pytest

from science.witness_test import model_blind_witness_test, summarize_witness_results
from science.compute_model_blind_witness import compute_witness_claim


def test_perfect_witness_matches_probe_auc():
    result = model_blind_witness_test(
        [0, 0, 1, 1], {"turn_count": [1, 1, 3, 3]}, probe_auc=1.0
    )[0]
    assert result.witness_auc == pytest.approx(1.0)
    assert result.gap == pytest.approx(0.0)
    assert result.matches_within_tolerance is True


def test_witness_auc_is_sign_invariant():
    forward = model_blind_witness_test([0, 0, 1, 1], {"x": [1, 1, 3, 3]}, 1.0)[0]
    reverse = model_blind_witness_test([0, 0, 1, 1], {"x": [3, 3, 1, 1]}, 1.0)[0]
    assert forward.witness_auc == pytest.approx(reverse.witness_auc)
    assert reverse.witness_auc == pytest.approx(1.0)


def test_nonmatching_witness_is_reported():
    result = model_blind_witness_test(
        [0, 1, 0, 1], {"constantish": [0, 0, 1, 1]}, probe_auc=1.0,
        tolerance=0.05,
    )[0]
    assert result.witness_auc == pytest.approx(0.5)
    assert result.matches_within_tolerance is False
    assert "No match" in result.verdict


def test_rejects_nonbinary_labels():
    with pytest.raises(ValueError, match="binary"):
        model_blind_witness_test([0, 2], {"x": [0, 1]}, probe_auc=1.0)


def test_rejects_misaligned_feature_length():
    with pytest.raises(ValueError, match="length"):
        model_blind_witness_test([0, 1], {"x": [0]}, probe_auc=1.0)


def test_summary_lists_matching_witnesses():
    results = model_blind_witness_test(
        [0, 0, 1, 1],
        {"turn_count": [1, 1, 3, 3], "noise": [0, 1, 0, 1]},
        probe_auc=1.0,
    )
    summary = summarize_witness_results(results)
    assert summary["any_witness_matches"] is True
    assert summary["matching_witnesses"] == ["turn_count"]
    assert summary["n_witnesses_tested"] == 2


def test_repository_wiring_computes_scoped_headline_claim():
    report = compute_witness_claim(n_pairs_per_class=8, seed=7)
    scoped = report["multi_turn_subset"]["results"][0]
    all_families = report["all_attack_families"]["results"][0]
    assert scoped["witness_auc"] == pytest.approx(1.0)
    assert scoped["matches_within_tolerance"] is True
    assert report["model_forward_pass_used"] is False
    # This guard prevents the scoped result from being silently generalized.
    assert all_families["witness_auc"] < 1.0


def test_all_family_result_is_labeled_partial_witness_and_localized():
    report = compute_witness_claim(n_pairs_per_class=8, seed=7)
    all_result = report["all_attack_families"]["results"][0]
    assert all_result["witness_auc"] == pytest.approx(0.75)
    assert report["all_family_partial_witness"] is True
    assert report["redesign_priority_families"] == ["semantic", "encoding"]
    assert report["per_attack_family"]["fragmentation"]["results"][0]["witness_auc"] == pytest.approx(1.0)
    assert report["per_attack_family"]["priming"]["results"][0]["witness_auc"] == pytest.approx(1.0)
    assert report["per_attack_family"]["semantic"]["results"][0]["witness_auc"] == pytest.approx(0.5)
    assert report["per_attack_family"]["encoding"]["results"][0]["witness_auc"] == pytest.approx(0.5)
    assert "partial model-blind witness" in report["interpretation"]


def test_witness_stack_does_not_import_torch_in_fresh_interpreter():
    import subprocess
    import sys

    code = (
        "import sys; import compute_model_blind_witness; "
        "assert 'torch' not in sys.modules; "
        "compute_model_blind_witness.compute_witness_claim(n_pairs_per_class=2); "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_dependency_light_payload_contract_matches_attack_family_shapes():
    from science.payload_templates import generate_witness_payloads

    payloads = generate_witness_payloads(n_pairs_per_class=2, seed=1)
    adversarial_counts = {
        family: {len(p.turns) for p in payloads if p.label == 1 and p.attack_class == family}
        for family in ("fragmentation", "semantic", "encoding", "priming")
    }
    assert adversarial_counts == {
        "fragmentation": {3}, "semantic": {1}, "encoding": {1}, "priming": {3}
    }
    assert {len(p.turns) for p in payloads if p.label == 0} == {1}
