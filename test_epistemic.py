from evidence_platform.epistemic import BC1Report, EpistemicStatus, maritime_counterexample_report, scientific_contract


def test_diagonal_support_blocks_semantic_interpretation():
    report = maritime_counterexample_report()
    assert report.construct_identifiable is False
    assert report.model_blind_match is True
    assert report.semantic_interpretation_supported is False
    assert report.experimental_status is EpistemicStatus.UNIDENTIFIED
    assert report.to_dict()["blocking_gate"] == "BC1"


def test_fully_crossed_design_can_pass_when_model_blind_match_absent():
    report = BC1Report.from_design(
        observed_cells={(0, 0), (0, 1), (1, 0), (1, 1)},
        model_blind_match=False,
    )
    assert report.construct_identifiable is True
    assert report.semantic_interpretation_supported is True
    assert report.strict_interpret()["experimental_status"] == "supported"


def test_strict_interpret_rejects_blocked_claim():
    report = maritime_counterexample_report()
    try:
        report.strict_interpret()
    except ValueError as exc:
        assert "blocked by BC1" in str(exc)
    else:
        raise AssertionError("blocked report must not serialize as an interpretation")


def test_scientific_contract_is_complete():
    assert set(scientific_contract()) == {"estimand", "assumptions", "uncertainty", "limits", "status"}
