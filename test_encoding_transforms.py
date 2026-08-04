"""Regression tests for encoding transforms and the content-safe normalizer."""
from science.environment import (
    ENCODING_VARIANTS,
    ENCODING_VARIANT_NORMALIZABLE,
    MaritimeEnvironment,
    _apply_encoding_surface,
    _ENC_INTENT_WORDS,
)

normalize_encoding = MaritimeEnvironment.normalize_encoding
BASE = (
    "IMO9000001 request waypoint update rt hub, cargo reclass gra, "
    "inspection waiver preferred, acknowledge."
)
INTENT = ("waypoint update", "inspection waiver")
ABBREV_SURFACE = "IMO9000001 rq: wp-upd rt-hub, cgo-cls gra, insp-wvr pref, std-ack."


def _recovers_intent(text: str) -> bool:
    return all(term in text for term in INTENT)


def test_variant_registry_is_complete():
    assert set(ENCODING_VARIANTS) == set(ENCODING_VARIANT_NORMALIZABLE)
    assert "abbreviation" in ENCODING_VARIANTS
    assert len(ENCODING_VARIANTS) == 8


def test_normalization_matches_declared_capability():
    for variant in ENCODING_VARIANTS:
        surface = ABBREV_SURFACE if variant == "abbreviation" else _apply_encoding_surface(
            BASE, variant, list(_ENC_INTENT_WORDS)
        )
        assert _recovers_intent(normalize_encoding(surface)) == ENCODING_VARIANT_NORMALIZABLE[variant]


def test_legitimate_text_is_invariant():
    legitimate = (
        "Route vessel IMO-9000001 Lagos->Rotterdam, cargo: agricultural commodity, "
        "standard manifest, no waypoint deviations."
    )
    assert normalize_encoding(legitimate) == legitimate
    assert "IMO-9000001" in normalize_encoding(legitimate)


def test_normalizable_surfaces_converge():
    plain = normalize_encoding(BASE)
    for variant in ("homoglyph", "fullwidth", "zero_width", "delimiter"):
        surface = _apply_encoding_surface(BASE, variant, list(_ENC_INTENT_WORDS))
        assert surface != BASE
        assert normalize_encoding(surface) == plain


def test_non_normalizable_variants_remain_visible_as_limit():
    for variant in ("leetspeak", "hex", "rot13"):
        surface = _apply_encoding_surface(BASE, variant, list(_ENC_INTENT_WORDS))
        assert not _recovers_intent(normalize_encoding(surface))
