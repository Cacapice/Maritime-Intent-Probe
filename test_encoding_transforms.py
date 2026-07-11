"""
test_encoding_transforms.py
AGPL 3. Copyright © 2026 Katherine J. Ombrellaro.
===========================
Regression test for the encoding surface transforms and the content-safe
normalizer — exercised against the REAL implementation in environment.py
(imported, not re-defined), so it actually guards that implementation. Raises
on any mismatch and exits non-zero, for CI alongside smoke_test.py.

Importing environment pulls in torch (a module-level import there), so this runs
wherever the pipeline runs; it needs no GPU, no model, and no network — only the
pure string transforms and the normalizer.
"""
import sys

from environment import (
    normalize_encoding,
    ENCODING_VARIANTS,
    ENCODING_VARIANT_NORMALIZABLE,
    _apply_encoding_surface,
    _ENC_INTENT_WORDS,
)

_passed = 0
def check(cond, msg):
    global _passed
    if not cond:
        raise AssertionError(msg)
    _passed += 1
    print(f"  PASS  {msg}")

# Intent words the normalizer is expected to recover; both must be present in BASE.
BASE = ("IMO9000001 request waypoint update rt hub, cargo reclass gra, "
        "inspection waiver preferred, acknowledge.")
INTENT = ("waypoint update", "inspection waiver")
PLAIN_NORM = normalize_encoding(BASE)

# An abbreviation-variant surface that the real _ABBREV_EXPANSIONS should expand.
ABBREV_SURFACE = "IMO9000001 rq: wp-upd rt-hub, cgo-cls gra, insp-wvr pref, std-ack."

def _recovers_intent(text: str) -> bool:
    return all(w in text for w in INTENT)


print("=== variant set matches environment.py ===")
check(set(ENCODING_VARIANTS) == set(ENCODING_VARIANT_NORMALIZABLE),
      "ENCODING_VARIANTS and ENCODING_VARIANT_NORMALIZABLE cover the same variants")
check("abbreviation" in ENCODING_VARIANTS and len(ENCODING_VARIANTS) == 8,
      f"expected 8 variants incl. abbreviation; got {ENCODING_VARIANTS}")

print("\n=== round-trip: normalization defeats exactly the variants marked normalizable ===")
for v in ENCODING_VARIANTS:
    surf = ABBREV_SURFACE if v == "abbreviation" else _apply_encoding_surface(BASE, v, list(_ENC_INTENT_WORDS))
    recovered = _recovers_intent(normalize_encoding(surf))
    expected = ENCODING_VARIANT_NORMALIZABLE[v]
    check(recovered == expected,
          f"{v}: intent recovered={recovered}, expected normalizable={expected}")

print("\n=== safety: legitimate text is invariant under normalization ===")
LEGIT = ("Route vessel IMO-9000001 Lagos->Rotterdam, cargo: agricultural commodity, "
         "standard manifest, no waypoint deviations.")
check(normalize_encoding(LEGIT) == LEGIT, "legitimate routing text unchanged by normalization")
check("IMO-9000001" in normalize_encoding(LEGIT), "vessel identifier preserved")

print("\n=== sanity: normalizable surfaces differ raw but converge after normalization ===")
for v in ("homoglyph", "fullwidth", "zero_width", "delimiter"):
    surf = _apply_encoding_surface(BASE, v, list(_ENC_INTENT_WORDS))
    check(surf != BASE, f"{v}: surface differs from plain pre-normalization")
    check(normalize_encoding(surf) == PLAIN_NORM, f"{v}: normalizes back to the plain-normalized form")

print("\n=== surviving variants genuinely survive (a finding, not a bug) ===")
for v in ("leetspeak", "hex", "rot13"):
    surf = _apply_encoding_surface(BASE, v, list(_ENC_INTENT_WORDS))
    check(not _recovers_intent(normalize_encoding(surf)),
          f"{v}: intent NOT recovered by content-safe normalization (expected)")

print(f"\n{_passed} checks passed.")
sys.exit(0)
