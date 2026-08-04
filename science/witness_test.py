"""
Model-blind witness test — general-purpose construct-validity diagnostic.
-----------------------------------------------------------------------------
Formalizes the generalization proposed in this repo's README ("A model-blind
readout can act as a witness, not merely a baseline"): when a statistic
computable WITHOUT a forward pass through the model -- a function of exogenous
surface structure only -- matches a probe's held-out AUC, that is strong
evidence the probe's apparent success is not tracking a model-internal
construct. Framed via the Rashomon set R_eps = {f : AUC(f) >= 1 - eps}
(Breiman 2001; Semenova, Rudin & Parr 2022) and predictive multiplicity
(Marx, Calmon & Ustun 2020): a model-blind witness lying in R_0 alongside the
probe means the two are indistinguishable on the observed support, and the
probe's AUC cannot be attributed to anything the model itself contributed
beyond what the witness already explains.

DELIBERATE DESIGN CHOICE: unlike BC1-BC11 (bias_controls.py), which are
specific to this study's pre-registered maritime paradigm (MaritimeEnvironment,
VaultConfig, the Pythia/TransformerLens pipeline), this module has NO
dependency on any of that. It operates on plain arrays: labels, one or more
model-blind feature vectors the caller computes however is appropriate for
their own benchmark, and the probe's own held-out AUC. That is what makes it
usable as a general methodological tool rather than a diagnosis tailored to
one benchmark -- the whole point of this extension.

WHAT THIS TEST IS NOT: it is a NECESSARY, not sufficient, check. Clearing it
(no witness matches) does not establish construct validity on its own -- it
only rules out this one failure mode (predictive multiplicity with an
exogenous statistic). Construct validity additionally requires the
design-level condition BC1-style structural audits check directly: that the
benchmark independently varies surface form and the downstream consequence
that operationalizes the target variable. A benchmark can clear the witness
test and still fail a structural audit, or vice versa -- they check different
things and neither substitutes for the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence



def _binary_roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute binary ROC AUC using average ranks, with no third-party dependency.

    This is equivalent to the Mann–Whitney U formulation and assigns average
    ranks to tied scores. Both classes must be present.
    """
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    positives = sum(label == 1 for label in labels)
    negatives = sum(label == 0 for label in labels)
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC requires at least one example from each class")

    ordered = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position][0]] = average_rank
        start = end

    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


@dataclass
class WitnessResult:
    """Result of comparing one model-blind witness against the probe's AUC."""
    witness_name: str
    witness_auc: float                 # folded: max(auc, 1-auc) -- sign of a
                                        # blind statistic vs. an arbitrary label
                                        # encoding is not meaningful
    probe_auc: float
    gap: float                         # probe_auc - witness_auc (signed)
    matches_within_tolerance: bool
    tolerance: float
    n: int                             # sample size the witness was scored on
    verdict: str                       # human-readable one-liner


def model_blind_witness_test(
    labels: Sequence[int],
    witness_features: dict[str, Sequence[float]],
    probe_auc: float,
    tolerance: float = 0.05,
) -> list[WitnessResult]:
    """
    Score each model-blind witness feature against `labels`, fold its AUC
    (sign-invariant, since an exogenous statistic has no inherent label
    polarity), and compare each to the probe's held-out AUC.

    Parameters
    ----------
    labels : array-like[n], binary (0/1)
        Same labels the probe was evaluated against.
    witness_features : dict[name -> array-like[n]]
        One or more scalar features computable WITHOUT running the model --
        e.g. turn count, token length, character count, punctuation count,
        presence/absence of a keyword. Each must be aligned to `labels` (same
        order, same n). The caller is responsible for making sure a witness is
        genuinely model-blind; this function has no way to verify that itself.
    probe_auc : float
        The probe's own held-out AUC on the same evaluation split.
    tolerance : float, default 0.05
        A witness "matches" the probe when |probe_auc - witness_auc| <=
        tolerance. 0.05 is a reasonable default (matches this repo's own
        BC11 null-exceedance margin) but is deliberately a parameter, not a
        constant -- pick a value defensible for your own benchmark's sample
        size and report it.

    Returns
    -------
    list[WitnessResult], one per witness_features entry, in insertion order.

    Raises
    ------
    ValueError if any witness_features array has a different length than
    labels, or if labels is not binary.
    """
    labels_arr = list(labels)
    uniq = set(labels_arr)
    if not uniq <= {0, 1}:
        raise ValueError(f"labels must be binary (0/1); got values {sorted(uniq)}")
    n = len(labels_arr)

    results: list[WitnessResult] = []
    for name, feat in witness_features.items():
        feat_arr = [float(value) for value in feat]
        if len(feat_arr) != n:
            raise ValueError(
                f"witness_features['{name}'] has length {len(feat_arr)}, "
                f"expected {n} (same as labels)."
            )
        raw_auc = _binary_roc_auc(labels_arr, feat_arr)
        folded_auc = max(raw_auc, 1.0 - raw_auc)
        gap = probe_auc - folded_auc
        matches = abs(gap) <= tolerance

        verdict = (
            f"WITNESS MATCH: '{name}' (AUC={folded_auc:.3f}) explains the probe's "
            f"AUC ({probe_auc:.3f}) within tolerance ({tolerance}) -- apparent "
            f"success is not attributable to anything the model contributed "
            f"beyond what '{name}' already predicts, without a forward pass."
            if matches else
            f"No match: '{name}' (AUC={folded_auc:.3f}) does not explain the "
            f"probe's AUC ({probe_auc:.3f}) (gap={gap:+.3f}, tolerance={tolerance})."
        )
        results.append(WitnessResult(
            witness_name=name, witness_auc=folded_auc, probe_auc=probe_auc,
            gap=gap, matches_within_tolerance=matches, tolerance=tolerance,
            n=n, verdict=verdict,
        ))
    return results


def summarize_witness_results(results: list[WitnessResult]) -> dict:
    """Roll up a list of WitnessResult into a single pass/fail-style summary."""
    any_match = any(r.matches_within_tolerance for r in results)
    return {
        "any_witness_matches": any_match,
        "matching_witnesses": [r.witness_name for r in results if r.matches_within_tolerance],
        "n_witnesses_tested": len(results),
        "per_witness": {
            r.witness_name: {
                "witness_auc": round(r.witness_auc, 4),
                "probe_auc": round(r.probe_auc, 4),
                "gap": round(r.gap, 4),
                "matches": r.matches_within_tolerance,
            }
            for r in results
        },
        "interpretation": (
            "At least one model-blind witness matches the probe's AUC -- "
            "construct-invalidity on the predictive-multiplicity axis, "
            "independent of any separate structural (BC1-style) finding."
            if any_match else
            "No model-blind witness tested matches the probe's AUC -- this "
            "check is cleared. Design-level construct validity (independent "
            "variation of surface form and downstream consequence) must "
            "still be verified separately; clearing this test alone does "
            "not establish it."
        ),
    }
