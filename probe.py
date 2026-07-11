"""
probe.py
========
Two probe implementations with a shared interface, plus comparison helpers:

  MassMeanProbe           — parameter-free, Marks & Tegmark (2023)
  LogisticRegressionProbe — fitted decision boundary, standard baseline
  compare_probes()        — runs directional patching with both and returns winner

Probe directions live in TWO spaces:

  - SAE feature space      — used for AUC scoring of SAE-encoded activations.
  - Raw residual space     — used for causal/directional patching, which operates
                             on the model's residual stream.

A direction fitted in SAE feature space cannot be applied directly to a raw
residual activation: the dimensionalities differ (n_features vs d_model). To
patch, the SAE-space direction is mapped back into residual space through the
SAE decoder (a feature direction f corresponds to the residual direction
W_dec @ f), then unit-normalised. Both probes expose:

  direction_for(layer)      → unit direction in the SCORING space (SAE if an SAE
                              was supplied at evaluate(), else raw)
  raw_direction_for(layer)  → unit direction in RAW residual space, suitable for
                              patching. Equal to direction_for() when no SAE was
                              used; the decoder-projected direction otherwise.

Probe interface (both classes implement):
  evaluate(activations, labels, turn_index, sae=None) → list[ProbeResult]
  direction_for(layer)     → torch.Tensor   (scoring space)
  raw_direction_for(layer) → torch.Tensor   (raw residual space, for patching)
  score(features, layer)   → float
AGPL 3. Copyright © 2026 Katherine J. Ombrellaro. 
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

if TYPE_CHECKING:
    from patching import CausalPatcher
    from sae import SparseAutoencoder

logger = logging.getLogger(__name__)


# ── Shared result dataclass ───────────────────────────────────────────────────

@dataclass
class ProbeResult:
    layer:      str
    turn_index: int
    accuracy:   float
    auc:        float            # cross-validated (held-out) AUC
    auc_in_sample: float         # in-sample AUC, for diagnostics only
    direction:  torch.Tensor     # unit probe direction in the scoring space
    n_samples:  int
    auc_cv_std: float = 0.0      # std of per-fold AUC (0.0 when not available)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _project_to_raw(
    sae: "SparseAutoencoder | None",
    feature_direction: torch.Tensor,
) -> torch.Tensor:
    """
    Map a direction in SAE feature space to a unit direction in raw residual
    space via the decoder: residual_dir = W_dec @ feature_dir.

    If sae is None the direction is already in raw space; it is returned
    unit-normalised unchanged.
    """
    if sae is None:
        fd = feature_direction.detach().cpu().float()
        return fd / (fd.norm() + 1e-8)
    W_dec = sae.decoder.weight.detach().cpu().float()      # [d_model, n_features]
    fd  = feature_direction.detach().cpu().float()         # explicit dtype/device
    raw = W_dec @ fd                                        # [d_model]
    return raw / (raw.norm() + 1e-8)


def _cv_auc(X_np: np.ndarray, y_np: np.ndarray, fit_fn, n_splits: int = 5) -> float:
    """
    Cross-validated AUC. fit_fn(X_train, y_train) must return a callable
    score_fn(X) -> 1-d score array. Falls back to in-sample if a split is
    degenerate (e.g. too few samples of a class).
    """
    n = len(y_np)
    classes, counts = np.unique(y_np, return_counts=True)
    if len(classes) < 2 or counts.min() < 2 or n < n_splits * 2:
        score_fn = fit_fn(X_np, y_np)
        return float(roc_auc_score(y_np, score_fn(X_np)))

    splits = min(n_splits, int(counts.min()))
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=0)
    scores, truths = [], []
    for tr, te in skf.split(X_np, y_np):
        if len(np.unique(y_np[tr])) < 2:
            continue
        score_fn = fit_fn(X_np[tr], y_np[tr])
        scores.append(score_fn(X_np[te]))
        truths.append(y_np[te])
    if not scores:
        score_fn = fit_fn(X_np, y_np)
        return float(roc_auc_score(y_np, score_fn(X_np)))
    return float(roc_auc_score(np.concatenate(truths), np.concatenate(scores)))


def _sae_encode_np(sae: "SparseAutoencoder", X_np: np.ndarray) -> np.ndarray:
    """Encode raw activations with a (frozen) SAE, returning a numpy feature matrix.
    Mirrors the encode path used elsewhere: move to the SAE's device, no grad."""
    with torch.no_grad():
        Z = sae.encode(
            torch.tensor(X_np, dtype=torch.float32).to(next(sae.parameters()).device)
        ).cpu()
    return Z.numpy()


def _cv_auc_sae_per_fold(
    X_raw_np: np.ndarray,
    y_np: np.ndarray,
    sae_factory: Callable[[np.ndarray], "SparseAutoencoder"],
    fit_fn,
    n_splits: int = 5,
) -> tuple[float, float, int]:
    """
    Leakage-free cross-validated AUC with the SAE fit INSIDE each fold.

    For every train/test split a fresh SAE is fit on the TRAIN fold's raw
    activations, frozen, and used to encode that fold's train and test
    partitions; the probe direction is then fit on encoded-train and scored on
    encoded-test. This closes the leakage path where a single SAE fit on the
    full set has already seen the held-out activations it is later scored on.

    Because each fold has a DIFFERENT feature basis, scores are not comparable
    across folds, so AUC is computed PER FOLD and averaged (concatenating
    cross-basis scores would be invalid). Returns (mean_auc, std_auc, n_folds).
    Falls back to a single in-sample fit (std 0.0, n_folds 1) when a stratified
    split is not possible.
    """
    n = len(y_np)
    classes, counts = np.unique(y_np, return_counts=True)
    if len(classes) < 2 or counts.min() < 2 or n < n_splits * 2:
        sae = sae_factory(X_raw_np)
        Z = _sae_encode_np(sae, X_raw_np)
        score_fn = fit_fn(Z, y_np)
        return float(roc_auc_score(y_np, score_fn(Z))), 0.0, 1

    splits = min(n_splits, int(counts.min()))
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=0)
    aucs: list[float] = []
    for tr, te in skf.split(X_raw_np, y_np):
        if len(np.unique(y_np[tr])) < 2:
            continue
        sae = sae_factory(X_raw_np[tr])          # fresh SAE, fit on TRAIN fold only
        Z_tr = _sae_encode_np(sae, X_raw_np[tr])
        Z_te = _sae_encode_np(sae, X_raw_np[te])
        score_fn = fit_fn(Z_tr, y_np[tr])
        try:
            aucs.append(float(roc_auc_score(y_np[te], score_fn(Z_te))))
        except ValueError:
            continue
    if not aucs:
        sae = sae_factory(X_raw_np)
        Z = _sae_encode_np(sae, X_raw_np)
        score_fn = fit_fn(Z, y_np)
        return float(roc_auc_score(y_np, score_fn(Z))), 0.0, 1
    arr = np.asarray(aucs)
    return float(arr.mean()), float(arr.std()), len(aucs)


# ── Shared probe base ─────────────────────────────────────────────────────────

@dataclass
class _LayerFit:
    """The probe-specific products of one layer that the shared evaluate loop
    consumes: the scoring-space unit direction, the CV fit_fn, the in-sample
    scores (for the in-sample AUC) and the hard predictions (for accuracy)."""
    unit_dir:         torch.Tensor
    fit_fn:           Callable
    in_sample_scores: np.ndarray
    preds:            torch.Tensor


class Probe:
    """Shared machinery for the linear probes.

    Subclasses implement `_fit_layer`, returning the probe-specific direction,
    CV fit_fn, in-sample scores and predictions for one layer. Everything else —
    encoding, direction storage, the raw-space projection, the (optionally
    SAE-folded) cross-validation, accuracy and ProbeResult assembly — is shared,
    so the scoring/CV plumbing lives in exactly one place.

    The per-layer numerics are unchanged from the original per-class loops: this
    is a pure extraction, verified by test_probe.py's equivalence harness.
    """

    def __init__(self):
        self._directions:     dict[str, torch.Tensor] = {}   # scoring space
        self._raw_directions: dict[str, torch.Tensor] = {}   # raw residual space

    def _fit_layer(
        self,
        X: torch.Tensor,
        X_np: np.ndarray,
        labels: torch.Tensor,
        y_np: np.ndarray,
        sae: Optional["SparseAutoencoder"],
        layer: str,
    ) -> _LayerFit:
        """Probe-specific: fit the direction for this layer and return the
        products the shared loop needs. Implemented by each subclass."""
        raise NotImplementedError

    def evaluate(
        self,
        activations: dict[str, torch.Tensor],
        labels: torch.Tensor,
        turn_index: int,
        sae: Optional["SparseAutoencoder"] = None,
        sae_factory: Optional[Callable[[np.ndarray], "SparseAutoencoder"]] = None,
    ) -> list[ProbeResult]:
        """Score each layer with cross-validated (held-out) AUC, plus an in-sample
        AUC for diagnostics.

        sae (full-fit, frozen) supplies the deployable direction and in-sample
        AUC; sae_factory, when given, drives a leakage-free held-out AUC with a
        fresh SAE fit inside each fold (_cv_auc_sae_per_fold). Without it, CV runs
        in the full-fit SAE's space (or raw space when sae is None)."""
        results = []
        y_np = labels.cpu().numpy()

        for layer, X_tensor in activations.items():
            X_raw = X_tensor.float()                 # keep RAW for per-fold SAE fitting
            if sae is not None:
                with torch.no_grad():
                    X = sae.encode(X_raw.to(next(sae.parameters()).device)).cpu()
            else:
                X = X_raw
            X_np = X.cpu().numpy()

            fit = self._fit_layer(X, X_np, labels, y_np, sae, layer)
            self._directions[layer]     = fit.unit_dir
            self._raw_directions[layer] = _project_to_raw(sae, fit.unit_dir)

            if sae is not None and sae_factory is not None:
                auc_cv, auc_cv_std, _ = _cv_auc_sae_per_fold(
                    X_raw.cpu().numpy(), y_np, sae_factory, fit.fit_fn
                )
            else:
                auc_cv, auc_cv_std = _cv_auc(X_np, y_np, fit.fit_fn), 0.0

            auc_in   = float(roc_auc_score(y_np, fit.in_sample_scores))
            accuracy = (fit.preds == labels).float().mean().item()

            results.append(ProbeResult(
                layer=layer, turn_index=turn_index,
                accuracy=round(accuracy, 4),
                auc=round(auc_cv, 4), auc_in_sample=round(auc_in, 4),
                direction=fit.unit_dir, n_samples=len(labels),
                auc_cv_std=round(auc_cv_std, 4),
            ))
        return results

    def direction_for(self, layer: str) -> torch.Tensor:
        if layer not in self._directions:
            raise ValueError(f"No direction for '{layer}'. Run evaluate() first.")
        return self._directions[layer]

    def raw_direction_for(self, layer: str) -> torch.Tensor:
        if layer not in self._raw_directions:
            raise ValueError(f"No raw direction for '{layer}'. Run evaluate() first.")
        return self._raw_directions[layer]

    def score(self, features: torch.Tensor, layer: str) -> float:
        return (features @ self.direction_for(layer)).item()


# ── Mass-Mean Probe ───────────────────────────────────────────────────────────

class MassMeanProbe(Probe):
    """
    Parameter-free linear probe following Marks & Tegmark (2023).

    Direction = mean(features | label=1) - mean(features | label=0), unit-normalised.
    No optimisation, no hyperparameters.

    AUC is reported cross-validated (the mass-mean direction is recomputed on each
    training fold and scored on the held-out fold) so that the H2 threshold is not
    evaluated on in-sample numbers. An in-sample AUC is also returned for diagnostics.

    Marks & Tegmark found this direction generalises better across datasets than
    logistic regression and hypothesise it is more causally implicated in outputs.
    Whether this holds in the maritime routing domain is tested via compare_probes().
    """

    @staticmethod
    def _mass_mean_dir(X: torch.Tensor, y_bool: torch.Tensor) -> torch.Tensor:
        direction = X[y_bool].mean(dim=0) - X[~y_bool].mean(dim=0)
        return direction / (direction.norm() + 1e-8)

    def _fit_layer(self, X, X_np, labels, y_np, sae, layer) -> _LayerFit:
        y = labels.bool()
        unit_dir = self._mass_mean_dir(X, y)

        def fit_fn(Xtr, ytr, _dir=self._mass_mean_dir):     # bind hook explicitly
            d = _dir(torch.tensor(Xtr), torch.tensor(ytr).bool())
            d_np = d.cpu().numpy()
            return lambda Xte: Xte @ d_np

        scores_in = (X @ unit_dir).cpu().numpy()
        preds     = (torch.tensor(scores_in) > np.median(scores_in)).long()
        return _LayerFit(unit_dir, fit_fn, scores_in, preds)


# ── Logistic Regression Probe ─────────────────────────────────────────────────

class LogisticRegressionProbe(Probe):
    """
    Logistic regression probe — standard baseline for comparison with MassMeanProbe.

    The probe direction is the unit-normalised weight vector of the fitted
    classifier. AUC is reported cross-validated; the stored direction is fit on
    the full data and used for patching (mapped to raw residual space via the SAE
    decoder when an SAE is supplied).

    Probability threshold (pre-registered): fixed at 0.50 across all classes and
    transfer combinations. Threshold sensitivity at 0.40/0.50/0.60 is reported as
    a diagnostic elsewhere.

    Regularization (C): unlike MassMeanProbe — a closed-form, parameter-free
    difference of means with no overfitting surface — this classifier fits a
    full weight vector in SAE feature space (n_features=8192 by default) at
    n_samples typically in the tens to low hundreds per class. That is a small-n,
    high-p regime in which sklearn's LogisticRegression default (C=1.0, light L2
    penalty calibrated for general-purpose use) under-regularises: the fitted
    direction can be substantially driven by noise dimensions rather than the
    signal MassMeanProbe is built to find directly. C defaults to 0.1 here
    (10x stronger L2 penalty than sklearn's default) as a conservative, fixed
    correction for this regime; pass a different value explicitly if a
    nested-CV sweep (see select_C_nested_cv) indicates otherwise for your data.
    Whatever value is used must be applied identically across every layer,
    attack class, and transfer-matrix cell (H2-H5) — varying C per cell would
    itself introduce a new, undocumented degree of freedom into the comparison.
    """

    def __init__(self, random_state: int = 42, C: float = 0.1):
        super().__init__()
        self.random_state  = random_state
        self.C             = C
        self._classifiers: dict[str, LogisticRegression] = {}

    def _fit_layer(self, X, X_np, labels, y_np, sae, layer) -> _LayerFit:
        clf = LogisticRegression(C=self.C, max_iter=1000, random_state=self.random_state)
        clf.fit(X_np, y_np)
        self._classifiers[layer] = clf

        w        = torch.tensor(clf.coef_[0], dtype=torch.float32)
        unit_dir = w / (w.norm() + 1e-8)

        def fit_fn(Xtr, ytr, _C=self.C, _rs=self.random_state):   # bind C/rs explicitly
            c = LogisticRegression(C=_C, max_iter=1000, random_state=_rs)
            c.fit(Xtr, ytr)
            return lambda Xte: c.predict_proba(Xte)[:, 1]

        scores_in = clf.predict_proba(X_np)[:, 1]
        preds     = torch.tensor(clf.predict(X_np))
        return _LayerFit(unit_dir, fit_fn, scores_in, preds)


def select_C_nested_cv(
    activations: dict[str, torch.Tensor],
    labels: torch.Tensor,
    layer: str,
    sae: Optional["SparseAutoencoder"] = None,
    C_grid: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0),
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Nested cross-validation for selecting LogisticRegressionProbe's C at one
    layer, rather than relying on the fixed default (0.1) documented on the
    class. Outer loop estimates held-out AUC per candidate C using the same
    _cv_auc machinery the probe itself uses; the C with the highest mean
    held-out AUC is returned.

    This is a DIAGNOSTIC / TUNING utility, not part of the pre-registered H2/H3
    measurement procedure. If used to choose a non-default C, the selected
    value must be fixed and applied identically across every layer, attack
    class, and transfer-matrix cell before any H2-H5 result is computed —
    selecting C per-cell would let the regularisation strength itself adapt to
    each comparison, undermining the comparison it is meant to stabilise.

    Returns: {'best_C': float, 'auc_by_C': {C: mean_auc}, 'layer': layer}
    """
    X_tensor = activations[layer].float()
    if sae is not None:
        with torch.no_grad():
            X_np = sae.encode(X_tensor.to(next(sae.parameters()).device)).cpu().numpy()
    else:
        X_np = X_tensor.cpu().numpy()
    y_np = labels.cpu().numpy()

    auc_by_C = {}
    for C in C_grid:
        def fit_fn(Xtr, ytr, _C=C):
            c = LogisticRegression(C=_C, max_iter=1000, random_state=random_state)
            c.fit(Xtr, ytr)
            return lambda Xte: c.predict_proba(Xte)[:, 1]
        auc_by_C[C] = _cv_auc(X_np, y_np, fit_fn, n_splits=n_splits)

    best_C = max(auc_by_C, key=auc_by_C.get)
    return {"best_C": best_C, "auc_by_C": auc_by_C, "layer": layer}



def compare_probes(
    mass_mean_probe: MassMeanProbe,
    lr_probe: LogisticRegressionProbe,
    patcher: "CausalPatcher",
    adv_payloads: list,
    legit_mean_acts: dict[str, torch.Tensor],
    env,
    best_layer: str,
    n_sample: int = 20,
    turn_index: int = -1,
) -> dict:
    """
    Select the empirically more causal probe direction at the best layer by
    comparing routing-change rates under directional patching with the
    mass-mean vs the logistic-regression direction.

    This is a method-SELECTION procedure, not a significance test of the Marks &
    Tegmark (2023) claim: it reports which direction flips more routing outputs,
    with an inconclusive band, rather than testing a null. A margin < 0.05 (5pp)
    is reported inconclusive and BOTH probes are carried forward — itself a
    finding (no privileged causal direction at this layer/sample size).

    Both directions are taken in RAW residual space (raw_direction_for) because
    directional patching operates on the residual stream, not SAE feature space.
    turn_index selects which conversation turn's tokens are patched (default -1,
    the final turn); pass the experiment's best turn to align the causal stage
    with the turn that won probe selection.

    Turn-index handling for single-turn payloads (clamp, not filter):
    This function's purpose is METHOD SELECTION — comparing mm vs lr on the same
    tokens. Internal consistency between the two directions matters more than
    strict turn alignment, so single-turn payloads (semantic, encoding) that
    cannot supply turn_index are clamped to their final turn and counted. Both
    directions see identical tokens, keeping the comparison valid.

    Contrast with _run_mean_patching in experiment.py, which reports a
    pre-registered causal rate (H5 evidence): there, mixing turn-0 and turn-2
    measurements under one estimand is incoherent, so structurally incompatible
    payloads are filtered rather than clamped.

    Also returns direction_cosine: the cosine similarity between the two raw-
    space directions at this layer. This is a supplementary diagnostic, not a
    pre-registered H2/H3 threshold — it indicates whether mass-mean and
    logistic regression are converging on the same causal direction (cosine
    near 1.0) or disagreeing (cosine near 0 or negative), which contextualises
    the causal-rate margin above.
    """
    sample     = adv_payloads[:n_sample]
    legit_mean = legit_mean_acts[best_layer]
    mm_dir     = mass_mean_probe.raw_direction_for(best_layer)
    lr_dir     = lr_probe.raw_direction_for(best_layer)

    # Direction agreement: cosine similarity between the two probe directions
    # at this layer, IN RAW RESIDUAL SPACE (both already projected there via
    # raw_direction_for / _project_to_raw, so this is a like-for-like
    # comparison — unlike a cross-scale SAE-feature-space comparison, where
    # the two spaces have no defined correspondence and cosine is not
    # meaningful). Both directions are already unit-normalised, so this is
    # simply their dot product. A high cosine (close to 1.0) means mass-mean
    # and logistic regression converge on the same causal direction despite
    # being fit by very different procedures; a low or negative cosine means
    # they disagree on what direction encodes the adversarial/legitimate
    # distinction, in which case a high causal-rate margin between them is
    # less surprising. This is a direct, falsifiable check on whether
    # LogisticRegressionProbe's C regularisation (see its docstring) is
    # pulling its direction toward, rather than away from, the closed-form
    # mass-mean estimate.
    direction_cosine = float((mm_dir.float() @ lr_dir.float()).item())

    mm_changed      = 0
    lr_changed      = 0
    total           = 0
    clamped_by_class: dict[str, int] = {}

    for payload in sample:
        turns = env.tokenize(payload)
        if not turns:
            logger.warning(
                "compare_probes: env.tokenize returned empty list for payload %r — skipping.",
                payload[:80] if isinstance(payload, str) else payload,
            )
            continue
        # Clamp turn_index to the last available turn rather than skipping.
        # Single-turn payloads (semantic, encoding) are valid patching targets
        # even when best_turn > 0; patching their only turn is the correct
        # behaviour for method selection: both directions see identical tokens,
        # keeping the mm-vs-lr comparison internally consistent.
        #
        # Contrast with _run_mean_patching (experiment.py), which reports a
        # pre-registered causal rate and therefore filters rather than clamps:
        # mixing turn-0 and turn-2 measurements under one H5 estimand would be
        # incoherent. Here, internal consistency of the comparison is the goal.
        #
        # The previous guard (abs(turn_index) > len(turns)) had an off-by-one:
        # turn_index=1 with len(turns)==1 passed the guard but raised IndexError
        # at turns[1]. The clamp makes behaviour explicit and logs when it fires.
        if turn_index >= len(turns) or turn_index < -len(turns):
            turn_index_eff = len(turns) - 1
            cls = getattr(payload, "attack_class", "unknown")
            clamped_by_class[cls] = clamped_by_class.get(cls, 0) + 1
            logger.debug(
                "compare_probes: turn_index=%d out of range for %d-turn payload "
                "(pair_id=%s, attack_class=%s) — clamping to turn %d.",
                turn_index, len(turns),
                getattr(payload, "pair_id", "?"), cls, turn_index_eff,
            )
        else:
            turn_index_eff = turn_index
        tokens = turns[turn_index_eff]

        unpatched, mm_patched = patcher.directional_patch_and_run(
            tokens, legit_mean, mm_dir, best_layer
        )
        if patcher.routing_changed(unpatched, mm_patched):
            mm_changed += 1

        _, lr_patched = patcher.directional_patch_and_run(
            tokens, legit_mean, lr_dir, best_layer
        )
        if patcher.routing_changed(unpatched, lr_patched):
            lr_changed += 1
        total += 1

    mm_rate      = mm_changed / total if total > 0 else 0.0
    lr_rate      = lr_changed / total if total > 0 else 0.0
    margin       = abs(mm_rate - lr_rate)
    inconclusive = margin < 0.05
    best_method  = (
        "inconclusive" if inconclusive
        else ("mass_mean" if mm_rate >= lr_rate else "logistic_regression")
    )

    if inconclusive:
        logger.info(
            "Probe comparison INCONCLUSIVE | mass-mean: %.1f%% | "
            "logistic regression: %.1f%% | margin: %.1f%% (< 5pp threshold) | "
            "direction cosine: %.4f. Both probes carried forward.",
            mm_rate * 100, lr_rate * 100, margin * 100, direction_cosine,
        )
    else:
        logger.info(
            "Probe comparison | mass-mean: %.1f%% | logistic regression: %.1f%% "
            "| winner: %s | margin: %.1f%% | direction cosine: %.4f",
            mm_rate * 100, lr_rate * 100, best_method, margin * 100, direction_cosine,
        )

    if clamped_by_class:
        logger.info(
            "compare_probes: clamped %d payloads to final turn (turn_index=%d "
            "out of range): %s",
            sum(clamped_by_class.values()), turn_index, clamped_by_class,
        )

    return {
        "mass_mean_causal_rate":           round(mm_rate, 4),
        "logistic_regression_causal_rate": round(lr_rate, 4),
        "best_probe_method":               best_method,
        "margin":                          round(margin, 4),
        "inconclusive":                    inconclusive,
        "n_sample":                        total,
        "layer":                           best_layer,
        "clamped_by_class":                clamped_by_class,
        "direction_cosine":                round(direction_cosine, 4),
    }


def compare_probes_multiseed(
    mass_mean_probe: MassMeanProbe,
    patcher: "CausalPatcher",
    adv_payloads: list,
    legit_mean_acts: dict[str, torch.Tensor],
    env,
    best_layer: str,
    sae_acts: dict[str, torch.Tensor],
    labels: torch.Tensor,
    sae: "SparseAutoencoder",
    seeds: list[int] = [42, 123, 7],
    n_sample: int = 20,
    C: float = 0.1,
) -> dict:
    """
    Direction stability check for LogisticRegressionProbe (Bias Control 6).

    The MassMeanProbe direction is deterministic; the LogisticRegressionProbe
    direction depends on random initialisation. If the comparison winner changes
    across seeds the result is stability-sensitive and both probes are carried
    forward regardless of the single-seed outcome.

    sae is required so the per-seed LR direction can be mapped from SAE feature
    space into raw residual space for patching, consistent with compare_probes().

    C: passed through to each per-seed LogisticRegressionProbe. Must match the
    value used for the main H2/H3 self.lr_probe (see MaritimeExperiment's
    lr_probe_C) — using a different C here than in the primary comparison would
    make this stability check answer a different question (stability of a
    differently-regularised estimator) than the one it is meant to validate.
    """
    sample     = adv_payloads[:n_sample]
    legit_mean = legit_mean_acts[best_layer]
    mm_dir     = mass_mean_probe.raw_direction_for(best_layer)
    mm_rate    = _causal_rate(patcher, sample, env, legit_mean, mm_dir, best_layer)

    winners:  list[str]   = []
    lr_rates: list[float] = []
    direction_cosines: list[float] = []

    for seed in seeds:
        seed_probe = LogisticRegressionProbe(random_state=seed, C=C)
        # sae_acts are already SAE-encoded; pass sae=None for scoring but supply
        # decoder projection explicitly afterwards.
        seed_probe.evaluate(sae_acts, labels, turn_index=0, sae=None)
        # sae_acts are in feature space, so direction_for() is a feature direction;
        # map it to raw space via the decoder for patching.
        lr_dir  = _project_to_raw(sae, seed_probe.direction_for(best_layer))
        lr_rate = _causal_rate(patcher, sample, env, legit_mean, lr_dir, best_layer)
        lr_rates.append(lr_rate)

        direction_cosine = float((mm_dir.float() @ lr_dir.float()).item())
        direction_cosines.append(direction_cosine)

        margin = abs(mm_rate - lr_rate)
        winner = (
            "inconclusive" if margin < 0.05
            else ("mass_mean" if mm_rate >= lr_rate else "logistic_regression")
        )
        winners.append(winner)
        logger.info("Stability seed %d: mm=%.1f%% lr=%.1f%% winner=%s cosine=%.4f",
                    seed, mm_rate * 100, lr_rate * 100, winner, direction_cosine)

    consistent = len(set(winners)) == 1
    return {
        "winner_consistent": consistent,
        "winners_per_seed":  dict(zip(seeds, winners)),
        "lr_rate_min":       round(min(lr_rates), 4),
        "lr_rate_max":       round(max(lr_rates), 4),
        "mm_rate":           round(mm_rate, 4),
        "direction_cosine_per_seed": dict(zip(seeds, [round(c, 4) for c in direction_cosines])),
        "direction_cosine_min":      round(min(direction_cosines), 4),
        "direction_cosine_max":      round(max(direction_cosines), 4),
        "seeds":             seeds,
    }


def _causal_rate(patcher, payloads, env, legit_mean, direction, layer) -> float:
    """Compute directional patch causal rate for a given raw-space direction.

    Always patches the final turn (turns[-1]) for every payload, regardless of
    the experiment's best_turn. This is intentional: _causal_rate is used by
    compare_probes_multiseed for BC6 direction-stability checks, where the goal
    is a consistent turn position across both mm and lr directions — not
    alignment to the probing cell. For the pre-registered H5 causal rate, see
    _run_mean_patching in experiment.py, which filters to best_turn-compatible
    payloads rather than clamping.
    """
    changed = 0
    total   = 0
    for p in payloads:
        turns = env.tokenize(p)
        if not turns:
            logger.warning(
                "_causal_rate: env.tokenize returned empty list for payload %r — skipping.",
                p[:80] if isinstance(p, str) else p,
            )
            continue
        total += 1
        changed += patcher.routing_changed(*patcher.directional_patch_and_run(
            turns[-1], legit_mean, direction, layer
        ))
    return changed / total if total > 0 else 0.0


def permutation_baseline_auc(
    train_acts: dict[str, torch.Tensor],
    train_labels: torch.Tensor,
    test_acts: dict[str, torch.Tensor],
    test_labels: torch.Tensor,
    layer: str,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> dict:
    """
    Permutation baseline for the TRANSFER AUC statistic (Bias Control 11).

    The statistic this defends is "train a mass-mean direction on one class,
    evaluate AUC on a held-out class". The correct null therefore permutes the
    TRAINING labels, refits the direction, and scores the (unpermuted) held-out
    class — mirroring the transfer procedure rather than an in-sample fit.

    Pre-registered criterion: a transfer AUC >= 0.60 is claimed as shared
    representation only if it exceeds the 95th percentile of this null by >= 0.05.
    If the null P95 > 0.55 the threshold is reported as insufficiently stringent
    for this dataset size and results are treated as exploratory.

    Args:
        train_acts/train_labels: SAE-encoded training-class activations + labels
                                 (labels are permuted to build the null).
        test_acts/test_labels:   SAE-encoded held-out-class activations + labels
                                 (used to score each permuted direction).
        layer:                   Layer at which to compute the statistic.
    """
    X_tr = train_acts[layer].float().cpu().numpy()
    y_tr = train_labels.cpu().numpy()
    X_te = test_acts[layer].float().cpu().numpy()
    y_te = test_labels.cpu().numpy()
    rng = np.random.RandomState(random_state)

    null_aucs = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y_tr)
        if len(np.unique(y_perm)) < 2:
            continue
        Xt = torch.tensor(X_tr)
        yb = torch.tensor(y_perm).bool()
        direction = Xt[yb].mean(dim=0) - Xt[~yb].mean(dim=0)
        direction = (direction / (direction.norm() + 1e-8)).cpu().numpy()
        scores = X_te @ direction
        try:
            null_aucs.append(roc_auc_score(y_te, scores))
        except ValueError:
            continue

    if not null_aucs:
        return {"error": "Permutation baseline failed — check activations and labels."}

    null_array = np.array(null_aucs)
    # Report the null AS-IS (do not fold AUC<0.5 onto >0.5). Permuting the
    # training labels yields a direction that is, on average, uninformative about
    # the held-out class, so the raw null is centred on ~0.50. A high upper tail
    # (P95) is itself the diagnostic BC11 exists to surface: it means a direction
    # fit on noise can still transfer-score highly by chance at this dataset size
    # and dimensionality, so the 0.60 threshold is not stringent and results must
    # be treated as exploratory. Folding the null (max(auc, 1-auc)) would conflate
    # "direction pointed the wrong way" with genuine signal and inflate P95.
    p95  = float(np.percentile(null_array, 95))
    mean = float(null_array.mean())
    std  = float(null_array.std())

    interpretation = (
        f"NULL P95={p95:.3f} — transfer threshold 0.60 is sufficiently stringent "
        f"(exceeds null by >= 0.05). Threshold is defensible."
        if p95 <= 0.55 else
        f"NULL P95={p95:.3f} EXCEEDS 0.55 — transfer threshold 0.60 is "
        f"insufficiently stringent for this dataset size. Transfer AUC results "
        f"should be treated as exploratory rather than confirmatory."
    )

    logger.info("Permutation baseline (transfer null): mean=%.4f p95=%.4f std=%.4f n=%d",
                mean, p95, std, len(null_aucs))

    return {
        "null_auc_mean":  round(mean, 4),
        "null_auc_p95":   round(p95, 4),
        "null_auc_std":   round(std, 4),
        "n_permutations": len(null_aucs),
        "layer":          layer,
        "interpretation": interpretation,
    }
