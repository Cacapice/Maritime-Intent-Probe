"""
surface_geometry.py — Exploratory surface-geometry characterization.
AGPL 3. Copyright © 2026 Katherine J. Ombrellaro.

PURPOSE & SCOPE
===============
This module is EXPLORATORY and NON-GATING. None of its functions return a
pass/fail, none carry a locked threshold, and nothing here feeds a confirmatory
(H1–H5 / BC) judgment. Every function returns measurements for a human to
interpret. This separation is deliberate: the confirmatory pipeline lives in
experiment.py and the BC gates live in null_validator.py; the geometry work is
physically separated so the confirmatory/exploratory line the pre-registration
amendment draws is also a line in the codebase.

WHY THIS MODULE EXISTS
======================
The corrected BC1 construction (turn-count-matched, neutral-surface) showed that
the probe linearly separates adversarial-surface from neutral-surface payloads
with benign content on both sides, slot values matched, and routing outcome
constant (surface register and non-slot content co-vary; see
environment._benign_surface_pair), separability increasing
monotonically with depth. Environment analysis established that in this
environment intent is operationalized LEXICALLY (adversarial vs. benign = intent-
word swaps) and routing outcome is never represented (true_destination == stated
destination everywhere; the routing readout reads full-vocab next-token, not a
destination). Consequently the separation the probe finds can only be SURFACE —
intent is not separable from surface in these stimuli to begin with.

This module therefore characterizes the SURFACE geometry the design does afford.
It cannot, and does not attempt to, speak to intent: a directional surface
separation is still surface. Nothing here reopens the intent question.

THE FIVE METRICS (in analysis order)
====================================
1. norm_vs_direction()        — is the separation directional or magnitude?
                                 (logically first: everything downstream runs in
                                 the space this certifies as meaningful)
2. centroid_geometry()        — per-class centroid, intra-cluster variance (σ),
                                 pairwise centroid distance (Δμ)
3. effective_dimensionality() — participation ratio per class
4. som_manifold_with_null()   — SOM multi-directional structure + matched-
                                 Gaussian null (the null ships WITH the SOM, never
                                 separately: "SOM found multiple directions" is
                                 only meaningful relative to what a SOM finds on a
                                 structureless ellipsoid of the same mean/cov)
5. compare_light_vs_som()     — reconciles centroid/PCA (light) vs. SOM (heavy):
                                 do they agree a class is single-direction, or does
                                 the SOM find structure the light path misses?

References
----------
Piras et al. 2025, "SOM Directions are Better than One" (arXiv:2511.08379):
  - 1-neuron-SOM converges to the centroid (their Prop. 1), so SOMs generalize
    the difference-in-means / single-direction construction. This is the bridge
    that makes "we used a single separating direction, now we map the manifold"
    rigorous rather than ad hoc.
  - Their mechanistic metrics — intra-cluster variance σ (mean Euclidean distance
    of points to their class centroid) and centroid distance Δμ — are exactly the
    centroid_geometry() readouts here.
  IMPORTANT DISANALOGY: Piras validate their directions BEHAVIORALLY (ablate →
  measure Attack Success Rate against a judge). This environment has NO behavioral
  ground truth for intent, so the SOM machinery imports here ONLY as a descriptive
  geometry tool. No suppression / causal claim is borrowed, and none should be
  read into these outputs.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# MiniSom is imported lazily inside som_manifold_with_null so the rest of the
# module is usable without it (e.g. metrics 1–3 in an environment that hasn't
# installed MiniSom yet).


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _as_numpy(x) -> np.ndarray:
    """Accept a torch.Tensor or array-like; return a float64 numpy array on CPU."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _unit_norm_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Project each row onto the unit sphere (norm-control transform)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def _auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Rank-based AUC (Mann–Whitney U), no sklearn dependency. labels in {0,1}.
    Returns 0.5 for degenerate single-class input.
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    tie_sum = np.zeros(len(counts))
    np.add.at(tie_sum, inv, ranks)
    avg = tie_sum / counts
    ranks = avg[inv]
    sum_pos = ranks[labels == 1].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _mean_diff_direction(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Difference-in-means direction (the single-direction baseline)."""
    mu1 = X[labels == 1].mean(axis=0)
    mu0 = X[labels == 0].mean(axis=0)
    d = mu1 - mu0
    n = np.linalg.norm(d)
    return d / n if n > 0 else d


# ─────────────────────────────────────────────────────────────────────────────
# Metric 1 — norm vs. direction
# ─────────────────────────────────────────────────────────────────────────────
def norm_vs_direction(
    X,
    labels,
    *,
    layer: int | str | None = None,
) -> dict[str, Any]:
    """
    Is the surface separation DIRECTIONAL (genuine geometric structure) or
    MAGNITUDE (a norm/length artifact)? This is logically prior to every other
    metric: if a class pair separates only by norm, its centroid distances and
    SOM directions must be computed on unit-normed activations or they are
    meaningless.

    Four readouts on the binary contrast in `labels` (1 vs. 0):
      auc_raw      — difference-in-means projection on raw activations
      auc_normed   — same, on unit-norm-projected activations (h/‖h‖)
      auc_norm_only— a 1-D classifier using ONLY ‖h‖ as the feature
      delta        — auc_raw − auc_normed (the norm-attributable component)

    Interpretation (no thresholds — the reader judges):
      auc_norm_only ≈ auc_raw     → separation IS the norm; almost no direction.
      auc_norm_only ≈ 0.5,
        auc_normed stays high     → separation is purely directional; norm story
                                    is dead for this class.
      auc_raw > auc_normed (Δ>0)  → norm carries part of the separation; downstream
                                    geometry should use normed space.

    NOTE: directional ≠ intent. A high auc_normed means real geometric structure,
    but it is still SURFACE structure. This check resolves "geometry vs. length",
    not "surface vs. intent" (which the environment already settles as surface).
    """
    X = _as_numpy(X)
    labels = _as_numpy(labels).astype(int).ravel()

    d_raw = _mean_diff_direction(X, labels)
    auc_raw = _auc_from_scores(X @ d_raw, labels)

    Xn = _unit_norm_rows(X)
    d_norm = _mean_diff_direction(Xn, labels)
    auc_normed = _auc_from_scores(Xn @ d_norm, labels)

    norms = np.linalg.norm(X, axis=1)
    auc_norm_only = _auc_from_scores(norms, labels)
    # AUC < 0.5 just means the norm is higher for the negative class; fold to ≥0.5
    auc_norm_only = max(auc_norm_only, 1.0 - auc_norm_only)

    delta = auc_raw - auc_normed
    out = {
        "metric": "norm_vs_direction",
        "layer": layer,
        "n": int(len(labels)),
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
        "auc_raw": round(auc_raw, 4),
        "auc_normed": round(auc_normed, 4),
        "auc_norm_only": round(auc_norm_only, 4),
        "delta_raw_minus_normed": round(delta, 4),
        # descriptive flag for the reader; not a gate
        "separation_space_hint": (
            "norm-dominated" if auc_norm_only >= auc_raw - 0.02
            else "directional" if auc_normed >= auc_raw - 0.05
            else "mixed"
        ),
    }
    logger.info(
        "norm_vs_direction[layer=%s]: raw=%.3f normed=%.3f norm_only=%.3f (%s)",
        layer, auc_raw, auc_normed, auc_norm_only, out["separation_space_hint"],
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2 — centroid geometry
# ─────────────────────────────────────────────────────────────────────────────
def centroid_geometry(
    class_acts: dict[str, Any],
    *,
    normed: bool = False,
    layer: int | str | None = None,
) -> dict[str, Any]:
    """
    Per-class centroid, intra-cluster variance σ, and pairwise centroid distance
    Δμ. Direct port of the Piras et al. mechanistic metrics to the three-class
    (or N-class) surface setting.

    class_acts: {class_name: activations [n_c, d]} — one entry per surface class
                (e.g. "adversarial", "neutral", "legitimate", and optionally an
                "encoding" group).
    normed:     if True, unit-norm rows first. Set this per the metric-1 result:
                norm-dominated classes must be measured in normed space or Δμ is a
                length artifact. (For the encoding group, norm control is
                effectively mandatory — its token explosion inflates raw Δμ.)

    σ (Piras): mean Euclidean distance of a class's points to that class centroid.
    Δμ (Piras): Euclidean distance between two class centroids.
    """
    prepared: dict[str, np.ndarray] = {}
    for cls, A in class_acts.items():
        A = _as_numpy(A)
        if normed:
            A = _unit_norm_rows(A)
        prepared[cls] = A

    centroids = {cls: A.mean(axis=0) for cls, A in prepared.items()}
    sigma = {
        cls: float(np.mean(np.linalg.norm(A - centroids[cls], axis=1)))
        for cls, A in prepared.items()
    }

    classes = list(prepared.keys())
    delta_mu: dict[str, float] = {}
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            a, b = classes[i], classes[j]
            delta_mu[f"{a}|{b}"] = float(
                np.linalg.norm(centroids[a] - centroids[b])
            )

    out = {
        "metric": "centroid_geometry",
        "layer": layer,
        "normed": normed,
        "n_per_class": {cls: int(len(A)) for cls, A in prepared.items()},
        "intra_cluster_variance_sigma": {k: round(v, 4) for k, v in sigma.items()},
        "centroid_distance_delta_mu": {k: round(v, 4) for k, v in delta_mu.items()},
    }
    logger.info("centroid_geometry[layer=%s, normed=%s]: σ=%s Δμ=%s",
                layer, normed, out["intra_cluster_variance_sigma"],
                out["centroid_distance_delta_mu"])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metric 3 — effective dimensionality (participation ratio)
# ─────────────────────────────────────────────────────────────────────────────
def effective_dimensionality(
    class_acts: dict[str, Any],
    *,
    normed: bool = False,
    layer: int | str | None = None,
) -> dict[str, Any]:
    """
    Participation ratio per class: PR = (Σλ_i)² / Σλ_i², where λ_i are the
    eigenvalues of the class covariance. PR ∈ [1, d]; it is the number of
    dimensions that meaningfully carry the class's variance (1 ≈ the class lies on
    a single direction; large ≈ variance spread across many dimensions).

    PR is a prerequisite for interpreting the SOM result: if a class is low-PR,
    "project out the separating direction" is clean and a SOM finding multiple
    directions is surprising/informative; if a class is high-PR, multi-direction
    findings are less surprising and the SOM null control carries more weight.
    """
    out_classes: dict[str, Any] = {}
    for cls, A in class_acts.items():
        A = _as_numpy(A)
        if normed:
            A = _unit_norm_rows(A)
        # covariance eigenvalues (use SVD on centered data for stability)
        Ac = A - A.mean(axis=0, keepdims=True)
        # singular values s relate to eigenvalues λ = s² / (n-1)
        s = np.linalg.svd(Ac, compute_uv=False)
        lam = (s ** 2) / max(len(A) - 1, 1)
        denom = float(np.sum(lam ** 2))
        pr = float((np.sum(lam) ** 2) / denom) if denom > 0 else 0.0
        out_classes[cls] = {
            "participation_ratio": round(pr, 3),
            "n": int(len(A)),
            "ambient_dim": int(A.shape[1]),
            "top5_eigenvalue_frac": [
                round(float(v), 4) for v in (lam[:5] / np.sum(lam)) if np.sum(lam) > 0
            ],
        }
    out = {
        "metric": "effective_dimensionality",
        "layer": layer,
        "normed": normed,
        "per_class": out_classes,
    }
    logger.info("effective_dimensionality[layer=%s, normed=%s]: %s",
                layer, normed,
                {c: v["participation_ratio"] for c, v in out_classes.items()})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metric 4 — SOM multi-directional structure + matched-Gaussian null
# ─────────────────────────────────────────────────────────────────────────────
def som_manifold_with_null(
    class_acts: np.ndarray | Any,
    reference_centroid: np.ndarray | Any,
    *,
    grid: tuple[int, int] = (4, 4),
    n_iter: int = 10_000,
    sigma: float = 0.3,
    learning_rate: float = 0.01,
    seed: int = 0,
    normed: bool = False,
    layer: int | str | None = None,
) -> dict[str, Any]:
    """
    Train a SOM on ONE class's representations (Piras train on the harmful class;
    the analogue here is the adversarial-surface class), derive a direction per
    neuron by subtracting `reference_centroid` (the analogue of the harmless
    centroid — e.g. the neutral-surface centroid), and report the multi-
    directional structure: how many DISTINCT directions span the class, and how
    related they are (pairwise cosine).

    THE NULL SHIPS WITH THE SOM, NOT SEPARATELY. A SOM will always produce several
    neurons with somewhat different directions — that is what it is built to do —
    so "the SOM found multiple directions" is NOT evidence of manifold structure
    on its own. We therefore fit an identical SOM to a Gaussian null matched on the
    class mean and covariance, and report both side by side. If the real class's
    directions are more structured (more distinct directions / lower mean cosine /
    wider direction spread) than the matched-Gaussian SOM's, the manifold
    structure is real; if they look the same, the SOM is imposing structure on what
    is geometrically an ellipsoid, and the single-centroid (light-path) description
    is the honest one. Without this comparison the SOM output is uninterpretable —
    it is the geometry analogue of the activation-norm confound.

    grid (4×4) and the training hyperparameters follow Piras et al. defaults.

    Returns real-vs-null structure; does NOT decide which is correct — that is the
    reader's call, aided by compare_light_vs_som().
    """
    try:
        from minisom import MiniSom
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "som_manifold_with_null requires MiniSom. Install with "
            "`pip install minisom`. (Metrics 1–3 do not need it.)"
        ) from e

    X = _as_numpy(class_acts)
    ref = _as_numpy(reference_centroid).ravel()
    if normed:
        X = _unit_norm_rows(X)
        ref = ref / max(np.linalg.norm(ref), 1e-12)

    d = X.shape[1]
    gx, gy = grid
    n_neurons = gx * gy

    def _fit_directions(data: np.ndarray, rng_seed: int) -> dict[str, Any]:
        som = MiniSom(
            gx, gy, d,
            sigma=sigma,
            learning_rate=learning_rate,
            neighborhood_function="gaussian",
            random_seed=rng_seed,
        )
        som.random_weights_init(data)
        som.train_random(data, n_iter)
        weights = som.get_weights().reshape(n_neurons, d)   # [n_neurons, d]
        # direction per neuron = neuron − reference centroid (Piras Line 5)
        dirs = weights - ref[None, :]
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        unit = dirs / np.maximum(norms, 1e-12)
        # pairwise cosine among the unit directions
        cos = unit @ unit.T
        iu = np.triu_indices(n_neurons, k=1)
        pair_cos = cos[iu]
        # effective number of distinct directions: PR of the direction set's
        # gram spectrum (how many orthogonal directions the neuron set spans)
        s = np.linalg.svd(unit, compute_uv=False)
        lam = s ** 2
        denom = float(np.sum(lam ** 2))
        dir_pr = float((np.sum(lam) ** 2) / denom) if denom > 0 else 0.0
        return {
            "mean_pairwise_cosine": round(float(np.mean(pair_cos)), 4),
            "min_pairwise_cosine": round(float(np.min(pair_cos)), 4),
            "max_pairwise_cosine": round(float(np.max(pair_cos)), 4),
            "direction_participation_ratio": round(dir_pr, 3),
            "n_neurons": n_neurons,
        }

    real = _fit_directions(X, seed)

    # matched-Gaussian null: same mean and covariance, structureless ellipsoid
    rng = np.random.default_rng(seed)
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    # regularize covariance for sampling stability
    cov_reg = cov + 1e-6 * np.eye(d)
    null_data = rng.multivariate_normal(mu, cov_reg, size=len(X))
    null = _fit_directions(null_data, seed)

    # structure is "more real" the more the data's directions span distinct
    # directions / are less mutually redundant than the matched-Gaussian baseline.
    # We report the gaps and let the reader judge — NO threshold is applied here,
    # consistent with this module's no-gate principle. A positive pr_gap and a
    # positive cos_gap both point toward real structure; their magnitude is the
    # evidence, read against the matched-Gaussian null on the same data.
    pr_gap = real["direction_participation_ratio"] - null["direction_participation_ratio"]
    cos_gap = null["mean_pairwise_cosine"] - real["mean_pairwise_cosine"]

    out = {
        "metric": "som_manifold_with_null",
        "layer": layer,
        "normed": normed,
        "grid": list(grid),
        "n": int(len(X)),
        "real": real,
        "matched_gaussian_null": null,
        "real_minus_null": {
            "direction_pr_gap": round(pr_gap, 3),
            "mean_cosine_gap_null_minus_real": round(cos_gap, 4),
        },
        "interpretation": (
            "Read both gaps against the matched-Gaussian null: a positive "
            "direction_pr_gap means the data's neuron directions span more "
            "distinct directions than a structureless ellipsoid of the same "
            "mean/covariance would; a positive mean_cosine_gap means the data's "
            "directions are less mutually redundant. Larger gaps are stronger "
            "evidence that the class carries manifold structure beyond a single "
            "centroid. Gaps near zero mean the SOM is imposing structure on an "
            "ellipsoid and the single-centroid (light-path) description is the "
            "honest one. No threshold is applied — the magnitude is the evidence, "
            "and compare_light_vs_som() cross-checks it against the participation "
            "ratio."
        ),
    }
    logger.info(
        "som_manifold_with_null[layer=%s]: real_PR=%.2f null_PR=%.2f pr_gap=%.2f "
        "real_cos=%.3f null_cos=%.3f cos_gap=%.3f",
        layer, real["direction_participation_ratio"],
        null["direction_participation_ratio"], pr_gap,
        real["mean_pairwise_cosine"], null["mean_pairwise_cosine"], cos_gap,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metric 5 — light vs. SOM reconciliation
# ─────────────────────────────────────────────────────────────────────────────
def compare_light_vs_som(
    eff_dim_result: dict[str, Any],
    som_result: dict[str, Any],
    *,
    target_class: str,
    layer: int | str | None = None,
) -> dict[str, Any]:
    """
    Reconcile the light path (centroid/PCA → effective_dimensionality) and the
    heavy path (SOM vs. its matched-Gaussian null) for one class. The disagreement,
    when it occurs, is the finding — it is exactly the phenomenon Piras built the
    SOM for: a class that looks low-dimensional to a single-centroid description
    but that a SOM resolves into multiple distinct directions beyond what an
    ellipsoid would yield.

    This function does NOT threshold. It lays the two pieces of evidence side by
    side — the participation ratio (relative to ambient dim) and the SOM-vs-null
    gaps — and describes the four conceptual quadrants so the reader can place the
    class. The numeric evidence is reported verbatim; the quadrant language is a
    reading aid, not a verdict.

    The four quadrants:
      light low-PR  + SOM gaps large  → SINGLE-CENTROID MAY MISS STRUCTURE
            the class reads as ~one direction to PCA but the SOM exceeds its null
            — manifold treatment may be warranted. (Piras's motivating case.)
      light low-PR  + SOM gaps ~0     → SINGLE DIRECTION ADEQUATE
            both agree the class is ~one direction; manifold framing adds nothing.
      light high-PR + SOM gaps large  → DISTRIBUTED + STRUCTURED
            variance is spread AND organized into distinct directions.
      light high-PR + SOM gaps ~0     → DISTRIBUTED, UNSTRUCTURED
            variance is spread but the SOM finds nothing a Gaussian wouldn't.
    """
    pc = eff_dim_result.get("per_class", {}).get(target_class, {})
    pr = pc.get("participation_ratio")
    ambient = pc.get("ambient_dim", None)

    gaps = som_result.get("real_minus_null", {})
    pr_gap = gaps.get("direction_pr_gap")
    cos_gap = gaps.get("mean_cosine_gap_null_minus_real")

    # Descriptive PR position relative to ambient (NOT a gate — a reading aid).
    # "low" if the class occupies a small fraction of available dimensions.
    pr_fraction = (pr / ambient) if (pr is not None and ambient) else None

    out = {
        "metric": "compare_light_vs_som",
        "layer": layer,
        "target_class": target_class,
        "participation_ratio": pr,
        "ambient_dim": ambient,
        "pr_fraction_of_ambient": (round(pr_fraction, 4)
                                   if pr_fraction is not None else None),
        "som_direction_pr_gap": pr_gap,
        "som_mean_cosine_gap": cos_gap,
        "reading": (
            "Place the class using two axes, both reported above and neither "
            "thresholded here: (1) participation ratio vs. ambient dimension — a "
            "small fraction means the light path sees ~one direction, a large "
            "fraction means variance is spread; (2) the SOM-vs-null gaps — gaps "
            "well above zero mean the SOM finds structure an ellipsoid would not, "
            "gaps near zero mean it does not. Low-PR + large gaps is the case "
            "where a single centroid misses real structure (Piras's motivating "
            "finding); low-PR + ~0 gaps means a single direction is adequate; "
            "high-PR + large gaps is distributed-and-structured; high-PR + ~0 gaps "
            "is distributed noise. The disagreement between the two axes, when it "
            "occurs, is itself the result."
        ),
    }
    logger.info(
        "compare_light_vs_som[layer=%s, class=%s]: PR=%s (frac=%s) "
        "pr_gap=%s cos_gap=%s",
        layer, target_class, pr, out["pr_fraction_of_ambient"], pr_gap, cos_gap,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metric 6 — supervised projection with permutation null
# ─────────────────────────────────────────────────────────────────────────────
def supervised_projection_with_null(
    class_acts: dict[str, Any],
    *,
    pair: tuple[str, str] | None = None,
    n_splits: int = 5,
    n_perm: int = 200,
    seed: int = 0,
    normed: bool = False,
    layer: int | str | None = None,
) -> dict[str, Any]:
    """
    Held-out supervised 2D projection (LDA) of two surface regions, WITH a
    label-permutation null. A supervised projection ALWAYS separates its labels
    in-sample (it is fit to do so), so an in-sample scatter proves nothing. Two
    protections, both required:

      1. HELD-OUT separation. Axes are fit on a training split; separation is
         measured on points the fit never saw. In-sample LDA always separates;
         held-out LDA separates only if the structure generalizes. The held-out
         coordinates are what should be plotted.
      2. LABEL-PERMUTATION null. Shuffle labels, refit, measure held-out
         separation, many times → the distribution of separation achievable from
         this sample/dimension ratio with MEANINGLESS labels. Real separation is
         a finding only if it sits clearly outside this band. (This tests the
         supervised-projection failure mode — label-fitting capacity in high
         dimension — so it is the correct null here, not the SOM's Gaussian one.)

    Separation metric: cross-validated held-out between-class AUC of the 1-D LDA
    score (rank-based; no sklearn). Returned with held-out 2-D coords for plotting
    and the permutation-null distribution. No threshold; read held_out_auc against
    null_auc_mean ± null_auc_std (and p95).

    pair defaults to ("adv_full","adv_surface") — same adversarial surface idioms,
    different content vocabulary; the sharpest surface/lexical contrast available.
    This held-out AUC is also the primary SCALE-INVARIANT quantity used for
    cross-model comparison by depth fraction (see compare_across_models).
    """
    rng = np.random.default_rng(seed)
    keys = list(class_acts.keys())
    a, b = pair if pair is not None else ("adv_full", "adv_surface")
    if a not in class_acts or b not in class_acts:
        raise KeyError(
            f"supervised_projection_with_null: pair ({a!r},{b!r}) not in keys {keys}"
        )

    Xa = _as_numpy(class_acts[a])
    Xb = _as_numpy(class_acts[b])
    if normed:
        Xa, Xb = _unit_norm_rows(Xa), _unit_norm_rows(Xb)

    X = np.vstack([Xa, Xb])
    y = np.concatenate([np.ones(len(Xa)), np.zeros(len(Xb))]).astype(int)
    n, d = X.shape

    def _lda_direction(Xtr: np.ndarray, ytr: np.ndarray) -> np.ndarray:
        mu1 = Xtr[ytr == 1].mean(axis=0)
        mu0 = Xtr[ytr == 0].mean(axis=0)
        Xc = Xtr.copy()
        Xc[ytr == 1] -= mu1
        Xc[ytr == 0] -= mu0
        cov = (Xc.T @ Xc) / max(len(Xtr) - 2, 1)
        trace_mean = np.trace(cov) / d
        alpha = 0.1  # light fixed shrinkage toward scaled identity (high-dim stability)
        cov_shr = (1 - alpha) * cov + alpha * trace_mean * np.eye(d)
        try:
            w = np.linalg.solve(cov_shr, mu1 - mu0)
        except np.linalg.LinAlgError:
            w = mu1 - mu0
        nrm = np.linalg.norm(w)
        return w / nrm if nrm > 0 else w

    def _cv_heldout_auc(Xd, yd, splits, rng_local) -> float:
        idx = rng_local.permutation(len(Xd))
        fold_sizes = np.full(splits, len(Xd) // splits)
        fold_sizes[: len(Xd) % splits] += 1
        aucs, start = [], 0
        for fs in fold_sizes:
            test_idx = idx[start : start + fs]
            train_idx = np.concatenate([idx[:start], idx[start + fs:]])
            start += fs
            if len(np.unique(yd[train_idx])) < 2 or len(np.unique(yd[test_idx])) < 2:
                continue
            w = _lda_direction(Xd[train_idx], yd[train_idx])
            aucs.append(_auc_from_scores(Xd[test_idx] @ w, yd[test_idx]))
        return float(np.mean(aucs)) if aucs else 0.5

    held_out_auc = _cv_heldout_auc(X, y, n_splits, np.random.default_rng(seed))

    null_aucs = np.asarray([
        _cv_heldout_auc(X, rng.permutation(y), n_splits, np.random.default_rng(1000 + p))
        for p in range(n_perm)
    ])

    # held-out 2-D coords for plotting
    perm = np.random.default_rng(seed + 7).permutation(n)
    cut = n // 2
    tr, te = perm[:cut], perm[cut:]
    w1 = _lda_direction(X[tr], y[tr])
    Xte = X[te]
    ld1 = Xte @ w1
    resid = Xte - np.outer(ld1, w1)
    resid -= resid.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(resid, full_matrices=False)
        w2 = vt[0]
    except np.linalg.LinAlgError:
        w2 = np.zeros(d)
    ld2 = Xte @ w2
    coords = {
        "axis1_lda": [round(float(v), 4) for v in ld1],
        "axis2_resid_pc": [round(float(v), 4) for v in ld2],
        "labels": [int(v) for v in y[te]],
        "label_map": {"1": a, "0": b},
    }

    out = {
        "metric": "supervised_projection_with_null",
        "layer": layer,
        "normed": normed,
        "pair": [a, b],
        "n_per_class": {a: int(len(Xa)), b: int(len(Xb))},
        "ambient_dim": int(d),
        "held_out_auc": round(held_out_auc, 4),
        "null_auc_mean": round(float(null_aucs.mean()), 4),
        "null_auc_std": round(float(null_aucs.std()), 4),
        "null_auc_p95": round(float(np.percentile(null_aucs, 95)), 4),
        "auc_above_null_p95": bool(held_out_auc > np.percentile(null_aucs, 95)),
        "heldout_coords": coords,
        "interpretation": (
            "Compare held_out_auc against null_auc_mean ± null_auc_std (and p95). "
            "Above the null band → the two regions separate beyond label-fitting "
            "capacity at this sample/dimension ratio (genuine surface separation). "
            "Inside the band → the in-sample scatter was overfitting and the "
            "regions are not reliably distinct at this layer. No threshold applied; "
            "auc_above_null_p95 is a convenience comparison, not a gate."
        ),
    }
    logger.info(
        "supervised_projection_with_null[layer=%s, %s|%s]: held=%.3f null=%.3f±%.3f "
        "p95=%.3f above=%s",
        layer, a, b, held_out_auc, null_aucs.mean(), null_aucs.std(),
        np.percentile(null_aucs, 95), out["auc_above_null_p95"],
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metric 7 — SOM multi-direction vs. raw difference-in-means (Piras SD-vs-MD)
# ─────────────────────────────────────────────────────────────────────────────
def compare_som_vs_raw_direction(
    class_acts: dict[str, Any],
    *,
    pair: tuple[str, str] | None = None,
    grid: tuple[int, int] = (4, 4),
    n_iter: int = 10_000,
    sigma: float = 0.3,
    learning_rate: float = 0.01,
    n_splits: int = 5,
    seed: int = 0,
    normed: bool = False,
    layer: int | str | None = None,
) -> dict[str, Any]:
    """
    Piras SD-vs-MD contrast, made honest for a no-behavioral-ground-truth setting.

    SD (single direction): the raw difference-in-means direction between the two
    regions — one direction. MD (multiple directions): the SOM-derived direction
    set (each neuron trained on region `a`, minus the region-`b` centroid).

    The question the SOM apparatus exists to answer: is a single difference-in-
    means direction sufficient to describe how these two surface regions separate,
    or does the manifold (multiple directions) capture separation SD misses?

    Two readouts, both against nulls already in this module's discipline:

      1. ALIGNMENT (Piras Fig. 4): cosine of each SOM direction to the SD
         direction. High alignment everywhere → the SOM merely re-finds SD and
         multi-direction buys nothing. A spread of alignments → the SOM spans
         facets SD does not.
      2. HELD-OUT SEPARATION: held-out between-class AUC using
           (a) the single SD direction alone,
           (b) the SOM subspace (project onto the span of the SOM directions,
               then a held-out LDA in that low-dim subspace),
         compared against
           (c) a single RANDOM-direction floor, and
           (d) the matched-Gaussian SOM null subspace (the same construction on a
               structureless ellipsoid of region `a`'s mean/covariance).
         "MD beats SD" counts only if (b) exceeds (a) AND (b) exceeds both (c) the
         random floor and (d) the Gaussian-null subspace — i.e. the SOM subspace
         separates better than the single direction AND better than what a
         structureless blob's SOM subspace would.

    No threshold; the deltas are reported for the reader. Default pair
    ("adv_full","adv_surface").
    """
    try:
        from minisom import MiniSom
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "compare_som_vs_raw_direction requires MiniSom (`pip install minisom`)."
        ) from e

    rng = np.random.default_rng(seed)
    a, b = pair if pair is not None else ("adv_full", "adv_surface")
    Xa = _as_numpy(class_acts[a])
    Xb = _as_numpy(class_acts[b])
    if normed:
        Xa, Xb = _unit_norm_rows(Xa), _unit_norm_rows(Xb)
    d = Xa.shape[1]
    gx, gy = grid
    n_neurons = gx * gy

    ref_centroid = Xb.mean(axis=0)

    # SD direction
    sd = Xa.mean(axis=0) - ref_centroid
    sd_unit = sd / max(np.linalg.norm(sd), 1e-12)

    def _som_dirs(train: np.ndarray, ref: np.ndarray, rng_seed: int) -> np.ndarray:
        som = MiniSom(gx, gy, d, sigma=sigma, learning_rate=learning_rate,
                      neighborhood_function="gaussian", random_seed=rng_seed)
        som.random_weights_init(train)
        som.train_random(train, n_iter)
        w = som.get_weights().reshape(n_neurons, d)
        dirs = w - ref[None, :]
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        return dirs / np.maximum(norms, 1e-12)

    som_unit = _som_dirs(Xa, ref_centroid, seed)

    # 1. alignment of each SOM direction to SD
    align = som_unit @ sd_unit
    alignment = {
        "mean_cosine_to_sd": round(float(np.mean(np.abs(align))), 4),
        "min_abs_cosine_to_sd": round(float(np.min(np.abs(align))), 4),
        "max_abs_cosine_to_sd": round(float(np.max(np.abs(align))), 4),
        "spread_std": round(float(np.std(align)), 4),
    }

    # 2. held-out separation: SD alone vs SOM subspace vs random vs gaussian-null subspace
    X = np.vstack([Xa, Xb])
    y = np.concatenate([np.ones(len(Xa)), np.zeros(len(Xb))]).astype(int)
    n = len(X)

    def _heldout_auc_in_subspace(basis: np.ndarray, rng_local) -> float:
        """basis: [k, d] rows span the subspace; held-out LDA within it."""
        # orthonormalize basis
        q, _ = np.linalg.qr(basis.T)   # [d, k]
        Z = X @ q                       # project all points → [n, k]
        idx = rng_local.permutation(n)
        fold_sizes = np.full(n_splits, n // n_splits)
        fold_sizes[: n % n_splits] += 1
        aucs, start = [], 0
        for fs in fold_sizes:
            te = idx[start:start + fs]; trn = np.concatenate([idx[:start], idx[start+fs:]]); start += fs
            if len(np.unique(y[trn])) < 2 or len(np.unique(y[te])) < 2:
                continue
            mu1 = Z[trn][y[trn] == 1].mean(axis=0); mu0 = Z[trn][y[trn] == 0].mean(axis=0)
            Zc = Z[trn].copy(); Zc[y[trn]==1]-=mu1; Zc[y[trn]==0]-=mu0
            k = Z.shape[1]
            cov = (Zc.T @ Zc)/max(len(trn)-2,1) + 1e-6*np.eye(k)
            try: w = np.linalg.solve(cov, mu1-mu0)
            except np.linalg.LinAlgError: w = mu1-mu0
            aucs.append(_auc_from_scores(Z[te] @ w, y[te]))
        return float(np.mean(aucs)) if aucs else 0.5

    auc_sd = _heldout_auc_in_subspace(sd_unit[None, :], np.random.default_rng(seed))
    auc_som = _heldout_auc_in_subspace(som_unit, np.random.default_rng(seed))

    # random-direction floor (single random direction)
    rdir = rng.standard_normal(d); rdir /= np.linalg.norm(rdir)
    auc_rand = _heldout_auc_in_subspace(rdir[None, :], np.random.default_rng(seed))

    # matched-Gaussian-null SOM subspace
    mu = Xa.mean(axis=0); cov = np.cov(Xa, rowvar=False) + 1e-6*np.eye(d)
    Xa_null = rng.multivariate_normal(mu, cov, size=len(Xa))
    som_null = _som_dirs(Xa_null, ref_centroid, seed)
    auc_som_null = _heldout_auc_in_subspace(som_null, np.random.default_rng(seed))

    md_beats_sd = auc_som - auc_sd
    md_beats_random = auc_som - auc_rand
    md_beats_gaussian = auc_som - auc_som_null

    out = {
        "metric": "compare_som_vs_raw_direction",
        "layer": layer,
        "normed": normed,
        "pair": [a, b],
        "grid": list(grid),
        "alignment_to_sd": alignment,
        "heldout_auc": {
            "sd_single_direction": round(auc_sd, 4),
            "som_subspace": round(auc_som, 4),
            "random_direction_floor": round(auc_rand, 4),
            "gaussian_null_som_subspace": round(auc_som_null, 4),
        },
        "som_minus": {
            "vs_sd": round(md_beats_sd, 4),
            "vs_random_floor": round(md_beats_random, 4),
            "vs_gaussian_null": round(md_beats_gaussian, 4),
        },
        "interpretation": (
            "MD (SOM subspace) is evidence of multi-directional structure only if "
            "som_subspace exceeds sd_single_direction AND exceeds BOTH the "
            "random_direction_floor and the gaussian_null_som_subspace. If "
            "som ≈ sd, a single difference-in-means is adequate. If the alignment "
            "cosines are all near 1, the SOM merely re-finds SD. No threshold "
            "applied — the three som_minus deltas are the evidence."
        ),
    }
    logger.info(
        "compare_som_vs_raw_direction[layer=%s, %s|%s]: sd=%.3f som=%.3f rand=%.3f "
        "gnull=%.3f | som-sd=%.3f som-rand=%.3f som-gnull=%.3f",
        layer, a, b, auc_sd, auc_som, auc_rand, auc_som_null,
        md_beats_sd, md_beats_random, md_beats_gaussian,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metric 8 — cross-model comparison (scale-invariant, depth-fraction aligned)
# ─────────────────────────────────────────────────────────────────────────────
def compare_across_models(
    model_results: dict[str, dict[Any, dict[str, Any]]],
    model_n_layers: dict[str, int],
    *,
    primary_key_path: tuple[str, ...] = ("supervised", "held_out_auc"),
) -> dict[str, Any]:
    """
    Compare two (or more) models' per-layer geometry on SCALE-INVARIANT quantities
    ONLY, aligned by DEPTH FRACTION (block / n_layers), never by raw layer index.

    Why this discipline is mandatory: Pythia-1.4B (d_model 2048, 24 layers) and
    6.9B (d_model 4096, 32 layers) live in non-commensurable spaces. Raw Δμ and
    raw σ CANNOT be compared across them — only bounded/ratio quantities can:
    AUCs (0–1), participation-ratio / ambient-dim, SOM-vs-null gaps (differences).
    And "block 11" is depth-fraction 0.46 in the small model but 0.34 in the
    large one, so alignment is by fraction with interpolation, not by index. This
    function refuses raw-magnitude inputs by construction: it only reads the
    scale-invariant fields named in primary_key_path (default the held-out AUC
    from supervised_projection_with_null, the cleanest scale-invariant axis).

    model_results: {model_name: {layer_index: per_layer_dict}} where each
                   per_layer_dict contains nested results keyed so that
                   primary_key_path resolves to a scalar (e.g.
                   per_layer["supervised"]["held_out_auc"]).
    model_n_layers: {model_name: total_layer_count} for depth-fraction conversion.

    Returns, per model: the (depth_fraction, value) trajectory; plus a paired
    comparison interpolated onto a common depth-fraction grid so the two models'
    trajectories can be overlaid and differenced. No threshold — the overlaid
    trajectories and their interpolated difference are the reported evidence.
    """
    def _resolve(d: dict, path: tuple[str, ...]):
        cur = d
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    trajectories: dict[str, list[tuple[float, float]]] = {}
    for model, per_layer in model_results.items():
        nL = model_n_layers[model]
        pts = []
        for layer_idx, res in per_layer.items():
            try:
                frac = float(layer_idx) / float(nL)
            except (TypeError, ValueError):
                continue
            val = _resolve(res, primary_key_path)
            if val is None:
                continue
            pts.append((round(frac, 4), float(val)))
        pts.sort()
        trajectories[model] = pts

    # common depth-fraction grid for overlay/difference (union of observed fracs,
    # then linear interpolation of each model's trajectory onto it)
    all_fracs = sorted({f for pts in trajectories.values() for f, _ in pts})
    interpolated: dict[str, list[float | None]] = {}
    for model, pts in trajectories.items():
        if not pts:
            interpolated[model] = [None] * len(all_fracs)
            continue
        fx = np.array([f for f, _ in pts])
        fy = np.array([v for _, v in pts])
        interp = []
        for g in all_fracs:
            if g < fx.min() or g > fx.max():
                interp.append(None)            # don't extrapolate beyond observed depth
            else:
                interp.append(round(float(np.interp(g, fx, fy)), 4))
        interpolated[model] = interp

    # pairwise difference where both models have a value
    models = list(trajectories.keys())
    pairwise_diff = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            mi, mj = models[i], models[j]
            diffs = []
            for k, g in enumerate(all_fracs):
                vi, vj = interpolated[mi][k], interpolated[mj][k]
                diffs.append(None if (vi is None or vj is None)
                             else round(vi - vj, 4))
            pairwise_diff[f"{mi}_minus_{mj}"] = diffs

    out = {
        "metric": "compare_across_models",
        "scale_invariant_quantity": list(primary_key_path),
        "depth_fraction_grid": all_fracs,
        "per_model_trajectory": {
            m: [{"depth_fraction": f, "value": v} for f, v in pts]
            for m, pts in trajectories.items()
        },
        "interpolated_on_common_grid": interpolated,
        "pairwise_difference": pairwise_diff,
        "interpretation": (
            "Overlay per_model_trajectory by depth_fraction (NOT layer index). "
            "Agreement across models → the surface-separation depth profile is a "
            "property of the task/stimuli, not of one model's scale. Divergence "
            "(see pairwise_difference) → the larger model organizes the surface "
            "regions differently with depth. Only scale-invariant quantities are "
            "compared; raw Δμ / σ are excluded by construction. 'None' entries are "
            "depth fractions outside a model's observed probe-layer range (no "
            "extrapolation)."
        ),
    }
    logger.info(
        "compare_across_models[%s]: models=%s grid=%d points",
        "/".join(primary_key_path), list(trajectories.keys()), len(all_fracs),
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metric 9 — separability decomposition (length / lexical / residual)
#   The Tier-1 quantitative headline. Partitions the cross-region separability
#   into the part attributable to TOKEN LENGTH, the part attributable to LEXICAL
#   SURFACE overlap, and the RESIDUAL — all in NORM-CONTROLLED (unit-normed)
#   activation space, with a LENGTH-MATCHED subset as a confirmatory survival
#   check. This is the metric that lets the work state, concretely, "X% of the
#   apparent separability is length, Y% lexical, Z% residual" — the quantified
#   form of the construct-validity / confound finding.
#
#   Design (per the agreed settings):
#     - Activation-norm artifact CONTROLLED: runs in unit-normed space (raw
#       reported alongside as the uncontrolled comparison).
#     - Stimulus length NOT equalized: length stays a measured predictor, entered
#       FIRST as the regressor whose explained separability is the "length
#       component"; what remains after removing it is attributed further.
#     - Confirmatory check: separability on a length-matched subset (pairs of
#       near-equal token length). Survives → non-length component is real;
#       collapses → length was carrying it.
#
#   NON-GATING. No thresholds. Reports attributions + the matched-subset survival.
# ─────────────────────────────────────────────────────────────────────────────
def separability_decomposition(
    class_acts: dict[str, Any],
    token_lengths: dict[str, Any],
    *,
    pair: tuple[str, str] = ("adv_full", "adv_surface"),
    lexical_overlap: dict[str, Any] | None = None,
    n_splits: int = 5,
    n_perm: int = 200,
    length_match_tol: int = 2,
    seed: int = 0,
    layer: int | None = None,
) -> dict[str, Any]:
    """
    Decompose held-out separability between two regions into length-, lexical-,
    and residual-attributable components, in unit-normed activation space, with a
    length-matched confirmatory check.

    Parameters
    ----------
    class_acts : {region: acts[n, d]}  raw activations (will be unit-normed here)
    token_lengths : {region: lengths[n]}  per-sample token counts (NOT equalized)
    pair : the two regions to decompose (default adv_full vs adv_surface)
    lexical_overlap : optional {region: overlap[n]} per-sample lexical-surface
        score (e.g. fraction of tokens shared with the plain/legit reference).
        If given, a lexical component is attributed AFTER length. If None, the
        decomposition is length-vs-residual only and the lexical share is folded
        into residual (flagged in the output).
    length_match_tol : max |length difference| (tokens) for the matched subset.

    Returns a dict with:
      full_auc_normed / full_auc_raw : separability with no control
      length_partialled_auc : separability after removing the length-predicted
        score component (the "residual after length")
      attribution : {length, lexical, residual} as FRACTIONS of (full-0.5)
      length_matched : {n_pairs, auc, null_mean, null_std, survives}
      Interpretation strings; no thresholds.

    Method
    ------
    1. Unit-norm all activations (control the norm artifact).
    2. Held-out LDA AUC = full separability (the quantity being decomposed).
    3. LENGTH component: regress the held-out projection score on token length;
       the AUC of the length-PREDICTED score is the length-attributable
       separability. Remove it (use residual-of-score) → length-partialled AUC.
    4. LEXICAL component (if lexical_overlap given): on the length-partialled
       score, regress on lexical overlap; its predicted-score AUC is the lexical
       component. Remainder is residual.
    5. Attribution fractions are reported over (full_auc - 0.5), the separability
       above chance.
    6. LENGTH-MATCHED subset: restrict to cross-region pairs with
       |len_a - len_b| <= tol, recompute held-out AUC + permutation null.
       survives = AUC above null band on the matched subset.
    """
    a, b = pair
    Xa_raw, Xb_raw = _as_numpy(class_acts[a]), _as_numpy(class_acts[b])
    La = _as_numpy(token_lengths[a]).ravel()
    Lb = _as_numpy(token_lengths[b]).ravel()
    rng = np.random.default_rng(seed)

    def _heldout_score(X, y, splits, rng_local):
        """Return per-sample held-out projection scores (not just AUC)."""
        idx = rng_local.permutation(len(X))
        fold_sizes = np.full(splits, len(X) // splits)
        fold_sizes[: len(X) % splits] += 1
        scores = np.full(len(X), np.nan)
        start = 0
        for fs in fold_sizes:
            test_idx = idx[start:start + fs]
            train_idx = np.concatenate([idx[:start], idx[start + fs:]])
            start += fs
            if len(np.unique(y[train_idx])) < 2:
                continue
            mu1 = X[train_idx][y[train_idx] == 1].mean(axis=0)
            mu0 = X[train_idx][y[train_idx] == 0].mean(axis=0)
            w = mu1 - mu0
            nrm = np.linalg.norm(w)
            w = w / nrm if nrm > 0 else w
            scores[test_idx] = X[test_idx] @ w
        return scores

    def _build(Xa, Xb):
        X = np.vstack([Xa, Xb])
        y = np.concatenate([np.ones(len(Xa)), np.zeros(len(Xb))])
        L = np.concatenate([La, Lb])
        return X, y, L

    # 1. norm-controlled (and raw, for comparison)
    Xa_n, Xb_n = _unit_norm_rows(Xa_raw), _unit_norm_rows(Xb_raw)
    Xn, y, L = _build(Xa_n, Xb_n)
    Xr, _, _ = _build(Xa_raw, Xb_raw)

    full_scores = _heldout_score(Xn, y, n_splits, np.random.default_rng(seed))
    ok = ~np.isnan(full_scores)
    full_auc_normed = _auc_from_scores(full_scores[ok], y[ok])
    raw_scores = _heldout_score(Xr, y, n_splits, np.random.default_rng(seed))
    okr = ~np.isnan(raw_scores)
    full_auc_raw = _auc_from_scores(raw_scores[okr], y[okr])

    # 3. length component: regress score ~ length (+intercept), in normed space
    def _ols_predict(x, target):
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, target, rcond=None)
        return A @ coef

    s = full_scores[ok]; yk = y[ok]; Lk = L[ok]
    length_pred = _ols_predict(Lk, s)
    length_auc = _auc_from_scores(length_pred, yk)         # length-attributable
    score_resid_len = s - length_pred
    length_partialled_auc = _auc_from_scores(score_resid_len, yk)

    # 4. lexical component (optional), on the length-residual score
    lexical_auc = None
    score_resid_lex = score_resid_len
    if lexical_overlap is not None:
        Oa = _as_numpy(lexical_overlap[a]).ravel()
        Ob = _as_numpy(lexical_overlap[b]).ravel()
        O = np.concatenate([Oa, Ob])[ok]
        lex_pred = _ols_predict(O, score_resid_len)
        lexical_auc = _auc_from_scores(lex_pred, yk)
        score_resid_lex = score_resid_len - lex_pred
    residual_auc = _auc_from_scores(score_resid_lex, yk)

    # 5. attribution rescaled to a TRUE PARTITION summing to (full_auc - 0.5).
    #    Each component's raw separability-above-chance (AUC - 0.5, floored at 0
    #    since a below-chance component contributes no positive separability) is
    #    renormalized so the shares sum to the actual full separability. This
    #    removes the >100% artifact (rank-based AUCs are not strictly additive, so
    #    raw fractions could exceed 1); the rescaled shares are genuine, bounded
    #    proportions of the separability actually present. Raw component AUCs are
    #    retained alongside so nothing is hidden by the rescaling.
    full_above = max(full_auc_normed - 0.5, 1e-9)
    raw_mag = {
        "length": max(length_auc - 0.5, 0.0),
        "residual": max(residual_auc - 0.5, 0.0),
    }
    if lexical_auc is not None:
        raw_mag["lexical"] = max(lexical_auc - 0.5, 0.0)
    mag_sum = sum(raw_mag.values()) or 1e-9
    # rescale so the shares sum to full_above (i.e. to the real separability),
    # then express each as a fraction of full_above (so fractions sum to 1.0)
    shares = {k: round(float(v / mag_sum), 4) for k, v in raw_mag.items()}
    shares_abs = {k: round(float(v / mag_sum * full_above + 0.0), 4)
                  for k, v in raw_mag.items()}  # in AUC-above-chance units
    attribution = {
        "length": shares.get("length"),
        "lexical": shares.get("lexical"),   # None-key absent if no lexical input
        "residual": shares.get("residual"),
        "shares_sum_to": 1.0,
        "shares_in_auc_units": shares_abs,   # these sum to (full_auc - 0.5)
        "raw_component_auc": {               # un-rescaled, for transparency
            "length": round(float(length_auc), 4),
            "lexical": (round(float(lexical_auc), 4)
                        if lexical_auc is not None else None),
            "residual": round(float(residual_auc), 4),
        },
        "note": ("shares are a TRUE partition: each component's separability-"
                 "above-chance (AUC-0.5, floored at 0) renormalized to sum to the "
                 "full separability. 'length'/'lexical'/'residual' sum to 1.0; "
                 "'shares_in_auc_units' sum to (full_auc_normed - 0.5); "
                 "'raw_component_auc' are the un-rescaled AUCs. The length-matched "
                 "survival check (below) remains the primary, partition-free result."
                 + ("" if lexical_overlap is not None else
                    " lexical_overlap not provided: lexical folded into residual.")),
    }

    # 6. length-matched confirmatory subset
    # pair up cross-region samples with near-equal length, recompute AUC + null
    matched_a, matched_b = [], []
    used_b = set()
    order_b = np.argsort(Lb)
    for i in range(len(La)):
        # find an unused b within tol of La[i]
        for j in order_b:
            if j in used_b:
                continue
            if abs(La[i] - Lb[j]) <= length_match_tol:
                matched_a.append(i); matched_b.append(j); used_b.add(j)
                break
    if len(matched_a) >= 10:
        Xm = np.vstack([Xa_n[matched_a], Xb_n[matched_b]])
        ym = np.concatenate([np.ones(len(matched_a)), np.zeros(len(matched_b))])
        m_scores = _heldout_score(Xm, ym, min(n_splits, 5),
                                  np.random.default_rng(seed + 3))
        mok = ~np.isnan(m_scores)
        matched_auc = _auc_from_scores(m_scores[mok], ym[mok])
        null = []
        for p in range(n_perm):
            ns = _heldout_score(Xm, np.random.default_rng(p).permutation(ym),
                                min(n_splits, 5), np.random.default_rng(700 + p))
            nok = ~np.isnan(ns)
            null.append(_auc_from_scores(ns[nok], ym[nok]))
        null = np.asarray(null)
        length_matched = {
            "n_pairs": len(matched_a),
            "auc": round(matched_auc, 4),
            "null_mean": round(float(null.mean()), 4),
            "null_std": round(float(null.std()), 4),
            "survives": bool(matched_auc > np.percentile(null, 95)),
            "interpretation": (
                "survives=True → separability persists when length is matched, so "
                "a non-length component is real. survives=False → separability "
                "collapses under length-matching, so length was carrying it."),
        }
    else:
        length_matched = {
            "n_pairs": len(matched_a),
            "auc": None,
            "survives": None,
            "interpretation": ("too few length-matched pairs (<10) at tol="
                               f"{length_match_tol}; widen tol or collect more "
                               "samples for the matched-subset check."),
        }

    out = {
        "metric": "separability_decomposition",
        "layer": layer, "pair": pair,
        "full_auc_normed": round(full_auc_normed, 4),
        "full_auc_raw": round(full_auc_raw, 4),
        "length_auc": round(length_auc, 4),
        "length_partialled_auc": round(length_partialled_auc, 4),
        "lexical_auc": (round(lexical_auc, 4) if lexical_auc is not None else None),
        "residual_auc": round(residual_auc, 4),
        "attribution": attribution,
        "length_matched": length_matched,
        "interpretation": (
            "full_auc_normed is separability with the norm artifact controlled "
            "(compare to full_auc_raw to see the norm inflation). attribution "
            "gives the length / lexical / residual shares of that separability. "
            "length_matched is the confirmatory survival check — the single most "
            "legible result: does separability survive when length is held equal?"),
    }
    logger.info(
        "separability_decomposition[L%s %s vs %s]: full_normed=%.3f (raw %.3f) "
        "length_share=%.2f residual_share=%.2f matched_survives=%s",
        layer, a, b, full_auc_normed, full_auc_raw,
        attribution["length"], attribution["residual"],
        length_matched.get("survives"),
    )
    return out
