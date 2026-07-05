"""
stats.py
AGPL 3. Copyright © 2026 Katherine J. Ombrellaro.
========
Shared uncertainty estimators for SECONDARY reporting alongside the
pre-registered point thresholds. None of these change a locked threshold or a
confirmatory test (PREREGISTRATION.md §4 / §8); they report whether an estimate
is *reliably* above its threshold (CI lower bound vs threshold) rather than
merely over it. Disclose in the writeup as supplementary uncertainty reporting.

All functions are numpy-only (no torch) so they are unit-testable off-GPU.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% score interval for a binomial proportion (k of n). Preferred
    over the normal approximation at the small n / extreme rates typical of
    bypass and causal rates. Returns (low, high) clamped to [0, 1]."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def normal_ci(mean: float, std: float, z: float = 1.96) -> tuple[float, float]:
    """Normal-approx 95% CI for a mean given its std, clamped to [0, 1].
    Used for the per-fold AUC mean (std = spread across CV folds)."""
    half = z * std
    return (round(max(0.0, mean - half), 4), round(min(1.0, mean + half), 4))


def beta_credible_interval(
    k: int,
    n: int,
    prior: str = "jeffreys",
    ci: float = 0.95,
) -> tuple[float, float]:
    """
    SECONDARY/EXPLORATORY Bayesian credible interval for a proportion, reported
    only as a companion to wilson_ci — never as a replacement and never on a
    confirmatory readout.

    Deliberately restricted to UNINFORMATIVE priors so no analyst degree of
    freedom enters a pre-registered quantity:
        "jeffreys" -> Beta(0.5, 0.5)   (default; reference prior)
        "uniform"  -> Beta(1, 1)
    Informative priors are intentionally unsupported here: §4.4 reads the causal
    rate against absolute locked bands, and a prior that pulls the posterior
    toward 0.5 at small n could shift that readout. Returns (low, high) in [0,1].
    """
    from scipy.stats import beta as _beta
    a0, b0 = {"jeffreys": (0.5, 0.5), "uniform": (1.0, 1.0)}.get(prior, (0.5, 0.5))
    a_post, b_post = a0 + k, b0 + (n - k)
    lo = (1 - ci) / 2
    return (round(float(_beta.ppf(lo, a_post, b_post)), 4),
            round(float(_beta.ppf(1 - lo, a_post, b_post)), 4))


def bootstrap_auc_ci(
    y_true,
    scores,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """
    Percentile bootstrap CI for ROC-AUC by resampling (y_true, scores) pairs.

    Returns {auc, ci_low, ci_high, n_boot, n}. Resamples that lose a class are
    skipped. This quantifies sampling uncertainty in a transfer-matrix cell or a
    detection AUC so the >= 0.60 threshold can be read against the CI lower bound
    rather than the point estimate alone. Secondary reporting only.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    n = len(y)
    point = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
    rng = np.random.RandomState(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        boots.append(roc_auc_score(yb, s[idx]))
    if not boots:
        return {"auc": round(point, 4), "ci_low": None, "ci_high": None,
                "n_boot": 0, "n": int(n)}
    arr = np.sort(np.asarray(boots))
    lo = float(np.quantile(arr, alpha / 2))
    hi = float(np.quantile(arr, 1 - alpha / 2))
    return {
        "auc": round(point, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "n_boot": len(boots),
        "n": int(n),
    }
