"""
grouped_stats.py
AGPL 3. Copyright © 2026 Katherine J. Ombrellaro.
================
Uncertainty estimated over the RIGHT unit.

The twin design has 24 frames across 6 relations; each frame can render many
payloads that differ only in slot fill. Those payloads are not independent
experimental units -- the effective sample size is the number of frames, and for
a relation-general claim, the number of RELATIONS.

An i.i.d. bootstrap over payloads is therefore INVALID here: it assumes an
independence the design does not have. Note carefully that "invalid" does not
reduce to "too narrow" -- measured on this project's own structures, the cluster
interval came out NARROWER than the i.i.d. one in both a between-relation and a
within-frame construction. Which way the error points depends on whether the
effect of interest lives between clusters or within them, and on the sign of the
intra-cluster correlation for the statistic being computed. The argument for
resampling at the group level is that it matches the design, not that it always
widens the interval; anyone reaching for it expecting a bigger number is reaching
for it for the wrong reason.

Everything here therefore resamples at a group level:

  loro_scores            -- leave-one-relation-out: train on the rest, score the
                            held-out relation. The distribution ACROSS relations,
                            not a pooled number, is the result.
  cluster_bootstrap_ci   -- resample RELATIONS with replacement (all frames within
                            a sampled relation travel together).
  sign_flip_permutation  -- null for the paired twin design: under H0 the sign of
                            Δ_i is arbitrary, so flip signs per FRAME and preserve
                            pair/frame structure exactly.

Nothing here fits an SAE or calls a GPU; all inputs are precomputed arrays.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


# ── Leave-one-group-out ───────────────────────────────────────────────────────

def loro_scores(X, y, groups, fit_fn, min_test: int = 4) -> dict:
    """Leave-one-relation-out (or any group level).

    For each unique group: fit on every OTHER group, score the held-out one.
    Returns the per-group AUCs and their spread. Report the DISTRIBUTION -- a
    mean alone hides the case your design is most exposed to, where one relation
    carries the whole result.

    Args:
        X:        [n, d] features.
        y:        [n] binary labels.
        groups:   [n] group id (use probe.group_keys(payloads, "relation")).
        fit_fn:   fit_fn(X_tr, y_tr) -> score_fn(X_te) -> 1-d scores.
        min_test: skip a held-out group with fewer than this many rows, or with
                  only one class present (AUC undefined) -- recorded in "skipped"
                  rather than silently dropped.

    Returns:
        {"per_group": {g: auc}, "mean", "std", "min", "max", "n_groups", "skipped"}
    """
    X = np.asarray(X); y = np.asarray(y); groups = np.asarray(groups)
    per, skipped = {}, {}
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if te.sum() < min_test:
            skipped[str(g)] = f"n_test={int(te.sum())} < {min_test}"; continue
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            skipped[str(g)] = "single class in train or test"; continue
        score_fn = fit_fn(X[tr], y[tr])
        per[str(g)] = float(roc_auc_score(y[te], score_fn(X[te])))
    vals = np.array(list(per.values()), dtype=float)
    return {
        "per_group": per,
        "mean": float(vals.mean()) if vals.size else float("nan"),
        "std":  float(vals.std(ddof=1)) if vals.size > 1 else float("nan"),
        "min":  float(vals.min()) if vals.size else float("nan"),
        "max":  float(vals.max()) if vals.size else float("nan"),
        "n_groups": int(vals.size),
        "skipped": skipped,
    }


# ── Cluster bootstrap ─────────────────────────────────────────────────────────

def cluster_bootstrap_ci(
    stat_fn,
    groups,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Percentile CI resampling GROUPS with replacement (cluster bootstrap).

    stat_fn(idx) -> float, computed on the row indices of a resampled dataset.
    All rows of a sampled group travel together, so within-group correlation
    (slot-fill near-duplicates) is preserved rather than broken -- which is the
    entire point: an i.i.d. bootstrap over rows would treat 12 fills of one frame
    as 12 independent observations and shrink the interval accordingly.

    With few groups (6 relations) the interval is driven by very little
    information, and whatever width results is the honest report -- not a defect
    to tune away, and not necessarily wider than an i.i.d. interval on the same
    data (see the module docstring; measured both ways on this project's
    structures).

    Returns {"point", "lo", "hi", "n_boot", "n_effective", "n_groups"}.
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)

    point = float(stat_fn(np.arange(len(groups))))
    boots = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in drawn])
        try:
            v = stat_fn(idx)
        except Exception:
            continue                      # degenerate draw (e.g. one class) -> skip
        if np.isfinite(v):
            boots.append(float(v))
    boots = np.array(boots)
    if boots.size == 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "n_boot": n_boot, "n_effective": 0, "n_groups": len(uniq)}
    return {
        "point": point,
        "lo": float(np.quantile(boots, alpha / 2)),
        "hi": float(np.quantile(boots, 1 - alpha / 2)),
        "n_boot": n_boot,
        "n_effective": int(boots.size),
        "n_groups": int(len(uniq)),
    }


# ── Paired sign-flip permutation ──────────────────────────────────────────────

def sign_flip_permutation(
    deltas: np.ndarray,
    relation_ids: list[str] | None = None,
    n_perm: int = 2000,
    seed: int = 0,
) -> dict:
    """Null for the twin design's direction-consistency claim.

    Statistic: mean pairwise cosine similarity between frames' Δ vectors -- i.e.
    "do independently authored realisations perturb the residual stream the same
    way".

    H0: the contrast has no consistent direction, so the SIGN of each Δ_i is
    arbitrary. The null is generated by flipping signs per FRAME, which preserves
    pair and frame structure exactly -- each frame keeps its own magnitude and
    feature profile, only its orientation is randomised. This is the right null
    for a paired design; shuffling labels across frames would additionally destroy
    the pairing and test a different (weaker) hypothesis.

    Note the statistic is invariant to a global sign flip, so the null is centred
    near 0 rather than at 0.5.

    Returns {"observed", "null_mean", "null_p95", "p_value", "n_perm"} where
    p_value is the one-sided proportion of null statistics >= observed
    (add-one corrected, so it is never exactly 0).
    """
    D = np.asarray(deltas, dtype=float)
    n = D.shape[0]
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    U = D / norms
    iu = np.triu_indices(n, k=1)

    def stat(sign_vec):
        V = U * sign_vec[:, None]
        return float((V @ V.T)[iu].mean())

    observed = stat(np.ones(n))
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.choice([-1.0, 1.0], size=n)) for _ in range(n_perm)])
    p = (1.0 + float((null >= observed).sum())) / (1.0 + n_perm)
    return {
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        "p_value": p,
        "n_perm": n_perm,
    }


# ── Per-relation effect sizes ─────────────────────────────────────────────────

def relation_effect_sizes(
    deltas: np.ndarray,
    relation_ids: list[str],
    feature_idx=None,
) -> tuple[np.ndarray, list[str]]:
    """Per-relation mean Δ per feature, standardized by the within-relation SD.

    Feeds twin_figures.relation_feature_heatmap. Standardizing WITHIN relation
    means a relation whose realisations disagree gets a small effect even if its
    mean Δ is large -- which is the property that makes the heatmap's sign-flip
    reading meaningful rather than a magnitude artefact.

    Returns (E, relations) where E is [n_relations, n_features].
    """
    D = np.asarray(deltas, dtype=float)
    rel = np.asarray(relation_ids)
    if feature_idx is not None:
        D = D[:, np.asarray(feature_idx)]
    relations = sorted(set(rel.tolist()))
    rows = []
    for r in relations:
        M = D[rel == r]
        sd = M.std(axis=0, ddof=1) if M.shape[0] > 1 else np.ones(M.shape[1])
        sd = np.where(sd < 1e-8, 1.0, sd)          # a constant feature -> effect 0, not inf
        rows.append(M.mean(axis=0) / sd)
    return np.array(rows), relations


# ── Pair x reference-relation agreement (the summary diagnostic) ─────────────

def pair_reference_agreement(deltas, relation_ids, fills=None, eps: float = 1e-9) -> dict:
    """Signed direction agreement and separation magnitude for every
    (matched pair) x (reference relation) cell, under a strict leakage rule.

    THE QUESTION: is the contrast broadly reproducible across template
    constructions, or is the headline separation generated by a few of them?

    For frame i and reference relation t:
        ref_{t,-i} = mean Δ over frames in relation t, EXCLUDING frame i
        A[i, t]    = cos(Δ_i, ref_{t,-i})          -- signed direction agreement
        B[i, t]    = SIGNED held-out projection of Δ_i on ref_{t,-i}

    FOLD LANGUAGE (precise). The diagonal cells are within-relation
    LEAVE-ONE-REALISATION-OUT: the reference is relation t's other realisations,
    with frame i removed. The off-diagonal cells are CROSS-RELATION TRANSFER, NOT
    leave-one-relation-out: the direction comes from relation t ALONE and is
    tested on a frame from a different relation. True leave-one-relation-out --
    a direction pooled over every relation except the row's -- is the extra
    "pooled (LORO)" column, which is the reference that actually tests for a
    relation-general direction. Conflating the two would overstate what an
    off-diagonal cell shows.

    LEAKAGE RULE. Every reference is estimated without the frame being plotted,
    including the pooled column. Without this the figure is descriptive fit, not
    generalisation evidence -- the same distinction as this project's synthetic
    example-CV 1.000 vs frame-held-out 0.345.

    WHY SIGNED, AND WHY TWO MATRICES. Direction and magnitude must not be
    conflated: two cells can both show large separation while moving activations
    in OPPOSITE directions, which is evidence against a shared representation even
    though a local classifier does well in both. A magnitude-only heatmap would
    show those two cells as agreeing. Hence A (does it move the same way?) is
    primary and B (how far does it move?) is the companion.

    READING:
        uniform rows    -> that pair separates against every reference: shared;
        uniform columns -> one relation's direction works for nearly every pair,
                           i.e. what looks like shared intent may be a
                           template-family mechanism;
        isolated cells  -> pair-specific lexical/construction effects dominate;
        block diagonal  -> within-relation agreement only; no relation-general
                           direction.

    Args:
        deltas: [n_frames, d] (one Δ per frame) or [n_frames, n_fills, d] (Δ per
            slot-fill). When fills are supplied, B becomes a PAIRED effect size --
            mean_j / sd_j of the projection across fills -- and A uses the
            per-frame mean. Caution: fills of one frame are not independent, so a
            fill-driven effect size looks more precise than the design supports;
            it is reported here for shape, not for a confidence claim.
        relation_ids: [n_frames] relation label per frame.

    Returns dict with "A", "B" (columns = relations + a trailing pooled-LORO
    column), "columns", "relations", "row_relation", "n_ref", and
    "gap_by_relation": G_r = mean within-relation cos - mean cross-relation cos,
    which quantifies what the eye reads off the block structure.
    """
    D = np.asarray(deltas, dtype=float)
    per_fill = (D.ndim == 3)
    if per_fill:
        D_mean = D.mean(axis=1)                  # [n_frames, d]
    else:
        D_mean = D
    rel = np.asarray(relation_ids)
    n = D_mean.shape[0]
    if len(rel) != n:
        raise ValueError(f"relation_ids has {len(rel)} entries for {n} frames.")
    relations = sorted(set(rel.tolist()))
    columns = relations + ["pooled (LORO)"]

    A = np.full((n, len(columns)), np.nan)
    B = np.full((n, len(columns)), np.nan)
    n_ref = np.zeros((n, len(columns)), dtype=int)
    scale = float(np.median(np.linalg.norm(D_mean, axis=1))) or 1.0

    def _cell(i, mask):
        """Signed cosine + signed projection of frame i against the mean Δ of
        `mask`. Sign is preserved throughout: a reversal must read as negative,
        not as a small magnitude."""
        if mask.sum() == 0:
            return np.nan, np.nan
        ref = D_mean[mask].mean(axis=0)
        nref = np.linalg.norm(ref)
        ni = np.linalg.norm(D_mean[i])
        if nref < eps or ni < eps:
            return np.nan, np.nan
        u = ref / nref
        a = float(D_mean[i] @ u / ni)
        if per_fill and D.shape[1] > 1:
            proj = D[i] @ u
            sd = proj.std(ddof=1)
            b = float(proj.mean() / sd) if sd > eps else np.nan
        else:
            b = float(D_mean[i] @ u) / scale       # SIGNED -- reversals stay visible
        return a, b

    for i in range(n):
        for t, r in enumerate(relations):
            mask = (rel == r)
            mask[i] = False                       # LEAKAGE RULE: never the plotted frame
            n_ref[i, t] = int(mask.sum())
            A[i, t], B[i, t] = _cell(i, mask)
        # pooled (LORO): every relation EXCEPT the row's own -> a genuinely
        # relation-general direction, which no single off-diagonal cell tests.
        pooled = (rel != rel[i])
        n_ref[i, -1] = int(pooled.sum())
        A[i, -1], B[i, -1] = _cell(i, pooled)

    gap = {}
    for r in relations:
        rows = np.flatnonzero(rel == r)
        within = [A[i, relations.index(r)] for i in rows]
        cross = [A[i, t] for i in rows for t, rr in enumerate(relations) if rr != r]
        def _nanmean(v):
            # An all-NaN slice is an expected state here (a singleton relation has
            # no reference left after the leakage rule removes the plotted frame),
            # so it returns nan quietly rather than raising a RuntimeWarning in CI.
            v = np.asarray(v, dtype=float)
            return float(np.nanmean(v)) if v.size and np.isfinite(v).any() else float("nan")

        gap[r] = {
            "within_mean": _nanmean(within),
            "cross_mean": _nanmean(cross),
            "pooled_loro_mean": _nanmean([A[i, -1] for i in rows]),
        }
        gap[r]["G"] = gap[r]["within_mean"] - gap[r]["cross_mean"]

    return {
        "A": A, "B": B, "columns": columns, "relations": relations,
        "row_relation": rel.tolist(), "n_ref": n_ref, "per_fill": per_fill,
        "gap_by_relation": gap,
        "B_meaning": ("SIGNED paired effect size across fills (mean/sd) -- fills are "
                      "NOT independent; read for shape, not precision"
                      if per_fill else
                      "SIGNED projection on the reference, scaled by median ||Δ||"),
    }


# ── Three-stage sequence: surface -> intent -> intent minus surface ──────────

def remove_direction(deltas, direction, eps: float = 1e-9):
    """Project `direction` out of every Δ. Returns the residual component."""
    D = np.asarray(deltas, dtype=float)
    u = np.asarray(direction, dtype=float)
    nu = np.linalg.norm(u)
    if nu < eps:
        return D.copy()
    u = u / nu
    if D.ndim == 3:                       # [frames, fills, d]
        return D - (D @ u)[..., None] * u
    return D - (D @ u)[:, None] * u


def three_stage_agreement(
    surface_deltas,
    intent_deltas,
    relation_ids,
    surface_is_independent: bool = False,
    eps: float = 1e-9,
) -> dict:
    """The sequence that connects the neutral-intent surface test to the
    matched-pair analysis, all on one row x reference-relation grid.

        Stage 1  SURFACE          Δ_surf = h(benign content, adversarial surface)
                                          − h(benign content, neutral surface)
                 Intent is held CONSTANT (both members benign), so this isolates
                 what adversarial SURFACE alone does to the residual stream.
        Stage 2  INTENT           Δ_int  = h(adv) − h(clean), the twin design.
        Stage 3  INTENT ⊥ SURFACE Δ_int with the learned surface direction
                                  projected out.

    THE QUESTION: does the cross-relation agreement pattern in Stage 2 SURVIVE
    once the measured surface direction is removed? If Stage 2 shows agreement and
    Stage 3 does not, the apparent shared direction was the surface manipulation's
    signature -- which is precisely what a stable Δ cannot rule out on its own.
    If it survives, that is the first evidence the displacement is not merely
    recovering the edited words, the syntax change, or the register shift.

    LEAKAGE RULE, EXTENDED TO THE SURFACE DIRECTION. The direction removed from
    frame i is estimated WITHOUT frame i:

      surface_is_independent=False (default) -- Δ_surf comes from the same frames
        (e.g. each twin frame also rendered in a neutral-surface variant), so a
        leave-one-frame-out surface direction is used per row. Removing a
        direction fitted on frame i from frame i would guarantee Stage 3 collapse
        and prove nothing.
      surface_is_independent=True -- Δ_surf comes from a SEPARATE dataset (the BC1
        null pairs, which share no rows with the twin frames), so one global
        direction is used. This is the cleaner design: the surface direction then
        carries no information about any twin frame at all.

    Args:
        surface_deltas: [n_frames, d] or [n_frames, n_fills, d] when tied to the
            frames; [m, d] or [m, n_fills, d] for an independent source.
        intent_deltas:  [n_frames, d] or [n_frames, n_fills, d], row-aligned with
            relation_ids.
        relation_ids:   [n_frames].

    Returns {"surface", "intent", "intent_minus_surface"} -- three
    pair_reference_agreement dicts on the same grid -- plus
    "surface_direction_norm" and "residual_fraction" (mean ||Δ_resid|| / ||Δ_int||;
    near 0 means the intent displacement lived almost entirely along the surface
    direction, and Stage 3 has almost nothing left to agree about -- itself a
    finding, and one to state rather than let the reader infer from pale cells).
    """
    S = np.asarray(surface_deltas, dtype=float)
    I = np.asarray(intent_deltas, dtype=float)
    rel = np.asarray(relation_ids)
    S_mean = S.mean(axis=1) if S.ndim == 3 else S
    I_mean = I.mean(axis=1) if I.ndim == 3 else I
    n = I_mean.shape[0]

    if surface_is_independent:
        u_global = S_mean.mean(axis=0)
        resid = remove_direction(I, u_global)
        dirs_norm = float(np.linalg.norm(u_global))
    else:
        if S_mean.shape[0] != n:
            raise ValueError(
                f"surface_deltas has {S_mean.shape[0]} frames but intent_deltas has {n}. "
                "Pass surface_is_independent=True when the surface direction comes from a "
                "separate dataset (e.g. the BC1 null pairs)."
            )
        resid = np.empty_like(I)
        norms = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False                      # LEAKAGE RULE on the surface direction
            u_i = S_mean[mask].mean(axis=0)
            norms.append(float(np.linalg.norm(u_i)))
            resid[i] = remove_direction(I[i][None, ...] if I.ndim == 3 else I[i][None, :],
                                        u_i)[0]
        dirs_norm = float(np.mean(norms))

    resid_mean = resid.mean(axis=1) if resid.ndim == 3 else resid
    num = np.linalg.norm(resid_mean, axis=1)
    den = np.linalg.norm(I_mean, axis=1)
    frac = float(np.mean(np.where(den > eps, num / np.maximum(den, eps), np.nan)))

    return {
        "surface": pair_reference_agreement(S, rel) if not surface_is_independent else None,
        "intent": pair_reference_agreement(I, rel),
        "intent_minus_surface": pair_reference_agreement(resid, rel),
        "surface_direction_norm": dirs_norm,
        "residual_fraction": frac,
        "surface_is_independent": surface_is_independent,
    }


# ── Leave-two-realisations-out sensitivity ──────────────────────────────────

def realisation_sensitivity(deltas, relation_ids, eps: float = 1e-9) -> dict:
    """How fragile is each within-relation reference direction?

    With 4 realisations per relation, a diagonal cell's reference is a
    THREE-realisation centroid (the plotted frame is excluded). A single unstable
    realisation could move it materially, and a near-perfect diagonal hides that
    completely -- the agreement can look decisive while resting on three examples.

    This drops ONE ADDITIONAL realisation (leave-two-out) in every possible way
    and recomputes the frame's within-relation cosine. A robust relation direction
    barely moves; a fragile three-example centroid swings.

    Reports per frame: the leave-one-out cosine, the leave-two-out values, their
    spread (max - min), and `carried_by_one` -- True when dropping some single
    additional realisation moves the cosine by more than `fragile_threshold`
    (0.2), i.e. one realisation is carrying the reference.

    Returns {"per_frame": [...], "max_swing", "n_fragile", "fragile_threshold"}.
    """
    D = np.asarray(deltas, dtype=float)
    D_mean = D.mean(axis=1) if D.ndim == 3 else D
    rel = np.asarray(relation_ids)
    n = D_mean.shape[0]
    FRAGILE = 0.2

    def _cos(i, mask):
        if mask.sum() == 0:
            return float("nan")
        ref = D_mean[mask].mean(axis=0)
        nr, ni = np.linalg.norm(ref), np.linalg.norm(D_mean[i])
        if nr < eps or ni < eps:
            return float("nan")
        return float(D_mean[i] @ (ref / nr) / ni)

    per_frame, swings = [], []
    for i in range(n):
        mates = np.flatnonzero((rel == rel[i]) & (np.arange(n) != i))
        loo = _cos(i, np.isin(np.arange(n), mates))
        ltro = []
        for drop in mates:
            keep = mates[mates != drop]
            ltro.append(_cos(i, np.isin(np.arange(n), keep)))
        ltro_arr = np.array([v for v in ltro if np.isfinite(v)], dtype=float)
        spread = float(ltro_arr.max() - ltro_arr.min()) if ltro_arr.size > 1 else float("nan")
        worst = (float(np.max(np.abs(ltro_arr - loo))) if ltro_arr.size and np.isfinite(loo)
                 else float("nan"))
        per_frame.append({
            "frame": int(i), "relation": str(rel[i]),
            "loo_cos": loo, "n_refs_loo": int(len(mates)),
            "ltro_cos": [float(v) for v in ltro], "ltro_spread": spread,
            "max_shift_from_loo": worst,
            "carried_by_one": bool(np.isfinite(worst) and worst > FRAGILE),
        })
        if np.isfinite(worst):
            swings.append(worst)
    return {
        "per_frame": per_frame,
        "max_swing": float(max(swings)) if swings else float("nan"),
        "mean_swing": float(np.mean(swings)) if swings else float("nan"),
        "n_fragile": int(sum(p["carried_by_one"] for p in per_frame)),
        "fragile_threshold": FRAGILE,
    }
