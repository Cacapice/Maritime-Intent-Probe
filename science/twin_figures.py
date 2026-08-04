"""
twin_figures.py
AGPL 3. Copyright © 2026 Katherine J. Ombrellaro.
===============
Heatmaps for the twin semantic design. Every figure here answers a
METHODOLOGICAL question -- stability, attribution, reproducibility across
independently constructed realisations -- rather than displaying clusters.

Deliberately no t-SNE / UMAP. Cluster structure in those projections varies with
perplexity, initialisation, normalisation and local density, so they are weak
evidence for construct-validity claims and are easy to overread. Everything here
plots a quantity with a defined estimator instead.

The signature figure is `pairwise_delta_heatmap`: it does not show that classes
separate, it shows whether the REPRESENTATION CHANGE itself reproduces across
independently authored realisations of the same intended contrast.

All functions take precomputed arrays and return (fig, data) so the numbers
behind a figure can be asserted in a test or written to the ledger; none of them
fit models or call a GPU.
"""
from __future__ import annotations

import numpy as np

_DIVERGING = "RdBu_r"
_SEQUENTIAL = "viridis"


def _hm(ax, M, cmap, vmin=None, vmax=None):
    return ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                     interpolation="nearest")


def _labels(ax, rows, cols, rot=90, colsize=7):
    if cols is not None:
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=rot, fontsize=colsize, ha="right" if rot else "center")
    if rows is not None:
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=7)


# ── 1. Pairwise activation-difference heatmap (the signature figure) ──────────

def select_feature_basis(deltas: np.ndarray, train_idx=None, top_k: int = 60) -> np.ndarray:
    """ONE feature basis, selected on TRAINING rows only, applied unchanged
    everywhere afterwards.

    Selecting features per-panel (or on all rows including held-out ones) makes
    every panel look strongly structured even when the panels emphasise unrelated
    dimensions -- the structure is then a property of the selection, not of the
    data. Both panels of the two-panel figure, and every held-out relation, must
    share the basis this returns.

    train_idx=None selects on all rows: acceptable ONLY for a purely descriptive
    single-panel plot with no held-out claim attached. If a fold, a held-out
    relation, or a comparison is involved, pass the training rows.

    Alternatives with the same guarantee, if preferred: a fixed SAE feature set,
    or features chosen on an independent development partition.
    """
    D = np.asarray(deltas, dtype=float)
    rows = D if train_idx is None else D[np.asarray(train_idx)]
    return np.argsort(-np.abs(rows.mean(axis=0)))[:min(top_k, D.shape[1])]


def pairwise_delta_heatmap(
    deltas: np.ndarray,
    relation_ids: list[str],
    realisation_ids: list[int] | None = None,
    provenance: list[str] | None = None,
    top_k: int = 60,
    feature_basis: np.ndarray | None = None,
    title: str = "Matched-pair activation difference (adv − clean)",
):
    """Δ_i = h(x_i^adv) − h(x_i^clean) per twin frame.

    rows    = twin frames, ORDERED BY RELATION so within-relation reproducibility
              reads as horizontal banding;
    columns = the top_k features by |mean Δ| (SAE features or neurons);
    cell    = the activation difference itself, on a symmetric diverging scale so
              a SIGN FLIP between realisations of one relation is visible as a
              colour flip rather than hidden in a magnitude.

    What to read:
      consistent vertical stripes within a relation block -> the perturbation
        reproduces across independently authored realisations of that contrast;
      stripes that persist ACROSS relation blocks -> a shared direction;
      block structure with no shared stripes -> each relation moved the residual
        stream its own way, i.e. no common representation of the intended
        construct, whatever the classifier AUC says;
      a row that looks like noise -> that realisation carries the contrast in
        name only.

    Args:
        deltas:          [n_frames, d_feature] adversarial minus clean, one row per frame.
        relation_ids:    [n_frames] relation label per row.
        realisation_ids: optional [n_frames] realisation index, used in row labels.
        provenance:      optional [n_frames] "seed"/"new"; marked in row labels so
                         a reader can see whether reproducibility is carried only
                         by the pre-registered templates.
        top_k:           columns to show when feature_basis is None.
        feature_basis:   explicit column indices from select_feature_basis(...).
                         PASS THIS whenever the figure sits beside another panel
                         or backs a held-out claim: self-selecting the basis from
                         all rows leaks the held-out frames into the visual and
                         guarantees apparent structure.

    Returns:
        (fig, data) where data holds the row order, the selected column indices,
        and the plotted matrix.
    """
    import matplotlib.pyplot as plt

    deltas = np.asarray(deltas, dtype=float)
    n = deltas.shape[0]
    if len(relation_ids) != n:
        raise ValueError(f"relation_ids has {len(relation_ids)} entries for {n} rows.")

    order = sorted(range(n), key=lambda i: (relation_ids[i],
                                            -1 if realisation_ids is None else realisation_ids[i]))
    D = deltas[order]
    rel_sorted = [relation_ids[i] for i in order]

    cols = (select_feature_basis(deltas, None, top_k) if feature_basis is None
            else np.asarray(feature_basis))
    M = D[:, cols]
    lim = float(np.abs(M).max()) or 1.0

    labels = []
    for i in order:
        lab = relation_ids[i]
        if realisation_ids is not None:
            lab += f":{realisation_ids[i]}"
        if provenance is not None:
            lab += f" [{provenance[i][:4]}]"
        labels.append(lab)

    fig, ax = plt.subplots(figsize=(11, max(3.0, 0.28 * n)))
    im = _hm(ax, M, _DIVERGING, vmin=-lim, vmax=lim)
    _labels(ax, labels, None)
    ax.set_xlabel(f"top {len(cols)} features by |mean Δ|")
    ax.set_title(title, fontsize=10)

    # Relation block separators — the unit of the primary held-out claim.
    for b in [k for k in range(1, len(rel_sorted)) if rel_sorted[k] != rel_sorted[k - 1]]:
        ax.axhline(b - 0.5, color="black", lw=1.2)
    fig.colorbar(im, ax=ax, fraction=0.02, label="Δ activation")
    fig.tight_layout()
    return fig, {"row_order": order, "cols": cols.tolist(), "matrix": M,
                 "relations": rel_sorted, "abs_max": lim}


# ── 7. Δ similarity matrix (the signature figure's companion) ─────────────────

def delta_similarity_matrix(
    deltas: np.ndarray,
    relation_ids: list[str],
    realisation_ids: list[int] | None = None,
    title: str = "Cosine similarity of Δ between twin frames",
):
    """Cosine similarity of Δ_i between every pair of frames, ordered by relation.

    This is the direct test of whether "intent" is represented CONSISTENTLY
    across independently authored realisations:
      uniformly high  -> one shared direction; the contrast is relation-general;
      block-diagonal  -> each relation has its own direction, and a probe trained
                         across relations is averaging over unrelated
                         perturbations -- exactly the case a high pooled AUC
                         would hide.

    Returns (fig, data) with the similarity matrix and the mean within-relation
    vs between-relation similarity, which is the number the figure is really
    about: within >> between is block structure, whatever the eye says.
    """
    import matplotlib.pyplot as plt

    D = np.asarray(deltas, dtype=float)
    n = D.shape[0]
    order = sorted(range(n), key=lambda i: (relation_ids[i],
                                            -1 if realisation_ids is None else realisation_ids[i]))
    D = D[order]
    rel = [relation_ids[i] for i in order]

    norms = np.linalg.norm(D, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    S = (D / norms) @ (D / norms).T

    iu = np.triu_indices(n, k=1)
    same = np.array([rel[i] == rel[j] for i, j in zip(*iu)])
    within = float(S[iu][same].mean()) if same.any() else float("nan")
    between = float(S[iu][~same].mean()) if (~same).any() else float("nan")

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = _hm(ax, S, _DIVERGING, vmin=-1, vmax=1)
    labs = [f"{rel[k]}:{k}" for k in range(n)]
    _labels(ax, labs, labs, rot=90)
    ax.set_title(f"{title}\nwithin-relation {within:.2f} vs between-relation {between:.2f}",
                 fontsize=10)
    for b in [k for k in range(1, n) if rel[k] != rel[k - 1]]:
        ax.axhline(b - 0.5, color="black", lw=1.0)
        ax.axvline(b - 0.5, color="black", lw=1.0)
    fig.colorbar(im, ax=ax, fraction=0.03, label="cosine similarity")
    fig.tight_layout()
    return fig, {"similarity": S, "within_relation_mean": within,
                 "between_relation_mean": between, "order": order}


# ── 2. Relation × feature effect-size heatmap ────────────────────────────────

def relation_feature_heatmap(
    effect: np.ndarray,
    relation_ids: list[str],
    feature_labels: list[str] | None = None,
    title: str = "Effect size by relation × feature",
):
    """rows = relation, cols = feature (probe coefficient / SAE feature /
    principal direction), cell = effect size, diverging around 0.

    A row of consistent signs across features means the learned direction is
    stable for that relation; sign flips between rows of the same column mean the
    feature does not encode the intended contrast in a relation-general way --
    the failure this figure exists to make visible at a glance.
    """
    import matplotlib.pyplot as plt

    E = np.asarray(effect, dtype=float)
    lim = float(np.abs(E).max()) or 1.0
    fig, ax = plt.subplots(figsize=(min(12, 1.1 + 0.5 * E.shape[1]), 1.2 + 0.42 * E.shape[0]))
    im = _hm(ax, E, _DIVERGING, vmin=-lim, vmax=lim)
    _labels(ax, relation_ids, feature_labels or [f"F{k}" for k in range(E.shape[1])], rot=0)
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            ax.text(j, i, f"{E[i, j]:+.2f}", ha="center", va="center", fontsize=6.5)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, label="effect size")
    fig.tight_layout()
    return fig, {"matrix": E}


# ── 3. Transfer matrix ───────────────────────────────────────────────────────

def transfer_matrix_heatmap(
    auc: np.ndarray,
    classes: list[str],
    title: str = "Cross-class transfer AUC (train → test)",
):
    """rows = train class, cols = test class, cells = AUC, centred at 0.5.

    Per the revised evidence hierarchy this is a ROBUSTNESS / external-validity
    analysis, not the principal construct test: the classes are heterogeneous
    operationalisations (semantic is a minimal-pair contrast; fragmentation and
    priming are family contrasts carrying a turn-count differential), so a
    failure to transfer means the contrast did not generalise ACROSS
    operationalisations with different surface geometry -- not that no shared
    intent representation exists.
    """
    import matplotlib.pyplot as plt

    A = np.asarray(auc, dtype=float)
    fig, ax = plt.subplots(figsize=(1.6 + 0.9 * len(classes), 1.4 + 0.8 * len(classes)))
    im = _hm(ax, A, _DIVERGING, vmin=0.0, vmax=1.0)
    _labels(ax, classes, classes, rot=0)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if 0.35 < A[i, j] < 0.65 else "white")
    ax.set_xlabel("test class"); ax.set_ylabel("train class")
    ax.set_title(title + "\n(secondary: robustness across operationalisations)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, label="AUC")
    fig.tight_layout()
    return fig, {"matrix": A, "classes": classes}


# ── 4. Signal source × evaluation split ──────────────────────────────────────

SPLIT_LADDER = ("Example CV", "Pair CV", "Realisation CV", "Relation CV")

def signal_by_split_heatmap(
    auc: np.ndarray,
    feature_families: list[str],
    splits: tuple[str, ...] = SPLIT_LADDER,
    title: str = "AUC by signal source × evaluation split",
):
    """rows = feature family (Length, Sequence surprisal, Variance, Tail, Max
    surprisal, External-LM features, Residual probe); cols = the validation
    ladder, left to right in increasing strictness.

    This is the decomposition of PREDICTIVE SIGNAL, not a baseline table: read
    right-to-left to see what survives progressively stricter evaluation.

      both probe and external-LM rows collapse at Relation CV
        -> the signal was tied to repeated construction structure;
      probe survives where external-LM features do not
        -> a residual not available from an external model's surprisal, which is
           the more interesting result.

    Every row must be computed under the SAME folds, or the comparison is a leaky
    number against clean ones. Note what a high Example CV column means here: it
    is the failure-prone baseline, retained precisely so its collapse is visible.
    """
    import matplotlib.pyplot as plt

    A = np.asarray(auc, dtype=float)
    fig, ax = plt.subplots(figsize=(1.8 + 1.4 * len(splits), 1.4 + 0.45 * len(feature_families)))
    im = _hm(ax, A, _DIVERGING, vmin=0.0, vmax=1.0)
    _labels(ax, feature_families, list(splits), rot=0)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = A[i, j]
            txt = "—" if not np.isfinite(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black" if (np.isfinite(v) and 0.35 < v < 0.65) else "white")
    ax.set_title(title + "\n(strictness increases →)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, label="AUC")
    fig.tight_layout()
    return fig, {"matrix": A, "rows": feature_families, "cols": list(splits)}


# ── 5. Layer × relation effect ───────────────────────────────────────────────

def layer_relation_heatmap(
    effect: np.ndarray,
    layers: list[str],
    relation_ids: list[str],
    title: str = "Effect size by layer × relation",
):
    """rows = scanned layer (depth-ordered), cols = semantic relation.

    Shows whether the contrast appears early, late, uniformly, or only for
    certain relations -- the last being the case a depth-pooled summary hides.
    Layers are the depth-matched scan grid (layers.scan_layers_for), so this is
    comparable across models at equal depth fraction.
    """
    import matplotlib.pyplot as plt

    E = np.asarray(effect, dtype=float)
    lim = float(np.abs(E).max()) or 1.0
    fig, ax = plt.subplots(figsize=(1.8 + 0.85 * len(relation_ids), 1.2 + 0.4 * len(layers)))
    im = _hm(ax, E, _DIVERGING, vmin=-lim, vmax=lim)
    _labels(ax, layers, relation_ids, rot=30)
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            ax.text(j, i, f"{E[i, j]:+.2f}", ha="center", va="center", fontsize=6.5)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, label="effect size")
    fig.tight_layout()
    return fig, {"matrix": E}


# ── 6. Generator audit ───────────────────────────────────────────────────────

GENERATOR_AUDIT_COLS = ("Length", "GPT-2 surprisal", "GPT-NeoX surprisal",
                        "Variance", "Tail", "Lexical overlap")

def generator_audit_heatmap(
    sep: np.ndarray,
    generator_families: list[str],
    columns: tuple[str, ...] = GENERATOR_AUDIT_COLS,
    title: str = "Generator audit: standardized class separation",
):
    """rows = generator family, cols = upstream signal, colour = STANDARDIZED
    separation (e.g. Cohen's d between labels on that feature).

    The upstream analogue of the residual-stream audit: it asks whether the
    signal can already be explained BEFORE a forward pass. Read it as "where does
    the generator leak information" -- a hot cell means that family's classes are
    distinguishable on that surface property alone.

    Interpretation guard: an external LM recovering the labels shows the
    predictive information is available WITHOUT interrogating Pythia's residual
    stream, and is therefore not unique to Pythia's internal representations. It
    does NOT establish that the signal originated in the generator -- an external
    model may share broad statistical regularities with it, or the labels may
    correlate with pervasive properties of natural language.
    """
    import matplotlib.pyplot as plt

    S = np.asarray(sep, dtype=float)
    lim = float(np.abs(S[np.isfinite(S)]).max()) if np.isfinite(S).any() else 1.0
    fig, ax = plt.subplots(figsize=(2.0 + 1.35 * len(columns), 1.3 + 0.45 * len(generator_families)))
    im = _hm(ax, S, _DIVERGING, vmin=-lim, vmax=lim)
    _labels(ax, generator_families, list(columns), rot=30)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            v = S[i, j]
            ax.text(j, i, "—" if not np.isfinite(v) else f"{v:+.2f}",
                    ha="center", va="center", fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.035, label="standardized separation (d)")
    fig.tight_layout()
    return fig, {"matrix": S, "rows": generator_families, "cols": list(columns)}


# ── Two-panel: absolute geometry vs matched displacement ─────────────────────

def two_panel_geometry(
    adv_acts: np.ndarray,
    deltas: np.ndarray,
    relation_ids: list[str],
    realisation_ids: list[int] | None = None,
    provenance: list[str] | None = None,
    feature_basis: np.ndarray | None = None,
    top_k: int = 60,
    title: str = "Absolute adversarial geometry vs matched displacement",
):
    """The comparison the paper's argument turns on. Two panels, SAME row order,
    SAME feature basis.

      Panel A -- within-adversarial structure: h(x_i^adv), standardized per
        feature. Asks what organises the labelled adversarial population. Strong
        relation banding here means "adversarial" is not one coherent activation
        population; topic or relation dominates absolute geometry.
      Panel B -- matched displacement: Δ_i = h(adv) − h(clean). Asks whether the
        clean→adversarial transition itself is shared once matched frame content
        is subtracted away.

    The four readings:
      A banded / B shared      -> absolute states stay semantic, but the
                                  intervention may induce a common direction;
      A banded / B per-relation-> the apparent contrast is probably tied to
                                  relation or wording;
      A flat   / B shared      -> best candidate for a common
                                  manipulation-associated direction;
      A flat   / B unstable    -> little evidence for either coherent state or
                                  coherent displacement.

    Even the most favourable cell is MANIPULATION-associated, not
    intent-associated: a stable Δ may consistently recover the edited words, the
    syntax change, or a surprisal shift. That is what the external-LM audit is
    for, and why this figure cannot settle the question by itself.

    Both panels are drawn on the same basis (`feature_basis`) so a reader can see
    whether Panel A's subgroup structure survives, vanishes, or reorganises under
    subtraction. Selecting columns independently per panel would manufacture
    structure in both -- the caution `select_feature_basis` exists to enforce.
    """
    import matplotlib.pyplot as plt

    A = np.asarray(adv_acts, dtype=float)
    D = np.asarray(deltas, dtype=float)
    if A.shape != D.shape:
        raise ValueError(f"adv_acts {A.shape} and deltas {D.shape} must align row-for-row.")
    n = A.shape[0]

    cols = (select_feature_basis(D, None, top_k) if feature_basis is None
            else np.asarray(feature_basis))
    order = sorted(range(n), key=lambda i: (relation_ids[i],
                                            -1 if realisation_ids is None else realisation_ids[i]))
    rel = [relation_ids[i] for i in order]

    Az = A[order][:, cols]
    Az = (Az - Az.mean(axis=0)) / np.where(Az.std(axis=0) < 1e-8, 1.0, Az.std(axis=0))
    Dz = D[order][:, cols]

    labels = []
    for i in order:
        lab = relation_ids[i]
        if realisation_ids is not None:
            lab += f":{realisation_ids[i]}"
        if provenance is not None:
            lab += f" [{provenance[i][:4]}]"
        labels.append(lab)

    fig, axes = plt.subplots(1, 2, figsize=(15, max(3.2, 0.3 * n)), sharey=True)
    for ax, M, sub in ((axes[0], Az, "A — within-adversarial h(adv), standardized"),
                       (axes[1], Dz, "B — matched displacement Δ = h(adv) − h(clean)")):
        lim = float(np.abs(M).max()) or 1.0
        im = _hm(ax, M, _DIVERGING, vmin=-lim, vmax=lim)
        ax.set_title(sub, fontsize=9)
        ax.set_xlabel(f"shared basis: {len(cols)} features")
        for b in [k for k in range(1, len(rel)) if rel[k] != rel[k - 1]]:
            ax.axhline(b - 0.5, color="black", lw=1.2)
        fig.colorbar(im, ax=ax, fraction=0.025)
    _labels(axes[0], labels, None)
    fig.suptitle(title + "  (same row order, same feature basis)", fontsize=10)
    fig.tight_layout()
    return fig, {"row_order": order, "cols": cols.tolist(),
                 "panel_a": Az, "panel_b": Dz, "relations": rel}


# ── Pair x reference-relation: the summary diagnostic (A primary, B companion) ─

def pair_reference_heatmaps(
    agreement: dict,
    realisation_ids: list[int] | None = None,
    provenance: list[str] | None = None,
    title: str = "Pair × reference-relation",
):
    """Two panels from grouped_stats.pair_reference_agreement.

      A (PRIMARY) — signed direction agreement, cos(Δ_i, ref_{t,−i}), scale pinned
        to [−1, +1] so a blue cell is a genuine sign REVERSAL rather than a small
        number. Answers: does the manipulation move representations the same way?
      B (COMPANION) — SIGNED held-out projection along the reference direction.
        Answers: how far, and which way? Never read alone: two cells can both be
        large while pointing opposite ways in A, which is evidence AGAINST a
        shared representation even though a local classifier succeeds in both.
        The sign is kept deliberately — absolute values would hide reversals.

    COLUMN GRAMMAR:
      boxed diagonal cell  — within-relation, LEAVE-ONE-REALISATION-OUT (the
                             reference is the relation's other realisations, this
                             frame removed);
      off-diagonal cell    — CROSS-RELATION TRANSFER: the direction is learned
                             from that relation ALONE and applied to a frame from
                             another. NOT leave-one-relation-out;
      "pooled (LORO)"      — true leave-one-relation-out: a direction pooled over
                             every relation except the row's. This is the column
                             that tests for a relation-general direction; no
                             single off-diagonal cell does.

    Every reference excluded the plotted frame. NaN cells (a singleton relation
    has no reference left) render blank, never 0.0 — "not estimable" must not read
    as "no agreement".

    The per-relation gap G_r = mean(within) − mean(cross) is printed beside each
    relation block, quantifying what the block structure shows visually.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    A, B = np.asarray(agreement["A"]), np.asarray(agreement["B"])
    columns = agreement.get("columns", agreement["relations"])
    relations = agreement["relations"]
    row_rel = agreement["row_relation"]
    gap = agreement.get("gap_by_relation", {})
    n = A.shape[0]

    order = sorted(range(n), key=lambda i: (row_rel[i],
                                            -1 if realisation_ids is None else realisation_ids[i]))
    rel_s = [row_rel[i] for i in order]
    labels = []
    for i in order:
        lab = row_rel[i]
        if realisation_ids is not None:
            lab += f":{realisation_ids[i]}"
        if provenance is not None:
            lab += f" [{provenance[i][:4]}]"
        labels.append(lab)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, max(3.4, 0.34 * n)), sharey=True)
    panels = (
        (axes[0], A[order], -1.0, 1.0,
         "A — signed direction agreement  cos(Δᵢ, ref₍t,−i₎)"),
        (axes[1], B[order], None, None,
         "B — signed held-out projection along reference direction"),
    )
    for ax, M, vmin, vmax, sub in panels:
        if vmin is None:
            lim = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 1.0
            vmin, vmax = -lim, lim
        im = _hm(ax, np.ma.masked_invalid(M), _DIVERGING, vmin=vmin, vmax=vmax)
        ax.set_title(sub, fontsize=9)
        _labels(ax, None, columns, rot=30)
        ax.set_xlabel("reference (plotted frame always excluded)")
        for b in [k for k in range(1, len(rel_s)) if rel_s[k] != rel_s[k - 1]]:
            ax.axhline(b - 0.5, color="black", lw=1.2)
        ax.axvline(len(relations) - 0.5, color="black", lw=1.6)   # split off pooled column
        for r_i, r in enumerate(rel_s):
            ax.add_patch(Rectangle((relations.index(r) - 0.5, r_i - 0.5), 1, 1,
                                   fill=False, edgecolor="black", lw=1.4))
        for r_i in range(M.shape[0]):
            for c_i in range(M.shape[1]):
                v = M[r_i, c_i]
                if np.isfinite(v):
                    ax.text(c_i, r_i, f"{v:+.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.025)
    _labels(axes[0], labels, columns, rot=30)

    # Per-relation gap G_r annotated at each block's midpoint.
    if gap:
        seen, mid = {}, {}
        for k, r in enumerate(rel_s):
            seen.setdefault(r, []).append(k)
        for r, ks in seen.items():
            mid[r] = float(np.mean(ks))
        for r, y in mid.items():
            g = gap.get(r, {})
            axes[1].text(A.shape[1] + 0.15, y,
                         f"G={g.get('G', float('nan')):+.2f}\nLORO {g.get('pooled_loro_mean', float('nan')):+.2f}",
                         fontsize=6.5, va="center")

    fig.suptitle(
        f"{title} — boxed = within-relation (leave-one-realisation-out); "
        f"off-diagonal = cross-relation TRANSFER; last column = pooled LORO.  "
        f"G_r = mean(within) − mean(cross)", fontsize=8.5)
    fig.tight_layout()
    return fig, {"order": order, "A": A[order], "B": B[order],
                 "columns": columns, "relations": relations, "gap_by_relation": gap}


# ── Three-stage sequence on the pair x reference grid ────────────────────────

def three_stage_heatmap(
    three_stage: dict,
    realisation_ids: list[int] | None = None,
    provenance: list[str] | None = None,
    title: str = "Surface → intent → intent ⊥ surface",
):
    """Panel A (signed direction agreement) for all three stages, same grid, same
    row order, IDENTICAL colour limits [-1, +1] so the panels are comparable by
    eye rather than only by number.

      Stage 1  SURFACE           what adversarial surface alone does (intent held
                                 constant: both members benign);
      Stage 2  INTENT            the matched-pair displacement;
      Stage 3  INTENT ⊥ SURFACE  Stage 2 with the learned surface direction
                                 projected out, the direction for each row
                                 estimated without that row.

    THE READ: does Stage 2's agreement SURVIVE into Stage 3? If it collapses, the
    apparent shared direction was the surface manipulation's signature -- the
    rival explanation a stable Δ cannot exclude on its own. If it survives, the
    displacement is not merely recovering the edited words or the register shift.

    Watch `residual_fraction` in the suptitle: when it is small, the intent
    displacement lived almost entirely along the surface direction, so Stage 3 has
    little left to agree ABOUT. That is a finding, but it is a different finding
    from "the direction disagrees", and the two look identical in pale cells.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    stages = [("1 — surface", three_stage.get("surface")),
              ("2 — intent", three_stage["intent"]),
              ("3 — intent ⊥ surface", three_stage["intent_minus_surface"])]
    stages = [(n, s) for n, s in stages if s is not None]

    ref = stages[-1][1]
    relations, columns = ref["relations"], ref.get("columns", ref["relations"])
    row_rel = ref["row_relation"]
    n = len(row_rel)
    order = sorted(range(n), key=lambda i: (row_rel[i],
                                            -1 if realisation_ids is None else realisation_ids[i]))
    rel_s = [row_rel[i] for i in order]
    labels = []
    for i in order:
        lab = row_rel[i]
        if realisation_ids is not None:
            lab += f":{realisation_ids[i]}"
        if provenance is not None:
            lab += f" [{provenance[i][:4]}]"
        labels.append(lab)

    fig, axes = plt.subplots(1, len(stages), figsize=(5.2 * len(stages), max(3.4, 0.34 * n)),
                             sharey=True)
    if len(stages) == 1:
        axes = [axes]
    for ax, (sname, st) in zip(axes, stages):
        M = np.asarray(st["A"])[order]
        im = _hm(ax, np.ma.masked_invalid(M), _DIVERGING, vmin=-1.0, vmax=1.0)
        pooled = float(np.nanmean(M[:, -1])) if M.shape[1] > len(relations) else float("nan")
        ax.set_title(f"Stage {sname}\npooled LORO {pooled:+.2f}", fontsize=9)
        _labels(ax, None, columns, rot=30)
        ax.axvline(len(relations) - 0.5, color="black", lw=1.6)
        for b in [k for k in range(1, len(rel_s)) if rel_s[k] != rel_s[k - 1]]:
            ax.axhline(b - 0.5, color="black", lw=1.2)
        for r_i, r in enumerate(rel_s):
            ax.add_patch(Rectangle((relations.index(r) - 0.5, r_i - 0.5), 1, 1,
                                   fill=False, edgecolor="black", lw=1.3))
        for r_i in range(M.shape[0]):
            for c_i in range(M.shape[1]):
                v = M[r_i, c_i]
                if np.isfinite(v):
                    ax.text(c_i, r_i, f"{v:+.2f}", ha="center", va="center", fontsize=5.5)
        fig.colorbar(im, ax=ax, fraction=0.025)
    _labels(axes[0], labels, columns, rot=30)
    frac = three_stage.get("residual_fraction", float("nan"))
    fig.suptitle(f"{title} — signed agreement cos(Δ, ref₍−i₎), shared scale [−1, +1]. "
                 f"residual fraction after surface removal = {frac:.2f} "
                 f"(small ⇒ little left to agree about, which is not the same as disagreement)",
                 fontsize=8.5)
    fig.tight_layout()
    return fig, {"order": order, "stages": [s for s, _ in stages]}
