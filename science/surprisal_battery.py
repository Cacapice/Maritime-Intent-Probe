"""
surprisal_battery.py
AGPL 3. Copyright © 2026 Katherine J. Ombrellaro.
====================
The upstream analogue of the residual-stream audit: can the class signal already
be explained BEFORE a forward pass through the model under study?

Structurally isomorphic to the residual audit, deliberately -- same attribution
ladder (length -> lexical -> distributional -> residual), same folds, same
grouping. That symmetry is the point: the two analyses then answer the same
causal question at two stages, rather than one being an appendix to the other.

WHAT AN EXTERNAL LM BUYS
------------------------
Pythia surprisal and Pythia activations are not independent witnesses -- surprisal
is a functional of the same forward pass, so a high surprisal AUC cannot separate
"the probe reads predictability" from "both read the same underlying signature".
An external LM (different family, different tokenizer) breaks that dependence.

INTERPRETATION GUARD -- read this before writing the claim.
If external-LM surprisal features recover the labels at ~probe AUC, the defensible
statement is:

    the predictive information is available from an external language model and is
    therefore not unique to Pythia's internal representations.

NOT "the signature is exogenous to Pythia" and not "the signal originated in the
generator". An external model may recover the labels because it shares broad
statistical regularities with the generator, or because the labels correlate with
pervasive properties of natural language. The design does not identify the origin,
and does not need to: showing the information is reachable without interrogating
the residual stream already relocates the burden.

MODEL-AGNOSTIC BY CONSTRUCTION
------------------------------
Everything takes a `logits_fn(token_ids) -> logits[1, T, V]` and an
`encode(text) -> list[int]`. Adapters for HookedTransformer and HF are provided.
That keeps the battery testable with a deterministic stub (no download, no GPU)
and lets the same code run GPT-2 and GPT-NeoX for the two-column comparison.
"""
from __future__ import annotations

import numpy as np

# Alignment-free families only. The edit-localized features
# (S_edited_token / S_pre-edit / S_post-edit) are DELIBERATELY absent: the main
# dataset's legit and adversarial members come from different template families
# with no shared frame, so there is no edit point to attach them to. They are
# well defined on the minimal-pair vault and (via a span diff) on the BC1 null
# twins -- compute them there, not here.
FEATURE_NAMES = (
    "n_tokens",        # length only
    "S_sequence",      # total NLL -- scales with length by construction
    "S_mean",          # length-normalized: the honest surprisal channel
    "S_var",           # Var_t S_t
    "S_max",           # max_t S_t
    "S_tail_p95",      # 95th percentile of per-token surprisal
)

FAMILY_COLUMNS = {
    "Length only":        ("n_tokens",),
    "Sequence surprisal": ("S_sequence",),
    "Mean surprisal":     ("S_mean",),
    "Variance":           ("S_var",),
    "Max surprisal":      ("S_max",),
    "Tail surprisal":     ("S_tail_p95",),
    "All surprisal":      ("S_mean", "S_var", "S_max", "S_tail_p95"),
    "Surprisal + length": FEATURE_NAMES,
}


# ── Core ──────────────────────────────────────────────────────────────────────

def token_surprisals(logits_fn, token_ids: list[int]) -> np.ndarray:
    """-log p(t_i | t_<i) in nats, for i = 1..T-1.

    Position 0 has no left context and is excluded -- including it would add a
    constant-ish unigram term that differs only by first token, which is a lexical
    artefact rather than a predictability signal.
    """
    import torch

    if len(token_ids) < 2:
        return np.zeros(0, dtype=float)
    toks = torch.tensor(token_ids).unsqueeze(0)
    with torch.no_grad():
        logits = logits_fn(toks)
        if hasattr(logits, "logits"):
            logits = logits.logits
        logp = torch.log_softmax(logits[0].float(), dim=-1)
    idx = torch.tensor(token_ids[1:])
    s = -logp[:-1].gather(1, idx.unsqueeze(1)).squeeze(1)
    return s.cpu().numpy().astype(float)


def surprisal_features(logits_fn, encode, texts: list[str]) -> np.ndarray:
    """[n_texts, len(FEATURE_NAMES)] feature matrix.

    A text too short to score (fewer than 2 tokens) yields zeros rather than NaN,
    so downstream folds do not silently drop rows; such inputs should not occur in
    this design and are worth checking for upstream.
    """
    rows = []
    for t in texts:
        ids = encode(t)
        s = token_surprisals(logits_fn, ids)
        if s.size == 0:
            rows.append([float(len(ids)), 0.0, 0.0, 0.0, 0.0, 0.0]); continue
        rows.append([
            float(len(ids)),
            float(s.sum()),
            float(s.mean()),
            float(s.var(ddof=1)) if s.size > 1 else 0.0,
            float(s.max()),
            float(np.quantile(s, 0.95)),
        ])
    return np.array(rows, dtype=float)


# ── Adapters ──────────────────────────────────────────────────────────────────

def hooked_transformer_adapter(model):
    """(logits_fn, encode) for a TransformerLens HookedTransformer."""
    return (lambda toks: model(toks.to(next(model.parameters()).device)),
            lambda text: model.tokenizer.encode(text))


def hf_adapter(model, tokenizer):
    """(logits_fn, encode) for a HuggingFace causal LM -- e.g. GPT-2, the
    external model for the cross-family comparison."""
    return (lambda toks: model(toks.to(next(model.parameters()).device)),
            lambda text: tokenizer.encode(text))


# ── The split x family matrix (feeds twin_figures.signal_by_split_heatmap) ────

def auc_by_family_and_split(
    X: np.ndarray,
    y,
    payloads,
    levels=("example", "pair", "realisation", "relation"),
    families=None,
    C: float = 1.0,
    n_splits: int = 5,
) -> tuple[np.ndarray, list[str], list[str]]:
    """AUC for each feature family under each split of the validation ladder.

    Every cell uses the SAME folds and the same estimator, so the comparison is
    like-for-like. A cell that cannot be computed (a level with too few groups to
    fold) is np.nan -- rendered "—" by the heatmap, never 0.0, which would read as
    "AUC 0.0" rather than "not run".

    Features are standardized inside each fold via the pipeline, so a family with
    a large raw scale (n_tokens, S_sequence) is not advantaged by regularisation.

    Returns (A, family_names, level_names) with A[i, j] = AUC.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    try:
        from science.probe import _cv_auc, group_keys
    except ImportError as e:
        # group_keys / _make_splitter arrived with the frame-level CV work. A
        # probe.py predating them raises a bare "cannot import name", which does
        # not say that the REPO is stale rather than the code wrong.
        raise ImportError(
            f"{e}\n\n"
            "surprisal_battery needs probe.group_keys (the validation-ladder "
            "grouping helper). The probe.py on this path predates it, so the repo "
            "copy is stale -- pull the current probe.py (it must export "
            "group_keys, _make_splitter and fold_composition) and re-run. Nothing "
            "here is wrong with the battery itself."
        ) from e

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    families = families or FAMILY_COLUMNS
    fam_names = list(families)
    cols = {n: i for i, n in enumerate(FEATURE_NAMES)}

    def fit_fn(Xtr, ytr):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, max_iter=2000, random_state=0))
        clf.fit(Xtr, ytr)
        return lambda Xte: clf.predict_proba(Xte)[:, 1]

    A = np.full((len(fam_names), len(levels)), np.nan)
    for i, fam in enumerate(fam_names):
        sub = X[:, [cols[c] for c in families[fam]]]
        for j, lvl in enumerate(levels):
            g = group_keys(payloads, lvl)
            if g is not None and len(set(g)) < 2:
                continue                       # not enough groups to fold at this level
            try:
                A[i, j] = _cv_auc(sub, y, fit_fn, n_splits=n_splits, groups=g)
            except Exception:
                pass                           # leave nan -> renders "—"
    return A, fam_names, list(levels)
