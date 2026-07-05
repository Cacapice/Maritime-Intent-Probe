"""
test_probe.py
=============
Two jobs in one file:

1. PROPERTY TESTS (pytest-compatible `test_*` functions; also runs standalone).
   Pins the *invariants* of the probe math — the mass-mean direction is the
   normalized difference of class means, degenerate CV splits fall back to an
   in-sample fit, LogisticRegression is reproducible under a fixed seed, the
   error paths raise, the permutation null centres on ~0.5. These don't depend
   on golden numbers, so they're robust across library versions.

2. EQUIVALENCE HARNESS (`dump` / `check` subcommands). Records the probe
   directions and AUCs for a fixed synthetic dataset to JSON, so a refactor can
   be proven behavior-preserving:

       python test_probe.py dump  golden_probe.json      # on CURRENT probe.py
       # ... apply a refactor (base Probe class, CV factoring, etc.) ...
       python test_probe.py check golden_probe.json      # must PASS

   This is the safety net the structural refactors in the review need: the
   mass-mean direction and the SAE-folded CV feed the locked BC1 gate and the H2
   AUCs, so "behaviour unchanged" must be demonstrated, not assumed. (The harness
   covers the sae=None path — the direction algebra and the plain CV runner. To
   also pin the SAE-folded CV path, pass a real SparseAutoencoder factory; left
   out here so the file needs no GPU/SAE to run.)

Requires torch + sklearn (i.e. the repo's normal environment); it is NOT runnable
in a torch-free sandbox. `python -m py_compile test_probe.py` checks syntax only.
AGPL 3. Copyright © 2026 Katherine J. Ombrellaro.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from probe import (
    MassMeanProbe,
    LogisticRegressionProbe,
    _cv_auc,
    _cv_auc_sae_per_fold,
    _project_to_raw,
    permutation_baseline_auc,
)

SEED = 0


def _deterministic_cpu():
    """Maximise SAE-fit reproducibility for the equivalence harness. The
    SAE-folded path fits a fresh SAE per fold, so before/after equivalence
    needs single-threaded, deterministic CPU math and a fixed seed."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


def _make_sae_factory(d_model: int, n_features: int = 64, n_steps: int = 120, seed: int = 0):
    """A small, deterministic SparseAutoencoder factory matching the
    Callable[[np.ndarray], SparseAutoencoder] contract _cv_auc_sae_per_fold
    expects: takes raw activations, returns a fitted SAE. Seeded so the per-fold
    fits are reproducible run-to-run (CPU)."""
    from sae import SparseAutoencoder

    def factory(X_raw_np: np.ndarray):
        torch.manual_seed(seed)
        sae = SparseAutoencoder(d_model, n_features)          # CPU
        sae.fit(torch.tensor(np.asarray(X_raw_np), dtype=torch.float32),
                n_steps=n_steps, batch_size=32, labels=None)
        return sae

    return factory


# ── deterministic synthetic data ──────────────────────────────────────────────

def _make_dataset(seed: int = SEED, n: int = 60, d: int = 32):
    """Balanced 2-class data with three layers of decreasing separability:
    'sep' (cleanly separable), 'weak' (small shift), 'noise' (no signal)."""
    rng = np.random.RandomState(seed)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    e = np.zeros(d, dtype=np.float32); e[0] = 1.0           # known signal axis

    def layer(shift):
        X = rng.randn(n, d).astype(np.float32)
        X[y == 1] += shift * e
        return torch.tensor(X)

    acts = {
        "blocks.sep.hook_resid_post":   layer(8.0),
        "blocks.weak.hook_resid_post":  layer(0.6),
        "blocks.noise.hook_resid_post": layer(0.0),
    }
    return acts, torch.tensor(y), e


def _lin_fit(Xtr, ytr):
    """Mass-mean-style numpy scorer, for exercising _cv_auc directly."""
    mu1, mu0 = Xtr[ytr == 1].mean(0), Xtr[ytr == 0].mean(0)
    d = mu1 - mu0
    d = d / (np.linalg.norm(d) + 1e-8)
    return lambda Xte: Xte @ d


# ── property tests ────────────────────────────────────────────────────────────

def test_mass_mean_direction_is_normalized_diff_of_means():
    acts, y, e = _make_dataset()
    X = acts["blocks.sep.hook_resid_post"]
    d = MassMeanProbe._mass_mean_dir(X, y.bool())
    # unit norm
    assert abs(float(d.norm()) - 1.0) < 1e-5
    # equals normalized (mean1 - mean0)
    raw = X[y.bool()].mean(0) - X[~y.bool()].mean(0)
    expect = raw / (raw.norm() + 1e-8)
    assert torch.allclose(d, expect, atol=1e-6)
    # on cleanly separable data, aligns with the known signal axis
    assert float(torch.dot(d, torch.tensor(e))) > 0.99


def test_mass_mean_separable_in_sample_auc_is_one():
    acts, y, _ = _make_dataset()
    res = {r.layer: r for r in MassMeanProbe().evaluate(acts, y, turn_index=0, sae=None)}
    sep = res["blocks.sep.hook_resid_post"]
    assert sep.auc_in_sample == 1.0
    assert sep.auc >= 0.99                       # held-out CV also separates
    assert abs(float(sep.direction.norm()) - 1.0) < 1e-5
    # the noise layer should be near chance, not separable
    assert res["blocks.noise.hook_resid_post"].auc <= 0.75


def test_raw_direction_equals_scoring_direction_without_sae():
    acts, y, _ = _make_dataset()
    p = MassMeanProbe(); p.evaluate(acts, y, turn_index=0, sae=None)
    layer = "blocks.sep.hook_resid_post"
    # with sae=None the raw direction is the scoring direction, re-normalized
    assert torch.allclose(p.direction_for(layer), p.raw_direction_for(layer), atol=1e-6)


def test_cv_auc_degenerate_split_falls_back_to_in_sample():
    rng = np.random.RandomState(SEED)
    X = rng.randn(8, 16).astype(np.float32)      # n=8 < n_splits*2 -> fallback branch
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    got = _cv_auc(X, y, _lin_fit)
    sf = _lin_fit(X, y)
    expect = float(roc_auc_score(y, sf(X)))      # the documented in-sample fallback
    assert abs(got - expect) < 1e-9


def test_cv_auc_min_class_count_one_falls_back():
    rng = np.random.RandomState(SEED)
    X = rng.randn(20, 16).astype(np.float32)
    y = np.array([0] + [1] * 19)                 # counts.min() == 1 < 2
    got = _cv_auc(X, y, _lin_fit)                # must not raise
    assert 0.0 <= got <= 1.0


def test_cv_auc_sae_per_fold_degenerate_falls_back_to_in_sample():
    _deterministic_cpu()
    rng = np.random.RandomState(SEED)
    d = 16
    X = rng.randn(8, d).astype(np.float32)       # n=8 < n_splits*2 -> fallback branch
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    factory = _make_sae_factory(d, n_features=32, n_steps=30)
    auc, std, n_folds = _cv_auc_sae_per_fold(X, y, factory, _lin_fit)
    assert n_folds == 1 and std == 0.0           # documented single in-sample fit
    assert 0.0 <= auc <= 1.0


def test_project_to_raw_none_is_unit():
    v = torch.randn(64)
    out = _project_to_raw(None, v)
    assert abs(float(out.norm()) - 1.0) < 1e-5


def test_project_to_raw_uses_decoder():
    import types
    d_model, n_features = 48, 96
    W = torch.randn(d_model, n_features)
    fake_sae = types.SimpleNamespace(decoder=types.SimpleNamespace(weight=W))
    feat_dir = torch.randn(n_features)
    out = _project_to_raw(fake_sae, feat_dir)
    expect = (W @ feat_dir)
    expect = expect / (expect.norm() + 1e-8)
    assert torch.allclose(out, expect, atol=1e-5)
    assert abs(float(out.norm()) - 1.0) < 1e-5


def test_logreg_is_reproducible_under_fixed_seed():
    acts, y, _ = _make_dataset()
    layer = "blocks.sep.hook_resid_post"
    a = LogisticRegressionProbe(random_state=42)
    b = LogisticRegressionProbe(random_state=42)
    ra = {r.layer: r for r in a.evaluate(acts, y, turn_index=0, sae=None)}[layer]
    rb = {r.layer: r for r in b.evaluate(acts, y, turn_index=0, sae=None)}[layer]
    assert torch.allclose(a.direction_for(layer), b.direction_for(layer), atol=1e-6)
    assert ra.auc == rb.auc and ra.auc_in_sample == rb.auc_in_sample


def test_direction_for_raises_before_evaluate():
    p = MassMeanProbe()
    for fn in (p.direction_for, p.raw_direction_for):
        try:
            fn("blocks.sep.hook_resid_post")
        except ValueError:
            pass
        else:
            raise AssertionError(f"{fn.__name__} should raise before evaluate()")


def test_permutation_null_centres_near_half():
    acts, y, _ = _make_dataset(n=40)
    layer = "blocks.noise.hook_resid_post"       # no real signal -> null ~ 0.5
    out = permutation_baseline_auc(
        acts, y, acts, y, layer, n_permutations=200, random_state=42,
    )
    assert "null_auc_mean" in out                # actual key: null_auc_{mean,p95,std}
    assert 0.35 <= out["null_auc_mean"] <= 0.65


# ── equivalence harness (golden dump / check) ─────────────────────────────────

def _full_sae(X_np: np.ndarray, n_features: int = 64, n_steps: int = 120, seed: int = 0):
    """A deployable SAE fit on the full layer (the `sae=` arg to evaluate)."""
    from sae import SparseAutoencoder
    torch.manual_seed(seed)
    sae = SparseAutoencoder(int(X_np.shape[1]), n_features)
    sae.fit(torch.tensor(np.asarray(X_np), dtype=torch.float32),
            n_steps=n_steps, batch_size=32)
    return sae


def _row(r) -> dict:
    return {
        "auc": r.auc, "auc_in_sample": r.auc_in_sample,
        "accuracy": r.accuracy, "auc_cv_std": r.auc_cv_std,
        "direction": [round(float(x), 8) for x in r.direction.tolist()],
    }


def _golden(include_sae: bool = False) -> dict:
    """Directions + AUCs for both probes on the fixed dataset.

    Default covers the sae=None path (direction algebra + plain CV runner).
    include_sae also exercises the SAE-folded CV (_cv_auc_sae_per_fold) — the
    path the BC1 gate runs — with a fresh seeded SAE per layer.
    """
    acts, y, _ = _make_dataset()
    out: dict = {}
    for name, probe in (("mass_mean", MassMeanProbe()),
                        ("logreg", LogisticRegressionProbe(random_state=42))):
        rows = probe.evaluate(acts, y, turn_index=0, sae=None)
        out[name] = {r.layer: _row(r) for r in rows}

    if include_sae:
        _deterministic_cpu()
        for name, make_probe in (("mass_mean_sae", lambda: MassMeanProbe()),
                                 ("logreg_sae", lambda: LogisticRegressionProbe(random_state=42))):
            section = {}
            for layer, A in acts.items():
                X_np = A.float().numpy()
                full = _full_sae(X_np)
                factory = _make_sae_factory(int(A.shape[1]))
                r = make_probe().evaluate(
                    {layer: A}, y, turn_index=0, sae=full, sae_factory=factory
                )[0]
                section[layer] = _row(r)
            out[name] = section
    return out


def _check_against(path: str) -> bool:
    ref = json.load(open(path))
    include_sae = any(k.endswith("_sae") for k in ref)
    cur = _golden(include_sae=include_sae)
    ok = True
    for probe, layers in ref.items():
        for layer, g in layers.items():
            c = cur[probe][layer]
            for k in ("auc", "auc_in_sample", "accuracy", "auc_cv_std"):
                if c[k] != g[k]:
                    ok = False
                    print(f"  MISMATCH {probe}/{layer}.{k}: golden={g[k]} current={c[k]}")
            dg, dc = np.array(g["direction"]), np.array(c["direction"])
            if dg.shape != dc.shape or not np.allclose(dg, dc, atol=1e-6):
                ok = False
                md = float(np.max(np.abs(dg - dc))) if dg.shape == dc.shape else float("nan")
                print(f"  MISMATCH {probe}/{layer}.direction: max|Δ|={md}")
    print("EQUIVALENCE: PASS — refactor preserved the probe numerics." if ok
          else "EQUIVALENCE: FAIL — refactor changed a locked quantity (see above).")
    return ok


def _run_property_tests() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\n{len(tests)} property tests passed.")


if __name__ == "__main__":
    args = sys.argv[1:]
    include_sae = "--sae" in args
    args = [a for a in args if a != "--sae"]
    if args and args[0] == "dump":
        path = args[1] if len(args) > 1 else "golden_probe.json"
        json.dump(_golden(include_sae=include_sae), open(path, "w"), indent=2)
        print(f"Golden probe values written to {path}"
              f"{' (incl. SAE-folded path)' if include_sae else ''}")
    elif args and args[0] == "check":
        path = args[1] if len(args) > 1 else "golden_probe.json"
        sys.exit(0 if _check_against(path) else 1)
    else:
        _run_property_tests()
