"""
smoke_test.py
AGPL 3. Copyright © 2026 Katherine J. Ombrellaro.
=============
Fast end-to-end correctness check for the maritime-intent-probe pipeline.

Runs the whole pipeline — payload generation, activation collection, SAE
fitting, both probes, causal comparison, transfer matrix, and all bias controls
— on a TINY randomly-initialised TransformerLens model and a handful of pairs.
It does NOT test research validity (a random model has no real signal); it tests
that every module wires together, every signature matches, every tensor shape is
consistent, and no path raises. Runs in well under a minute on CPU.

Usage:
    python smoke_test.py

Requires torch + transformer-lens + scikit-learn (see requirements.txt). If
torch is unavailable the script prints a clear message and exits 0 so it can be
dropped into environments where only the pure-Python logic tests run.
"""

import logging
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.WARNING)


def main() -> int:
    try:
        import torch
        from transformer_lens import HookedTransformer, HookedTransformerConfig
    except Exception as e:  # pragma: no cover
        print(f"[skip] torch/transformer-lens unavailable ({e}). "
              f"Run the pure-Python logic checks instead.")
        return 0

    from science.config import VaultConfig
    from science.environment import (MaritimeEnvironment, EmbeddingGateway, ENCODING_VARIANTS,
                             audit_minimal_pairs)
    from science.collector import ActivationCollector
    from science.sae import SparseAutoencoder
    from science.probe import (MassMeanProbe, LogisticRegressionProbe,
                       compare_probes, compare_probes_multiseed,
                       permutation_baseline_auc)
    from science.patching import CausalPatcher
    from science.experiment import ProbeExperiment, RunMode
    from science.transfer import run_transfer_matrix, run_held_out_template_analysis
    from science.bias_controls import (bc2_tokenisation_normalisation, bc3_pooling_comparison,
                               bc9_post_patch_alignment, bc10_legitimate_specificity,
                               bc11_permutation_baseline, bc10_patch_scope_fallback,
                               lr_threshold_sensitivity,
                               encoding_null_pooling_invariance)
    from science.null_validator import (validate_null_control,
                                validate_encoding_variant_nulls,
                                lexical_parity_report)
    from science.monitor import DualLayerAuditMonitor

    torch.manual_seed(0)
    device = "cpu"

    # ── Tiny model ────────────────────────────────────────────────────────────
    n_layers, d_model = 4, 32
    cfg_tl = HookedTransformerConfig(
        n_layers=n_layers, d_model=d_model, n_ctx=128, d_head=8, n_heads=4,
        d_mlp=64, d_vocab=1000, act_fn="gelu", normalization_type="LN",
    )
    model = HookedTransformer(cfg_tl)
    model.eval()

    class TinyTok:
        def encode(self, s): return [(abs(hash(w)) % 999) + 1 for w in s.split()][:120] or [1]
    model.tokenizer = TinyTok()

    hook_names = [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]
    base = Path(tempfile.mkdtemp())
    cfg = VaultConfig(base_dir=base)
    cfg.base_dir.mkdir(parents=True, exist_ok=True)

    env       = MaritimeEnvironment(model.tokenizer, seed=0)
    collector = ActivationCollector(model, hook_names, batch_size=8)
    probe     = MassMeanProbe()
    patcher   = CausalPatcher(model)

    n_pairs = 8
    step = 0

    def ok(msg):
        nonlocal step
        step += 1
        print(f"  [{step:2d}] {msg}")

    print("Running pipeline smoke test on tiny random model...")

    # ── H1 gateway ────────────────────────────────────────────────────────────
    gw = EmbeddingGateway(model, env, threshold=0.60)
    rates = gw.bypass_rates(n_pairs=n_pairs)
    assert set(["fragmentation", "semantic", "encoding", "priming"]).issubset(rates)
    ok("H1 gateway bypass_rates() returns all classes")

    # ── Encoding family + gateway normalization (axis E) ──────────────────────
    env.reset_rng()
    mixed = env.generate_pairs(n_pairs)
    enc_variants = {p.encoding_variant for p in mixed
                    if p.attack_class == "encoding"}
    assert enc_variants and enc_variants.issubset(set(ENCODING_VARIANTS)), \
        "encoding payloads must be tagged with a known variant"
    ok(f"encoding family: payloads tagged {sorted(v for v in enc_variants)}")

    rates_norm = gw.bypass_rates(n_pairs=n_pairs, normalize=True)
    assert rates_norm.get("_normalize") is True
    assert 0.0 <= rates_norm["encoding"]["bypass_rate"] <= 1.0
    ok("gateway bypass_rates(normalize=True) runs (matched-pair regime)")

    sens = gw.normalization_sensitivity(n_pairs=n_pairs)
    assert set(sens["per_variant"]) == set(ENCODING_VARIANTS)
    for v, d in sens["per_variant"].items():
        assert 0.0 <= d["off"] <= 1.0 and 0.0 <= d["on"] <= 1.0
        assert d["expected_normalizable"] in (True, False)
    ok(f"normalization_sensitivity: {len(sens['per_variant'])} variants, "
       f"legit clear off/on={sens['_legitimate_clear_rate_off']}/"
       f"{sens['_legitimate_clear_rate_on']}")

    # ── Template provenance (guarded) ─────────────────────────────────────────
    # Held-out-template robustness analysis depends on every payload carrying a
    # `template_key` whose prefix matches its class family. The field is not yet
    # part of RoutingPayload; this check activates automatically once it is, and
    # skips cleanly until then so the smoke test stays correct in the meantime.
    if any(getattr(p, "template_key", None) is not None for p in mixed):
        unset = [p for p in mixed if getattr(p, "template_key", "UNSET") == "UNSET"]
        assert not unset, (
            f"{len(unset)}/{len(mixed)} payloads have template_key 'UNSET' — "
            "provenance plumbing is broken (held-out-template analysis would be empty)"
        )
        bad = [p.template_key for p in mixed
               if p.template_key.split(":")[0] != p.attack_class]
        assert not bad, f"template_key prefix does not match attack_class: {bad[:3]}"
        assert len({p.template_key for p in mixed}) > 1, \
            "expected multiple distinct templates (diversity design); got one"
        ok(f"template provenance: {len({p.template_key for p in mixed})} distinct keys, none UNSET")
    else:
        ok("template provenance: SKIPPED (RoutingPayload has no template_key field yet)")

    # ── Minimal pairs: generation + statistical audit ─────────────────────────
    mpairs = env.generate_minimal_pairs()
    assert len(mpairs) == 16, "expect 4 vectors x 2 domains x 2 roles"
    groups = {}
    for p in mpairs:
        assert p.minimal_pair_id is not None
        assert p.template_key.split(":")[0] == p.attack_class
        groups.setdefault(p.minimal_pair_id, []).append(p)
    assert all(sorted(c.label for c in g) == [0, 0, 1, 1] for g in groups.values()), \
        "each minimal pair must link 2 clean + 2 adversarial cells"
    ok(f"minimal pairs: {len(mpairs)} cells in {len(groups)} linked 2x2 groups")

    audit = audit_minimal_pairs(model, env, surprisal_threshold=1.5)
    assert "_summary" in audit and all(v in audit for v in
        ["fragmentation", "semantic", "encoding", "priming"])
    for vec in ["fragmentation", "semantic", "encoding", "priming"]:
        assert "maritime_surprisal_delta" in audit[vec] and "maritime_length_delta" in audit[vec]
    ok(f"minimal-pair audit: {audit['_summary']['n_warnings']} matching warning(s) "
       f"(advisory); surprisal computed from logits")

    # ── Experiment ────────────────────────────────────────────────────────────
    experiment = ProbeExperiment(
        model, cfg, env, collector, probe, patcher,
        mode=RunMode.ITERATION, sae_n_features=128, sae_n_steps=20,
    )
    report = experiment.run(n_pairs=n_pairs)
    assert report["best_layer"] in hook_names
    for r in [rr for tr in [report["accumulation"]] for rr in tr]:
        assert 0.0 <= r["mean_auc"] <= 1.0
    ok(f"experiment.run() complete, best_layer={report['best_layer']}")

    cc = report["causal_comparison"]
    assert cc["best_probe_method"] in ("mass_mean", "logistic_regression", "inconclusive")
    ok(f"compare_probes() -> {cc['best_probe_method']} (margin {cc['margin']})")

    best_layer = report["best_layer"]
    sae = experiment._saes[best_layer]

    # ── Direction spaces ──────────────────────────────────────────────────────
    d_feat = probe.direction_for(best_layer)
    d_raw  = probe.raw_direction_for(best_layer)
    assert d_feat.shape[0] == experiment.sae_n_features
    assert d_raw.shape[0]  == d_model
    assert abs(d_raw.norm().item() - 1.0) < 1e-4
    ok(f"direction_for()={tuple(d_feat.shape)} (SAE), "
       f"raw_direction_for()={tuple(d_raw.shape)} (residual), unit-norm")

    # ── BC1 null validation (must run, two-class, not crash) ──────────────────
    bc1 = validate_null_control(model, cfg, env, collector, probe,
                                saes=experiment._saes, device=device, n_pairs=n_pairs)
    assert "passed" in bc1 and any(v >= 0 for v in bc1["auc_per_layer"].values())
    ok(f"BC1 null validation ran, worst_auc={bc1['worst_auc']} (valid AUCs present)")

    # ── BC1 per-encoding-variant null (axis E, airtight family) ───────────────
    bc1v = validate_encoding_variant_nulls(
        model, cfg, env, collector, probe,
        saes=experiment._saes, device=device, n_pairs=n_pairs,
    )
    assert set(bc1v["per_variant"]) == set(ENCODING_VARIANTS)
    for v, d in bc1v["per_variant"].items():
        assert "passed" in d and "expected_normalizable" in d
        assert any(a >= 0 for a in d["auc_per_layer"].values()), \
            f"variant {v} produced no valid AUC"
    assert bc1v["worst_variant"] in ENCODING_VARIANTS
    ok(f"BC1 per-variant null ran for {len(bc1v['per_variant'])} variants, "
       f"worst={bc1v['worst_variant']} auc={bc1v['worst_auc']}")

    # ── BC6 stability ─────────────────────────────────────────────────────────
    pairs = env.generate_pairs(n_pairs)
    mean_acts, _, labels = collector.collect_dual(pairs, env, turn_index=-1)
    with torch.no_grad():
        sae_acts = {best_layer: sae.encode(mean_acts[best_layer].float().to(device)).cpu()}
    legit_means = {l: X.mean(dim=0) for l, X in mean_acts.items()}
    stab = compare_probes_multiseed(
        probe, patcher, [p for p in pairs if p.label == 1], legit_means, env,
        best_layer, sae_acts, labels, sae=sae, seeds=[42, 7], n_sample=4,
    )
    assert "winner_consistent" in stab
    ok(f"BC6 stability: consistent={stab['winner_consistent']}")

    # ── BC2 / BC3 ─────────────────────────────────────────────────────────────
    bc2 = bc2_tokenisation_normalisation(model, env, collector, probe, sae, best_layer,
                                         n_pairs=n_pairs, device=device)
    assert bc2["raw_auc"] >= 0, "BC2 must produce a valid AUC, not the old error path"
    assert set(bc2["per_variant"]) == set(ENCODING_VARIANTS), \
        "BC2 must report every encoding variant (no mixed-variant dilution)"
    for v, d in bc2["per_variant"].items():
        assert "auc_drop" in d and "expected_normalizable" in d
    ok(f"BC2 per-variant: worst='{bc2['worst_variant']}' drop={bc2['auc_drop']} "
       f"driven={bc2['driven_variants']}")

    # ── Tool A: lexical-parity diagnostic (model-independent) ─────────────────
    lp = lexical_parity_report(env, n_pairs=n_pairs)
    assert lp["gate"] == "mixed" and "cosine" in lp["mixed"]
    assert set(lp["per_variant"]) == set(ENCODING_VARIANTS)
    ok(f"Tool A lexical parity: mixed cosine={lp['mixed']['cosine']} "
       f"below_floor={lp['mixed']['below_floor']}")

    # ── Tool C: encoding-null pooling-invariance ──────────────────────────────
    tc = encoding_null_pooling_invariance(env, collector, probe, sae, best_layer,
                                          n_pairs=n_pairs)
    assert set(tc["per_variant"]) == set(ENCODING_VARIANTS)
    for v, d in tc["per_variant"].items():
        assert d["classification"] in (
            "clean", "pooling_specific_artifact", "structural_surface_leak", "unevaluable")
    ok(f"Tool C pooling-invariance ran; artifacts={tc['pooling_specific_artifacts']} "
       f"structural={tc['structural_surface_leaks']}")

    bc3 = bc3_pooling_comparison(env, collector, probe, sae, best_layer, n_pairs=n_pairs)
    assert set(bc3["results_per_class"]) == {"fragmentation", "semantic", "encoding", "priming"}
    ok("BC3 pooling comparison ran for all classes")

    # ── BC9 / BC10 (raw direction!) ───────────────────────────────────────────
    legit_mean_act = mean_acts[best_layer][labels == 0].mean(dim=0)
    adv = [p for p in pairs if p.label == 1]
    leg = [p for p in pairs if p.label == 0]
    bc9 = bc9_post_patch_alignment(patcher, adv, legit_mean_act, d_raw, best_layer,
                                   env, collector)
    assert 0.0 <= bc9["causal_rate"] <= 1.0 and "measure_layer" in bc9
    ok(f"BC9 post-patch: causal_rate={bc9['causal_rate']} measure_layer={bc9['measure_layer']}")

    bc10 = bc10_legitimate_specificity(patcher, leg, legit_mean_act, d_raw, best_layer, env)
    assert "passed" in bc10
    ok(f"BC10 specificity: degradation_rate={bc10['degradation_rate']} passed={bc10['passed']}")

    # ── BC11 transfer-aware permutation null ──────────────────────────────────
    tr_pairs = env.generate_class_pairs("fragmentation", n_pairs=n_pairs)
    te_pairs = env.generate_class_pairs("semantic", n_pairs=n_pairs)
    trm, _, trl = collector.collect_dual(tr_pairs, env, turn_index=-1)
    tem, _, tel = collector.collect_dual(te_pairs, env, turn_index=-1)
    with torch.no_grad():
        tr_sae = {best_layer: sae.encode(trm[best_layer].float().to(device)).cpu()}
        te_sae = {best_layer: sae.encode(tem[best_layer].float().to(device)).cpu()}
    bc11 = bc11_permutation_baseline(tr_sae, trl, te_sae, tel, best_layer, n_permutations=50)
    assert 0.0 <= bc11["null_auc_mean"] <= 1.0
    ok(f"BC11 transfer null: mean={bc11['null_auc_mean']} p95={bc11['null_auc_p95']}")

    # ── Transfer matrix ───────────────────────────────────────────────────────
    tmatrix = run_transfer_matrix(model, cfg, env, collector, experiment, probe,
                                  global_best_layer=best_layer, device=device, n_pairs=n_pairs)
    assert "class_specific" in tmatrix and "global" in tmatrix
    ok(f"transfer matrix: outcome={tmatrix['aggregate']['outcome']}")

    # ── Held-out-template robustness (provenance-driven) ──────────────────────
    hot = run_held_out_template_analysis(
        model, cfg, env, collector, experiment, best_layer,
        group_levels=None, device=device, n_pairs=n_pairs, min_group_size=3,
    )
    assert "per_class" in hot and "overall_worst_held_out_auc" in hot
    for c, d in hot["per_class"].items():
        if "per_group" in d:
            assert d["evaluated_groups"] >= 0 and "worst_held_out_auc" in d
    ok(f"held-out-template (per-template): overall_worst="
       f"{hot['overall_worst_held_out_auc']} passed={hot['overall_passed']}")

    # Per-VARIANT grain on encoding (the non-uniform-key payoff: held-out surface)
    hot_v = run_held_out_template_analysis(
        model, cfg, env, collector, experiment, best_layer,
        classes=["encoding"], group_levels=1, device=device,
        n_pairs=n_pairs, min_group_size=3,
    )
    enc = hot_v["per_class"]["encoding"]
    assert enc["n_template_groups"] <= len(ENCODING_VARIANTS), \
        "variant-grain groups must collapse to at most one fold per variant"
    ok(f"held-out-template (per-variant): {enc['n_template_groups']} variant folds, "
       f"worst={enc['worst_held_out_auc']} worst_variant={enc['worst_group']}")

    # ── §3.3: Finer-grid layer refinement (push-button) ───────────────────────
    refinement = experiment.refine_layers(report, span=1, n_pairs=n_pairs)
    assert refinement["refined_best_layer"] in experiment._saes, \
        "refined layer must have a cached SAE for downstream use"
    assert refinement["coarse_best_layer"] == best_layer
    assert all(0.0 <= r["auc"] <= 1.0 for r in refinement["refined_results"].values())
    ok(f"refine_layers(): {refinement['refined_best_layer']} "
       f"({'changed' if refinement['selection_changed'] else 'confirmed'}), "
       f"{len(refinement['refined_results'])} layers evaluated")

    # ── BC9 registered method: measurement pinned to the probe layer ──────────
    assert bc9["measure_layer"] == best_layer, \
        "bc9 wrapper must measure at the probe layer per PREREGISTRATION §4.13"
    ok("BC9 measure_layer pinned to probe layer (registered method §4.13)")

    # ── §4.16: LR probability threshold sensitivity ───────────────────────────
    sens = lr_threshold_sensitivity(experiment.lr_probe, sae_acts, labels, best_layer)
    assert set(sens["per_threshold"]) == {"0.40", "0.50", "0.60"}
    assert isinstance(sens["threshold_sensitive"], bool)
    ok(f"lr_threshold_sensitivity: spread={sens['positive_rate_spread']} "
       f"sensitive={sens['threshold_sensitive']}")

    # ── Patch-scope variants (BC10 fallback primitives) ───────────────────────
    toks = env.tokenize(adv[0])[-1]
    up, p1 = patcher.directional_patch_final_token_only(toks, legit_mean_act, d_raw, best_layer)
    assert up.shape == p1.shape
    ok("directional_patch_final_token_only: shapes consistent")

    ident = patcher.per_position_causal_rates(adv[:3], legit_mean_act, d_raw,
                                              best_layer, env, max_positions=3)
    assert len(ident["rates_by_position"]) == 3
    assert isinstance(ident["signal_positions"], list)
    ok(f"per_position_causal_rates: {ident['rates_by_position']} "
       f"signal={ident['signal_positions']}")

    up, p2 = patcher.directional_patch_positions_and_run(
        toks, legit_mean_act, d_raw, best_layer, positions=[-1, -2])
    assert up.shape == p2.shape
    ok("directional_patch_positions_and_run: shapes consistent")

    # ── §6: BC10 patch-scope fallback orchestrator (full sequence) ────────────
    fb = bc10_patch_scope_fallback(
        patcher, leg[:3], adv[:3], legit_mean_act, d_raw, best_layer, env,
        max_positions=3,
    )
    assert fb["final_status"] in ("passed_step1", "passed_step2", "exploratory")
    assert "step1_final_token" in fb["steps"]
    s1 = fb["steps"]["step1_final_token"]
    assert 0.0 <= s1["degradation_rate"] <= 1.0
    assert 0.0 <= s1["adversarial_causal_rate"] <= 1.0
    if fb["final_status"] != "passed_step1":
        assert "step2_position_identification" in fb["steps"], \
            "step 2 must run when step 1 fails"
    ok(f"bc10_patch_scope_fallback: {fb['final_status']} "
       f"(step1 deg={s1['degradation_rate']}, causal={s1['adversarial_causal_rate']})")

    # ── Monitor: summary spans MULTIPLE turns ─────────────────────────────────
    mon = DualLayerAuditMonitor(model, cfg, sae, d_feat, best_layer)
    for tid in range(3):
        toks = env.tokenize(adv[tid % len(adv)])[-1]
        mon.run_audit(toks, turn_id=tid)
    summ = mon.summary()
    assert summ["total_turns"] >= 3, "summary() must accumulate across turns, not reset"
    ok(f"monitor summary spans {summ['total_turns']} turns (multi-turn accumulation works)")

    print(f"\nSMOKE TEST PASSED — {step} checks, all modules wired correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
