"""
notebook_setup.py — make the analysis cells self-contained.
===========================================================
The notebook used to be *cumulative*: every cell after "Cell 3: Experimental
phase" relied on names that an earlier cell had left in memory (`env`,
`collector`, `SCAN_LAYERS`, `hook_names`, and the heavy `experiment` / `probe`
directions / `report`). That meant you could not load the model and jump
straight to, say, Cell 7 (surface geometry) — you had to re-run Cell 3 and the
whole 3b chain first.

This module supplies two entry points so each cell can stand alone given only
the model load (Cell 2):

  ensure_shared(model)
      Rebuilds the CHEAP shared objects from `model` alone — instant, no GPU
      work, no disk. Returns (env, SCAN_LAYERS, hook_names, collector, probe).
      The scan grid is depth-matched per model via layers.scan_layers_for
      (identity on 1.4B; deep-inclusive on 6.9B). Enough on its own for the
      exploratory geometry battery (geometry / decomposition / cross-class /
      directional / cross-model), which never touches the trained probe.

  restore_analysis_state(model, CFG)
      ensure_shared + lazily loads the HEAVY artifacts that only Cell 3 produces
      (per-layer SAEs, mass-mean directions, the run report) from the vault
      checkpoint Cell 3 wrote with save_probe_state(..., version='probe_v1').
      Returns (env, collector, probe, experiment, report) with experiment._saes,
      probe._directions and probe._raw_directions populated (raw directions are
      rebuilt from the SAE decoders via _project_to_raw, exactly as evaluate()
      caches them) — exactly the state the 3b diagnostics,
      bias controls, transfer matrix and probe-comparison cells expect.
      Cached per (model, version) so repeated calls across cells don't re-read
      the SAEs from Drive each time.

Precondition for restore_analysis_state: Cell 3 must have run ONCE on this model
(any session) so the 'probe_v1' checkpoint exists on Drive. n_features is read
back from the checkpoint itself, so an ITERATION (8192) vs FINAL (16384) run is
handled automatically — no need to remember which produced the saved state.
"""

from __future__ import annotations

from science.layers import scan_layers_for, hook_names_for

# Heavy deps are imported lazily inside the functions so this module imports in a
# torch-free context (e.g. unit tests); the names below are the seams a test can
# monkeypatch. They are resolved for real on first call.
try:                                              # pragma: no cover - env dependent
    from science.environment import MaritimeEnvironment
    from science.collector import ActivationCollector
    from science.probe import MassMeanProbe
except Exception:                                 # torch / repo not importable yet
    MaritimeEnvironment = ActivationCollector = MassMeanProbe = None


def ensure_shared(model):
    """Cheap, idempotent rebuild of the shared objects from `model` alone.

    Returns:
        (env, SCAN_LAYERS, hook_names, collector, probe)

    `probe` is a fresh (untrained) MassMeanProbe — load directions via
    restore_analysis_state for cells that read them. Construction mirrors Cell 3
    exactly, so an analysis cell that starts with this line behaves identically
    to one run straight after Cell 3.
    """
    if MaritimeEnvironment is None:               # real deps not importable
        raise RuntimeError(
            "notebook_setup: repo modules not importable — run Cell 1a "
            "(install) and Cell 2 (which puts /content/Repo on sys.path) first."
        )
    n_layers    = int(model.cfg.n_layers)
    SCAN_LAYERS = list(scan_layers_for(n_layers))
    hook_names  = hook_names_for(n_layers)
    env         = MaritimeEnvironment(model.tokenizer)
    collector   = ActivationCollector(model, hook_names)
    probe       = MassMeanProbe()
    return env, SCAN_LAYERS, hook_names, collector, probe


# ── Diagnosis switch ─────────────────────────────────────────────────────────
# restore_analysis_state refuses probe state whose BC1 gate FAILED, because a
# failed gate means payload-construction leakage: the probe reads the adversarial
# surface rather than intent, so every AUC/direction/Δ from it is contaminated.
#
# When BC1 has failed and the state is being loaded to DIAGNOSE WHY, set this
# once instead of editing the ~15 cells that call restore_analysis_state:
#
#     import notebook_setup
#     notebook_setup.ALLOW_FAILED_GATE = True   # diagnosis mode
#
# It is a module global on purpose: turning it on is a deliberate, visible act
# that applies to the whole session, and leaving it in a saved notebook is
# legible in review. Nothing computed while it is True is reportable.
ALLOW_FAILED_GATE = False

# (id(model), version) -> (experiment, report) so repeated cell runs don't
# re-read the SAE checkpoint from Drive each time.
_STATE_CACHE: dict = {}


def _infer_n_features(CFG, version: str) -> int:
    """Read dictionary size back from the saved SAE checkpoint, so the caller
    need not remember whether the state came from an ITERATION (8192) or FINAL
    (16384) run. encoder.weight is [n_features, d_model]."""
    import torch
    sae_dir = CFG.checkpoint_dir / version / "saes"
    files = sorted(sae_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(
            f"No saved SAEs at {sae_dir}. Run Cell 3 once (it calls "
            f"save_probe_state(..., version='{version}')) before any cell that "
            f"needs the trained probe, or restore a checkpoint to that path."
        )
    sd = torch.load(files[0], map_location="cpu", weights_only=True)
    return int(sd["encoder.weight"].shape[0])


def restore_analysis_state(
    model,
    CFG,
    *,
    version: str = "probe_v1",
    mode=None,
    force: bool = False,
    allow_failed_gate: bool | None = None,
):
    """ensure_shared + lazy load of the trained probe/SAEs/report from the vault.

    Returns:
        (env, collector, probe, experiment, report)
        with experiment._saes, probe._directions and probe._raw_directions
        populated (raw directions rebuilt via _project_to_raw, matching
        evaluate()'s own caching).

    Mirrors the "Cell 4: Resume after session drop" logic so any heavy cell can
    self-restore instead of depending on Cell 3 still being in memory. Cached per
    (model, version); pass force=True to reload.

    GATE ENFORCEMENT (allow_failed_gate). A checkpoint saved with
    save_probe_state(..., gate_status=bc1_result) carries BC1's verdict in
    gate_status.json. If that verdict is FAIL, this refuses to hand the state to
    an analysis cell: BC1 failing means payload-construction leakage was detected,
    so every AUC, direction and Δ computed from this probe is contaminated by
    construction rather than measuring intent -- and none of the ~15 cells calling
    this function would otherwise know.

    Previously the protection was accidental: BC1 raised, Cell 3 aborted before
    the save, and downstream cells failed on a missing checkpoint. That coupled
    the gate to EXECUTION ORDER and threw away the state you most need in order to
    diagnose the failure. The verdict now travels with the checkpoint instead.

    allow_failed_gate=True loads a failed-gate state deliberately -- for
    diagnosis. Nothing computed from it may support a claim; that is the whole
    point of the gate. None (the default) falls back to the module-level
    ALLOW_FAILED_GATE, so a diagnosis session is one assignment rather than an
    edit to every calling cell.
    """
    if allow_failed_gate is None:
        allow_failed_gate = ALLOW_FAILED_GATE
    key = (id(model), version)
    if not force and key in _STATE_CACHE:
        env, collector, probe, experiment, report = _STATE_CACHE[key]
        return env, collector, probe, experiment, report

    import json
    from evidence_platform.vault import load_probe_state, probe_state_gate
    from science.experiment import ProbeExperiment, RunMode
    from science.patching import CausalPatcher

    gate = probe_state_gate(CFG, version)
    if gate is None:
        print(
            f"NOTE: checkpoint '{version}' carries no gate_status.json, so the "
            "BC1 verdict behind it is unknown. Re-run Cell 3 with "
            "save_probe_state(..., gate_status=bc1_result) to attribute it."
        )
    elif not gate.get("passed"):
        if not allow_failed_gate:
            raise RuntimeError(
                f"Probe state '{version}' was saved with a FAILED gate "
                f"({gate.get('scope')}):\n"
                f"  {gate.get('message')}\n"
                f"  worst_auc={gate.get('worst_auc')} deviation="
                f"{gate.get('deviation')} tolerance={gate.get('tolerance')}\n\n"
                "BC1 failing means payload-construction leakage was detected: the "
                "probe separates null pairs whose content is benign, so it is "
                "reading the adversarial SURFACE, not intent. Anything computed "
                "from this state is contaminated by construction.\n\n"
                "Pass allow_failed_gate=True to load it FOR DIAGNOSIS ONLY. No "
                "claim may rest on it until the generator is redesigned and the "
                "null control passes."
            )
        print(
            f"WARNING: loading '{version}' despite a FAILED gate "
            f"({gate.get('scope')}, worst_auc={gate.get('worst_auc')}). "
            "Diagnosis only — no result from this state is reportable."
        )

    env, _scan, _hooks, collector, probe = ensure_shared(model)

    n_features = _infer_n_features(CFG, version)
    saes, directions = load_probe_state(
        CFG, version=version, d_model=int(model.cfg.d_model), n_features=n_features
    )
    probe._directions = directions
    # Rehydrate raw-space directions alongside the SAE-space ones. evaluate()
    # writes _directions and _raw_directions in the same loop iteration
    # (probe.py: _raw_directions[layer] = _project_to_raw(sae, fit.unit_dir)),
    # so a live-evaluated probe always has both; only restored sessions lacked
    # _raw_directions, and raw_direction_for() raised on them (Cells 3b-1b,
    # 3b-1f). Rebuilding via the same projection is numerically identical to
    # what evaluate() would have cached. Strict indexing (saes[l]) is
    # deliberate: a direction layer with no saved SAE would mean a corrupt or
    # hand-edited checkpoint, and a loud KeyError beats a silent wrong-space
    # direction.
    from science.probe import _project_to_raw
    probe._raw_directions = {
        l: _project_to_raw(saes[l], d) for l, d in directions.items()
    }

    experiment = ProbeExperiment(
        model, CFG, env, collector, probe, CausalPatcher(model),
        mode=mode or RunMode.FINAL, sae_l1_coeff=2e-4,
    )
    experiment._saes = saes

    report_path = CFG.base_dir / "maritime_probe_results.jsonl"
    if report_path.exists():
        report = json.loads(report_path.read_text())
    else:
        # No exported report (e.g. only a checkpoint was restored). Fall back to
        # a minimal report so best_layer is available; cells needing more should
        # re-run Cell 3's export.
        report = {"best_layer": next(iter(saes))}

    _STATE_CACHE[key] = (env, collector, probe, experiment, report)
    return env, collector, probe, experiment, report
