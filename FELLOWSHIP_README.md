# Maritime Intent Probe

> **Document status — read this first (updated per OSF Amendment 1, 2026).**
> This document is the original fellowship proposal and preregistration
> narrative for the confirmatory program. During implementation, bias control
> **BC1 established a construct-validity failure**: adversarial intent and
> surface form are perfectly confounded in the stimulus design, so the
> preregistered semantic question is not answerable with these stimuli.
> The preregistered hypotheses **H1–H5 remain locked, unchanged, and
> unevaluated**. The confirmatory plan below — success criteria, thresholds,
> transfer matrix, and timeline — is preserved **unmodified as the archival
> record** and is *not* being executed as written. The current analytical
> plan is **[OSF Amendment 1](https://osf.io/pnaxk/overview)**, summarized in
> the [Amendment 1 section](#amendment-1-exploratory-analytical-plan-current-status)
> near the end of this document. Status annotations below are marked
> **“Status note (Amendment 1)”** and are the only post-hoc edits.

Empirical scaffold for investigating whether internal mechanistic probing can
detect adversarial routing intent in an autonomous maritime logistics agent
**prior to system execution**, and whether that detection generalises across
surface-level attack strategies (the cross-attack-class transfer experiment).

## Research question

> To what extent can Reconstruction Attacks successfully bypass gateway-level
> safety filters to manipulate autonomous logistics routing agents in
> high-stakes maritime environments, and can internal mechanistic probing
> identify this malicious intent prior to system execution?

> **Status note (Amendment 1):** the mechanistic-probe half of this question
> was blocked at construct validity by BC1. No result from the current
> stimuli is interpreted as evidence regarding adversarial intent.

The question has two testable halves. This project addresses both in sequence:
the gateway bypass rate (month 1) and the mechanistic probe (months 2–3),
with the probe comparison and transfer experiment as the primary contribution.
Month 3 incorporates a targeted scale probe on Pythia-6.9B alongside the
transfer matrix, making the four-month output a scale-comparative finding
rather than a single-model result. Month 4 is reserved for interpretation
structured around that comparison and writeup. Cross-domain extension is
future work, not a four-month deliverable.

## Experimental scope

**Models** — Pythia-1.4B (EleutherAI), primary model. Chosen for full
TransformerLens support, public availability, manageable size on a Colab T4
(fits in ~6GB VRAM with SAE on GPU), and an established mechanistic
interpretability literature to situate findings. Pythia-6.9B is used for a
targeted scale probe in month 3 — not a full replication, but the single
best-signal transfer combination run at 6.9B scale to test whether the probe
direction generalises geometrically across model scale. **The 6.9B scale probe
requires an A100-80GB GPU instance (~26GB VRAM); this is not available on
Colab at any tier and requires a one-session cloud GPU rental (~$10–15). The
primary experiment is fully reproducible on Colab Pro; only the scale probe
requires external compute.** Kulkarni et al. (2026) find that probes trained
on one model architecture do not transfer across architectures; results from
this experiment are therefore specific to the Pythia model family and should
not be assumed to generalise to other architectures without replication.

**Dataset** — 200 matched pairs per experiment (50 for iteration, 200 for
final runs). Each pair: one legitimate routing payload, one adversarial payload
sharing the same true destination and cargo but expressed via one of the four
attack classes. Attack classes balanced: 50 pairs per class in the full dataset.

**Layer subset** — initial scan over `[4, 8, 12, 16, 20, 24]` (6 of 24
layers). Once the best layer is identified, a finer grid of ±2 layers around
it is evaluated in the final run.

**Compute environment and budget**

The fellowship provides $15,000 monthly compute budget. This project as scoped for
Pythia-1.4B does not require anywhere near that figure — the majority is held
in reserve or funds an optional scaling experiment in month 4 (see timeline).

| Phase | Runtime | Est. cost | Notes |
|-------|---------|-----------|-------|
| Pre-month 1 — infrastructure preparation | Local / free tier | $0 | Bias controls built and tested; 6.9B notebook adapted |
| Month 1 — gateway, null control, iteration runs | Colab free tier (T4) | $0 | Null dataset validation is prerequisite before month 2 |
| Month 2 — SAE fitting with bias controls, probe comparison | Colab Pro (~$12/month) | ~$12 | Bias controls run in parallel; class-specific layer selection from start |
| Month 3 — Transfer matrix + targeted 6.9B scale probe | Colab Pro + A100-80GB cloud GPU | ~$25–45 | 1.4B transfer matrix + single best-signal combination at 6.9B |
| Month 4 — Scale-comparative interpretation + writeup | Colab free tier | $0 | Structured around 1.4B vs 6.9B comparison |
| **Total** | | **~$37–57** | |
| **Remaining from $15k/month** | | **~$14,943–14,963** | Available for future work or cohort use |

Colab Pro costs approximately $12/month (billed monthly, cancellable). Subscribe
for months 2 and 3 only; free-tier T4 is sufficient for months 1 and 4. The
6.9B scale probe in month 3 requires an A100-80GB instance (~$3/hr, 3–5 hours,
~$10–15 total) — a single run on the strongest transfer signal, not a full
pipeline replication.

Drive I/O is slower than local storage — saving 6 SAE state dicts after a run
takes 2–5 minutes over a mounted Drive connection. This is expected; logging
confirms completion.

**Timeline**

| Month | Focus | Primary deliverable | Compute |
|-------|-------|---------------------|---------|
| Pre-1 | Bias control infrastructure (BC1–BC11); 6.9B notebook adaptation; synthetic validation | All eleven bias controls tested on synthetic data; 6.9B notebook confirmed runnable; BC11 permutation baseline confirmed on synthetic activations | Local / free tier |
| 1 | Environment setup, gateway filter, null dataset validation, bypass rate baseline; BC11 permutation baseline on month 1 activations | Null control pass/fail (BC1 gate) + bypass rate table across four attack classes; BC11 null P95 established before month 2 begins | Free tier |
| 2 | SAE fitting (iteration → final) with BC2–BC8 built in; probe comparison; class-specific layer selection; BC9 post-patch alignment; BC10 legitimate specificity check | Causal comparison report; best_probe_method selected; bias diagnostic values logged; BC9 alignment rate reported; BC10 pass/fail — if BC10 fails (degradation > 0.10), redesign patch scope before proceeding to month 3 | Colab Pro |
| 3 | 1.4B transfer matrix (all 4×3 combinations) with BC11 permutation null P95 reported alongside; targeted Pythia-6.9B scale probe on best-signal pair | Transfer AUC matrix with permutation null P95; 6.9B scale comparison on strongest signal; all eleven bias control results complete | Colab Pro + A100-80GB |
| 4 | Scale-comparative interpretation; BC9/BC10/BC11 result integration; SAE feature analysis if transfer strong; writeup | Draft paper or technical report framed around 1.4B vs 6.9B comparison with full eleven-control bias diagnostic section; negative transfer writeup structure executed | Free tier |

**Pre-month 1 — infrastructure preparation**

This phase is not additional work — it is work that would happen in month 2
moved earlier, where it protects the four-month timeline from the two
concentrated risks the revised design introduces.

The four bias control implementations are built and tested on a small synthetic
dataset before month 1 begins: the null dataset generator in `environment.py`,
the tokenisation normalisation pipeline, the per-class reconstruction error
logging addition to `sae.py`, and the dual mean-pooled/final-token collection
path in `collector.py`. Testing on synthetic data confirms each control
produces the expected diagnostic output without requiring a full SAE run.
If any control reveals a design problem, it is caught before the month 1 gate
rather than mid-month 2.

The Pythia-6.9B notebook adaptation is drafted in parallel: the Drive mount
and keepalive cells are verified for the A100-80GB environment, the VRAM
budget is confirmed (~26GB required, 80GB available), and a minimal test run
on a single payload pair establishes that the inference pipeline runs cleanly
before month 3 depends on it. This takes one session, not one month, and
eliminates the risk of encountering a novel compute environment for the first
time when the transfer matrix results are already in hand.

**Pre-specified fallback:** if Pythia-6.9B is unavailable, Pythia-2.8B is
the pre-specified fallback for the scale probe without requiring a registered
pre-registration amendment. This is confirmed during pre-month 1 preparation
so the month 3 timeline is not contingent on A100-80GB availability.

Null dataset validation is a prerequisite, not an option. Generate matched
pairs where both members use adversarial templates but carry legitimate labels.
Score these with a probe trained on real data; AUC must be ≈ 0.50 before
proceeding to month 2. If it is not, payload construction is leaking signal
and must be redesigned. Gateway bypass rates and threshold sensitivity sweep
(0.50–0.70 in 0.05 increments) run in parallel.

**Pre-specified dataset expansion:** if BC11 null P95 > 0.55 on the 200-pair
dataset, the dataset expands to 400 pairs (100 per class) without requiring
a registered amendment — provided expansion occurs before any primary analysis
results are examined. This is confirmed at the end of month 1 when BC11 null
P95 is first established, so the expansion decision is made before month 2
analysis begins.

**Month 2 — primary experiment with bias controls integrated**

SAE fitting and probe comparison run on Pythia-1.4B with bias controls BC2–BC8
active from the start: tokenisation normalisation comparison (normalised vs
unnormalised AUC per attack class), per-class SAE reconstruction error logging
(one addition to `sae.py`), final-token versus mean-pooled activation
collection (both collected, compared per class), and class-specific layer
selection replacing global layer selection throughout. These run in parallel
with the primary experiment and add minimal compute overhead. Results are
reported as diagnostic values in the methods section, not as findings, unless
one reveals a problem requiring redesign.

Once the best probe and best layer are selected, three additional controls run
in sequence:

**BC9 — post-patch alignment rate:** for adversarial payloads where the
directional patch produces a routing flip, cosine similarity of the post-patch
residual stream activation is measured against the legitimate class mean.
Alignment rate < 0.70 for the majority of flipped outputs is reported as a
limitation on the production intervention claim but does not stop month 3.

**BC10 — legitimate payload specificity check:** the directional patch is
applied to legitimate routing payloads. Degradation rate > 0.10 triggers the
pre-specified three-step patch scope adjustment sequence: (1) narrow to final
token position only and re-run BC10; (2) if still failing, narrow to
adversarial-signal-bearing positions and re-run BC10; (3) if neither passes,
causal patching result is reported as exploratory and the transfer matrix
proceeds without causal validation claims. Each step is logged and reported.
This sequence is pre-registered in PREREGISTRATION.md Section 8 and does not
require a registered amendment.

**Layer boundary extension:** if the best layer for any training class falls
at layer 4 or layer 24 (scan boundary), the scan extends ±4 layers without
requiring a registered amendment. Extension is logged and reported alongside
primary results.

**BC11 — permutation baseline:** the within-experiment null distribution for
transfer AUC is already established from month 1 activations. Month 2
confirms the null P95 is ≤ 0.55 before the transfer matrix begins. If null
P95 > 0.55, transfer results in month 3 are treated as exploratory.

**Month 3 — transfer matrix and targeted scale probe**

The 1.4B transfer matrix (all 4×3 attack class combinations) runs first using
class-specific layer selection. Once complete, identify the train/test pair
with the strongest transfer AUC. Run that single combination on Pythia-6.9B
using the same probe infrastructure — no new SAE fitting procedure, just the
best-signal experiment at larger scale (~$10–15, 3–5 hours). The question it
answers is whether the probe direction exists at 6.9B in a geometrically
comparable location. If yes: evidence for scale generalisation of the finding.
If no: evidence that the direction is specific to 1.4B-scale representations,
which is itself a finding about the relationship between model scale and
adversarial intent geometry.

**Month 4 — scale-comparative writeup**

The writeup is structured around the 1.4B vs 6.9B comparison rather than
treating the scale result as an appendix. Primary sections: (1) what the 1.4B
probe found across attack classes; (2) whether the same direction exists at
6.9B; (3) what the similarity or difference tells us about the geometry of
adversarial intent representation across scale. If transfer AUC was
consistently high at 1.4B, an optional SAE feature analysis identifies which
features drive the probe direction — a mechanistic interpretation that deepens
the contribution without new compute.

The bias diagnostics section covers all eleven controls with pre-registered
interpretations. BC9 alignment rate and BC10 specificity check are reported
alongside causal rate — a high causal rate with low alignment or failed
specificity check is reported as a primary limitation, not in supplementary
material. BC11 permutation null P95 is reported alongside every transfer AUC
claim.

**Negative transfer writeup structure (pre-specified):** if transfer AUC <
0.55 for the majority of combinations, the writeup is restructured around
the negative result as the primary finding — consistent with Nordby et al.
(2026) layer rotation findings extending to domain-specific adversarial
concepts in agentic systems. Section structure: (1) null dataset validation
and bypass rate results; (2) per-class probe AUC at best layer; (3) transfer
matrix with permutation null; (4) negative result interpretation against
Nordby; (5) BC9/BC10 results as secondary findings; (6) successor methodology
recommendations (multi-layer ensembling → Winninger subspace characterisation).
This structure is pre-specified to prevent the writeup being shaped by the
direction of results.

Cross-domain extension remains future work.

**Amendment protocol:** a formal amendment protocol (PREREGISTRATION.md
Section 8) distinguishes infrastructure decisions that may be adjusted
without amendment from inferential thresholds that are fully locked. All
amendments are filed on OSF before the affected analysis runs and reported
alongside primary results. The locked items — all AUC thresholds, causal
rate thresholds, probability threshold, primary model, BC1/BC9/BC10/BC11
thresholds, and hypotheses H1–H5 — are not subject to amendment without
full disclosure.

## Gateway filter definition

The research question asks whether attacks bypass "gateway-level safety
filters." To operationalise this, the project implements a simple embedding-
distance gateway in `environment.py` alongside the routing payloads.

**Implementation** — the gateway embeds each incoming turn using the model's
token embeddings (mean-pooled) and computes cosine similarity against a
reference set of known-legitimate routing instructions. Turns below a
similarity threshold (`gateway_threshold = 0.6` by default) are flagged and
blocked before reaching the model.

**Bypass rate** — for each attack class, bypass rate is the fraction of
adversarial payloads that clear the gateway (similarity ≥ threshold). A high
bypass rate confirms the attack class is a genuine evasion threat. A low
bypass rate means the gateway already catches it and the mechanistic probe is
redundant for that class — also a useful finding.

**Why this scope** — a transparent, interpretable baseline was chosen
deliberately over a fine-tuned moderation model or LiteLLM classifier. Using
an existing moderation API would make the gateway more realistic, but would
obscure *why* specific attack classes evade detection — which is precisely
what this project needs to understand. The embedding-distance gateway has
legible failure modes: you can directly observe which attack classes reduce
cosine similarity scores and by how much, making the comparison with the
mechanistic probe meaningful. Results should be read as a lower bound on what
a more capable gateway could catch, not as a claim that the embedding-distance
approach is the appropriate production defence.

This is not the only glass-box approach available. Alternatives that preserve
interpretability while being more realistic include sparse logistic regression
on TF-IDF or n-gram features (fully interpretable coefficients, more realistic
than embedding distance) and probing a frozen moderation model's internal
representations using the same linear probe infrastructure already built for
Pythia (a frozen moderation model probe would make the gateway realistic *and*
preserve interpretable failure mode analysis). The embedding-distance baseline
is chosen first because it generates a specific insight unavailable from more
complex formulations: cosine similarity in mean-pooled token embedding space
measures surface lexical overlap with legitimate instructions, so when an
attack class evades it, the reason is geometrically legible — the adversarial
payload's token distribution sits close enough to legitimate routing language
that the distance collapses. This creates a two-stage filter separating surface
representational failure (gateway) from deeper computational failure (probe).
An attack class evading both implicates the model's internal structure, not
just surface form; an attack class evading the gateway but caught by the probe
confirms the probe is detecting something the surface representation misses.

That separation directly informs any subsequent frozen moderation model
experiment: rather than a blind layer sweep, the simple baseline identifies
which attack classes carry signal at which representational depth, providing
principled layer-targeting hypotheses before the more complex experiment begins.

**Bypass rate is reported per attack class:**

| Attack class | Expected gateway behaviour | Reason |
|---|---|---|
| Fragmentation | Low bypass (caught) | Individual turns are anomalous in isolation |
| Semantic substitution | High bypass (evades) | Surface form resembles legitimate instructions |
| Encoding shift | Moderate bypass | Abbreviated form reduces similarity score |
| Contextual priming | High bypass (evades) | Early turns are genuinely benign |

These are predictions to be tested, not assumptions.

## Success criteria

> **Status note (Amendment 1):** the thresholds below are preserved as the
> preregistered record. Because BC1 rendered the construct non-identifiable,
> H1–H5 were never evaluated against them and remain locked; no threshold
> has been altered.

Results are interpreted against the following pre-registered thresholds.
Setting these before running experiments prevents post-hoc threshold
adjustment. This experimental design — including all thresholds, null control
criteria, layer selection procedure, and transfer matrix structure — is
deposited as a pre-registration prior to month 1 data collection.

> Pre-registration: Ombrellaro, K. J. (2026). OSF. https://osf.io/xuq5v
> (DOI: 10.17605/OSF.IO/XUQ5V)

**Literature basis for thresholds**

Probe AUROC scales at approximately 5% per 10× increase in parameters across
models from 0.5B to 176B (Pais et al. 2026). Probes on SAE feature activations
perform somewhat worse than probes on raw residual stream activations in
direct comparisons (Goldowsky-Dill et al. 2025) — relevant here because this
project probes SAE feature space. Detection difficulty also varies substantially
by attack type: explicit instructed deception achieves 0.89–1.00 AUROC across
model sizes, while implicit intent detection on small models falls to 0.56–0.89.
Decomposition attacks (the closest published analogue to fragmentation) achieve
only AUC 0.60 even with a purpose-built detector (Zheng et al. 2025). Thresholds
below are set per model and per attack class to reflect these distinctions.

**Probe AUC thresholds — by model**

Thresholds apply to the best single layer in SAE feature space. The project
runs on Pythia-1.4B; the Pythia-2.8B and Pythia-6.9B columns are included
as reference points if the experiment is scaled in month 4.

| Result level | Pythia-1.4B | Pythia-2.8B | Pythia-6.9B |
|---|---|---|---|
| Null (no signal) | < 0.60 | < 0.65 | < 0.70 |
| Positive (signal present) | 0.60–0.75 | 0.65–0.80 | 0.70–0.85 |
| Strong | > 0.75 | > 0.80 | > 0.85 |

**Probe AUC thresholds — by attack class (Pythia-1.4B)**

Attack classes differ in how detectable their intent is expected to be,
based on analogous tasks in the literature.

| Attack class | Expected range | Basis |
|---|---|---|
| Contextual priming | 0.70–0.85 | Closest to instructed deception — clear intent signal in final turn |
| Semantic substitution | 0.65–0.80 | Intent present but surface-camouflaged; implicit detection task |
| Encoding shift | 0.60–0.75 | Abbreviated form may reduce residual stream signal |
| Fragmentation | 0.55–0.70 | Decomposition attacks are hardest class; AUC 0.60 is realistic floor |

If fragmentation AUC falls below 0.55, this is a finding: the attack class
successfully evades both gateway and mechanistic detection at this scale.

**Directional patching causal rate thresholds**

| Result level | Pythia-1.4B | Pythia-2.8B | Pythia-6.9B |
|---|---|---|---|
| Null | < 20% | < 25% | < 30% |
| Positive | 20–35% | 25–40% | 30–50% |
| Strong | > 35% | > 40% | > 50% |

Causal rates are expected to be lower than the 77→92% flip rates Marks &
Tegmark report for LLaMA-13B on truth-value classification, which is a
simpler binary task on a much larger model.

**Probe comparison margin**

≥ 5 percentage points causal rate difference to declare a winner between
MassMeanProbe and LogisticRegressionProbe. Below 5pp is within noise for
this dataset size and should be reported as inconclusive.

**Cross-attack-class transfer (Pythia-1.4B)**

| Metric | Threshold |
|--------|-----------|
| Transfer AUC (held-out class) | ≥ 0.60 to claim shared representation |
| Transfer AUC (held-out class) | < 0.55 to claim class-specific representation |
| 0.55–0.60 | Inconclusive |

Transfer thresholds are set lower than detection thresholds because transfer
tests generalisation to unseen surface forms — a harder problem — and uses
50 pairs per class rather than 200 combined. Strong transfer is ≥ 9 of 12
train/test combinations exceeding 0.60. Partial transfer is 5–8. No transfer
is fewer than 5. The fragmentation class is expected to show the weakest
transfer, as it differs most structurally from other attack classes.

**Gateway bypass**

| Metric | Threshold |
|--------|-----------|
| Bypass rate (semantic, priming) | ≥ 60% expected — confirms evasion threat |
| Bypass rate (fragmentation) | ≤ 30% expected — gateway should catch this |
| Bypass rate (encoding shift) | 30–60% expected — intermediate case |

If bypass rates do not match predictions, the gateway threshold or
implementation should be examined before proceeding to the mechanistic probe.

**Null results are findings.** Fragmentation attacks evading both gateway and
probe at 1.4B scale motivates scaling to 2.8B. Weak transfer motivates
attack-class-specific probe training. A gateway catching all four classes
narrows the research question to a harder adversarial regime. All are
publishable contributions if framed with respect to these pre-registered
thresholds.

## Production-scale implications

This project operates at Pythia-1.4B–6.9B. Kulkarni et al. (2026) demonstrate
that an activation-level signal — adversarial restlessness, the excess path
length of residual stream activations across adversarial turns — replicates
across four model families at 24B–70B. The mechanistic findings from this
project have production-scale implications under both positive and negative
transfer results, but the strength and form of those implications differ. The
three implications below are presented with their conditionality stated
explicitly rather than collapsed into a single contingency.

---

**Implication 1 — Layer targeting for efficient monitoring**
*Robust to the transfer result. Supported regardless of whether transfer is
positive or negative.*

Monitoring every layer of a 70B+ model in production is computationally
expensive. The layer scan in this experiment answers a question that is
independent of whether the transfer matrix is positive: at which layer does
the residual stream carry the most diagnostic information about adversarial
intent? This question has a meaningful answer even if each attack class has
its own class-specific probe direction, because those directions may all
concentrate signal at the same layer or narrow layer range despite being
geometrically distinct.

If the best layer is consistent across attack classes under weak transfer, that
implies the model has a preferred computational depth at which intent-relevant
information is processed regardless of attack surface — even if the specific
subspace varies. A production monitoring architecture targeting that layer
with class-specific classifiers running in parallel is more efficient than
full-stack instrumentation and is directly supported by the layer scan result
regardless of transfer outcome.

The proportional layer mapping used in the 6.9B scale probe tests whether the
preferred depth scales predictably with model depth. If confirmed, layer
targeting can be extrapolated to production-scale models without requiring a
full layer scan at 70B+ scale — a practically significant result under either
transfer outcome.

---

**Implication 2 — Targeted intervention rather than binary detection**
*Conditional on positive transfer and geometric comparability at 6.9B.*

Kulkarni's trajectory approach produces a detection signal. A linear probe
direction in SAE feature space is a different kind of artefact: a handle on
the representation itself. Under positive transfer, directional patching
becomes a real-time non-destructive intervention — suppressing the adversarial
component along a single probe direction without altering any other aspect of
the model's output. That capability does not follow from detection performance
alone; it requires knowing that adversarial intent is encoded in a consistent
direction regardless of surface form.

Under weak or class-specific transfer, the intervention argument becomes
class-conditional: a classifier first identifies the attack class from
early-turn activations, then applies the appropriate class-specific directional
patch. More complex operationally, but still more precise than binary detection.
The maritime project's class-specific probe directions provide the patch
targets; the additional classification step is what weak transfer imposes.

---

**Implication 3 — Detection robustness under adversarial surface variation**
*Transforms rather than disappears under weak transfer. A finding exists in
either case; the form and the production recommendation differ.*

Under positive transfer: probe directions trained on one attack class detect
held-out attack classes without re-fitting. An attacker switching attack
surface remains detectable by the same probe direction — the strongest
robustness claim, directly actionable at production scale.

Under weak transfer: each attack class requires training data covering that
class. This maps directly onto Kulkarni's finding that generalisation is
source-dependent and combined multi-source training is necessary. The maritime
project provides the mechanistic explanation for why: each attack surface
activates a distinct subspace, so detection systems cannot rely on
representation overlap to generalise. That explanation strengthens the argument
for comprehensive training data coverage and distinguishes it from a data
sufficiency argument — it is a structural claim about what the model's internal
representations require, grounded in the geometry of the probe directions.

Under either outcome the finding is practically relevant. The difference is
whether the production recommendation is a single trained probe or a suite of
class-specific probes with sufficient class coverage.

---

**Relationship to Kulkarni (2026)**

The adversarial restlessness finding and the mechanistic findings from this
project are complementary rather than competing. An open question connecting
them is whether Kulkarni's trajectory signal is driven by movement along a
single linear axis in SAE feature space or through a higher-dimensional
subspace. Under positive transfer from this project, the single-axis hypothesis
is supported — all attack classes move along the same direction, producing
large path length. Under weak transfer, the trajectory signal is more likely
driven by shared layer dynamics: all attack classes produce large activation
movement at the same computational depth even though the directions differ.
That distinction matters for intervention design at production scale and is not
resolvable from Kulkarni's data alone.

A planned additional analysis projects the probe direction onto the trajectory
path length statistic for each attack class and reports the correlation. High
correlation indicates the two signals track the same underlying axis; low
correlation indicates they are independent contributions. This analysis is
inexpensive — it requires only the activations already collected — and would
directly characterise the mechanistic relationship between the two projects.

This project is upstream of the production-scale question, not a smaller-scale
version of it. A positive result provides a mechanistic hypothesis testable at
24B–70B. A negative result provides a mechanistic explanation for why
production-scale detection requires comprehensive training coverage and
class-specific monitoring architectures. Both outcomes advance the field.



1. **Null dataset validation** — before any SAE fitting or probe training,
   `environment.py` generates a null control dataset: matched pairs using
   adversarial templates but legitimate routing outcomes and labels. A probe
   trained on real data scored on this null set must return AUC ≈ 0.50.
   Failure indicates payload construction leakage and requires redesign before
   proceeding. This is the validity prerequisite for all subsequent results.

2. **Matched-pair generation** — `environment.py` produces legitimate and
   adversarial routing payloads sharing the same true routing outcome but
   expressed via four attack classes: fragmentation, semantic substitution,
   encoding shift, and contextual priming.

3. **Activation collection** — `collector.py` runs payloads through a
   HookedTransformer and captures both mean-pooled and final-token residual
   stream activations at a sparse layer subset. Both are retained for bias
   comparison in step 4.

4. **SAE decomposition with bias controls** — `sae.py` trains a sparse
   autoencoder (Bricken et al. 2023) per layer, decomposing polysemantic
   residual stream vectors into interpretable monosemantic features. Per-class
   reconstruction error is logged throughout. Tokenisation normalisation
   comparison runs in parallel on a normalised copy of the dataset.

5. **Probe comparison** — `probe.py` implements two probes with a shared
   interface: `MassMeanProbe` (Marks & Tegmark 2023, parameter-free) and
   `LogisticRegressionProbe` (fitted decision boundary, standard baseline).
   Both are trained on the same SAE-encoded activations. Layer selection is
   performed per training class, not globally, to avoid circularity in the
   transfer results. Marks & Tegmark hypothesise that the mass-mean direction
   may be more causally implicated in model outputs than a fitted boundary;
   this is tested empirically rather than assumed (see step 6).

6. **Causal comparison** — `patching.py` provides two patching methods.
   `patch_and_run()` replaces the full activation with the legitimate class
   mean (baseline causal check). `directional_patch_and_run()` applies a
   targeted patch along a specific probe direction only, leaving all other
   activation components unchanged. `compare_probes()` in `probe.py` runs
   the directional patch with both probe directions and compares causal rates
   — the fraction of adversarial routing outputs that flip. The probe with
   the higher causal rate is selected as `best_probe_method` and carried
   forward into all subsequent transfer analyses, grounding the method choice
   in an empirical result specific to this domain.

7. **Cross-attack-class transfer** — using the probe selected in step 6,
   directions trained on one attack class are scored on held-out classes
   without re-fitting. Layer selection uses the training class only. High
   transfer AUC is evidence of a shared internal representation of adversarial
   intent regardless of surface form.

8. **Targeted scale probe** — the train/test pair with the strongest transfer
   AUC from step 7 is run once on Pythia-6.9B using the same probe
   infrastructure. No new SAE fitting procedure. The question is whether the
   probe direction exists at 6.9B in a geometrically comparable location.

9. **Runtime deployment** — `monitor.py` wires the selected probe direction
   into live hook infrastructure as a non-destructive audit layer.

## Project structure

```
maritime-intent-probe/
├── README.md
├── PREREGISTRATION.md
├── OPTIMISATION_TODO.md
├── LICENSE            AGPL-3.0 (current); LICENSE-MIT retains prior terms
├── requirements.txt
├── config.py          VaultConfig — all file paths and run metadata
├── vault.py           Integrity verification, model loading, checkpointing,
│                      probe state save/restore (session-drop resilience)
├── environment.py     Routing vocabulary, RoutingPayload, MaritimeEnvironment
│                      (multi-template matched-pair + null-control generators),
│                      EmbeddingGateway (H1) — swap this file for a new domain
├── collector.py       ActivationCollector — hook-based activation capture
├── sae.py             SparseAutoencoder — Bricken et al. architecture,
│                      per-class reconstruction MSE logging (BC4)
├── probe.py           MassMeanProbe, LogisticRegressionProbe (dual direction
│                      spaces: SAE for scoring, decoder-projected raw for
│                      patching; cross-validated AUC), compare_probes(),
│                      compare_probes_multiseed() (BC6),
│                      permutation_baseline_auc() (transfer-matched null, BC11)
├── patching.py        CausalPatcher — full-mean, directional, final-token-only,
│                      and position-restricted patching; per-position causal
│                      rates; post_patch_alignment_rate() (BC9),
│                      legitimate_specificity_check() (BC10)
├── experiment.py      ProbeExperiment — pipeline orchestrator, RunMode,
│                      refine_layers() (±2 finer grid, §3.3)
├── transfer.py        run_transfer_matrix() — 4×3 transfer matrix with BC5/BC7
├── bias_controls.py   BC2–BC11 consolidated runner, individual controls,
│                      lr_threshold_sensitivity() (§4.16),
│                      bc10_patch_scope_fallback() (§6 sequence)
├── null_validator.py  validate_null_control(), quick_null_check() — BC1 gate
├── scale_probe.py     check_vram(), run_scale_probe() — 6.9B / 2.8B fallback
├── monitor.py         DualLayerAuditMonitor — runtime deployment
├── smoke_test.py      End-to-end pipeline smoke test on a tiny random model
├── maritime_intent_probe.ipynb        Primary experiment (Cells 0–6 incl. 2b
│                                      gateway, 3b refinement, 4d BC10 fallback)
└── maritime_intent_probe_scale.ipynb  Scale probe — Pythia-6.9B / 2.8B fallback
```

## Notebook cell guide

### `maritime_intent_probe.ipynb` — primary experiment

| Cell | Purpose |
|------|---------|
| 0 | Keepalive + runtime check — VRAM detection, JS keepalive |
| 1 | Drive mount, dependency install, W&B login, quick null check (BC1 lightweight gate) |
| 1b | BC11 permutation baseline — establishes null P95 and triggers dataset expansion if > 0.55 |
| 2 | Config, device detection, model load (integrity verification) |
| 2b | H1 — embedding-distance gateway bypass rates per attack class (`EmbeddingGateway.bypass_rates()`); writes `gateway_bypass_rates.json` |
| 3 | SAE fitting, both probes trained, full BC1 null validation, report generated |
| 3b | Finer-grid layer refinement (±2, §3.3) via `experiment.refine_layers()` — FINAL runs; refined layer carried forward |
| 4 | Resume after session drop — restores SAE weights and probe directions without refitting |
| 4b | Probe comparison + BC6 direction stability check — selects `best_probe` |
| 4c | Gap closure controls — BC9 post-patch alignment, BC10 legitimate specificity, BC11 permutation baseline |
| 4d | BC10 patch-scope fallback sequence (§6) via `bc10_patch_scope_fallback()` — run only if BC10 failed in 4c |
| 5 | Cross-attack-class transfer matrix using `best_probe` — BC5/BC7 active, null P95 reported alongside |
| 6 | Scale probe VRAM check, result inspection, runtime monitor setup |

### `maritime_intent_probe_scale.ipynb` — scale probe

| Cell | Purpose |
|------|---------|
| S0 | VRAM check — determines 6.9B or 2.8B fallback, starts keepalive |
| S1 | Drive mount + install |
| S2 | Load 1.4B results, identify best combination, load probe state |
| S3 | Load scale model (6.9B or 2.8B fallback) |
| S4 | Run scale probe — payload generation, activation collection, SAE fitting, AUC scoring |
| S5 | Inspect results + month 4 write-up framing |

## Probe comparison and method selection

Cell 4b is the mechanism by which the Marks & Tegmark hypothesis is tested
rather than assumed. It prints a result of the form:

```
Layer:                      blocks.16.hook_resid_post
Mass-mean causal rate:      65.0%
Logistic regression rate:   40.0%
Selected method:            mass_mean
```

The selected method is then used automatically in Cell 5 (transfer experiment)
and Cell 6 (monitor), so all downstream results rest on an empirically
validated choice. If logistic regression wins, that itself is a finding worth
reporting — it would suggest the geometry of adversarial intent in this domain
differs from Marks & Tegmark's truth-value datasets.

## Cross-attack-class transfer experiment

> **Status note (Amendment 1):** under BC1, any transfer measured here is
> transfer of a surface-form axis, not of an intent representation; the
> confirmatory transfer matrix was not run as a hypothesis test.

The core fellowship contribution. Cell 5 runs this automatically using
`best_probe`, but the logic can also be run standalone:

```python
# Train on fragmentation only
frag_payloads = [p for p in all_payloads
                 if p.label == 0 or p.attack_class == "fragmentation"]
frag_acts, frag_labels = collector.collect(frag_payloads, env)
frag_sae_acts = {layer: experiment._saes[layer].encode(X)
                 for layer, X in frag_acts.items()}
train_probe = type(best_probe)()
train_probe.evaluate(frag_sae_acts, frag_labels, turn_index=0)

# Score on semantic — direction unchanged, no re-fitting
sem_payloads = [p for p in all_payloads
                if p.label == 0 or p.attack_class == "semantic"]
sem_acts, sem_labels = collector.collect(sem_payloads, env)
sem_sae_acts = {layer: experiment._saes[layer].encode(X)
                for layer, X in sem_acts.items()}
sem_results = train_probe.evaluate(sem_sae_acts, sem_labels, turn_index=0)
# High AUC = shared internal representation across attack classes
```

The transfer matrix (all train/test class combinations) is exported to
`causal_comparison.json` and `transfer_results.json` for downstream analysis.

## Extending to a new domain

Replace `environment.py` with a new environment file (e.g.
`medical_environment.py`) exposing `RoutingPayload` and a class with
`generate_pairs()` and `tokenize()`. No other file needs to change.

To carry the causal comparison result into a cross-domain analysis, run
Cell 4b in the new domain with the same probe directions — if `best_probe_method`
differs between domains, that is a finding about how intent representation
varies across task structures.

## References

- Boxo, G., Neelappa, A., & Raval, S. (2025). Linear Probes Rely on Textual
  Evidence: Results from Leakage Mitigation Studies in Language Models.
  *arXiv:2509.21344.* <https://arxiv.org/abs/2509.21344>
- Bricken, T., et al. (2023). Towards Monosemanticity: Decomposing Language
  Models with Dictionary Learning. *Transformer Circuits Thread.*
  <https://transformer-circuits.pub/2023/monosemantic-features/index.html>
- Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits.
  *Transformer Circuits Thread.*
  <https://transformer-circuits.pub/2021/framework/index.html>
- Kulkarni, A., et al. (2026). Latent Adversarial Detection: Adaptive Probing
  of LLM Activations for Multi-Turn Attack Detection. *arXiv:2604.28129.*
  <https://arxiv.org/abs/2604.28129>
- Marks, S., & Tegmark, M. (2023). The Geometry of Truth: Emergent Linear
  Structure in LLM Representations of True/False Datasets. *arXiv:2310.06824.*
  <https://arxiv.org/abs/2310.06824>
- Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and Editing
  Factual Associations in GPT. *Advances in Neural Information Processing
  Systems 35 (NeurIPS 2022).*
  <https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html>
- Rajamanoharan, S., Conmy, A., Smith, L., Lieberum, T., Varma, V., Kramár, J.,
  Shah, R., & Nanda, N. (2024). Improving Dictionary Learning with Gated Sparse
  Autoencoders. *arXiv:2404.16014.* <https://arxiv.org/abs/2404.16014>
- Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of
  Rare Words with Subword Units. *Proceedings of the 54th Annual Meeting of the
  Association for Computational Linguistics*, pp. 1715–1725.
  <https://aclanthology.org/P16-1162/>
- Smith, L., Rajamanoharan, S., Conmy, A., McDougall, C., Kramar, J.,
  Lieberum, T., Shah, R., & Nanda, N. (2025). Negative Results for Sparse
  Autoencoders on Downstream Tasks and Deprioritising SAE Research.
  *DeepMind Safety Research.*
  <https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9>
- Templeton, A., et al. (2024). Scaling Monosemanticity: Extracting
  Interpretable Features from Claude 3 Sonnet. *Transformer Circuits Thread.*
  <https://transformer-circuits.pub/2024/scaling-monosemanticity/>
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
  Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in
  Neural Information Processing Systems 30 (NeurIPS 2017).*
  <https://arxiv.org/abs/1706.03762>

## Bias analysis and experimental controls

The following methodological rationale identifies the sources of bias most
likely to produce false positives or inflated results in this experimental
design, grounded in mechanistic analysis of how Pythia processes token
sequences and in the probing literature. Each source is paired with the
control that addresses it. Controls are listed in implementation priority order;
the null dataset validation is a prerequisite for interpreting any other result.

---

**1. Payload construction leakage (highest priority)**

The most consequential threat to validity is unintended surface regularities
in matched-pair construction. Boxo et al. (2025) demonstrate systematically
that linear probes trained on model activations frequently exploit textual
evidence — surface features of the prompt rather than internal representations
of the target behaviour — producing high AUC that does not reflect detection
of the intended construct. If the matched-pair generator in `environment.py`
produces adversarial payloads that differ from legitimate payloads in
sentence-initial token distributions, routing vocabulary frequencies, cargo
type patterns, or punctuation, the residual stream will encode those
regularities from the earliest layers (Elhage et al. 2021) — prior to any
semantic processing — and the probe will learn to classify on surface form
rather than adversarial intent. The causal patching rate will be elevated,
and the transfer matrix will appear to show generalisation, all without
detecting intent.

**Control:** generate a null dataset where both members of each matched pair
are constructed using adversarial templates but carry legitimate routing
outcomes and labels. A probe trained on the real dataset and scored on the
null dataset must return AUC ≈ 0.50. If it does not, payload construction
leakage is present and the experiment must be redesigned before any further
analysis proceeds.

---

**2. Tokenisation artifacts treated as classification signal**

Pythia processes token sequences produced by BPE tokenisation, not raw text.
BPE is a compression algorithm that segments text based on corpus frequency
rather than semantic boundaries (Sennrich et al. 2016); the same semantic
content expressed in abbreviated versus full form produces different token
sequences, different sequence lengths, and therefore different positional
embeddings (Vaswani et al. 2017) and different attention patterns before
any semantic layer is active. Encoding shift attacks use abbreviated and
non-standard forms by design. The residual stream representation of
"ETA 0600" and "estimated time of arrival 06:00" will differ at the token
embedding layer — the initial state of the residual stream (Elhage et al.
2021) — independently of any difference in adversarial intent. High probe
AUC on encoding shift may therefore reflect tokenisation-level signal.
Critically, this would also produce poor transfer from encoding shift to other
attack classes in the transfer matrix: the probe direction would encode a
tokenisation artifact rather than a shared representation of intent.

**Control:** run the probe on a tokenisation-normalised copy of the dataset
in which all abbreviations are expanded to full forms before tokenisation,
and compare AUC per attack class against the unnormalised condition. A
substantial AUC drop after normalisation identifies the result as
tokenisation-driven.

---

**3. Mean-pooling misalignment with sequential intent structure**

In decoder-only transformers, the final token position is the only position
that has attended over the entire prompt and therefore carries the model's
full contextual encoding of the input prior to generation (Elhage et al.
2021; Geva et al. 2022). Mean-pooling residual stream activations across the
full token sequence averages this final-position signal with earlier positions
that have not yet integrated the complete context. For contextual priming
attacks — where early turns are genuinely benign and adversarial intent is
expressed only in the final turn — mean-pooling dilutes the signal the probe
needs to detect: early benign tokens pull the mean toward legitimate activation
space, and the final adversarial turn contributes only its proportional share
of the pooled vector.

Kulkarni et al. (2026) demonstrate that mean-pooled representations miss
multi-turn adversarial signal that is recoverable from turn-level trajectory
features, directly motivating position-sensitive activation collection as a
methodological choice rather than a post-hoc adjustment. For fragmentation
attacks, the situation is structurally different: adversarial content is
distributed across turns and mean-pooling may aggregate signal that no single
turn contains. The expected pattern is therefore asymmetric: mean-pooling
disadvantages contextual priming detection and may assist fragmentation
detection. Deviations from this pattern are informative about the pooling
mechanism's interaction with attack structure.

**Control:** collect activations at both mean-pooled and final-token positions
for each payload. Compare probe AUC per attack class between the two
collection strategies. Results are reported for both; the primary analysis
uses final-token collection for contextual priming and mean-pooled for
fragmentation unless the comparison yields a different finding.

---

**4. SAE reconstruction error as an uncontrolled confound**

The SAE decomposes residual stream activations into sparse feature
representations via a learned encoder-decoder pair trained to minimise
reconstruction loss subject to a sparsity penalty (Bricken et al. 2023).
The SAE error term ε(h) — the component of the activation not captured by
the sparse reconstruction — is non-zero and varies across inputs (Rajamanoharan
et al. 2024). Smith et al. (2025) find that probes trained on SAE feature
activations perform substantially worse than probes trained on raw residual
stream activations for harmful intent detection tasks, and that reconstruction
fidelity is a key mediating factor: the SAE must reconstruct both classes
faithfully for the probe to operate on equivalent representations. If the SAE
reconstructs adversarial activations less faithfully than legitimate ones,
the probe learns a classification boundary that reflects reconstruction
quality rather than intent, and per-class AUC results are not comparable.

**Control:** log per-class mean squared reconstruction error for each attack
class during SAE fitting (one addition to `sae.py`, zero compute overhead).
If the highest-AUC attack class also shows the highest reconstruction error,
the AUC result is confounded and must be reported with that caveat.

---

**5. Layer selection circularity inflating transfer AUC**

Selecting the best probing layer using performance across all four attack
classes and then reporting transfer AUC on held-out classes at that layer
introduces circularity: the layer selection criterion includes the held-out
class, so the chosen layer is partly optimised for that class before the
transfer experiment begins. This is a form of evaluation bias documented in
the probing literature (Boxo et al. 2025; Nordby et al. 2026) in which
multiple comparisons on the same data overstate generalisation. The magnitude
of the inflation is unknown a priori but is directly testable.

**Control:** perform layer selection using only the training class for each
transfer combination. The transfer loop in Cell 5 runs four layer scans (one
per training class) rather than one global scan. The AUC difference between
global and class-specific layer selection is reported alongside the transfer
matrix as a measure of how much circularity affected the original design.

---

**6. Scale as a threat to external validity**

The internal validity controls above address whether the probe detects what
it claims to detect in Pythia-1.4B. A separate and unresolvable threat from
within the current design is whether a genuine linear direction encoding
adversarial intent at 1.4B scale exists in a geometrically comparable location
at scales relevant to deployed systems. Nordby et al. (2026) find that linear
probe accuracy scales with model size but that the best layer varies across
models and tasks, meaning a direction found at one scale cannot be assumed
to transfer across scales without empirical verification. The targeted
Pythia-6.9B scale probe in month 3 partially addresses this by testing
geometric comparability of the strongest transfer signal at larger scale,
moving the finding from a single-model result toward a scale-comparative one.

---

**7. Probe direction stability (Bias Control 6)**

The LogisticRegressionProbe direction depends on random initialisation and
may vary across seeds. `compare_probes_multiseed()` in `probe.py` runs the
comparison across three seeds (42, 123, 7). If the winner changes across any
seed, both probes are carried forward regardless of the single-seed outcome
and the instability is reported as a limitation. The MassMeanProbe direction
is deterministic given fixed activations and requires no stability check.

**Control:** run `compare_probes_multiseed()` in Cell 4b immediately after
`compare_probes()`. If `winner_consistent` is False, override `best_probe`
to `mass_mean` and report the instability explicitly.

---

**8. Within-class activation variance (Bias Control 7)**

If one attack class has systematically higher within-class variance in SAE
feature space, a probe trained on that class may learn an unstable direction
that separates held-out classes due to variance rather than shared intent
geometry. High transfer AUC from a high-variance training class is therefore
less evidential than the same AUC from a low-variance class.

`_within_class_variance()` in `transfer.py` computes per-attack-class mean
variance before the transfer matrix runs. Any class with mean per-sample
variance > 1.5× the cross-class mean is flagged. Transfer AUC from flagged
training classes is reported with a variance caveat and is not used as primary
evidence for H4. If variance correlates with transfer AUC across classes, that
correlation is reported as a confound rather than a finding.

---

*References for this section:*
Boxo, G., Neelappa, A., & Raval, S. (2025). Linear probes rely on textual
evidence: Results from leakage mitigation studies in language models.
*arXiv:2509.21344.* <https://arxiv.org/abs/2509.21344> —
Elhage, N., et al. (2021). A mathematical framework for transformer circuits.
*Transformer Circuits Thread.*
<https://transformer-circuits.pub/2021/framework/index.html> —
Geva, M., et al. (2022). Transformer feed-forward layers are key-value memories.
*EMNLP 2022.* —
Kulkarni, A., et al. (2026). Latent adversarial detection: Adaptive probing of
LLM activations for multi-turn attack detection. *arXiv:2604.28129.*
<https://arxiv.org/abs/2604.28129> —
Nordby, E., Pais, T., & Parrack, A. (2026). Linear probe accuracy scales with
model size and benefits from multi-layer ensembling. *arXiv:2604.13386.*
<https://arxiv.org/abs/2604.13386> —
Rajamanoharan, S., et al. (2024). Improving dictionary learning with gated
sparse autoencoders. *arXiv:2404.16014.* <https://arxiv.org/abs/2404.16014> —
Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of
rare words with subword units. *ACL 2016.*
<https://aclanthology.org/P16-1162/> —
Smith, L., et al. (2025). Negative results for sparse autoencoders on downstream
tasks and deprioritising SAE research. *DeepMind Safety Research.*
<https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9> —
Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 2017.*
<https://arxiv.org/abs/1706.03762>

## Amendment 1: exploratory analytical plan (current status)

Filed on OSF ([current documentation source: osf.io/pnaxk](https://osf.io/pnaxk/overview)) following the BC1 construct-validity result. It modifies no
preregistered hypotheses, decision criteria, or confirmatory analyses —
H1–H5 remain locked and unevaluated — and specifies what the project does
instead.

**Phase I — exploratory characterization (current project).** Conducted
exclusively on the existing dataset; no new data collection, no stimulus
modification, and no additional analyses after publication of the amendment
without a further amendment. The question is deliberately narrower than the
preregistered one: *what geometric properties does the current
(construct-invalid) stimulus design measure, and which mechanistic
explanations remain consistent with them?* Core completion criteria are E1
and E3–E6; E2 and E7 are conditional on compute (E2 for forward passes over
an external corpus, E7 for activation-patching campaigns across depth and,
where resources permit, both model scales).

- **E1 — LayerNorm mechanistic test.** Apply the model's native post-layer
  LayerNorm, as implemented at inference (not a reimplementation or generic
  unit-variance normalization), to residual activations; measure class
  separation before and after, held-out probe AUC, raw centroid distance,
  and normalized centroid distance (per E3). Directly evaluates the
  LayerNorm-attenuation working interpretation.
- **E2 — Ambient residual-geometry baseline** *(conditional)*. Activation
  norm, covariance anisotropy, eigenspectrum, centroid-distance growth, and
  random-direction separability on unrelated text. Compared against the
  maritime-stimulus growth slope via the E4 regression framework, with
  dataset × depth-fraction interaction as a formal test rather than a
  qualitative curve comparison. Deferred to Phase II if compute does not
  permit, without affecting the interpretation of the other Phase I
  analyses.
- **E3 — Scale-controlled geometry.** Normalized centroid distance (by
  within-class norm), Fisher ratio, and Mahalanobis distance, separating
  activation-scale growth from scale-invariant separation.
- **E4 — Formal statistical modeling.** Mixed-effects models wherever
  repeated observations violate independence (template-level dependence),
  with regression estimates and confidence intervals for probe AUC,
  centroid distance, normalized centroid distance, Fisher ratio, Mahalanobis
  distance, and random-direction separability; generalized linear or
  generalized additive mixed models substituted, and documented, where
  diagnostics indicate assumption violations.
- **E5 — Continuous geometry analyses.** Individual-observation
  relationships among representational displacement, activation norm, probe
  decision margin, random-direction separability, and null leakage, using
  mixed-effects or cluster-robust methods, in both raw residual space and
  SAE feature space.
- **E6 — Abbreviation follow-up: null leakage as a continuous endpoint.** A
  focused follow-up using the displacement–leakage relationship already
  estimated under E5, rather than re-deriving it: does abbreviation's
  observed null leakage fall within the range its displacement predicts, or
  deviate as an outlier? Determines whether abbreviation occupies the
  expected endpoint of E5's relationship rather than constituting an
  independent, unexplained exception.
- **E7 — Causal ablation of the surface-associated geometry** *(conditional
  on compute)*. E1 is analytic — it describes the effect of a
  transformation on frozen activations, not an intervention on the model's
  computation — and does not by itself establish whether the geometry
  characterized elsewhere in Phase I is causally load-bearing for
  downstream processing, as opposed to representationally present but
  behaviorally inert. Activation patching intervenes directly on residual
  activations at the layers/directions identified as separable, exchanging
  activations between paired stimuli from the two confounded BC1
  conditions, measured against matched random-direction and random-layer
  controls, and — where feasible — patching applied before versus after
  the layer's LayerNorm. Draws exclusively on the existing maritime
  stimulus set (new analysis of existing activations, not new data
  collection); where resources permit, extends the comparison across the
  1.4B model and the 6.9B replication (within-Pythia scale, distinct from
  the beyond-Pythia replication in Phase II's V3). Like every Phase I
  analysis, bears on geometry, not adversarial intent, regardless of
  outcome. Deferred to Phase II if compute does not permit, without
  affecting the interpretation of the other Phase I analyses.

**Representation and interpretation policy.** Raw residual analyses are the
primary representation; SAE analyses are corroborative in an alternative
basis, not independent validation. No Phase I result — including E7's
causal-ablation outcome — will be interpreted as evidence regarding
adversarial intent; positive findings characterize the geometry induced by
the present stimuli and its causal role in downstream computation, not the
semantic variable responsible for it. All semantic claims about adversarial
intent remain reserved for identifiable designs (Phase II). Where multiple
mechanisms remain compatible with the observations, the least committal
interpretation consistent with the evidence is favored. Failure to
discriminate among mechanisms — including a null or ambiguous E7 result —
is itself informative.

**Statistical reporting.** Effect sizes; 99% confidence intervals unless
otherwise specified; exact p-values where applicable; model specifications
and robustness analyses. Phase I is a single exploratory program: nominal
significance is descriptive, convergent findings are consistency rather than
confirmation (all analyses share one dataset), and exploratory effect sizes
are treated as potentially optimistic for prospective power calculations.
Any future preregistration derived from these findings will disclose that
its hypotheses emerged from this exploratory program.

**Phase II — future validation program (subsequent project).**
- **V1 — Pre-specified methodological positive control.** The identical
  exploratory pipeline applied to one pre-specified identifiable construct,
  specified before analysis in a future amendment or preregistration, with
  no substitution of alternate controls without documentation. Committed in
  advance: a meaningfully different geometric profile would support the
  current interpretation of BC1 as a stimulus-design property; the *same*
  profile counts as evidence **against** the current exploratory
  interpretation.
- **V2 — Orthogonal Intent / Action-Space pilot.** A new stimulus set
  explicitly crossing surface manipulation with downstream policy — the
  pilot of the identifiability framework this document proposes.
- **V3 — Cross-architecture replication.** Principal analyses on at least
  one transformer family beyond Pythia. (Within-Pythia scale comparison,
  1.4B vs. 6.9B, is addressed conditionally in Phase I by E7, not here.)

All Phase I analytical code is version controlled and released with the
corresponding project update; all prespecified analyses are reported
regardless of outcome.

## License and citation

**License — changed 2026-06-10 (not retroactive).** Code obtained under the prior
license remains available under that license to anyone who received it.

- **As of 2026-06-10:** GNU Affero General Public License v3.0 (AGPL-3.0).
  Development from this date forward is AGPL-3.0; under §13 (network use), anyone
  running a modified version as a network service must offer corresponding source
  to that service's users. SPDX: `AGPL-3.0-or-later`.
- **Through 2026-06-09:** MIT License. Versions released on or before this date
  remain available under their original MIT terms.

Full text in `LICENSE` (AGPL-3.0) and `LICENSE-MIT` (prior terms).
Copyright © 2026 Katherine J. Ombrellaro.

**Cite this repository**

- *MLA (9th ed.):* Ombrellaro, Katherine J. *Maritime Intent Probe*. 2026.
  GitHub, github.com/Cacapice/Maritime-Intent-Probe.
- *Chicago (17th ed.):* Ombrellaro, Katherine J. *Maritime Intent Probe*. GitHub
  repository. 2026. https://github.com/Cacapice/Maritime-Intent-Probe.

**Cite the pre-registration**

> Ombrellaro, K. J. (2026). *Detecting Adversarial Routing Intent via Mechanistic
> Probing: A Cross-Attack-Class Transfer and Scale Comparison Study*
> [Pre-registration]. OSF. https://doi.org/10.17605/OSF.IO/XUQ5V

**Cite Amendment 1**

> Ombrellaro, K. J. (2026). *OSF Amendment 1: Exploratory analytical plan
> following construct-validity failure (BC1)* [Pre-registration amendment].
> OSF. https://osf.io/pnaxk
