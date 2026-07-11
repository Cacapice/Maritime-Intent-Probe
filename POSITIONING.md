# Positioning — Maritime Intent Probe

> **Document status.** This file situates the project relative to the 2026
> research and deployment landscape. It is interpretive, not part of the
> scientific record: all factual claims about the study defer to
> [`README.md`](README.md), the preregistration
> ([osf.io/xuq5v](https://osf.io/xuq5v/overview)), and
> [`FELLOWSHIP_README.md`](FELLOWSHIP_README.md), which is preserved as an
> unmodified archival record. Nothing in this file modifies or reinterprets
> any locked hypothesis, gate, or amendment. Claims about work outside this
> project are made at the level of structure or logic, not empirical
> generalization, and are marked as such. Landscape statements are current
> as of **2026-07-10** and should be re-verified before reuse — in
> particular the adjacent-work assessment, which is based on public project
> titles and descriptions and will change as new cohorts of funded work are
> announced.

## The 2026 agentic-security context

The security conversation around language models has shifted from harmful
text generation to harmful *actions*: jailbreaks and indirect prompt
injection against agents that execute code, call APIs, and route real-world
operations (OWASP
Top 10 for Agentic Applications 2026, released Dec 2025; *Careful Adoption
of Agentic AI Services*, joint NSA/CISA/ACSC/NCSC international guidance,
Apr 2026). Input-side defenses face a large and evolving
attack space, and monitoring of model internals has been proposed as a
complementary runtime safeguard: activation probes for deception and harmful
intent report strong held-out performance in recent work (e.g., MacDiarmid
et al., 2024; Goldowsky-Dill et al., 2025), and probe-based deception
detection is the focus of dedicated research funding programs as of 2026.

This study documents a failure mode directly relevant to that proposal. A
probing pipeline produced robust, depth-stable, held-out geometric
separation — passing permutation nulls, random-direction baselines, and
raw-residual/SAE cross-basis checks — under a stimulus design in which the
intended construct is unidentifiable by any estimator (**BC1**). The study
makes no claim about how common this failure mode is in other datasets or in
deployed systems; it establishes that the failure mode exists and that the
standard categories of statistical evidence can all pass without the
semantic conclusion following. The practical consequence is a conditional
one: to the extent internals-based monitors are considered for agentic
deployments, the construct validity of a probe's training contrast is worth
verifying before performance metrics are interpreted as evidence of intent
detection. The Construct-Validity Gate (**BC1–BC11**) was developed for this
study's setting and may be adaptable to that use; it has not been validated
outside this setting. The Orthogonal Intent / Action-Space design (**V2**)
proposes one route to training contrasts in which intent is operationalized
as downstream policy — the variable targeted by goal-hijacking and
indirect-injection attacks — rather than surface form.

## Bearing on probe-based deception detection

Probe-based detection of deceptive model behavior — the question of whether
interpretability methods can detect when a model's output contradicts what
it internally represents to be true, or to be its intended action — relies
on supervised training contrasts, most commonly instructed-honest versus
instructed-deceptive prompts. Three outcomes of the present study bear on
that question as it is currently pursued, each at the level of structure or
existence rather than empirical generalization:

- **The BC1 confound structure recurs in instructed-deception contrasts.**
  BC1 identified a design in which the labeled construct and a surface
  manipulation co-vary in every training pair while the downstream quantity
  of interest is never independently evidenced. An instructed-deception
  contrast has the same formal structure: the instruction is a surface
  feature that co-varies exactly with the label, and the model's internal
  state — the construct the probe is meant to read — is assumed rather than
  measured. The BC1 identifiability analysis therefore applies to such
  contrasts as-is. Whether any particular deception dataset passes it is an
  empirical question this study does not answer; existing work partially
  addresses the concern by evaluating probes on scenarios where deception
  arises without instruction, but identifiability of the *training* contrast
  is a distinct question from off-distribution evaluation, and it is the
  former that BC1 targets.
- **Standard statistical evidence does not discharge the identifiability
  requirement.** The exploratory geometry shows held-out probe AUC well
  above a permutation null, stable across depth, and robust across
  random-direction baselines and across raw-residual and SAE bases — all
  under a design in which the construct is unidentifiable in principle.
  These are the same categories of evidence customarily offered for
  deception probes. The result is an existence proof that every such check
  can pass without the semantic conclusion following.
- **Mechanical confound removal is partial.** NFKC normalization collapses
  four orthographic surface families to chance while leaving the remaining
  families confounded per BC1. By analogy — offered as a hypothesis, not a
  finding — paraphrase and style controls in deception datasets may remove
  some surface confounds without addressing the instruction–label confound
  itself.

Framed this way, the V2 remedy takes a specific form for the deception
question: operationalizing deception as an independently evidenced mismatch
between the model's internal representation and its output, rather than as
the presence of a deception instruction.

## Position within the science of evaluations

These outcomes locate the study within the science of evaluations rather
than within any single detection application. An internals-based evaluation
is decision-relevant only to the extent that its construct validity is
established: a probe metric computed over a confounded contrast measures the
contrast, not the deployment-relevant variable. Construct validity is in
turn upstream of predictive validity — an instrument cannot be expected to
predict behavior it does not measure, such as probe performance under novel
attack encodings or un-instructed deception — and this study makes no
predictive-validity claims; the proposed identifiable designs (**V1–V3**)
are what would make such claims testable. The same requirement governs
intervention: a claim that a training- or representation-level intervention
has changed what a system has learned, rather than what it says, can be
verified only by a measurement instrument whose construct is identifiable.
The present study contributes the measurement half of that pairing — an
identifiability gate (**BC1–BC11**) and an existence proof that standard
statistical evidence does not substitute for it — while **E7** (conditional
on compute) asks the adjacent representational question: whether the
observed geometry is causally load-bearing for downstream computation or
representationally present but behaviorally inert.

The study's procedural discipline serves the same aim. Locked hypotheses, a
formal amendment trail, and reporting of all prespecified analyses
regardless of outcome operationalize an explicit evidence standard for when
results do — and, as here, do not — justify semantic interpretation. The
archival preservation of the original confirmatory plan in
[`FELLOWSHIP_README.md`](FELLOWSHIP_README.md), with post-hoc changes
confined to marked status annotations, is part of that standard rather than
an incidental documentation choice.

## Relation to adjacent research directions

Contemporary evaluation-science and safety research includes several
directions adjacent to this project: measurement-theoretic construction and
validation of behavioral benchmarks; analyses of how contamination and
generalization confound what evaluation scores mean; behavioral elicitation
and testing of deceptive model behavior; and uses of interpretability as an
instrument for other validity problems (for example, detecting test-set
contamination from model internals). This project occupies a cell those
directions leave open: it applies measurement theory — construct validity
and identifiability — to internals-based instruments themselves. Benchmark
validity work asks whether behavioral tests measure what they claim; this
project asks the same question of probes.

Three qualifications bound that claim. First, adjacency judgments based on
public project descriptions are provisional; scope not visible in a title or
abstract may overlap more than it appears. Second, the underlying concern is
not novel in scattered form — awareness of surface confounds is part of why
existing deception-probe work evaluates off-distribution — so the
contribution claimed here is the *package*: a preregistered identifiability
gate, an existence proof that standard statistical checks do not substitute
for it, and identifiable designs as the remedy. Third, this landscape is
moving; the positioning above describes it as of mid-2026 and should be
expected to date.

## What this document does not claim

- No predictive-validity claims: the study does not demonstrate that any
  probe predicts behavior under novel attacks, un-instructed deception, or
  deployment conditions.
- No empirical claims about other datasets: whether any particular deception
  or intent dataset exhibits the BC1 confound is untested here.
- No deployment claims: the CVG has not been validated as an audit
  instrument for production monitoring systems.
- No semantic claims about adversarial intent from the present stimuli:
  per **BC1** and Amendment 1, all such claims are reserved for the
  identifiable designs of Phase II.

## Cross-references

- Scientific summary and current status — [`README.md`](README.md)
- Archival confirmatory record and current analytical plan —
  [`FELLOWSHIP_README.md`](FELLOWSHIP_README.md)
- Preregistration — [osf.io/xuq5v](https://osf.io/xuq5v/overview) ·
  DOI [10.17605/OSF.IO/XUQ5V](https://doi.org/10.17605/OSF.IO/XUQ5V)
- Amendment 1 (current analytical plan) —
  [osf.io/pnaxk](https://osf.io/pnaxk/overview)
