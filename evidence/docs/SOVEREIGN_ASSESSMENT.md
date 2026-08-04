# Assessment: Sovereign Evidence Architecture

Four repositories, a portfolio conformance layer, a shared `ScientificResult`
schema at version 1.1, and a stated architecture invariant.

## What works

**The publication bundle is real.** `publish_bundle` writes the result, both JSON
schemas, a provenance record and a manifest carrying a SHA-256 for every file.
`verify_bundle` recomputes them and returns `valid: True` with an empty error
list. This is a verifiable artifact rather than a described one.

**The Phase 1 guard is enforced, not merely declared.** `strict_publish` refuses
when a `phase_1` result carries a supported conclusion containing deployment or
completed-validation language. The maritime repository is pinned to
`phase_1_construct_validity_diagnostic` in the portfolio manifest, so the
constraint follows the repository rather than depending on an author remembering
it. This is the strongest single piece of the new layer.

**The architecture invariant is stated in the right form.** No raw record crosses
a trust boundary because an analytical service can consume it; computation moves
to the data; scientific, privacy, owner and decision authority remain separate.
That is a falsifiable claim about a system rather than a value statement.

**All 4 suites pass**, and the sovereign and contract modules are dependency-light.

## The central finding

**The qualification algebra had no call site.**

`QUALIFICATION_ALGEBRA.md` states the portfolio's shared monotonicity rule:
without `new_evidence`, applying a qualification may never increase publication
strength. `qualification_is_monotone` implements it correctly, and I verified all
4 truth-table cases behave as specified.

It was never called. The only references in the corpus were the definition and 2
assertions in a test file. No publication path consulted it.

Worse, the type it operates on could not reach a bundle. `ScientificResult`
declared `qualifications: Sequence[str]`, so a `Qualification` object attached to
a result raised `TypeError: Object of type Qualification is not JSON
serializable` on publish. The rule governed a class that could not be published,
through a function nothing called.

There was also no state transition. `publication_status` was set at construction
and never changed, so there was no operation for a monotonicity rule to constrain.

## What was changed

**Qualifications are structured.** Bare strings are still accepted and are coerced
to `Qualification(effect="weaken", rationale="declared as free text")`, so every
qualification now carries an effect the rule can act on. They serialize as objects
rather than crashing.

**`ScientificResult.apply(qualification, new_status)` is the call site.** It
consults `qualification_is_monotone` and raises on a non-monotone transition. A
`weaken` from qualified to publishable is refused; a `new_evidence` from blocked
to publishable is allowed.

**A blocking qualification cannot accompany a publishable status.** Constructing a
result with `effect="block"` and `publication_status="publishable"` now raises,
which closes the obvious route around the rule.

**9 tests per repository**, including negative controls for silent strengthening,
for `preserve` used as a strengthening effect, and for the block/publishable
contradiction.

## The structural problem underneath

`scientific_contract.py` is 344 lines duplicated across all 4 repositories, and
`sovereign_architecture.py` is 252 lines duplicated the same way. Before this
patch the 4 copies already differed: the maritime copy carries `from_bc1_report`
where the others carry `from_restricted_modulus_result`.

Divergence has therefore already begun, on the module that defines the shared
schema. A portfolio that pins 4 repositories to schema 1.1 cannot verify that
claim while each holds its own copy of the schema.

`portfolio/verify_portfolio.py` asserts that the manifest says 1.1. It does not
open any repository, so it would pass unchanged if all 4 copies drifted to
different schemas tomorrow.

The remedy is a single installable package, `qualification-contract`, with the 4
repositories depending on it and holding only their own adapter functions. Until
then, a conformance test should hash the shared modules across repositories and
fail on divergence, which is 10 lines and would have caught the existing drift.

## Smaller observations

- `HIGH_TRUST_PRIVACY_MODEL.md` and the architecture documents are declarative
  only. No code enforces aggregate-only release, direct-identifier exclusion, or
  residual disclosure-risk declaration. Given the pattern above, these should be
  assumed unenforced until a call site exists.
- `verify_portfolio.py` checks 2 manifest fields and prints `"repositories": 4`
  as a literal rather than counting them. It would report 4 if the manifest
  listed 1.
- `research_phase` defaults to `mature_method`. A default that grants the
  strongest phase is the wrong direction for a schema whose purpose is
  qualification before interpretation; `exploratory` would be the safer default.
