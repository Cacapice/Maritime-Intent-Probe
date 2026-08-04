# Commit notes

5 repositories, each independently committable. All suites green against the
split evidence_platform.

| Repository | Tests | Platform dependency |
|---|---|---|
| `high-trust-evidence` | 14 passed | n/a, it is the platform |
| `Inferential-Fidelity-Framework` | 208 passed | `>=1.4,<2` |
| `Benchmark-Stewardship_NOHARM_MAST` | 155 passed | `>=1.4,<2` |
| `Bayesian-Inferential-Fidelity` | 67 passed | `>=1.4,<2` |
| `Maritime-Intent-Probe` | 87 passed, 8 skipped | `>=1.4,<2` |

Each research suite emits exactly 1 `DeprecationWarning`, from the
`qualification_contract` compatibility alias. That warning is the migration
signal and clears when a repository updates its import.

## A correction to my own earlier finding

While merging I reported that v1.3.1 had regressed on the qualification algebra,
because all 4 adapters passed a literal `publication_status` rather than calling
`derive_publication_status`.

**That was wrong, and I withdraw it.** v1.3 replaced derivation with a different
mechanism, and the replacement is stronger in the dimension that matters most.

`AdapterPolicy` and `assert_adapter_result` enforce **adapter obligations**: the
required native fields must be present, a minimum number of structured
qualifications must be emitted, a qualification whose predicate fires must
actually appear, and a requirement may cap `maximum_publication_status`. An
adapter that under-qualifies its evidence fails closed.

Pure derivation cannot do that. Deriving status from whatever qualifications an
adapter chose to emit is silent when the adapter omits one it owed. Assert-then-
verify catches exactly that omission, which is the more likely failure in
practice: an adapter is far more likely to forget a qualification than to
mis-fold the ones it remembered.

`derive_publication_status` remains, and is called by `ScientificResult.apply`,
so the monotone transition path still runs through the algebra.

**The residual, stated plainly.** Status is now a literal checked against a
partial specification rather than a value computed from a total one. A condition
not covered by any `AdapterRequirement`, or a requirement with no
`maximum_publication_status`, leaves the literal unchecked. Closing that would
mean deriving in `__post_init__` and treating the adapter's literal as an
assertion to verify against the derivation. That is a small change and it would
make the 2 mechanisms complementary rather than alternative.

I reverted my own overwrites. The adapters ship as v1.3.1 wrote them.

## What changed in this integration

**All 4 research repositories**

- `README.md`: **Shared evidence platform** section naming the dependency, the
  specific adapter, and the rule that adapters stay in the research repositories
- `README.md`: **Repository metadata** section carrying the proposed GitHub
  description and topics, so the project page and the README cannot drift
- `pyproject.toml`: dependency moved from `qualification-contract` to
  `high-trust-evidence>=1.4,<2`
- `scientific_contract.py` and `sovereign_architecture.py`: imports repointed at
  the renamed platform
- `test_portfolio_conformance.py`: reformulated so it does not depend on the
  upstream package name, since the previous version asserted
  `__module__.startswith("qualification_contract")` and would have failed
  misleadingly after the rename
- `.gitignore` added where missing

**Inferential Fidelity Framework**

- README release corrected from v1.2.2 to v1.3.0, matching the package

**Maritime Intent Probe**

- `pyproject.toml` created. It previously had none, so its dependency on the
  platform was undeclared and it would have broken on a clean checkout. The
  heavy research stack stays in `requirements.txt`, which pins exact versions
  for session reproducibility; only the platform is declared as a resolvable
  dependency, under a `research` extra for the rest.

**high-trust-evidence, new**

- `src/high_trust_evidence/`: contract, sovereign, policy, figures
- `src/qualification_contract/`: deprecation alias, so the 4 dependents migrate
  on their own schedule
- `tests/`: 14 tests covering release policy, minimization metrics, disclosure
  accounting and the alias
- `tools/`: release signer, portfolio verifier, cross-repository compatibility
  verifier
- `docs/`: algebra, adapter discipline, policy migration, architecture
- LICENSE, `.gitignore`, CI across Python 3.10 to 3.13

## Adapters and obligation policies

The platform is generic. The obligations are domain-specific, and each policy is
written by the repository that knows what its own evidence owes.

| Repository | Adapter | Obligation policy | Min. qualifications | Requirements |
|---|---|---|---:|---:|
| Inferential Fidelity Framework | `from_restricted_modulus_result` | `IFF_ADAPTER_POLICY` | 1 | 3 |
| Benchmark Stewardship | `from_evidence_profile` | `BENCHMARK_ADAPTER_POLICY` | 2 | 3 |
| Bayesian Inferential Fidelity | `from_fidelity_summary` | `BAYESIAN_ADAPTER_POLICY` | 1 | 3 |
| Maritime Intent Probe | `from_bc1_report` | `MARITIME_PHASE1_ADAPTER_POLICY` | 2 | 3 |

Maritime's policy is the most constrained, which is correct for a Phase 1
repository. Beyond the shared shape it declares
`allowed_research_phases=("phase_1_construct_validity_diagnostic",)` and
`prohibited_supported_claim_fragments` covering "deployable monitor",
"validated intent detector", "operational intent monitor" and
"completed validation". A BC1 identifiability failure carries a `blocked`
ceiling rather than a weakening, so a failed gate cannot be qualified around.

## Before pushing

1. **Push the Bayesian LICENSE.** It exists locally and is absent from the
   published repository, while the published `pyproject.toml` declares AGPL. The
   repository is currently all rights reserved and says otherwise.
2. **Create `high-trust-evidence` first.** All 4 research READMEs link to it.
3. **Apply the descriptions and topics** from each README's metadata section.
   Only Maritime currently has topics.

---

## Follow-up: the roadmap item was mostly already built

Checking whether the proposed derive-and-verify pipeline would pass on the
shipped adapters turned up something better than expected, and 1 thing worse.

**Derivation already exists.** `ScientificResult.__post_init__` computes
`publication_status` from the base status and the qualifications, with the
comment "Status is derived, never asserted independently." So the pipeline

    adapter -> native evidence -> required qualifications -> adapter assertion
    -> derived publication status

is implemented up to the last arrow. An adapter cannot assert a status the
algebra does not support, because whatever it passes is replaced.

**Verification did not.** The adapter's `publication_status` argument was
*silently discarded*. The Bayesian adapter writes `publication_status =
"publishable"` and the constructed result carries `"qualified"`, with no signal
that the stated value was overridden.

That is the opposite of the gap described. Status was not being accepted as an
adapter literal; the literal was being thrown away without comment. Silent
agreement and silent disagreement were indistinguishable, so an adapter could
drift from its own stated intent and nothing would notice.

**Now verified.** The argument is treated as an *expectation*. Where it differs
from the derivation, the derivation still wins and the disagreement is recorded
on the result as `status_expectation_mismatch` and serialized into the payload.

An adapter passing the base as its status, which is what all 4 currently do, is
read as "no expectation" and is not flagged. Only a genuine disagreement is:

| Expectation | Base | Qualification | Derived | Flagged |
|---|---|---|---|---|
| publishable | publishable | weaken | qualified | no, placeholder |
| exploratory | publishable | weaken | qualified | yes |
| publishable | qualified | weaken | exploratory | yes |
| qualified | publishable | block | blocked | yes |

8 tests in `tests/test_status_expectation.py`, including a regression guard
asserting that no shipped adapter currently disagrees, so any future flag is
new drift rather than a pre-existing condition.

This closes the last arrow. It is not a behaviour change: the derived status was
already authoritative, and it remains so. What changed is that a disagreement is
now visible instead of resolved in silence.


## Repository layout: science separated from infrastructure

Each research repository now groups its evidence-platform integration under
`evidence/`, leaving the root to the science.

| Repository | Root entries before | After |
|---|---:|---:|
| Bayesian Inferential Fidelity | 44 | 15 |
| Benchmark Stewardship | 42 | 15 |
| Inferential Fidelity Framework | 43 | 22 |
| Maritime Intent Probe | 82 | 53 |

Bayesian is the clearest case: 44 top-level entries, of which 4 were the
science. 23 were platform documents and 9 were contract tests. The analysis was
findable only by knowing which filenames to ignore.

`evidence/` holds `adapter.py`, `sovereign.py`, `examples/`, `tests/` and
`docs/`. In repositories shipping an installable package the modules sit inside
it, at `<package>/evidence/`, so the package stays self-contained.

### 2 things the move surfaced

**9 tests had never run.** The Inferential Fidelity Framework set
`testpaths = ["tests"]` while 2 contract test files sat at the repository root,
so they were never collected. Grouping them under `evidence/tests/` and adding
that path took the suite from 208 to 217. The tests were passing. Nobody was
running them.

**A link checker earned its place.** Moving the documents broke README links,
and IFF's own link-validation test failed immediately with the specific broken
target. Every affected link was updated. A repository without that test would
have shipped the broken links.

### On relocation-proofing

The moved tests computed paths by counting parents from `__file__`, which breaks
whenever a file moves. They now locate the repository root by marker, walking up
until `pyproject.toml` is found, so the next relocation will not repeat this.
