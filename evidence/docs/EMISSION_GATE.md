# Making the privacy plane ambient

## The defect

Enforcement depended on the caller. `evaluate` and `strict_release` checked
purpose, direct identifiers, aggregate-only disclosure, residual risk and
approvals, and they checked them well. They checked nothing when a caller did
not route through them.

`publish_bundle` took a `ScientificResult` and wrote it. No classification, no
decision, no envelope. Any adapter, script or demo that constructed a result
could emit it with the privacy plane untouched.

This was not hypothetical. Every repository shipped 2 example scripts side by
side, `publish_contract_example.py` and `publish_sovereign_example.py`, 1 gated
and 1 not, with nothing to say which was correct or when. That is how an
enforced rule becomes an optional one.

## The fix

**A declaration is required, and there is no safe default.** `ScientificResult`
carries `data_classification`, 1 of `unrestricted`, `owner_data` or `derived`,
and it defaults to `None`. `unrestricted` as a default would silently exempt
owner data from the privacy plane; `owner_data` would silently claim a
provenance the result may not have. Emission refuses until the caller declares.

**Emission is the choke point.** `publish_bundle` calls `_authorise_emission`
before it touches the payload:

| Classification | Requirement |
|---|---|
| undeclared | refused |
| `unrestricted` | emitted; the declaration is recorded |
| `owner_data`, `derived` | a `TrustDecision` that permits release is required |
| any, with a blocking decision | refused, with the decision's reasons |

**The bundle carries its own warrant.** The manifest records
`data_classification` and the trust decision's status, purpose and outcome, so a
reader of the bundle can see which plane it passed through without consulting
the code that produced it.

**There is no ungated example any more.** `publish_contract_example.py` now shows
both legitimate paths, and both require a declaration. The unrestricted path
declares and emits; the owner-data path builds an envelope, evaluates it for a
purpose, and passes the decision through.

## The ambient part

A gate at 1 function is still bypassable by a script that serialises a payload
itself. Each repository now carries `test_emission_gate.py`, which checks the
gate directly and then scans every non-test module for:

- a reference to `scientific_result.json` or `sovereign_evidence.json` in a
  module that does not use a gated writer;
- an AST pattern of `json.dump(x.to_dict(), fh)` or
  `path.write_text(x.to_dict())`.

Negative control: planting a `leaky_export.py` that opens
`scientific_result.json` and dumps a result into it fails both checks. Removing
it restores them.

## A correction to the check itself

The first version of the AST scan included `json.dumps` alongside `json.dump`.
It flagged 3 validation scripts that print a summary object to stdout. Printing
is not emission, and the object was not a `ScientificResult`. A check that fires
on correct code is a check that gets disabled, so `dumps` was removed and the
reasoning recorded in the docstring.

## Verification

| | |
|---|---|
| Inferential Fidelity Framework | 184 passed |
| Benchmark Stewardship | 155 passed |
| Bayesian Inferential Fidelity | 67 passed |
| Maritime Intent Probe | 86 passed, 8 skipped |

The gate also caught a pre-existing test in every repository that published
without a classification. Those tests were exercising the ungated path, and each
now declares `unrestricted` explicitly and is joined by a test asserting the gate
refuses an undeclared result.

## What remains caller-dependent

The scan is structural. A module that constructs the payload dictionary by hand,
without calling `to_dict`, and writes it under a different filename would pass.
Closing that fully would require taint tracking on the result object rather than
a pattern scan. The gate now covers the realistic paths and the documented ones;
it is not a proof.
