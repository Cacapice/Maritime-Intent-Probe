# Publication checklist

Actions that cannot be made from inside the repositories, in the order they
should be taken.

## 1. Push the Bayesian LICENSE

`Bayesian-Inferential-Fidelity` shows no license badge on GitHub, while the
other 3 show AGPL-3.0. The file exists locally, 34,523 bytes, byte-identical to
the Maritime and IFF copies. It is not on the published repository.

This matters more than a missing badge. `pyproject.toml` **is** published and
carries `License :: OSI Approved :: GNU Affero General Public License v3`, so
the repository declares AGPL while being legally all rights reserved. Nobody can
lawfully use, fork or vendor it, and the declaration says otherwise.

    git add LICENSE && git commit -m "Add AGPL-3.0 licence" && git push

The Benchmark copy differs from the other 3 by 1 line wrap in the appendix
boilerplate. Cosmetic, not substantive, and it explains the hash mismatch.

## 2. Create the platform repository

`high-trust-evidence`, matching the distribution name so the `pip install`
target and the repository URL agree.

The 4 READMEs now link to
`https://github.com/Cacapice/high-trust-evidence`. That link is dead until the
repository exists.

## 3. Apply the descriptions and topics

Each README now ends with a **Repository metadata** section carrying the
proposed GitHub description and topic list, kept in the repository so the
project page and the README cannot drift apart.

Currently only Maritime has topics. The other 3 have none, which forfeits the
discovery path that matters most for methods work.

## 4. Check the truncated descriptions

The GitHub descriptions for Benchmark Stewardship and Bayesian Inferential
Fidelity appear cut mid-sentence, at "supporting inv..." and "introduce post...".
Confirm whether that is display truncation on the listing page or actual
truncation in the repository settings. The replacements in section 3 are within
the limit.

## What was changed in the repositories

| Change | Repositories |
|---|---|
| **Shared evidence platform** section: dependency line, named adapter, and the statement that adapters stay in the research repositories | all 4 |
| **Repository metadata** section: proposed description and topics | all 4 |
| README release version corrected from v1.2.2 to v1.3.0 | Inferential Fidelity Framework |
| Inline code backticks normalised | all 4 |

## On adapters and the platform

The adapters stay on GitHub, in the research repositories, and should. Each is
70 to 74 lines holding exactly 1 `from_` function that translates a native
result into a `ScientificResult`.

An adapter encodes domain knowledge about what a native result means. The
platform should never need to know what a Fiedler value, an information basis, a
posterior recurrence probability or a BC1 gate is. Moving adapters upstream
would put domain knowledge in the layer that exists to be domain-neutral.

The split is: the platform holds what is common to all 4, and each repository
holds what only it knows. `test_portfolio_conformance.py` enforces the first
half by failing if contract types are defined locally, and the second half by
failing if a repository contributes no adapter.
