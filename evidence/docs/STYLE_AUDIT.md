# Cambridge style guide: audit and application

Applied to the prose in all 4 repositories against the University of Cambridge
web style guide. Fenced code, indented code, inline code spans, LaTeX maths and
link targets were left untouched throughout.

## Result

| Rule | Before | After | Status |
|---|---:|---:|---|
| Dashes | 519 | 0 | fixed |
| Semicolons | 365 | 14 | fixed, except enumerations |
| Italics | 125 | 48 | emphasis removed, notation kept |
| Numeric ranges taking a dash | 56 | 7 | fixed |
| Ampersands | 19 | 0 | fixed |
| eg, ie, etc | 5 | 0 | fixed |
| Negative contractions | 4 | 0 | fixed |
| Long words | 5 | 0 | fixed |
| Block capitals | 78 | 76 | initialisms, correct as written |
| Spelled numbers 2 to 9 | 145 | 145 | see conflicts below |
| Slashes | 91 | 91 | see conflicts below |
| Heading case | 13 | 13 | needs an author's judgement |

## Where the guide conflicts with mathematical writing

The guide governs content written for the cam.ac.uk web estate. 3 of its rules
cannot be applied wholesale to a mathematical document without loss of meaning.

**Italics.** The guide bans italics on accessibility grounds and exempts only
gene and species names. Mathematical variables are set in italics by universal
convention, and removing that convention makes expressions such as h(1, L) and
I = L unreadable as notation. 48 instances were preserved on this basis. Pure
emphasis italics, such as a word italicised for stress, were removed.

**Numbers 2 to 9 as numerals.** The guide's own text allows words in common
phrases. Technical compounds such as two-sided, three-tier and four-fold read as
common phrases rather than counts, and rendering them as 2-sided or 3-tier would
be unidiomatic. 145 instances were left as written. The rule was applied where a
genuine count appears.

**Slashes.** The guide prefers 'or'. Many of the 91 instances are file paths,
package names, ratios and axis labels rather than prose alternatives. These need
reading one at a time.

## What blind application broke on the first pass

Recorded because it shows why the fixer had to be constrained.

| Original | Naive output | Damage |
|---|---|---|
| H1 to H5 remain locked | H1, H5 remain locked | range destroyed |
| (1) define X. (2) show Y | (1) define X. (2) show Y | sentence fragments |
| R = h(I, L) | R = h(I, L) | notation lost |
| the cells I=0, L=1 | the cells I=0, L=1 | notation lost |

The corrected fixer preserves alphanumeric ranges, skips semicolons inside
enumerations, and keeps any italic span containing a mathematical operator.

## Remaining items for an author

- 13 headings in title case. The guide asks for sentence case in headings and
  page titles, and title case only for proper nouns, qualifications, term names
  and College names.
- 14 semicolons inside enumerated clauses. These are correct as punctuation but
  the guide would prefer the enumeration be broken into a numbered list.
- Page titles should be 65 characters or less and summaries 160 or less. Several
  README headings exceed this.
- Paragraphs should carry a maximum of 5 sentences. Several in the design paper
  and the fellowship document run longer.

---

# Scientific communication conventions

A second pass applied the conventions organised in Bennett and Jennings,
*Successful Science Communication: Telling It Like It Is* (Cambridge University
Press, 2011). Note the scope: the linked frontmatter supplies the contents,
author biographies and Sir Walter Bodmer's foreword. The chapter-level guidance
was not available, so the conventions applied here are the standard ones the
volume is organised around, not quotations from it.

Bodmer's foreword rejects the reading that scientists need only supply the public
with facts, the position that became known as the deficit model, and argues that
dialogue is impossible unless the scientist can explain the issues to a
non-expert. The onus, on that argument, falls on the scientist.

## What the audit found

| Signal | Count | Verdict |
|---|---:|---|
| Passive without a named agent | 228 | mostly the impersonal mathematical register, correct as written |
| Common language of research | 222 | error, bias, uncertainty, risk, robust, valid; used correctly in the technical sense, but they carry a different meaning for a general reader |
| Vague hedges | 100 | 102 of these were 'rather than', a contrast rather than a hedge; about 5 genuine |
| Point estimates without an interval | 48 | worth checking individually |
| Specialist terms undefined at first use | 46 | identifiability alone accounts for 26 |
| Overclaiming | 20 | all instances of prove, proved, proven in the mathematical sense; correct |
| Causal language | 13 | worth checking each against the design |
| Sentences over 40 words | 9 | worth splitting |

The prose was already disciplined on the conventions that usually go wrong.
Claims are kept separate from interpretation, uncertainty is stated rather than
implied, and the evidence tables make the warrant for each claim explicit.

## The gap that mattered

All 4 documents opened in specialist register. None stated, in language a reader
outside the subfield could follow, what problem the work addresses and why the
result matters. This is the point Bodmer's foreword turns on, and it was the
single largest communication gap in the set.

Two additions were made to each repository.

**An 'In brief' summary** at the head of each README. Three short paragraphs, no
undefined jargon, stating the problem, the finding and its consequence.

**A GLOSSARY.md** defining the specialist terms in plain language, covering the
46 flagged instances. Identifiability, estimand, surrogate, censoring, certified
floor, modulus of continuity, information basis, saturation and the rest.

## Left for an author

- 48 point estimates without an interval. Some are exact by construction and need
  no interval. The rest would be stronger with one.
- 13 uses of causal language. Each needs checking against whether the design
  supports a causal reading.
- 9 sentences over 40 words.
- The words error, bias, uncertainty, risk, robust and valid are used correctly in
  their technical senses throughout. A general reader will take a different sense
  from each. The glossary addresses this for the terms most likely to mislead.

---

# Brown University Science Center, Quick Guide to Science Communication (2014)

A third pass applied the Brown guide. Unlike the Cambridge frontmatter, this
document contains checkable prescriptions, so it was possible to test compliance
directly.

## What the writing already did well

| Prescription | Result |
|---|---|
| Avoid cliches | 0 across 5,900 lines |
| Avoid wordplay and puns | 0 |
| Avoid euphemism | 2 |
| Use active verbs | mostly, allowing for the mathematical register |
| Only include critical details | yes |
| Cite your sources | yes |
| Tell a story but stay true to the facts | yes, in the maritime pivot record |

A zero cliche count over 5,900 lines is the strongest single result of the whole
audit. The guide's warnings about padding language do not apply to this writing.

## What was missing, and has been added

Every structural prescription failed on the first pass.

**Who is my audience?** The guide opens with 3 questions, and the first is this
one. No document named its reader. A 'Who this is for' section was added to each
of the 4 READMEs, naming the reader and stating that familiarity with the worked
example is not required.

**Use analogies and examples.** The guide prescribes both, and the corpus had 1
across all 4 repositories. Each README now opens with a concrete case measured in
its own code, with the script that regenerates it named:

| Repository | Measured example | Script |
|---|---|---|
| Maritime Intent Probe | Legitimate items hold exactly 1 turn, adversarial items exactly 3, so a turn count reaches AUC 1.000 with no forward pass. Across all 4 attack families the same count falls to 0.750. | `compute_model_blind_witness.py` |
| TransferMod | The same surrogate carries Silent Risk of 0.891 decades under a spectral gate and 0.013 decades under a Fiedler decision, about 70 times narrower. | `scripts/run_graph_instance.py` |
| Benchmark Stewardship | The mean rises smoothly, 0.740 to 0.880, the standard deviation is identical at 0.0311 throughout, and rank correlation drops from +1.000 to -1.000. | `benchmark_tier_transfer_check.py` |
| Bayesian Inferential Fidelity | A plug-in estimate of 0.016 against a posterior mean of 0.236, driven by 16.6% of posterior mass on non-positive drift. Silent Risk 0.270. | `demo_sar_reentry.py` |

Concrete measured cases were chosen over invented analogies. Each figure was
regenerated and checked against the code before being written into the prose. One
draft misstated the benchmark case as an identical mean across releases. The mean
in fact rises; what is held constant by construction is the standard deviation.
The correction is recorded here because a document about honest reporting should
show its own.

**Answer 'so what?' early.** Each analogy is followed by an explicit statement of
the consequence, in the opening screen rather than the discussion.

**Create an outline.** 4 documents run past 700 lines with no contents list. One
was added to each, between 15 and 24 entries.

**State your message at the beginning and at the end.** Only 1 document restated
its message in the closing section. An 'In closing' section was added to each
README, restating the single message in 1 short paragraph.

## Not addressed

**Talk about the process, not just the results.** The guide asks for the
challenges, the successes and the collaborations. These documents present
finished results in an impersonal register. The maritime repository is the
exception: its pivot from a confirmatory to an exploratory programme is fully
recorded, though written impersonally. Converting that record into a first-person
account of what was expected, what was found and what changed would satisfy the
guide, and is an editorial decision rather than a mechanical one.

## Note on the 3 sources

The Cambridge style guide and the Brown guide agree on front-loading, plain
language and avoiding jargon. They differ on 1 point of substance. Bodmer's
foreword and the Brown guide's models section both reject the deficit model, the
assumption that public doubt is a knowledge shortfall the expert can fill. The
additions above are written for the contextual model instead: they state who the
reader is, what that reader already knows, and why the result would matter to
them.

---

# Harvard, A Student's Guide to Writing in the Life Sciences (2007)

A fourth pass applied Chapter III, 'Common Mistakes in Scientific Writing', and
the scientific usage conventions in the same chapter. This is the most specific
of the 4 sources and the most directly checkable.

## The finding that mattered

**First person plural in a single-authored work.** The guide is explicit: use 'I'
for a single-authored work and 'we' for 2 or more. All 4 repositories are
authored by 1 person, and the prose used 'we', 'our' and 'us' in 11 places.

All 11 were revised. Not all became 'I', because the same guide asks for verbal
variety and warns against opening every sentence the same way. Claims about what
the work does were reattributed to the work: 'we propose treating BC1 as a
precondition' became 'this work proposes treating BC1 as a precondition'.
First-person statements of what the author did were kept in the first person
singular: 'we asked whether adversarial intent is linearly represented' became
'I asked whether'.

## Other corrections

| Rule | Found | Action |
|---|---:|---|
| 'Data' is plural | 1 | 'none of the data is offered' became 'are offered' |
| Cite as (Figure 1), not (see Figure 1) | 0 | already correct |
| Combine adjacent parentheses | 2 | both mathematical, left |
| 'however' needs a preceding full stop | 1 | parenthetical, not a run-on, left |
| 'between' for 2, 'among' for 3 or more | 0 | already correct |
| 'fewer' for countable nouns | 0 | already correct |
| Citation inside end punctuation | 0 | already correct |
| Abbreviations defined at first use | pass | already correct |

## The guide vindicates an earlier decision

The Cambridge pass flagged 22 instances of prove, proved and proven as
overclaiming, and they were left unchanged on the grounds that they refer to
theorems. This guide states the rule with exactly that exception: prove is
avoided outside mathematics, and in science the strongest available claim is that
the data are consistent with the hypothesis. The instances here are mathematical
proofs, so the exception applies and the earlier decision stands.

## A direct conflict between 2 of the sources

The Harvard guide requires Latin terms and their abbreviations to be italicised,
including et al. The Cambridge style guide bans italics for accessibility, with a
carve-out only for gene and species names. There are 66 instances of et al. in
these documents.

Neither source can be followed without breaking the other. The instances were
left unitalicised, following Cambridge, because the accessibility argument
concerns readers rather than convention, and because Harvard notes that some
journals ignore the italics rule. The conflict is recorded so the decision is
visible rather than accidental.

A second, smaller conflict: Harvard requires a full stop or a semicolon before
'however' used as a conjunction; Cambridge bans semicolons. Where this arises,
the full stop satisfies both.

## Not addressed

**Tense conventions.** The guide sets out a specific scheme: others' results in
the past tense, methods in the past tense, the results section in the present
tense, and proposals in the future tense. These documents are READMEs and design
papers rather than research articles, so the section boundaries the scheme
depends on are not present. Applying it would require restructuring the documents
into the research-paper format, which is a larger decision than a style pass.

**Abstract conventions.** The guide asks that an abstract stand alone and be
written last. The 'In brief' sections added in the previous pass serve this
function and were checked against the requirement, but they are not labelled
abstracts and are not bound by the 150 to 500 word range the guide gives.

---

# Paradis and Zimmerman, The MIT Guide to Science and Engineering Communication

A fifth pass applied the chapter 'Revising for Organization and Style', pages 55
to 60. The volume is under controlled digital lending, so only these pages were
available. The graphics section begins at the foot of page 60 and was not
supplied, so the figure audit remains open.

## The rule that produced the finding

Page 59 sets a specific threshold: long sentences, often amounting to more than
30 words, are usually too complicated. The remedy given is to determine the main
actions of the sentence and sort them into 2 or more shorter ones.

Measured across all prose, joining wrapped lines into paragraphs first, because
splitting on source lines fragments sentences and understates their length:

| | Before | After |
|---|---:|---:|
| Sentences | 2,429 | 2,429 |
| Mean length | 13.4 words | 13.3 words |
| Median | 11 | 11 |
| 90th percentile | 28 | 28 |
| Longest | 79 words | 73 words |
| Over 30 words | 166 | 164 |
| Over 45 words | 33 | 30 |
| Over 55 words | 8 | 5 |

The distribution is healthy. A median of 11 words and a 90th percentile of 28 sit
comfortably inside the guide's threshold. The problem is the tail, and it is
concentrated: the fellowship document, the design paper and the methods document
hold most of it.

The 3 worst sentences were revised using the technique the guide prescribes. The
79 word passage on the marginal-signal scenario became 4 sentences. The 62 word
passage contrasting 2 confidence intervals became 5. The 65 word passage on
CCS-style probing became 2.

## What the guide's other tests found

| Test | Count | Verdict |
|---|---:|---|
| Excessive nominalizing | 4 | 'regeneration of the resin bed is achieved by' pattern, largely absent |
| Wordy constructions | 4 | 'by the use of', 'in order to' and similar, largely absent |
| Vague words where a number belongs | 2 | 'high temperatures' pattern, largely absent |
| Passive with the agent buried | 9 | mostly the mathematical register |
| Stacked modifiers | flagged 143 | almost all instrument error; the regex matched prepositions rather than genuine noun piles |

The writing does not have the faults this chapter is written to correct. It has 1
fault, the long-sentence tail, and it has that fault in a small number of places.

## Left for an author

164 sentences remain over 30 words. Splitting them is editorial work rather than
mechanical: the guide's method requires identifying the main actions of each
sentence, which cannot be automated without changing meaning. The 30 over 45
words are the ones worth doing first, and they sit almost entirely in
docs/research_narrative.md, DESIGN_PAPER.md and methods.md.

## Figure conventions: still open

Checking whether the graphics chapter was worth requesting surfaced 2 problems
that the other 4 sources did catch.

**A documentation mismatch.** `results/README.md` in the TransferMod repository
describes `inferential_fidelity.png` and `benchmark_logic.png` as schematic
figures used in the README. The README contains no image reference at all.

**An orphan figure.** `diagnostics.png` is written by `run_scaling.py` but is
absent from the provenance table that `results/README.md` otherwise maintains
carefully.

Of 10 figures across the 4 repositories, 2 are embedded and captioned. Those 2,
the maritime depth profiles, carry alt text stating the finding rather than
naming the file, which satisfies both the accessibility requirement and the
standalone-legend rule at once. They are the standard the other 8 should be
brought to.
