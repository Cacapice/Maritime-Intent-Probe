# Qualification Algebra

## Mathematical object

The qualification algebra is a **monotone transition system over publication
status**. Qualifications are transitions on epistemic strength, not annotations
attached after an independently chosen status.

Let

```text
blocked < exploratory < qualified < publishable
```

be the publication-strength order. For every qualification `q` and status `s`,
the load-bearing invariant is

```text
strength(apply(q, s)) <= strength(s)
```

unless `q.effect == "new_evidence"`.

A blocked state is absorbing under `preserve`, `weaken`, and `block`.
Strengthening is intentionally isolated because it requires additional evidence,
not reinterpretation of evidence already present. A `new_evidence` transition
must therefore identify its evidence source and its resulting publication status.

## Architectural boundary

Adapters no longer decide final publication status. They provide:

1. domain evidence;
2. a base publication status;
3. structured qualifications.

`derive_publication_status()` alone determines the resulting status.

```text
Before
Evidence -> Adapter -> Status
Qualification --------> Metadata

Now
Evidence -> Qualification -> Publication algebra -> Publication status
```

This makes publication strength a derived property of qualified evidence and
prevents adapters, demos, or exporters from silently overstating a claim.
