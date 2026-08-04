# Sovereign Evidence Architecture

**Enterprise architecture:** Sovereign Data → High-Trust Privacy → Qualified Evidence → Human-Governed Decision Support.

**Scientific doctrine:** *Qualification Before Interpretation* remains the rule governing analytical claims. It is one layer inside the broader enterprise architecture, not a substitute for data sovereignty, privacy, consent, or accountable decision authority.

## Enterprise layers

1. **Mission and owner policy** — the data owner defines purpose, jurisdiction, consent basis, custodianship, retention, and publication authority.
2. **Sovereign data plane** — raw records remain under owner control. Computation moves to the data through owner-hosted, federated-local, or explicitly approved trusted-enclave execution.
3. **High-trust privacy plane** — identity separation, data minimization, aggregate-only disclosure, purpose limitation, and residual-risk review determine whether a computation or release may occur.
4. **Scientific qualification plane** — repository-native methods state the estimand, uncertainty, assumptions, qualifications, supported conclusions, and unsupported conclusions.
5. **Decision-support plane** — qualified evidence is routed to accountable human review. Analytical outputs do not execute policy by themselves.
6. **Application plane** — community, enterprise, research, and public-sector applications consume qualified evidence rather than centralized raw records.

## Non-negotiable trust boundaries

- Raw data movement is prohibited by default.
- Ownership is not transferred by computation.
- Purpose must be declared before computation and checked again before release.
- Direct identifiers are blocked from publication envelopes.
- Publication requires both data-owner and publication-authority approval.
- Scientific publishability cannot override privacy or sovereignty restrictions.
- Privacy approval cannot strengthen an unsupported scientific claim.
- Human decision authority remains outside the analytical service.

## Provenance chain

A release carries distinct provenance for:

- data ownership and source;
- identity and executing principal;
- consent and permitted purpose;
- computation and numerical environment;
- scientific qualification;
- disclosure and publication approval.

## Repository role

**Construct-Validity Engine — Phase 1.** Tests whether semantic interpretation is identifiable. Phase 1 does not produce an operational intent monitor.

**Phase boundary:** This repository is Phase 1 only. It may publish construct-validity diagnostics and model-blind witness evidence; it may not license deployment, operational monitoring, or completed intent validation.

## Canonical release path

```text
owner-controlled data
    ↓ local / federated computation
privacy qualification
    ↓ aggregate-only evidence
scientific qualification
    ↓ publishability and limitations
owner + publication approval
    ↓
sovereign evidence bundle
    ↓
human-governed decision support
```

The implementation is in `sovereign_architecture.py`. `SovereignEvidenceEnvelope` deliberately excludes raw records and evaluates purpose, privacy, scientific status, approvals, and research phase before release.
