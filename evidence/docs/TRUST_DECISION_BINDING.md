# Trust-decision binding and release authenticity

The sovereign release gate authorizes **a specific canonical payload**, not a
class of vaguely similar results. A `TrustDecision` is bound to:

- the SHA-256 digest of the scientific result;
- the SHA-256 digest of the sovereign envelope;
- the exact release purpose;
- data provenance and disclosure class;
- policy version and decision issuer;
- issue, expiry, and revocation state.

The release sequence is:

```text
canonical payload
    -> payload digest
    -> purpose/classification evaluation
    -> bound TrustDecision
    -> atomic emission of the authorized bytes
    -> integrity manifest (always)
    -> release-class enforcement
         -> unrestricted/public research artifact: integrity-only publication permitted
         -> controlled release: managed signature required before publication
```

Hashes establish integrity for every release. Manifest authenticity depends on
release classification: unrestricted public research artifacts may use an
integrity-only manifest, while controlled releases—including owner-held data,
results derived from owner-held data, and internal, controlled, or
aggregate-release disclosures—**must** include a valid managed signature and
cannot opt out of authenticity verification. In enterprise mode this produces
`manifest.sig`, authenticating the manifest and the files it references.

> **Normative requirement.** Any controlled release **MUST** include a valid
> managed signature. Signature creation and verification are part of the
> publication contract, not optional post-processing.

Classification is split into provenance and disclosure class: being derived
does not by itself make a result safe to release.

The repository scan for bypasses remains a secondary control. The primary
control is capability-based emission through `publish_bundle()` or
`publish_sovereign_bundle()`.
