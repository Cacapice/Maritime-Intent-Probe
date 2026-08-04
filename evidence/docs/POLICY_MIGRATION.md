# Policy-Version Migration

Trust decisions are bound to the exact policy version and canonical policy digest under which they were issued. A policy bump invalidates prior decisions for release under the new policy.

A new policy that supersedes an earlier version must:

1. declare `supersedes_policy_version`;
2. carry a `PolicyMigrationRecord` identifying the previous decision and both policy digests;
3. record an approver and migration rationale;
4. re-run the full trust evaluation under the new policy;
5. issue a new `TrustDecision` bound to the current result, envelope, purpose, classification, policy version, and policy bytes.

Migration records do not grandfather prior authorization. They document why re-evaluation occurred.
