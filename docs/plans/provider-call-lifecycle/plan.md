# Provider-Call Lifecycle

Status: design plan for local refinement; implementation has not started.

## Planning sources

For this planning stage, [plan-terms.toml](plan-terms.toml) defines the shared
vocabulary and [plan-contracts.toml](plan-contracts.toml) defines the proposed
standing behavior. This document uses those definitions and contracts rather
than restating them. It records the implementation design, migration, and
validation plan.

Any accepted vocabulary or standing-behavior change must be made in those files
before its entry replaces or extends the repository's authoritative `.defs`
entries.

## Goal and current gap

Make `dr-providers` the single owner of the generic lifecycle for one provider
call while leaving workflow orchestration and evaluation policy outside the
package as specified by the planning contracts.

Today the public `Provider` protocol returns a provider transport outcome,
`HttpProvider` separately exposes provider invocation evidence, and both HTTP
transport and research callers may retry. The implementation will hard-cut
these overlapping paths to one observable provider-call lifecycle.

## Agreed design

### Canonical provider invocation

The public provider operation becomes one evidence-producing invocation:

```python
Provider.invoke(request: ProviderCallRequest) -> ProviderInvocationEvidence
```

`HttpProvider` and `ScriptedProvider` will implement this operation. The
existing public `complete` path and package-controlled native retry loop will
be removed rather than retained as aliases or compatibility paths.

An invocation may return evidence without sending a provider wire request, for
example when an expected transport failure occurs before dispatch. Invalid
inputs, classifier defects, and unexpected programming or infrastructure
errors remain exceptions. They are not provider invocation outcomes and are
never retried by the provider-call executor.

### Provider-call execution and result shape

Add one synchronous provider-call executor. It accepts:

- a provider invocation operation;
- a provider call retry policy;
- a semantic response classifier; and
- an injectable, cancellation-aware wait/clock boundary.

The executor owns waiting between invocations. It does not expose a resumable
next-invocation state machine, an async execution API, or a separate call-level
deadline.

A provider call result contains an ordered nested record for each completed
invocation. Each record binds its provider invocation evidence to its provider
invocation outcome and, when another invocation will occur, the retry decision
and selected delay. The terminal record has no retry decision. The result also
contains exactly one provider call outcome.

The result does not duplicate per-invocation usage or cost as call-level
aggregates. Callers can derive aggregates from the ordered evidence without
introducing semantics for partially unavailable values.

### Provider invocation outcomes

Keep protocol parsing strict. A semantic response classifier receives only a
valid provider transport response; protocol failures do not become semantic
classifier inputs. Map evidence into a flat, exhaustive provider invocation
outcome set:

- success;
- blank response;
- malformed response;
- provider rejection;
- semantic rejection;
- permanent provider or transport failure;
- transient provider or network failure;
- rate limiting;
- resource exhaustion;
- timeout; and
- unknown transport failure.

Use the following boundary mappings:

- a valid response accepted by the classifier is success;
- a response with no generation text is blank response;
- invalid provider protocol or payload structure is malformed response;
- an explicit provider refusal is provider rejection;
- caller rejection of a valid response is semantic rejection; and
- other provider transport failures map from their failure class and code.

Success terminates the provider call as accepted. If the final permitted
invocation produces an otherwise retryable outcome, the provider call outcome
is policy exhaustion. Any other non-retryable invocation outcome terminates the
call with that outcome.

### Retry policy and delays

The provider call retry policy is the sole authority for retry eligibility.
Remove the transport failure `retryable` field, native HTTP retry count, and
package-owned retryability sets so that evidence and policy cannot disagree.

Policy construction validates a positive integer invocation limit and finite,
non-negative delays. The initial implementation supports only deterministic,
declared policy delays. Each retry record identifies its delay source as the
policy. Server hints, jitter, adaptive backoff, and replay of delay computation
remain out of scope.

Provide one frozen, identity-bearing standard policy. It retries transient
provider or network failure, rate limiting, and timeout. It does not retry
provider rejection, permanent failure, resource exhaustion, or unknown
failure. Blank response, malformed response, and semantic rejection are
retryable only when an explicitly selected policy says so. Retryability is
never an intrinsic property of an outcome.

### Semantic response classifier

Define the semantic response classifier as a narrow behavioral protocol over a
typed provider transport response. It returns success or semantic rejection;
protocol-derived outcomes are assigned before this boundary.

The classifier supplies an opaque, caller-owned stable identifier. The caller
owns keeping that identifier synchronized with classifier behavior;
`dr-providers` validates and records the identifier but does not infer or
verify its meaning.

### Identity

Represent two different identities:

- provider call identity binds the provider call request, provider call retry
  policy, semantic response classifier identifier, and provider-call schema
  version; and
- provider call result identity is the content identity of the complete
  serialized provider call result.

Repeated executions with the same provider call identity may produce different
provider call results. Workflow run membership, rather than a random occurrence
identifier in this package, distinguishes executions when necessary.

Package, dependency, and runtime versions remain result diagnostics. An
optional monotonic invocation duration may also be recorded as a diagnostic;
wall-clock timestamps are not added. Neither affects provider call identity,
although serialized result diagnostics participate in provider call result
identity.

### Cooperative cancellation

Cancellation is observed only at the synchronous executor boundary:

- before an invocation starts, cancellation terminates the provider call
  without starting that invocation;
- during retry waiting, cancellation terminates the provider call and preserves
  all completed invocation records; and
- when cancellation is requested during an in-flight invocation, the executor
  allows that invocation to return under its normal timeout contract, records
  and classifies its evidence, and then terminates without starting another
  invocation.

In the last case, the provider call outcome is cooperative cancellation while
the final provider invocation outcome records whether the invocation succeeded
or failed. Cancellation therefore does not discard a result for work that may
already have incurred provider cost, and it does not claim to terminate the
provider wire request.

Abrupt process or thread interruption does not produce a provider call result
or imply persistence of partial history.

### Evidence retention

Preserve the existing evidence boundary: retain complete decoded response and
failure bodies under the current credential exclusion and header-redaction
rules. Do not add raw wire-byte capture, truncation, or artifact
externalization. Any future bounded or externalized representation requires a
separate evidence-loss and storage design.

## Implementation sequence

1. Inventory provider-client retry behavior and confirm that every supported
   client can disable package-hidden retries.
2. Align planning terms and contracts with the agreed design, then freeze
   provider call identity, result, record, outcome, and policy schemas.
3. Hard-cut the public provider protocol to the canonical evidence-producing
   invocation and remove native HTTP retry configuration and retryability
   fields.
4. Implement the synchronous provider-call executor, classifier protocol,
   standard policy, controlled wait boundary, and scripted-provider coverage.
5. Apply the invocation contract to OpenAI Chat Completions, OpenAI Responses,
   OpenRouter, Gemini, and Anthropic routes.
6. Remove controlled downstream duplicate lifecycle paths in their owning
   repositories after the package API is released.
7. Update public exports, authoritative terms and contracts, package
   documentation, and wire-format golden tests.
8. Release and pin the package before downstream workflow migration.

## Validation bar

- Scripted-provider tests cover every provider invocation outcome, the standard
  retry matrix, explicit semantic retries, policy exhaustion, and terminal
  record invariants.
- Tests prove the exact ordered association among invocation evidence,
  invocation outcome, retry decision, and provider call outcome.
- Cancellation tests use explicit gates to cover cancellation before an
  invocation, during waiting, and during an in-flight invocation. The last case
  proves that completed evidence is retained and no subsequent invocation
  starts.
- Clock and wait tests synchronize on controlled state; elapsed time is only a
  watchdog.
- Identity tests distinguish stable provider call identity from provider call
  result content identity and pin all persisted keys and discriminators.
- Credential tests preserve the planning contracts' bounded exclusion and
  redaction claims.
- Live tests cover OpenAI Chat Completions, OpenAI Responses, OpenRouter,
  Gemini, and Anthropic through the same lifecycle contract.
- Public documentation makes no durability, exactly-once, containment, global
  rate-control, artifact-storage, or evaluation claim.

## Downstream handoff

After release, inspect the final lifecycle API before planning `dr-platform`,
`dr-code`, or Whetstone changes. Downstream code should delete its generic
provider retry and classification machinery while retaining domain response
interpretation, evaluation decisions, and workflow orchestration.
