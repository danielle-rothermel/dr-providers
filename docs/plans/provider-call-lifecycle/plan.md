# Provider-Call Lifecycle

Status: design plan incorporating the performance and cross-plan audit;
implementation has not started.

## Planning sources

For this planning stage, [plan-terms.toml](plan-terms.toml) defines the shared
vocabulary and [plan-contracts.toml](plan-contracts.toml) defines the proposed
standing behavior. This document uses those definitions and contracts rather
than maintaining a second specification. It records implementation structure,
migration, and validation evidence.

Accepted entries replace or extend the repository's authoritative `.defs`
entries in the implementation stack before public schemas are frozen.

Historical note: the performance and cross-repository revisions were requested
by the [performance and cross-plan audit](perf-and-plan-audit-review-doc.md).

## Goal and ownership

Make `dr-providers` the single owner of provider invocation, transport evidence,
outcome classification, retry-state transitions, and terminal provider-call
results.

The planning ownership contract fixes the cross-repository boundary:

- `dr-store` may persist provider lifecycle values without defining their
  schemas or becoming a `dr-providers` dependency;
- `dr-platform` persists provider call state, schedules declared retry delays,
  controls global admission, and assigns workflow membership without
  reimplementing retry decisions;
- `dr-exec` owns local process isolation, not provider HTTP deadline handling;
  and
- applications own semantic response rules, evaluation, rewards, and analysis.

The async `dr-store` boundary does not require a second async provider API. A
serializable retry transition releases durable workers during delays while the
provider invocation itself remains synchronous.

## Current gap

Today the public `Provider` protocol returns a provider transport outcome,
`HttpProvider` separately exposes provider invocation evidence, HTTP transport
may hide native retries, and callers may implement another retry layer. The
default HTTP path also creates a client and worker for each wire request, while
evidence repeats large request and failure fields. The implementation hard-cuts
these paths to one observable, bounded provider-call lifecycle.

## Agreed design

### Canonical provider invocation

The public provider operation becomes one evidence-producing invocation:

```python
Provider.invoke(request: ProviderCallRequest) -> ProviderInvocationEvidence
```

`HttpProvider` and `ScriptedProvider` implement this operation. The existing
public `complete` path, native retry loop, native retry count, and transport
`retryable` field are removed rather than retained as compatibility paths.

An invocation may return evidence without sending a provider wire request, for
example when an expected pre-dispatch transport failure occurs. Invalid inputs,
classifier defects, and unexpected programming or infrastructure errors remain
exceptions; the retry transition never converts them into outcomes.

### Serializable retry state machine

Define closed, versioned `ProviderCallState` and `ProviderRetryInstruction`
models plus one deterministic provider call transition.

The state contains the shared call-level components and ordered completed
invocation records needed to finish the result without external storage. It
validates:

- request, provider call retry policy, and classifier identities;
- invocation ordinals and the next permitted ordinal;
- each completed record's content hash; and
- every legal state transition.

Given current state and one completed invocation's evidence and classified
outcome, the transition returns exactly one of:

- a terminal provider call result; or
- a provider retry instruction with the declared delay, next invocation
  ordinal, and next serializable state.

Cancellation uses a deterministic terminal transition over the same state. With
no active invocation, it terminalizes the current state. With an active
invocation, the transition accepts its completed record and the cancellation
observation together, appends the record, and returns draining cancellation
without selecting a successor. Serializing and restoring state between every
invocation must produce the same result as uninterrupted execution.

Build one thin synchronous local driver over this transition. It invokes the
provider, classifies valid responses, advances state, and performs event-based
cancellation-aware waiting when instructed. A later durable driver persists the
same state and lets `dr-platform` schedule the declared delay. Neither driver
contains retry eligibility or terminal-outcome logic.

Interruption after a provider effect but before the next state is persisted may
cause another invocation after recovery. Each completed invocation therefore
remains independently attributable; serializable state does not imply exactly
once provider effects.

### Result and evidence normalization

A provider call result is self-contained but normalized:

- the complete provider call request identity document appears once;
- each invocation record refers to that request by content hash;
- provider HTTP request evidence alone stores the translated HTTP request body;
- provider transport failures do not repeat that body;
- failure messages are bounded structured summaries and never copies of
  response bodies;
- one decoded response body appears per invocation;
- extracted generation text may intentionally duplicate text in that body; and
- other repeated immutable components appear once and are referenced by
  validated content hashes.

Each ordered invocation record binds its provider invocation evidence to its
provider invocation outcome and, when another invocation is allowed, its retry
decision and selected delay. A terminal record has no retry decision. The
result contains one provider call outcome and does not duplicate usage or cost
as call-level aggregates.

### Provider invocation outcomes

Protocol translation remains strict and precedes semantic classification. The
closed invocation outcomes are:

- success;
- blank response;
- malformed response;
- provider rejection;
- semantic rejection;
- permanent provider or transport failure;
- transient provider or network failure;
- rate limiting;
- resource exhaustion;
- contained transport timeout;
- uncontained deadline expiration; and
- unknown transport failure.

A contained transport timeout records that the local HTTP operation ended and
no local worker remains. An uncontained deadline expiration records that a
watchdog returned while a worker or socket may still be active. The package
does not start a successor while it knows an earlier invocation for the call is
locally uncontained.

Success terminates the provider call as accepted. If the final permitted
invocation produces an otherwise eligible outcome, the provider call outcome
is policy exhaustion. Any other policy-terminal invocation outcome terminates
the call with that outcome.

### Retry policy and rate-limit evidence

The provider call retry policy is the sole retryability authority. The standard
policy is frozen and exact:

- at most two provider invocations;
- at most one retry;
- one deterministic one-second delay;
- one second of maximum cumulative delay in policy identity; and
- retry eligibility only for contained transient provider or network failure
  and contained transport timeout.

Rate limiting, uncontained deadline expiration, resource exhaustion, provider
rejection, permanent failure, and unknown failure are terminal under the
standard policy. Blank response, malformed response, and semantic rejection
require an explicitly selected custom policy. Custom policies retain positive
finite invocation limits and finite non-negative declared delays.

Capture a bounded normalized `Retry-After` hint in invocation evidence when
present, but do not let the standard policy act on it. Coordinated rate-limit
waiting belongs to global admission policy in `dr-platform`.

### Semantic response classifier

Define the semantic response classifier as a narrow behavioral protocol over a
valid typed provider transport response. It returns success or semantic
rejection; protocol-derived outcomes are assigned before this boundary.

The classifier supplies an opaque caller-owned stable identifier. The caller
owns keeping that identifier synchronized with classifier behavior;
`dr-providers` validates and records the identifier without inferring its
meaning.

### Identity composition

Provider call identity binds provider call request identity, provider call
retry policy identity, semantic response classifier identifier, and
provider-call schema version.

Provider call result identity composes call-level fields with the ordered
content hashes of invocation evidence. Each embedded component is validated
against its declared hash, and frozen components cache their identities. These
are storage-neutral content identities, not `dr-store` references.

Package, dependency, and runtime versions remain result diagnostics. Optional
monotonic invocation duration is diagnostic; wall-clock timestamps are not
added. Diagnostics may affect provider call result identity but not provider
call identity.

### Bounded transport evidence

Provider transport policy adds positive `max_request_bytes` and
`max_response_bytes` limits. Both participate in transport-policy identity and
invocation evidence.

Translate and encode each HTTP request once, count the exact encoded bytes
before dispatch, and send those same bytes. Stream and count response bytes
before decoding so success and error responses cannot be buffered without
limit.

Exceeding either limit produces bounded resource-exhaustion evidence with the
declared limit and observed accounting. It retains no partial body while
claiming complete evidence. Complete decoded response and failure bodies remain
available within the declared limits under the existing credential exclusion
and header-redaction rules. Raw wire capture, truncation presented as complete,
and artifact externalization remain outside this plan.

### HTTP provider lifecycle and deadlines

One `HttpProvider` owns one long-lived bounded `httpx.Client` and reuses its
connection pool across invocations. Construction fixes explicit connection and
keep-alive limits in transport-policy identity. `close()` drains active
operations as required, closes an owned client exactly once, rejects new
invocations, and leaves the provider terminal.

Retain caller-owned client injection only as an explicit separate construction
path. `HttpProvider` never closes the caller-owned client.

Prefer direct synchronous HTTP execution with native connect, read, write, and
pool timeouts. Keep a whole-operation worker deadline only if its stronger
caller-return bound remains necessary. If retained, track and drain its live
work explicitly and report an uncontained deadline expiration whenever its
worker may remain active. Do not start fresh child processes for provider HTTP
deadlines.

### Draining cancellation

Draining cancellation is not overload shedding. Callers stop admission first.
The local driver cancels pending retry waits through an event-based wait without
polling and starts no successor.

An active invocation reaches its normal contained or uncontained terminal
observation. The driver records and classifies any returned evidence, then
passes that record and the cancellation observation through the terminal
transition. This preserves results for work that may already have incurred
provider cost without promising prompt release of sockets, workers, or provider
capacity.

Abrupt process or thread interruption produces no provider call result unless a
durable caller has already persisted the preceding state.

## Implementation stack

### PR 1: schemas, normalization, and retry transition

- Promote the accepted vocabulary and contracts.
- Define timeout outcomes, evidence bounds, normalized evidence, provider call
  state, provider retry instruction, exact standard policy, identities, and the
  deterministic transition.
- Use scripted provider invocations only.

### PR 2: HTTP lifecycle and provider routes

- Hard-cut the public provider protocol and remove native retries and duplicate
  failure request bodies.
- Add the long-lived bounded HTTP client, native timeout handling, optional
  tracked whole-operation deadline, request/response byte enforcement, and
  explicit close lifecycle.
- Route OpenAI Chat Completions, OpenAI Responses, OpenRouter, Gemini, and
  Anthropic through the same invocation contract.

### PR 3: local driver and downstream handoff

- Add the thin synchronous driver and controlled cancellation-aware wait
  boundary.
- Add serialize/restore demonstrations showing how a durable driver consumes
  the same retry instructions without duplicating retry semantics.
- Update public exports, package documentation, wire-format golden tests, and
  downstream handoff evidence.
- Release and pin the package before downstream repositories delete their
  duplicate lifecycle paths.

## Validation bar

- A scripted two-round call serializes and restores state between every
  invocation and produces the same result as uninterrupted execution.
- State tests reject skipped or repeated ordinals, mismatched identities,
  mismatched component hashes, and illegal terminal transitions.
- Timeout tests prove that contained timeout may retry, uncontained deadline
  expiration starts no successor, and tracked live work is drained as declared.
- Many invocations reuse one HTTP client and remain within explicit connection
  and keep-alive limits.
- Concurrent close tests prove active-operation handling, close-once behavior,
  terminal rejection of new invocations, and caller-owned client preservation.
- Request and response limit tests cover success and error bodies, exact byte
  accounting, bounded resource-exhaustion evidence, and absence of partial
  bodies.
- Serialized-size checks prove that request identity, HTTP request body,
  failure request data, and decoded response bodies are not duplicated across
  one invocation record or amplified across retries.
- Standard-policy tests cover the exact two-invocation, one-second-delay matrix,
  including terminal rate limiting and uncontained deadline expiration.
- Cancellation tests use explicit gates before invocation, during an event-based
  retry wait, and during active invocation; no test uses elapsed time as success
  evidence.
- Identity tests distinguish provider call identity from compositional result
  identity, validate embedded hashes, cache component identities, and pin all
  persisted keys and discriminators.
- Credential tests preserve the planning contracts' bounded exclusion and
  redaction claims.
- Live smoke tests cover every supported provider route through the same
  lifecycle.
- Public documentation makes no persistence, exactly-once, containment, global
  admission, artifact-storage, or evaluation claim beyond the planning
  contracts.

## Downstream handoff

After release, inspect the final lifecycle API before planning downstream
cutovers. `dr-platform` persists provider call state and schedules provider
retry instructions; it does not decide eligibility. `dr-store` persists the
provider-owned serialized models without redefining them. `dr-code` and
Whetstone delete generic provider retry and classification machinery while
retaining domain response interpretation, evaluation decisions, and workflow
orchestration.
