# Performance and Cross-Plan Audit Instructions

Status: required plan revisions before schema freeze or implementation.

## Purpose

Finalize one provider-call lifecycle that works for local execution and later
durable orchestration without duplicating retry semantics or coupling provider
models to storage and workflow packages.

The plan, proposed terms, and proposed contracts must incorporate the decisions
below before persisted models, outcome discriminators, or identities are frozen.

## Cross-repository ownership

- `dr-providers` owns provider invocation, transport evidence, outcome
  classification, retry-state transitions, and terminal provider-call results.
- `dr-store` may persist those serializable values but must not define their
  schema. `dr-providers` must not depend on `dr-store`.
- `dr-platform` will persist call state, schedule retry delays, control global
  admission, and assign workflow membership. It must not reimplement provider
  retry decisions.
- `dr-exec` owns local process isolation. Provider HTTP calls should not be put
  in fresh child processes merely to enforce deadlines; doing so defeats
  connection pooling and does not establish remote exactly-once behavior.
- Applications own semantic response acceptance, evaluation, rewards, and
  analysis.

The async `dr-store` cutover does not by itself require a second async provider
API. A resumable pure transition boundary removes blocked durable retry waits;
the transport invocation may remain synchronous in this plan.

## Decisions to retain

Retain these plan selections:

- at most one visible wire request per provider invocation;
- strict protocol translation before semantic classification;
- closed, evaluation-neutral invocation and call outcomes;
- one caller-owned semantic classifier identity;
- distinct call identity and result identity;
- complete ordered evidence for every completed invocation;
- draining cancellation for in-flight work; and
- no workflow scheduling, global admission, caching, storage, or exactly-once
  claim.

## Required lifecycle revisions

### Replace the non-resumable retry loop with one state machine

Define a closed serializable `ProviderCallState` and a deterministic transition
operation. Given the current state and one completed invocation's evidence and
classified outcome, the operation returns exactly one of:

- a terminal `ProviderCallResult`; or
- a `ProviderRetryInstruction` containing the declared delay, next invocation
  ordinal, and next serializable call state.

The state validates ordered ordinals, policy identity, classifier identity,
request identity, completed record hashes, and the next legal transition. A
serialize/restore boundary between every invocation must not change the result.

Build one thin synchronous local driver over this state machine. The local
driver invokes, advances, and performs cancellation-aware waiting. A later
durable driver persists the same state and lets `dr-platform` schedule the
declared delay without occupying a worker. There must not be a second retry
implementation in `dr-platform` or Whetstone.

Interruption after a provider effect but before state persistence may cause a
new invocation after recovery. Preserve the existing no-exactly-once claim and
make every completed invocation independently attributable.

### Distinguish contained and uncontained timeouts

Replace the single timeout outcome with two observable cases:

- a contained transport timeout, where the local HTTP operation has returned
  or raised and no worker remains active; and
- an uncontained deadline expiration, where a watchdog returned while a worker
  or socket may still be active.

The standard policy may retry a contained timeout. An uncontained deadline
expiration is terminal under the standard policy. Do not start another request
while the package knows an earlier request for the call remains locally
uncontained.

Prefer direct synchronous HTTP execution with native connect, read, write, and
pool timeouts. Retain a whole-operation worker deadline only if its stronger
caller-return bound remains necessary; if retained, track and drain its live
work explicitly and preserve the uncontained outcome.

### Make one connection pool live for the provider lifecycle

One `HttpProvider` owns one long-lived bounded `httpx.Client` and reuses it for
all invocations. Construction fixes explicit connection and keep-alive limits.
`close()` drains active operations as required, closes the owned client exactly
once, rejects new invocations, and leaves the provider terminal. Do not create a
client per invocation.

If caller-owned client injection remains supported, its ownership must be an
explicit separate construction path and `HttpProvider` must never close it.
Otherwise remove injection and keep the single provider-owned lifecycle.

### Normalize evidence before adding persistence

The self-contained call result stores the complete semantic request identity
document once. Each invocation refers to its request identity by hash.

- `ProviderHttpRequestEvidence` is the sole owner of the translated HTTP request
  body.
- `ProviderTransportFailure` does not repeat the request body.
- Failure messages are bounded structured summaries and never copies of a
  response body.
- A decoded response body appears once per invocation.
- Extracted generation text may intentionally duplicate text in that body for
  usability; this is the only accepted large duplication.
- Repeated immutable components are represented once in the call result and
  referenced by their validated content hashes where doing so removes retry
  amplification.

Cache component identities on frozen models. Define result identity
composition over the call-level fields and ordered invocation-evidence hashes,
and validate that every embedded component matches its declared hash. Do not
re-dump complete nested evidence repeatedly merely to compute parent identity.

These hashes are storage-neutral content identities, not `dr-store` object
references.

### Bound request and response evidence

Add explicit positive `max_request_bytes` and `max_response_bytes` transport
limits. Count the encoded request before dispatch. Stream and count the
response so the process never buffers an unbounded success or error body.

An exceeded limit produces a typed, bounded resource-exhaustion outcome with
observed accounting; it does not retain a partial body while claiming complete
evidence. The plan's complete-evidence guarantee applies only within the
declared byte limits. Pin the size-accounting and over-limit evidence keys with
golden tests.

### Freeze a small standard retry policy

The standard policy has exactly two maximum invocations and therefore permits
at most one retry. Use one exact deterministic one-second policy delay for that
retry and expose the maximum cumulative delay in policy identity.

It retries only contained transient provider/network failure and contained
transport timeout. Rate limiting, uncontained deadline expiration, resource
exhaustion, provider rejection, permanent failure, and unknown failure are
terminal. Blank, malformed, and semantic rejection remain retryable only under
an explicitly selected custom policy.

Capture a bounded normalized `Retry-After` hint in invocation evidence when
present, but do not let the standard policy act on it. Coordinated rate-limit
handling belongs to later global admission policy in `dr-platform`.

### Describe cancellation as draining

Stop admission first. Cancel pending retry waits without polling. Let active
invocations reach their normal contained or uncontained terminal observation,
record them, and start no successor. Do not describe this as overload shedding
or prompt release of sockets, workers, or provider capacity.

## Suggested implementation stack

### PR 1: schemas, normalization, and retry reducer

Amend terms and contracts, define the timeout taxonomy, bounded normalized
evidence, call state, retry instruction, exact standard policy, identities, and
the pure transition operation. Use scripted invocations only.

### PR 2: HTTP lifecycle and provider routes

Add the long-lived bounded client, native timeout handling, request/response
byte enforcement, explicit close lifecycle, and route all supported providers
through the same invocation contract.

### PR 3: local driver and downstream handoff evidence

Add the thin synchronous driver, controlled cancellation-aware wait boundary,
and serialize/restore tests that demonstrate how a durable driver consumes the
same transition outputs. Release before downstream packages delete duplicate
retry paths.

## Validation evidence

Use explicit gates and controlled waits, never sleeps, to prove timeout,
cancellation, close, and transition behavior. The implementation handoff must
include:

- a two-round serialize/restore trace through the state reducer;
- proof that an uncontained deadline starts no retry;
- proof that many invocations reuse one HTTP client and bounded pool;
- request/response limit tests for success and error bodies;
- serialized-size checks showing no request/failure duplication per retry;
- exact standard-policy matrix coverage, including terminal rate limiting; and
- live smoke coverage for every supported route through the same lifecycle.
