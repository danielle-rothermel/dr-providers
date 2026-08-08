# Logical Provider-Call Lifecycle

Status: design plan for local refinement; implementation has not started.

## Purpose

Make `dr-providers` the single owner of the generic lifecycle for one logical
model-provider call: provider invocations, response classification, bounded
retries, and terminal evidence.

This moves provider-specific resilience out of research applications while
leaving evaluation policy and durable workflow scheduling with their proper
owners.

## Existing foundation

`dr-providers` already owns provider request configuration, transport behavior,
and invocation evidence. Research callers currently add a second semantic
retry/classification layer, while HTTP transport may also retry natively. That
split makes invocation and wire-request counts, evidence, and failure meaning
difficult to audit.

The intended hard boundary is:

- one **provider invocation** performs at most one provider wire request and
  yields one complete invocation-evidence record;
- one **logical provider call** applies a declared policy to an ordered
  sequence of provider invocations and yields one terminal result; and
- a research application maps the terminal result to its own evaluation row,
  reward, or experiment decision.

## Intended ownership

### Provider invocation

A provider invocation may fail before sending a provider wire request;
otherwise it performs exactly one observable wire request. Its evidence should
retain the sanitized request identity, provider/model route, timing and usage
data when available, response or failure classification, and the transport
diagnostics needed for replay and debugging.

The local design should hard-cut any implicit HTTP retry behavior that would
hide multiple wire requests inside one invocation. If a transport library
cannot disable such behavior, the public guarantee and evidence shape must be
reconsidered before implementation proceeds.

### Logical call executor

Add one small logical-call executor that accepts:

- a provider invocation operation;
- a frozen retry policy with an explicit identity;
- a semantic response classifier; and
- an injectable wait/clock boundary suitable for deterministic tests and
  durable callers.

It returns a closed, serializable result containing ordered invocation evidence
and corresponding outcomes and retry decisions, plus exactly one logical
provider call outcome. The result must distinguish at least:

- success;
- blank response;
- malformed response;
- semantically rejected response;
- transient provider or network failure;
- rate limiting;
- timeout; and
- exhausted policy.

The final taxonomy and which provider invocation outcomes are retryable belong
here because they describe provider-call behavior. The interpretation of
success content and its value to an evaluation remain outside this package.

Retry schedules must be bounded and explicit. Backoff and jitter choices must
be reproducible in evidence without making wall-clock timing part of request
identity. Cancellation and interruption must preserve every completed
invocation.

### Identity and redaction

Logical-call identity must be derived from stable, credential-free inputs. It
must be possible to distinguish the request, route, policy, and implementation
version without persisting API keys, authorization headers, or raw secret
configuration.

The exact persisted keys and discriminators are wire-format contracts and need
golden tests.

## Explicit non-goals

This foundation does not own:

- workflow scheduling, durable timers, fan-out, fan-in, or run membership;
- global provider admission control or adaptive account-wide rate limiting;
- evaluation task selection, candidate ranking, reward, or statistical power;
- result caching or artifact storage;
- exactly-once external provider effects; or
- application-specific parsing beyond the generic semantic classifier
  boundary.

`dr-platform` may durably invoke this operation later, but it should not learn
provider retry semantics. Research applications may supply semantic
classification rules, but they should not reimplement the invocation
lifecycle.

## Design questions to finalize locally

1. What is the closed logical provider call outcome model, and which provider
   invocation outcomes are retryable by default versus only by explicit
   policy?
2. Is the semantic classifier a protocol, a closed configuration, or a narrow
   callable boundary, and what evidence from it is serializable?
3. What exactly replaces the current native HTTP retry configuration in the
   hard cutover?
4. How are server-provided retry delays combined with policy backoff and
   deterministic jitter?
5. How does cancellation surface when it occurs before, during, or between
   invocations?
6. Which raw response fields are retained, bounded, redacted, or externalized
   as artifacts?
7. Does the executor perform waiting itself, or return a resumable
   next-invocation decision for a caller that owns durable waiting?

The last question should be resolved against the smallest stable provider
contract, without importing a particular workflow engine into this package.

## Implementation sequence

1. Inventory every existing transport and logical retry path and establish one
   wire-request contract.
2. Freeze invocation and logical-result schemas, taxonomy, and identities.
3. Implement the logical executor over an injected scripted transport.
4. Convert concrete OpenAI/OpenRouter routes to the same provider-invocation
   contract.
5. Delete duplicate native or application-level lifecycle paths in the same
   hard-cutover stack where their consumers are controlled.
6. Update public exports, terminology, contracts, and package documentation.
7. Release a pinned version before downstream migration.

## Validation bar

- Scripted transports prove exact invocation order and logical-call termination
  for every provider invocation outcome.
- One provider invocation always corresponds to one invocation-evidence record
  and at most one provider wire request.
- Cardinality, timeout, interruption, and retry-exhaustion behavior are exact.
- Tests control interleavings and clocks explicitly; elapsed time is only a
  watchdog.
- All representations and persisted evidence pass credential-redaction tests.
- OpenAI and OpenRouter integrations conform to the same lifecycle contract.
- Public docs make no exactly-once or global rate-control claim.

## Downstream handoff

After release, inspect the final lifecycle API before planning `dr-platform`,
`dr-code`, or Whetstone changes. Downstream code should delete its generic
provider retry/classification machinery and retain only domain response
interpretation and evaluation-row construction.
