# Potential future features

This document records directions dr-providers deliberately does not build
today. Nothing here is planned work or a commitment. Each entry names the
need that would justify building it and what the current design already
guarantees, so a future decision starts from the constraints rather than
rediscovering them.

The package's frozen surface is deliberately small. A future need may add any
of the following; until one does, the absence is a decision, not an oversight.

## Native async transport

A future need may add a native asynchronous transport built on
`httpx.AsyncClient`, so provider calls reach the wire without a worker thread.

This is a recorded non-build. Asynchronous access today is thread offload:
`run_local_provider_call_async` submits the unchanged synchronous driver to the
provider's executor and awaits its future. That keeps exactly one
implementation of the retry, classification, and evidence logic, and one
bounded HTTP client to reason about. A native async path would add a second
wire implementation whose bounds, timeout phases, and failure taxonomy must be
proven to match the synchronous one, and it earns that cost only when offload
thread count is a demonstrated bottleneck rather than a theoretical one.

## Opt-in auto-retry operator policy

A future need may add an operator-level retry policy that consumes the
`Retry-After` hint and `RecoverabilityClass` already recorded on evidence, and
chooses its own delays.

The pieces exist and stay unread on purpose. `ProviderRetryAfterHint` is
recorded as bounded evidence, and recoverability is classified per invocation,
but no policy consults either: a provider call retry policy declares its delays
up front so that a recorded call's delay schedule is a property of its declared
identity rather than of what a server asked for at run time. An auto-retry
policy reading server hints makes replay non-deterministic, so it belongs
outside the identity-bearing policy, in an operator layer that decides how much
to trust a provider's pacing request.

## Anthropic `refusal` stop reason as a typed signal

A future need may promote the Anthropic `refusal` stop reason to a typed
signal, distinct from the protocol stop reasons already modeled.

`ProviderStopReason` is a closed persisted enum of `stop`, `length`, and
`content_filter`. Adding a member is a recorded-data format change: it needs a
golden-pin update and a decision about whether existing consumers reading three
values should see a fourth, or whether a refusal is better carried as a
conformance warning beside an existing stop reason.

## Structured observability surface

A future need may add a structured observability surface — metrics, spans, or a
callback protocol — beyond the single logging hook this package has today.

That hook is deliberate and singular: the driver reports an offloaded provider
call that raises after its awaiting task was cancelled, because no caller
remains to receive it. Every other failure is a value, either recorded evidence
or a raised exception. A general observability surface changes that discipline,
so it needs a design pass covering what is emitted, at which boundary, and how
a consumer opts in — not a second logger added where one is convenient.

## Library-level variance and sampling helper

A future need may add a helper for repeated sampling of one provider call and
the variance across its results.

Nothing in this package prevents a caller from invoking one request many times;
a helper would standardize how the resulting evidence is grouped and compared.
It stays out of the frozen surface until the grouping and comparison a consumer
actually wants are known, because a premature shape here would be harder to
correct than its absence.

## dr-http surface candidates

These belong to [dr-http](https://github.com/danielle-rothermel/dr-http), which
owns the wire capability. dr-providers records them because it is the consumer
that would need them.

- **Multi-valued response headers.** `WireResponse` exposes headers as a
  single-valued mapping. A provider sending repeated headers that matter — a
  second `Retry-After`, or repeated rate-limit headers — is not representable
  today.
- **Typed client-factory protocol.** The client factory seam is currently an
  untyped callable, which tests use to inject a transport. A precise protocol
  would let a consumer implement it against a checked contract.
- **`HttpClientConfig` phase-timeout validator.** A configuration whose connect
  or idle timeout exceeds its general timeout is accepted by the client today;
  dr-providers avoids it by clamping in its transport policy before construction.
  Moving that check into the client would make the invariant hold for every
  consumer. **This one needs ratification before it is built:** clamping and
  rejecting are different contracts, and dr-providers depends on clamping
  because the clamped value is what its recorded policy identity commits to.
- **Bounded incremental decompression.** The response byte cap bounds retained
  decoded bytes and stops at the first crossing chunk, but httpx decodes each
  raw transport chunk fully before the bound sees decoded output, so transient
  decode memory for one raw chunk can exceed the cap by up to the codec's
  ratio. Reading raw bytes through a bounded incremental decoder would close
  that gap; it means reimplementing response decoding, and the providers this
  package calls are not an adversarial-decompression threat model, so it is
  recorded rather than built.
- **Cleanup-error-tolerant complete reads.** A determined wire result (a size
  refusal, or a response whose body finished) survives an error raised while
  the stream closes afterward. A stream that fails during its final
  exhaustion, where httpx entangles the trailing-chunk flush with the close,
  still reports as a wire failure even when nearly all bytes arrived —
  recovering the complete body there requires decoding raw chunks in this
  package's own frame via httpx internals, which the frozen wire core
  deliberately avoids.
