# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add `run_local_provider_call_async` and the `OffloadingProvider` structural
  type: the async entry point offloads the synchronous driver onto
  `HttpProvider.offload()`.
- Add `HttpProvider.offload()`, backed by a provider-owned executor created on
  first use with `policy.max_connections` workers; `close()` drains offloaded
  work before draining active invocations and closing the client.

### Removed

- Remove `dr_providers.surfaces.serve` and the `[serve]` optional extra
  (FastAPI/uvicorn).
- Remove post-build wheel install/smoke verification from CI and release
  workflows.
- Remove the ponytail audit-corpus benchmark fixtures, generator script, and
  tests.
- Remove the six public `DEFAULT_*` transport sizing constants
  (`DEFAULT_TIMEOUT_SECONDS`, `DEFAULT_IDLE_TIMEOUT_SECONDS`,
  `DEFAULT_MAX_CONNECTIONS`, `DEFAULT_MAX_KEEPALIVE_CONNECTIONS`,
  `DEFAULT_MAX_REQUEST_BYTES`, `DEFAULT_MAX_RESPONSE_BYTES`).
- Remove the CLI Anthropic 4096 `max_tokens` fallback.
- Remove unused failure exception hierarchy, `raise_failure`,
  `FAILURE_ERROR_TYPES`, recoverability frozensets, `ProviderFailure.retryable`,
  and public `FailureClass`.

### Changed

- Default the standard provider-call retry policy to one invocation with no
  auto-retry; opt-in retry remains on `CustomProviderCallRetryPolicy`.
- Record `Retry-After` hints as bounded invocation evidence only; remove
  canonical-form re-validation on the hint model.
- Require explicit transport-policy sizing at `ProviderTransportPolicy`
  construction and `policy_for()` (timeouts, connection pool limits, byte caps).
- Require explicit `connect_timeout_seconds` on transport policy; remove the
  hidden 30-second connect cap from `HttpProvider`.
- Rename `FailureClass` to `RecoverabilityClass` and rename serialized field
  `failure_class` to `recoverability` on transport failures and failure
  records.
- Advance Provider Invocation Evidence to schema version 5 (hard cutover).
- Advance Provider Invocation Evidence to schema version 6 with an optional
  `traceback` field on transport failures for wire-path httpx exceptions.
- Remove the 256-character failure message cap on provider-constructed transport
  failures.
- Remove `ProviderFailureError.underlying`; raised validation errors retain the
  original cause through normal exception chaining only.
- Classify wire-path httpx errors with `classify_httpx_error` instead of
  treating every non-timeout `HTTPError` as transient.
- Use static transport failure summary messages with `metadata.exception_type`
  for wire-path httpx exceptions; tracebacks carry diagnostic detail.

## [0.3.0] - 2026-08-08

### Added

- Add closed provider-invocation and provider-call outcomes, a caller-identified
  semantic response classifier, serializable provider-call state and retry
  instructions, deterministic retry and cancellation transitions, and a thin
  cancellation-aware local lifecycle driver.
- Export the provider-call lifecycle models, policies, classifiers, transitions,
  identities, and local driver from the stable top-level package surface.
- Publish the provider-call lifecycle vocabulary and standing contracts in the
  repository definitions.

### Changed

- Reject unsupported provider/protocol model routes, bind transport policies to
  one provider kind in identity, and reject route-policy mismatches before
  payload, credential, evidence, or dispatch work.
- Advance Provider Invocation Evidence to schema version 4 for the persisted
  provider-kind transport-policy binding.
- Configure the CLI, live matrix, and README quickstart with one open and one
  keep-alive connection for each single-invocation provider. Configure the
  local server the same way per request-created provider; its limit and
  connection reuse are not server-wide.
- State explicitly that native phase and response-read idle timeouts do not
  impose a total wall-clock deadline on a response that keeps producing bytes.
- Cut the public `Provider` protocol, `HttpProvider`, and `ScriptedProvider` over
  to one evidence-producing `invoke()` operation. CLI, serve, and all five live
  provider routes now execute calls through the same standard local lifecycle.
- Make serve query results retain the complete terminal `ProviderCallResult`,
  including the ordered decided invocation records and their evidence.

### Fixed

- Reject completed invocation observations whose declared failure outcome does
  not exactly match deterministic classification of the embedded evidence.
- Reject custom retry policies with non-finite cumulative delay and split
  accepted large waits into platform-bounded cancellation-aware chunks.

### Removed

- Remove `Provider.complete()`, hidden native transport retries, transport-level
  `retryable` fields, repeated failure request bodies, and the CLI `--retries`
  option. The provider-call retry policy is now the sole retry authority.

## [0.2.2] - 2026-08-05

### Added

- Publish the repository's authoritative terms and contracts as a
  client-rendered GitHub Pages reference.
- Run the same locked repository pre-check from local commits and CI, and
  verify built distributions before release publication.

### Fixed

- Collect every live-provider credential name during corpus promotion and
  redact sensitive URL fragment parameters as well as query parameters.
- Reject credential-bearing URL userinfo before transport policy identity or
  invocation evidence can retain it.
- Honor explicit unsupported-control dropping for Anthropic reasoning while
  continuing to reject unmappable values for supported reasoning controls.
- Reject provider call definitions that advertise reasoning without a usable
  wire mapping.
- Reject contradictory raised-error and carried-record failure
  classifications.
- Preserve internal CLI import failures instead of misreporting them as a
  missing optional dependency.
- Restore the complete BSD-3-Clause disclaimer on the vendored TOML parser
  published by the definitions site.
- Keep ordinary live pytest verification read-only; live corpus replacement is
  now an explicit complete-capture validation, redaction, and promotion step.
- Saturate socket and watchdog waits at the platform timeout ceiling while
  preserving requested finite timeout policy values, preventing large values
  from raising `OverflowError` during provider invocation.
- Reject non-finite generation controls and extension mappings outside strict
  finite JSON during model construction, before identity or HTTP encoding.
- Disable redirects and classify every non-2xx HTTP response as a typed
  transport failure.
- Exercise the FastAPI test client through its current `httpx2` path without
  the deprecated compatibility fallback.

### Changed

- Cut over the implementation to functional-area module paths under
  `dr_providers.modeling`, `dr_providers.translation`,
  `dr_providers.transport`, `dr_providers.outcomes`, `dr_providers.core`, and
  `dr_providers.surfaces`. The top-level `dr_providers` exports remain the
  stable import surface; the removed module paths have no compatibility
  aliases.
- Provider Call Definition and Provider Invocation Evidence are bare payload
  models; their `IdentityDocument` envelopes are the sole owners of `schema`
  and `schema_version`, and both envelopes now use schema version 2. This is a
  hard cutover with no compatibility reader: supplying `schema_version` to
  either payload model is rejected, and Definition, Config, and Request
  identity hashes change.
- Provider transport policies require strict finite positive timeout values
  and a strict nonnegative native retry count at model construction.
- Provider Invocation Evidence v2 names retained HTTP request evidence and
  constructed request and decoded response bodies directly, without
  `raw_*`/`stable_*` terminology or compatibility aliases.
- Require `dr-serialize>=0.1.2,<0.2` for the current identity-envelope
  contracts.
- Qualify and declare Python 3.14 alongside Python 3.12 and 3.13.
- Constrain releases to version-matching tags on merged `main` commits, pin
  release tooling, and isolate trusted PyPI publishing on a protected
  GitHub-hosted job.

### Removed

- Remove the obsolete model-specific live-test shell script; the maintained
  mise-aware live matrix is the provider-level verification path.

## [0.2.1] - 2026-08-05

### Fixed

- Provider definition JSON serialization orders supported controls, required
  controls, and extension keys deterministically while preserving their
  Python-mode frozensets and existing sorted identity payloads.

### Changed

- Require `dr-serialize>=0.1.1,<0.2` for canonical ordering of unordered JSON
  values.

## [0.2.0] - 2026-07-24

Complete rewrite. There is **no API compatibility with 0.1.x**: the legacy
OpenRouter query client has been fully removed and replaced by a typed
provider-call transport kernel. Code written against 0.1.x will not import.

### Added

- Typed Provider Call Definition -> Config -> Request identity, each carrying
  its own full 64-char SHA-256 Identity Hash via `dr-serialize`
  canonicalization.
- `HttpProvider` returns expected outcomes as a closed
  `ProviderTransportResponse | ProviderTransportFailure` instead of raising,
  while unexpected programming or infrastructure errors may still raise.
  Idle-stall and watchdog deadlines bound each native attempt's caller-visible
  wait; a caller-injected synchronous client can leave a daemon worker and
  socket lingering until the caller-owned operation eventually ends.
- Provider presets for OpenAI (`chat_completions` and `responses`), Anthropic
  (`anthropic_messages`), OpenRouter, and Gemini, fixing each route's
  protocol, token-limit parameter, and reasoning wire shape.
- Provider Invocation Evidence records binding request + policy identities to
  the outcome, structured request metadata, the constructed JSON request-body
  mapping, and response bodies decoded as JSON when possible or retained as
  text otherwise (the standard `HttpProvider` path redacts known credential
  header names before binding request evidence; direct raw-request inputs
  remain trusted).
- `ScriptedProvider` for network-free testing against the same `Provider`
  interface.
- `dr-providers` console script and a typer CLI in the optional `[cli]` extra;
  an optional FastAPI facade in the `[serve]` extra.
- Published vocabulary sheet documenting the provider-call transport contract.

## [0.1.1] - Legacy

- Legacy OpenRouter LLM query client (superseded by the 0.2.0 rewrite).

## [0.1.0] - Legacy

- Initial release: legacy OpenRouter LLM query client (superseded by the
  0.2.0 rewrite).
