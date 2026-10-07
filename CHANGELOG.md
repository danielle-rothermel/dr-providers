# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.4] - Unreleased

### Added

- Add `ProviderCallKind` (`generate`, `score`), `ProviderGenerateRequest`,
  and `ProviderScoreRequest`. Score requests carry exact context, ordered
  continuations, and an identity-bearing flag requiring per-token values.
- Add `ContinuationScore`, `ProviderScoreResponse`, `is_score_response`, and
  the kind-specific request schema names and version constants.
- Invocation evidence carries required `kind` and optional `score_response`;
  exactly one generate response, score response, or failure is retained.
  Score responses classify as success without semantic classification.
- `ScriptedOutcome.scores` and `ScriptedProvider` support scoring, including
  expected failures and the shared retry lifecycle. HTTP scoring raises before
  admission; the CLI remains generate-only. No production scoring backend,
  torch, transformers, or new runtime dependency is added.
- Add permanent validation failure codes `no_supported_kinds`,
  `unsupported_call_kind`, `score_request_rejects_controls`, and
  `score_request_rejects_extensions`.

- Add identity-bearing `ProviderCallDefinition.prompt_rendering` and the
  exported `PromptRendering` enum: `role_messages` (the default) preserves
  existing protocol bodies; `flat_text` concatenates every transcript
  message's content in order with no inserted separator and sends it as one
  user message. Responses carries that message in `input` without
  `instructions`; Anthropic Messages omits `system`. Controls, reasoning
  mappings, and extensions are unchanged. Callers own all whitespace.
- Add the `prompt_rendering` keyword to all five preset factories and the
  `--prompt-rendering` CLI option, both defaulting to `role_messages`.
  Transcript identity retains its original roles; rendering participates in
  request identity through the definition and config hash references.

### Changed

- Replace the concrete `ProviderCallRequest` with a discriminated union of
  `ProviderGenerateRequest` and `ProviderScoreRequest`. Generation callers
  construct `ProviderGenerateRequest`. Remove `PROVIDER_CALL_REQUEST_SCHEMA`
  and `PROVIDER_CALL_REQUEST_SCHEMA_VERSION`; the two request kinds each own
  a named version-1 identity schema. No aliases or compatibility shims remain.
- Require nonempty `ProviderCallDefinition.supported_kinds`, sorted in JSON
  and identity. HTTP presets declare generate only. Requests refuse undeclared
  kinds; scoring also refuses assigned controls and nonempty extensions.
  Definitions used for scoring therefore need empty `required_controls`.
- Generate request identity includes `kind`; score identity includes kind,
  config hash, exact context, ordered continuations, and `token_logprobs`.
  Definition, config, request, call, evidence, observation, record, and result
  hashes change. Config schema stays at 2, provider call at 2, and retry policy
  at 1 because those identity payload shapes are unchanged.

- Rename identity-hash reference fields and payload keys as a persisted-format
  hard cutover, including corresponding keyword parameters and exports:

  | Old | New |
  | --- | --- |
  | `request_identity_hash` | `request_hash` |
  | `retry_policy_identity_hash` | `retry_policy_hash` |
  | `call_identity_hash` | `call_hash` |
  | `evidence_identity_hash` | `evidence_hash` |
  | `definition_identity_hash` (config identity payload) | `definition_hash` |
  | `config_identity_hash` (request identity payload) | `config_hash` |
  | `provider_call_identity_hash()` | `provider_call_hash()` |

  Model `identity_hash` properties, `identity_payload()` and
  `identity_document()` methods, `provider_call_identity_document()`, and
  evidence `policy_identity` retain their names. No compatibility aliases
  are provided; recorded payloads with the old format remain historical.
- Advance the identity and persisted lifecycle schemas:

  | Schema | Previous released | Current |
  | --- | --- | --- |
  | Provider call definition | 3 | 5 |
  | Provider call config | 1 | 2 |
  | Provider call request | 1 | Removed; replaced by kind-specific schemas |
  | Provider generate request | — | 1 |
  | Provider score request | — | 1 |
  | Provider invocation evidence | 8 | 10 |
  | Provider call | 1 | 2 |
  | Completed invocation observation | 1 | 3 |
  | Decided invocation record | 1 | 3 |
  | Provider call state | 1 | 3 |
  | Provider retry instruction | 1 | 3 |
  | Provider call result | 1 | 3 |
  | Provider call retry policy | 1 | 1 |

  Definition identity includes prompt rendering and supported call kinds;
  evidence records the request kind and an exclusive generate response, score
  response, or failure. Lifecycle schemas advance for renamed reference keys
  and changed nested persisted shapes.
  Retry policy schema remains at 1 and retry-policy hashes are unchanged.
- Every definition, config, request, call, completed-observation,
  decided-record, result, and evidence hash changes, including with the
  default rendering. State records have a changed schema and embedded
  hashes, with no separate state identity hash. Existing `role_messages`
  wire bodies and the committed wire corpus are unchanged.
- Bump the package version to 0.3.4 and refresh `uv.lock`. This version is
  unreleased: no tag, GitHub release, or PyPI publication is created.

## [0.3.3] - 2026-10-06

### Added

- Add first-class `RequestControl.VERBOSITY`, the `Verbosity` enum (`low`,
  `medium`, `high`), and `GenerationControls.verbosity`. Verbosity
  participates in config identity when set, uses the single top-level wire
  key `verbosity`, and is reserved against `extra_body` smuggling. OpenAI
  Chat Completions and OpenRouter advertise it; Gemini OpenAI-compat,
  Anthropic Messages, and OpenAI Responses (which nests verbosity under
  `text`) do not. Definitions that advertise verbosity on Responses or
  Anthropic Messages raise `verbosity_protocol_unsupported`.
- Add `--verbosity` to the `query` CLI command.

### Changed

- The `openai.chat_completions` and `openrouter.chat_completions` preset
  definitions now advertise verbosity, so their definition identity hashes
  and every downstream config, request, call, and evidence hash change.
  Definition and evidence schema versions are unchanged.

## [0.3.2] - 2026-08-20

### Added

- Add first-class `RequestControl.SEED` and `GenerationControls.seed`. Seed
  participates in config identity when set, uses the single wire key `seed`,
  and is reserved against `extra_body` smuggling. OpenAI Chat Completions,
  OpenRouter, and Gemini OpenAI-compat advertise it; Anthropic Messages and
  OpenAI Responses do not. Definitions that advertise seed on a protocol
  with no seed wire key raise `seed_protocol_unsupported`.
- Record `system_fingerprint` on `ProviderTransportResponse` when an
  OpenAI-protocol body supplies it, so seed-plus-fingerprint audit can read
  a typed field rather than scraping `response_body`.

### Removed

- Remove `allow_unsupported_control_drop`. Construction is unconditionally
  loud: any set control the definition cannot transport raises
  `ControlValidationError(code="unsupported_control")`. A constructable
  `ProviderCallConfig` transports every control it carries; there is no drop
  mode.

### Changed

- Advance provider-call definition schema 2 to 3 because the constraints
  identity payload no longer carries a drop-mode flag. Every definition and
  config identity hash changes; recorded evidence keyed by old hashes is
  historical.
- Translate every set control unconditionally. Validation is the sole
  enforcement point that a config cannot carry an unsupported control.
- Advance Provider Invocation Evidence to schema version 8: transport
  responses persist `system_fingerprint`. Every success-response evidence
  identity hash changes.

## [0.3.1] - 2026-08-12

### Added

- Add `run_local_provider_call_async` and the `OffloadingProvider` structural
  type: the async entry point offloads the synchronous driver onto
  `HttpProvider.offload()`.
- Add `HttpProvider.offload()`, backed by a provider-owned executor created on
  first use with `policy.max_connections` workers; `close()` drains offloaded
  work before draining active invocations and closing the client, and an
  interrupted close still reaches the terminal closed state and releases the
  executor and client without completing the drain.
- Pin every persisted enum value, failure code, wire-format prefix, identity
  schema name, and schema version with golden tests against hand-written
  literals, and record the pinning rule as a standing contract.
- Add `ScriptedProvider.offload()` and `close()`, so the shipped network-free
  testing surface satisfies the `OffloadingProvider` surface and drives the
  asynchronous entry point directly. The executor is a single owned worker
  created on first offload, keeping scripted offloaded calls serial.
- Add `docs/future-features.md`, recording directions this package
  deliberately does not build today, linked from the README.

### Removed

- Remove the public `classify_httpx_error` export; wire-error classification is
  now the internal kind-to-code boundary mapping.
- Remove the unused `httpx2` development dependency.
- Remove `TIMEOUT_KINDS`; the wire-failure boundary matches timeout kinds
  structurally instead.
- Remove `httpx` as a runtime dependency; it remains a dev dependency for
  `MockTransport`-based tests.
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

- Rename `dr_providers.transport.httpx_errors` to
  `dr_providers.transport.wire_failures`; the module maps wire failure kinds and
  no longer touches httpx. Every literal it defines is unchanged.
- Dispatch the wire-failure boundary on the structured `WireFailureKind` rather
  than on persisted code strings, and record an explicit `REQUEST_TOO_LARGE`
  branch shaped like the response-size one.
- Record in the boundary map and its contract that `CONNECT_ERROR` and
  `NETWORK_ERROR` both map to the single literal `transport_error` with
  `TRANSIENT` recoverability, and `UNKNOWN` maps to `transport_error` with
  `UNKNOWN` recoverability, so the collapse is a standing obligation.
- Cap `pydantic` below 3, matching the frozen-package convention.
- Correct the `provider call retry policy`, `provider HTTP request evidence`,
  and `provider transport policy` terms: the standard retry policy declares only
  a one-invocation limit, header redaction is a `build()` step rather than a
  type invariant, and transport policy records connect and idle timeouts after
  clamping them to the general timeout.
- Advance Provider Invocation Evidence to schema version 7: `identity_payload()`
  is an explicit projection that excludes transport-failure `traceback` and
  `message` from the hash preimage. Both fields remain fully persisted on the
  model; excluding them keeps one provider behavior's identity stable across
  machines and lets a summary message be reworded without rehashing recorded
  evidence.
- Scrub the user home-directory prefix to `~` in captured transport-failure
  tracebacks when evidence is built, mirroring header redaction: deserializing
  third-party evidence does not re-scrub it.
- Report an offloaded provider call that raises after its awaiting task was
  cancelled, at `ERROR` on the `dr_providers.lifecycle.driver` logger, so a
  systematic driver defect stays visible in an unattended run.
- Depend on `dr-wire` for the bounded HTTP client core. `HttpProvider` now
  composes `dr_wire.BoundedHttpClient`, which owns the lifecycle state machine,
  the offload executor, connection-pool and byte bounds, native timeout phases,
  and wire-error detection. `dr_providers` no longer imports `httpx` anywhere
  in `src/`.
- Admit one whole invocation through the client rather than one wire call, so a
  clean close drains complete invocations and their evidence.
- Map `dr_wire.WireFailureKind` to persisted recoverability and failure codes at
  one exhaustive boundary; every recorded literal is unchanged.
- Bound `Retry-After` hints on top of the wire boundary's uncapped parse with
  one rule applied wherever a hint arrives, so retained
  `ProviderRetryAfterHint` values keep their existing bounds — an oversized raw
  header is refused whether the body was read or refused for size.
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
- Classify blank generation text by stop reason on every protocol: token-limit
  truncation is truncated-no-text, a genuine stop is empty-generation, and only
  an absent stop signal remains missing-generation-text.
- Classify success-status provider error envelopes with absent, empty, or
  text-free containers as provider rejections on chat-completions and Anthropic
  Messages; an envelope alongside real generation text remains a success.
- Validate `base_url` before dispatch and classify undispatchable URLs as
  never-sent failures carrying no HTTP request evidence, instead of letting
  `httpx` raise an unclassified `ValueError` out of `invoke()`.
- Classify HTTP 413 as resource exhaustion and remote protocol errors as
  transient; local protocol errors remain permanent.
- Name redirect statuses (301, 302, 303, 307, 308) with dedicated
  `http_redirect_*` codes and pool-acquisition timeouts with `pool_timeout`, in
  the wire path and the exported `classify_httpx_error` alike.
- Require `containment` on transport failures that carry timeout codes.

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
