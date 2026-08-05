# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Keep ordinary live pytest verification read-only; live corpus replacement is
  now an explicit complete-capture validation, redaction, and promotion step.
- Saturate socket and watchdog waits at the platform timeout ceiling while
  preserving requested finite timeout policy values, preventing large values
  from raising `OverflowError` during provider invocation.

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
