# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- No-throw `HttpProvider`: expected outcomes return a closed
  `ProviderTransportResponse | ProviderTransportFailure` instead of raising,
  with idle-stall and overall wall-clock (per-invocation deadline) timeout
  enforcement.
- Provider presets for OpenAI (`chat_completions` and `responses`), Anthropic
  (`anthropic_messages`), OpenRouter, and Gemini, fixing each route's
  protocol, token-limit parameter, and reasoning wire shape.
- Provider Invocation Evidence records binding request + policy identities to
  the outcome and the complete least-processed raw request/response bodies
  (credentials and authorization headers are never persisted).
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
