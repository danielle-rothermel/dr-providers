# Changelog

All notable changes to this project will be documented in this file.

## 0.2.0

Breaking cleanup release: the kernel is now the package.

- Removed the entire v0.1.x `query` API (`dr_providers.query`, `config.py`,
  `names.py`, `cli.py`) with no compatibility shims.
- Flattened `dr_providers.kernel` into the package root; `import
  dr_providers` is the single canonical Python surface.
- Reasoning is now a `ReasoningEffort` enum with per-config wire shapes
  (`reasoning` object vs. `reasoning_effort` field) instead of a free-form
  spec.
- Added `top_p` as a first-class request control.
- Every `Provider.complete()` applies the conformance contract
  (reasoning-not-observed, token-limit-exceeded, model-substitution
  warnings) uniformly.
- `HttpProvider` is a context manager with an explicit transport
  lifecycle; owned clients close on exit.

## 0.1.2

Breaking cleanup release.

- Made `LlmRequest` pure request data; `ApiProvider` now builds endpoint, headers, idempotency key, and JSON payload internally.
- Removed `LlmRequest.prepare()`, `endpoint()`, `headers()`, and `json_payload()`.
- Removed unused provider availability and capability config surface.
- Removed the `ProviderTransport` ABC while keeping public `ApiProvider` for upcoming providers.
- Simplified `ProviderName`; OpenRouter API constants now live beside the enum.

## 0.1.1

Breaking cleanup release.

- Removed public `LlmConfig`; construct `LlmRequest` directly.
- Removed public `RequestControls` and `ReasoningWarning`; reasoning payloads are built internally by `LlmRequest.prepare()`.
- Removed `LlmRequest.prepare(..., controls=...)`; call `prepare(config)`.
- Removed unused request/response warning fields and the `warnings=` argument from `llm_response_from_http()`.
- Removed the redundant `scripts/query_provider.py` wrapper; use `uv run python -m dr_providers.cli`.

## 0.1.0

Initial release.

- OpenRouter provider via `OpenRouterProvider` and generic `ApiProvider` transport
- Unified `LlmRequest` with `prepare()`, `endpoint()`, `headers()`, and `json_payload()`
- Minimal chat-completions response parsing into `LlmResponse`
- Typed config models: `ReasoningSpec`, `SamplingControls`, `LlmConfig`
- Module layout: `errors`, `transport_config`, `providers/openrouter`, `from_prompt`
- Public API exported from top-level `dr_providers` package
- Optional CLI extra: `pip install dr-providers[cli]` provides `query-provider`
