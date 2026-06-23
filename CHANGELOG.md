# Changelog

All notable changes to this project will be documented in this file.

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
