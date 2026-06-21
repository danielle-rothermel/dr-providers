# Changelog

All notable changes to this project will be documented in this file.

## 0.1.0

Initial release.

- OpenRouter provider via `OpenRouterProvider` and generic `ApiProvider` transport
- Unified `LlmRequest` with `prepare()`, `endpoint()`, `headers()`, and `json_payload()`
- Minimal chat-completions response parsing into `LlmResponse`
- Typed config models: `ReasoningSpec`, `SamplingControls`, `LlmConfig`
- Module layout: `errors`, `transport_config`, `providers/openrouter`, `from_prompt`
- Public API exported from top-level `dr_providers` package
- Optional CLI extra: `pip install dr-providers[cli]` provides `query-provider`
