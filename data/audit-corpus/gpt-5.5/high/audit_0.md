~/drotherm/repos/dr-providers (lower-python-to-3.12)
↑57k ↓5.4k R137k CH93.3% $0.515 (sub) 13.1%/272k (auto)                        (openai-codex) gpt-5.5 • high
done — 111 tok/s

yagni generic provider stack for one provider. Replace `ProviderTransport`/`ApiProvider`/`ProviderConfig` with one concrete `OpenRouterProvider` plus constants. `[src/dr_providers/query/transport.py, src/dr_providers/query/transport_config.py]`

shrink prepared-request state on `LlmRequest`. Build URL/headers/idempotency locally in provider instead of storing `base_url/api_key/chat_path/extra_body/warnings` on the model. `[src/dr_providers/query/request.py]`

yagni `RequestControls`, `ReasoningWarning`, and warnings plumbing that always stays empty. Replace with `_reasoning_extra_body(reasoning) -> dict`. `[src/dr_providers/query/reasoning.py]`

delete empty generated marimo notebook. Replacement: create a notebook when there’s real notebook content. `[nbs/hello_world.py, pyproject.toml]`

shrink `tenacity` for one HTTP retry decorator. Replace with a tiny bounded loop around `httpx` transport exceptions. `[src/dr_providers/query/transport.py, pyproject.toml]`

delete unused `LlmConfig`. Replacement: construct `LlmRequest` directly. `[src/dr_providers/config.py]`

delete unused `ControlRequestStyle`, `OpenRouterReasoningKey`, and `CallError`. Replacement: nothing. `[src/dr_providers/names.py, src/dr_providers/query/response.py]`

stdlib custom `HttpStatusCode` enum. Replace with `http.HTTPStatus` or local literals. `[src/dr_providers/query/response.py]`

shrink sampling convenience properties used once. Inline `self.sampling.temperature/top_p` in payload building. `[src/dr_providers/query/request.py]`

delete CLI wrapper script. Replacement: `uv run python -m dr_providers.cli ...`. `[scripts/query_provider.py, README.md]`

net: ~-230 lines, -2 deps possible.
