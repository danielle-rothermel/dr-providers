~/drotherm/repos/dr-providers (lower-python-to-3.12)
↑45k ↓2.6k R204k CH98.1% $0.405 (sub) 12.8%/272k (auto)                      (openai-codex) gpt-5.5 • medium
done — 109 tok/s

yagni cut `ProviderTransport` ABC; `ApiProvider` is the only implementation. Use concrete class directly. [src/dr_providers/query/transport.py]

yagni cut provider availability/executable/structured-output config; only API-key availability is real today. Replace with `resolve_api_key`. [src/dr_providers/query/transport_config.py, src/dr_providers/query/transport.py]

delete empty marimo notebook + `notebooks` extra. Replacement: nothing until there’s a real notebook. [nbs/hello_world.py, pyproject.toml]

delete unused response/warning plumbing; no code populates warnings. Replacement: add it back when a provider emits warnings. [src/dr_providers/query/reasoning.py, request.py, response.py]

delete unused public `LlmConfig`; callers use `LlmRequest`/`query_from_prompt`. Replacement: nothing. [src/dr_providers/config.py, __init__.py]

delete `CallError`; never referenced. Replacement: nothing. [src/dr_providers/query/response.py]

delete unused `ControlRequestStyle` and TODO `OpenRouterReasoningKey`. Replacement: nothing. [src/dr_providers/names.py]

shrink one-provider enum metadata; custom `ProviderName.__new__` is overkill. Replace with constants or a simple property. [src/dr_providers/names.py]

stdlib cut `HttpStatusCode` enum. Replace with `http.HTTPStatus`. [src/dr_providers/query/response.py]

shrink `tenacity` for one retry decorator if dependency count matters. Replace with a tiny bounded `for`/`time.sleep` loop. [src/dr_providers/query/transport.py, pyproject.toml]

net: -150 lines, -2 deps possible.
