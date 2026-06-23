~/drotherm/repos/dr-providers (lower-python-to-3.12)
↑52k ↓7.6k R197k CH87.3% $0.588 (sub) 14.3%/272k (auto)                       (openai-codex) gpt-5.5 • xhigh
done — 117 tok/s

yagni: generic multi-provider framework around one OpenRouter implementation. Replace `ProviderTransport`/`ApiProvider`/`ProviderConfig` with one concrete `OpenRouterProvider`; add generic layer when provider #2 lands. [src/dr_providers/query/transport.py, src/dr_providers/query/transport_config.py, src/dr_providers/query/providers/openrouter.py]

yagni: two-phase prepared request state on `LlmRequest`. Keep request as data; build endpoint/headers/payload as locals in the provider. [src/dr_providers/query/request.py]

delete: empty marimo notebook plus `notebooks` extra. Replacement: nothing until a real notebook exists. [nbs/hello_world.py, pyproject.toml]

stdlib: `tenacity` used once for one retry decorator. Replace with a tiny bounded stdlib loop, or caller-owned retry. [src/dr_providers/query/transport.py, pyproject.toml]

shrink: `RequestControls` + `ReasoningWarning` plumbing carries one dict and always-empty warnings. Replace with `reasoning_extra_body(reasoning)`. [src/dr_providers/query/reasoning.py]

delete: unused public `LlmConfig`. Replacement: `LlmRequest` or direct args. [src/dr_providers/config.py, src/dr_providers/__init__.py]

delete: unused `ControlRequestStyle`, `OpenRouterReasoningKey`, and `CallError`. Replacement: nothing. [src/dr_providers/names.py, src/dr_providers/query/response.py]

delete: `scripts/query_provider.py` wrapper. Use `uv run python -m dr_providers.cli ...`. [scripts/query_provider.py]

net: ~-350 lines, -2 deps possible.
