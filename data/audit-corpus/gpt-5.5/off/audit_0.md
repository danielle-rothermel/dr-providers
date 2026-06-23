~/drotherm/repos/dr-providers (lower-python-to-3.12)
↑63k ↓3.5k R217k CH91.6% $0.530 (sub) 11.6%/272k (auto)                (openai-codex) gpt-5.5 • thinking off
done — 445 tok/s

yagni: provider framework for one provider. Inline `ProviderTransport`/`ApiProvider` into `OpenRouterProvider`. [src/dr_providers/query/transport.py, src/dr_providers/query/providers/openrouter.py]

yagni: generic availability/executable/structured-output config nobody uses. Keep only API key/base URL/path. [src/dr_providers/query/transport_config.py]

stdlib: `tenacity` for one retrying POST. Replace with a small bounded loop using `time.sleep`; drop dependency. [src/dr_providers/query/transport.py, pyproject.toml]

delete: placeholder marimo notebook and `notebooks` extra. Replacement: nothing until there is a real notebook. [nbs/hello_world.py, pyproject.toml]

delete: warnings plumbing that never emits warnings. Replacement: add it back when a provider actually warns. [src/dr_providers/query/reasoning.py, src/dr_providers/query/request.py, src/dr_providers/query/response.py]

delete: unused public `LlmConfig`. Replacement: construct `LlmRequest` directly. [src/dr_providers/config.py, src/dr_providers/__init__.py]

delete: `query_from_prompt` helper wraps 6 lines of request construction. Replacement: build `LlmRequest` in the CLI/demo. [src/dr_providers/query/from_prompt.py]

delete: unused `CallError`, `ControlRequestStyle`, `OpenRouterReasoningKey`. Replacement: nothing. [src/dr_providers/query/response.py, src/dr_providers/names.py]

shrink: `ProviderName.__new__` stores one base URL for one enum member. Replacement: plain enum + module constant. [src/dr_providers/names.py]

delete: `scripts/query_provider.py` wrapper. Replacement: `uv run python -m dr_providers.cli`. [scripts/query_provider.py, README.md]

net: ~-220 lines, -2 deps possible.
