~/drotherm/repos/dr-providers (lower-python-to-3.12)
↑39k ↓2.1k R163k CH97.6% $0.339 (sub) 10.6%/272k (auto)                (openai-codex) gpt-5.5 • thinking off
done — 270 tok/s

delete: placeholder marimo notebook + `notebooks` extra + nbs ruff/ty config. Replacement: nothing until a real notebook exists. [nbs/hello_world.py, pyproject.toml]

stdlib: `tenacity` for one POST retry. Replacement: tiny bounded `for range(3)` loop around `httpx` exceptions, or no retry if callers own retry policy. [src/dr_providers/query/transport.py, pyproject.toml]

yagni: `ProviderTransport` ABC with one concrete implementation. Replacement: make `ApiProvider` the base/concrete class directly, or just `OpenRouterProvider`. [src/dr_providers/query/transport.py]

yagni: provider availability/executable/structured-output machinery. Replacement: keep only API-key resolution; no executables or structured-output flags are used. [src/dr_providers/query/transport_config.py]

yagni: `RequestControls`, `ReasoningWarning`, `warnings`, and `extra_body` plumbing for one reasoning payload. Replacement: inline reasoning payload into `LlmRequest.json_payload()`. [src/dr_providers/query/reasoning.py, src/dr_providers/query/request.py, src/dr_providers/query/response.py]

delete: unused `LlmConfig` public model. Replacement: pass `LlmRequest` directly; it already contains the same config fields. [src/dr_providers/config.py, src/dr_providers/__init__.py]

delete: unused `CallError`. Replacement: nothing. [src/dr_providers/query/response.py]

delete: dead enums `ControlRequestStyle` and `OpenRouterReasoningKey` (`TODO: remove this`). Replacement: nothing. [src/dr_providers/names.py]

shrink: `ProviderName` custom `StrEnum.__new__` only stores one base URL. Replacement: plain enum plus `OPENROUTER_API_BASE_URL` constant, or just constant. [src/dr_providers/names.py]

shrink: `scripts/query_provider.py` wrapper. Replacement: call `uv run python -m dr_providers.cli` from README/live script. [scripts/query_provider.py]

net: ~-250 lines, -2 deps possible.
