~/drotherm/repos/dr-providers (lower-python-to-3.12)
↑41k ↓2.8k R236k CH98.7% $0.406 (sub) 10.8%/272k (auto)                      (openai-codex) gpt-5.5 • medium
done — 104 tok/s

yagni: generic provider stack for one provider (`ProviderTransport` ABC, provider package shim, `ProviderName` base-url machinery). Inline into `ApiProvider`/`OpenRouterProvider` until a second provider exists. [src/dr_providers/query/transport.py, src/dr_providers/query/providers/*, src/dr_providers/names.py]

delete: availability/structured-output/executable probing with no current provider using it. Keep only API-key resolution. [src/dr_providers/query/transport_config.py]

yagni: warnings pipeline never emits warnings (`ReasoningWarning`, `RequestControls.warnings`, request/response warning plumbing). Add back when a real warning exists. [src/dr_providers/query/reasoning.py, src/dr_providers/query/request.py, src/dr_providers/query/response.py]

stdlib: `tenacity` for one HTTP retry. Replace with a tiny three-attempt loop using `time.sleep`; drop dependency. [pyproject.toml, src/dr_providers/query/transport.py]

delete: unused public-ish config/model leftovers: `LlmConfig`, `ControlRequestStyle`, `OpenRouterReasoningKey`, `CallError`, `has_sampling_controls`. Replacement: nothing. [src/dr_providers/config.py, src/dr_providers/names.py, src/dr_providers/query/response.py, src/dr_providers/query/request.py]

delete: blank generated marimo notebook plus optional `notebooks` extra. Replacement: create notebooks when needed. [nbs/hello_world.py, pyproject.toml]

shrink: `scripts/query_provider.py` only delegates to `dr_providers.cli.main`. Use `uv run python -m dr_providers.cli ...`. [scripts/query_provider.py, README.md, scripts/live_test_llm_configs.sh]

net: ~-180 lines, -2 deps possible.
