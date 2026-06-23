~/drotherm/repos/dr-providers (lower-python-to-3.12)
↑37k ↓1.7k R108k CH96.9% $0.292 (sub) 8.8%/272k (auto)                      (openai-codex) gpt-5.5 • minimal
done — 117 tok/s

yagni: cut `ProviderTransport` ABC; `ApiProvider` is the only implementation, so make it the base directly. [src/dr_providers/query/transport.py]

stdlib: cut `tenacity`; a 3-attempt `for` loop with `time.sleep()` covers the one retry site. [pyproject.toml, src/dr_providers/query/transport.py]

delete: unused `CallError` model. Replacement: nothing. [src/dr_providers/query/response.py]

delete: unused `ControlRequestStyle` and TODO-marked `OpenRouterReasoningKey`. Replacement: nothing. [src/dr_providers/names.py]

yagni: cut executable/supports-structured-output availability plumbing; no provider uses executables and only OpenRouter exists. Keep env var check only if needed. [src/dr_providers/query/transport_config.py]

delete: empty generated marimo notebook and `notebooks` extra if it’s not part of the package story. Replacement: nothing. [nbs/hello_world.py, pyproject.toml]

shrink: inline `sampling_temperature`, `sampling_top_p`, and unused `has_sampling_controls`; the properties add indirection for single-use values. [src/dr_providers/query/request.py]

yagni: `LlmConfig` is exported but unused internally; either wire it into request construction or delete it from public API. [src/dr_providers/config.py]

shrink: `scripts/query_provider.py` only calls `dr_providers.cli.main`; `python -m dr_providers.cli` already exists. Replacement: README command. [scripts/query_provider.py]

net: ~-120 lines, -2 deps possible.
