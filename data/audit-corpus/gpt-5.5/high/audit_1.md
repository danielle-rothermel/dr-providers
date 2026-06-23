~/drotherm/repos/dr-providers (lower-python-to-3.12)
↑55k ↓5.3k R331k CH97.4% $0.602 (sub) 20.0%/272k (auto)                        (openai-codex) gpt-5.5 • high
done — 136 tok/s

delete: Empty generated marimo notebook + notebooks extra. Replacement: nothing; create a notebook when it has content. [nbs/hello_world.py, pyproject.toml]

yagni: Single-provider transport hierarchy (`ProviderTransport` ABC, `ApiProvider`, `providers/` package) for one OpenRouter backend. Replacement: one concrete provider class; add a base when provider #2 lands. [src/dr_providers/query/transport.py, src/dr_providers/query/providers/]

yagni: Availability/executable/structured-output config for unused provider status checks. Replacement: keep only API-key/base-url/timeout config. [src/dr_providers/query/transport_config.py]

delete: Dead models/enums/helpers (`LlmConfig`, `CallError`, `ControlRequestStyle`, `OpenRouterReasoningKey`, `has_sampling_controls`). Replacement: nothing; remove public `LlmConfig` only with an API break. [src/dr_providers/config.py, src/dr_providers/names.py, src/dr_providers/query/response.py, src/dr_providers/query/request.py]

delete: Warning plumbing where no warning is ever produced. Replacement: remove `ReasoningWarning`, `warnings` fields, and pass-through args. [src/dr_providers/query/reasoning.py, src/dr_providers/query/request.py, src/dr_providers/query/response.py]

stdlib: Tenacity for one retrying POST. Replacement: small bounded loop with `time.sleep`/jitter; drop `tenacity`. [src/dr_providers/query/transport.py, pyproject.toml]

shrink: Sampling accessor properties only wrap two optional fields. Replacement: inline `self.sampling.temperature/top_p` in `json_payload`. [src/dr_providers/query/request.py]

delete: Dev wrapper script only calls package main. Replacement: `uv run python -m dr_providers.cli ...`; update live test/README. [scripts/query_provider.py, scripts/live_test_llm_configs.sh, README.md]

delete: `tests/.gitkeep` in a non-empty tests dir. Replacement: nothing. [tests/.gitkeep]

net: -190 lines, -2 deps possible.
