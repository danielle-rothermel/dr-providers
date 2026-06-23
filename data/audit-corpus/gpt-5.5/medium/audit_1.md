~/drotherm/repos/dr-providers (lower-python-to-3.12)
↑63k ↓2.4k R214k CH91.5% $0.496 (sub) 11.5%/272k (auto)                      (openai-codex) gpt-5.5 • medium
done — 163 tok/s

delete: availability framework (`ProviderAvailabilityStatus`, executables, structured-output flags, `availability_status/is_available`). Resolve API key at send time. [src/dr_providers/query/transport_config.py, src/dr_providers/query/transport.py]

yagni: `ProviderTransport` ABC with one implementation. Make `ApiProvider` the concrete base. [src/dr_providers/query/transport.py]

delete: empty warning plumbing (`ReasoningWarning`, `warnings` on controls/request/response). Add back when a provider emits real warnings. [src/dr_providers/query/reasoning.py, src/dr_providers/query/request.py, src/dr_providers/query/response.py]

delete: unused public `LlmConfig`. Request already carries the same fields. [src/dr_providers/config.py, src/dr_providers/__init__.py]

stdlib: `tenacity` for one retrying POST. Replace with a tiny bounded loop around `httpx` exceptions if you want fewer deps. [src/dr_providers/query/transport.py, pyproject.toml]

delete: unused `CallError`. Replacement: nothing. [src/dr_providers/query/response.py]

delete: unused `ControlRequestStyle` and TODO `OpenRouterReasoningKey`. Replacement: literals already live in one function. [src/dr_providers/names.py]

shrink: sampling proxy properties (`has_sampling_controls`, `sampling_temperature`, `sampling_top_p`). Inline `self.sampling.temperature/top_p`. [src/dr_providers/query/request.py]

delete: blank marimo notebook + `notebooks` optional dependency. Add when there’s a real notebook. [nbs/hello_world.py, pyproject.toml]

shrink: `scripts/query_provider.py` wrapper. Use `uv run python -m dr_providers.cli`. [scripts/query_provider.py, README.md, scripts/live_test_llm_configs.sh]

delete: ignored build/cache artifacts in working tree. Replacement: nothing. [dist/, .cache/, .pytest_cache/, .ruff_cache/, **/__pycache__/]

net: ~-230 lines, -2 deps possible.
