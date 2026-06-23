~/drotherm/repos/dr-providers (lower-python-to-3.12)
↑70k ↓7.5k R154k CH47.6% $0.654 (sub) 15.4%/272k (auto)                       (openai-codex) gpt-5.5 • xhigh
done — 140 tok/s

delete: placeholder marimo notebook + notebooks extra. Remove `nbs/hello_world.py` and `marimo[recommended]`. [nbs/hello_world.py, pyproject.toml]

stdlib: Typer for one command. Replace with `argparse`; delete `typer` optional/dev dep. [src/dr_providers/cli.py, pyproject.toml]

yagni: “prepared” `LlmRequest` copy carrying `base_url`, `api_key`, `headers()`, `endpoint()`, `json_payload()`. Build transport details inside `ApiProvider.generate()`. [src/dr_providers/query/request.py, src/dr_providers/query/transport.py]

yagni: provider availability/config surface for one provider. Cut `ProviderAvailabilityStatus`, executables, structured-output flag, availability methods. [src/dr_providers/query/transport_config.py]

yagni: `ProviderTransport` ABC with one concrete implementation. Keep `ApiProvider` concrete; add a protocol only when a second transport appears. [src/dr_providers/query/transport.py]

shrink: `RequestControls`/`ReasoningWarning`/`warnings` layer is a no-op wrapper. Replace with one `reasoning_extra_body()` function. [src/dr_providers/query/reasoning.py]

delete: dead symbols: `LlmConfig`, `ControlRequestStyle`, `OpenRouterReasoningKey`, `CallError`, `has_sampling_controls`. Replacement: nothing. [src/dr_providers]

yagni: Tenacity retry decorator around one POST. Let callers retry or use a tiny local loop later; delete `tenacity`. [src/dr_providers/query/transport.py, pyproject.toml]

shrink: custom `ProviderName.__new__` stores one URL. Use constants. [src/dr_providers/names.py]

delete: `scripts/query_provider.py` just calls `dr_providers.cli.main`. Use `uv run python -m dr_providers.cli`. [scripts/query_provider.py]

net: -250 lines, -3 direct deps possible.
