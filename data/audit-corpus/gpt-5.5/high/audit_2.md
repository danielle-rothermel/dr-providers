~/drotherm/repos/dr-providers (lower-python-to-3.12)
↑61k ↓5.5k R207k CH97.0% $0.573 (sub) 15.3%/272k (auto)                        (openai-codex) gpt-5.5 • high
done — 100 tok/s

yagni delete empty marimo notebook scaffold and `notebooks` extra. Nothing until a real notebook exists. [nbs/hello_world.py, pyproject.toml]

stdlib replace one-command Typer CLI with `argparse`. Removes Typer/Rich chain. [src/dr_providers/cli.py, pyproject.toml]

yagni delete availability/status machinery (`ProviderAvailabilityStatus`, executables, structured-output flag, `is_available`). Request path only needs API-key resolution. [src/dr_providers/query/transport_config.py, src/dr_providers/query/transport.py]

yagni delete warnings plumbing (`ReasoningWarning`, `RequestControls.warnings`, request/response warnings fields). No code emits warnings. [src/dr_providers/query/reasoning.py, src/dr_providers/query/request.py, src/dr_providers/query/response.py]

yagni merge `ProviderTransport` ABC into `ApiProvider`. One subclass, no real polymorphism. [src/dr_providers/query/transport.py]

delete unused public-ish models/enums: `CallError`, `ControlRequestStyle`, `OpenRouterReasoningKey`, `LlmConfig`. Replacement: nothing until a caller exists. [src/dr_providers/query/response.py, src/dr_providers/names.py, src/dr_providers/config.py]

stdlib replace Tenacity retry decorator with a tiny local loop, or no retry policy. One call site does not need a dependency. [src/dr_providers/query/transport.py, pyproject.toml]

delete `dr_providers.query.__init__` re-export mirror. Top-level `dr_providers.__init__` already defines the package API. [src/dr_providers/query/__init__.py]

shrink `live_test_llm_configs.sh` repeated `run_config` blocks into a simple array loop. Same behavior, fewer lines. [scripts/live_test_llm_configs.sh]

delete unused request helpers `has_sampling_controls`, `sampling_temperature`, `sampling_top_p`. Inline `self.sampling` in `json_payload`. [src/dr_providers/query/request.py]

delete `scripts/query_provider.py` wrapper. Use `uv run python -m dr_providers.cli ...`. [scripts/query_provider.py]

net: -300 lines, -3 deps possible.
