~/drotherm/repos/dr-providers (lower-python-to-3.12)
↑49k ↓1.9k R112k CH96.5% $0.358 (sub) 7.5%/272k (auto)                          (openai-codex) gpt-5.5 • low
done — 168 tok/s

`delete:` tracked build artifacts. Nothing in `dist/` belongs in source; rebuild on release. `[dist/*]`

`delete:` unused enum `OpenRouterReasoningKey` already marked TODO. Replacement: nothing. `[src/dr_providers/names.py]`

`delete:` unused enum `ControlRequestStyle`. Replacement: nothing. `[src/dr_providers/names.py]`

`delete:` unused model `CallError`. Replacement: nothing. `[src/dr_providers/query/response.py]`

`yagni:` `ProviderTransport` ABC has one real implementation and mostly delegates. Collapse into `ApiProvider` until there’s a second transport. `[src/dr_providers/query/transport.py]`

`yagni:` `LlmConfig` is exported/tested but unused by request/query flow. Delete or wire it into construction; right now it’s a parked shape. `[src/dr_providers/config.py]`

`delete:` `ReasoningWarning` / `RequestControls.warnings` plumbing never produces warnings; every path passes `[]`. Remove until a warning exists. `[src/dr_providers/query/reasoning.py, request.py, response.py, transport.py]`

`yagni:` `ProviderAvailabilityStatus.supports_structured_output`, `ProviderConfig.supports_structured_output`, and `required_executables` have no current provider/user. Keep env-var availability only. `[src/dr_providers/query/transport_config.py]`

`shrink:` `LlmRequest.sampling_temperature`, `sampling_top_p`, `has_sampling_controls` add indirection for two field reads. Inline `self.sampling.temperature/top_p`. `[src/dr_providers/query/request.py]`

`shrink:` `ApiProvider.generate()` computes `RequestControls` then `prepare()` can compute it again. Pass reasoning once; let `prepare()` own it. `[src/dr_providers/query/transport.py, request.py]`

`stdlib:` `tenacity` is used for one POST retry. A tiny `for attempt in range(3)` loop cuts a dependency if retry policy stays simple. `[src/dr_providers/query/transport.py, pyproject.toml]`

`delete:` committed caches / local env dirs are untracked but present: `.pytest_cache`, `.ruff_cache`, `.venv`. Add/keep ignores and clean locally. `[repo root]`

net: ~-180 lines, -1 dep possible.
