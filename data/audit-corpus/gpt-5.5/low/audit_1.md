~/drotherm/repos/dr-providers (lower-python-to-3.12)
↑61k ↓1.3k R102k CH70.4% $0.398 (sub) 9.5%/272k (auto)                                                            (openai-codex) gpt-5.5 • low
done — 144 tok/s

yagni `ProviderTransport` ABC has one implementation. Delete ABC; keep `ApiProvider` concrete. [src/dr_providers/query/transport.py]

yagni `tenacity` for one POST retry path. Replace with a tiny 3-attempt loop; drop dependency. [src/dr_providers/query/transport.py, pyproject.toml]

delete dead enums `ControlRequestStyle` and `OpenRouterReasoningKey`. Replacement: nothing. [src/dr_providers/names.py]

delete unused `CallError`. Replacement: nothing. [src/dr_providers/query/response.py]

yagni executable/structured-output availability plumbing is unused. Cut `required_executables`, `missing_executables`, `supports_structured_output`; env check is enough. [src/dr_providers/query/transport_config.py]

shrink `sampling_temperature` / `sampling_top_p` properties. Inline `self.sampling.temperature/top_p` in `json_payload`; delete unused `has_sampling_controls`. [src/dr_providers/query/request.py]

delete `scripts/query_provider.py` wrapper. Replacement: `python -m dr_providers.cli`. [scripts/query_provider.py, README.md]

net: ~-120 lines, -1 dep possible.
