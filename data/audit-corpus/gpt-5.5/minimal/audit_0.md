~/drotherm/repos/dr-providers (lower-python-to-3.12)
↑42k ↓1.3k R62k CH90.8% $0.282 (sub) 8.9%/272k (auto)                       (openai-codex) gpt-5.5 • minimal
done — 221 tok/s

yagni: cut `ProviderTransport` ABC; one implementation can just be `ApiProvider`. Replacement: concrete class only. [src/dr_providers/query/transport.py]

yagni: cut `ProviderConfig.required_executables`, `missing_executables`, `supports_structured_output`; no provider uses them. Replacement: env-key availability only. [src/dr_providers/query/transport_config.py]

yagni: cut custom `ProviderName.__new__`/embedded base URL for one provider. Replacement: plain `StrEnum` plus constants in `OpenRouterProvider`. [src/dr_providers/names.py, src/dr_providers/query/providers/openrouter.py]

delete: remove unused `ControlRequestStyle`, `OpenRouterReasoningKey`, and `CallError`. Replacement: nothing. [src/dr_providers/names.py, src/dr_providers/query/response.py]

stdlib: cut `tenacity` dependency for one retrying POST. Replacement: tiny `for attempt in range(3)` + `time.sleep`. [pyproject.toml, src/dr_providers/query/transport.py]

shrink: remove `sampling_temperature`, `sampling_top_p`, `has_sampling_controls` properties. Replacement: access `self.sampling.temperature/top_p` inline. [src/dr_providers/query/request.py]

yagni: `RequestControls` is a wrapper around `extra_body` with always-empty warnings. Replacement: `reasoning_extra_body(reasoning)`. [src/dr_providers/query/reasoning.py]

yagni: `LlmConfig` is exported/tested but not used by request flow. Replacement: delete unless external users rely on it. [src/dr_providers/config.py]

net: ~-120 lines, -1 dep possible.
