~/drotherm/repos/dr-providers (lower-python-to-3.12)
↑48k ↓2.2k R131k CH83.8% $0.372 (sub) 10.9%/272k (auto)                (openai-codex) gpt-5.5 • thinking off
done — 283 tok/s

yagni: cut generic provider availability scaffolding (`ProviderAvailabilityStatus`, `required_executables`, `supports_structured_output`, `availability_status`, `is_available`). Replacement: fail on missing OpenRouter API key at request time. [src/dr_providers/query/transport_config.py, src/dr_providers/query/transport.py]

yagni: cut `ProviderTransport` ABC; only `ApiProvider` implements it and it is not exported. Replacement: make `ApiProvider` the concrete base. [src/dr_providers/query/transport.py]

stdlib: cut `tenacity` dependency used once. Replacement: 3-attempt `for` loop + `time.sleep`/simple exponential delay, or no retry if callers should own retry. [pyproject.toml, src/dr_providers/query/transport.py]

delete: cut unused `LlmConfig`; requests already carry the same fields and nothing consumes it. Replacement: nothing. [src/dr_providers/config.py]

delete: cut warning plumbing (`ReasoningWarning`, `RequestControls.warnings`, request/response `warnings`) because nothing ever emits a warning. Replacement: add it back when a provider actually returns non-fatal diagnostics. [src/dr_providers/query/reasoning.py, src/dr_providers/query/request.py, src/dr_providers/query/response.py]

delete: cut unused `CallError`. Replacement: nothing. [src/dr_providers/query/response.py]

delete: cut TODO enum `OpenRouterReasoningKey`. Replacement: nothing. [src/dr_providers/names.py]

shrink: simplify `ProviderName` custom enum payload (`__new__`, `_api_base_url`, `Self`). Replacement: plain `StrEnum` + constant/property for the one OpenRouter URL. [src/dr_providers/names.py]

shrink: inline unused sampling helper properties (`has_sampling_controls`, `sampling_temperature`, `sampling_top_p`). Replacement: direct `self.sampling.temperature/top_p` checks in `json_payload`. [src/dr_providers/query/request.py]

yagni: cut `query_from_prompt(provider_name=...)`; only OpenRouter exists. Replacement: default OpenRouter or derive from provider config. [src/dr_providers/query/from_prompt.py, src/dr_providers/cli.py]

net: ~-150 lines, -1 dep possible.
