~/drotherm/repos/dr-providers (lower-python-to-3.12)
↑45k ↓1.3k R125k CH96.7% $0.327 (sub) 9.5%/272k (auto)                      (openai-codex) gpt-5.5 • minimal
done — 140 tok/s

yagni `ProviderTransport` ABC has one implementation and only forwards config/availability/close. Fold into `ApiProvider`. [src/dr_providers/query/transport.py]

delete `ProviderAvailabilityStatus`, `availability_status()`, `is_available()`, `required_executables`, `supports_structured_output`; only tested locally, not used by request flow. Replacement: fail on missing API key in `prepare()`. [src/dr_providers/query/transport_config.py, tests/test_transport_config.py]

delete `LlmConfig`; exported but unused in code paths. Replacement: callers already pass `LlmRequest`. [src/dr_providers/config.py]

delete `ControlRequestStyle` and `OpenRouterReasoningKey`; dead enums, one already marked TODO. Replacement: nothing. [src/dr_providers/names.py]

delete `CallError`; dead model. Replacement: existing exceptions. [src/dr_providers/query/response.py]

shrink sampling properties on `LlmRequest`; `has_sampling_controls`, `sampling_temperature`, `sampling_top_p` add indirection for two reads. Replacement: inline `self.sampling.temperature/top_p` in `json_payload()`. [src/dr_providers/query/request.py]

shrink `RequestControls.from_reasoning` plumbing; `prepare()` and `generate()` both build/pass controls for one field. Replacement: let `prepare()` compute reasoning directly. [src/dr_providers/query/transport.py, src/dr_providers/query/request.py]

yagni `scripts/query_provider.py` is a wrapper around `python -m dr_providers.cli`. Replacement: call module directly from README/live-test script. [scripts/query_provider.py]

net: ~-120 lines, -0 deps possible.
