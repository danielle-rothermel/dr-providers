~/drotherm/repos/dr-providers (lower-python-to-3.12)
↑40k ↓1.9k R166k CH98.4% $0.340 (sub) 11.0%/272k (auto)                         (openai-codex) gpt-5.5 • low
done — 141 tok/s

yagni: cut `ProviderTransport` ABC; `ApiProvider` is the only real implementation. Inline the shared methods there. [src/dr_providers/query/transport.py]

yagni: cut `ProviderConfig.required_executables`, `supports_structured_output`, availability plumbing; OpenRouter only needs API key + URL. [src/dr_providers/query/transport_config.py]

stdlib: cut `tenacity`; one retry site can be a tiny bounded `for` loop around `httpx` exceptions. [src/dr_providers/query/transport.py, pyproject.toml]

delete: remove unused `CallError`. Replacement: nothing. [src/dr_providers/query/response.py]

delete: remove unused `ControlRequestStyle` and `OpenRouterReasoningKey`. Replacement: nothing. [src/dr_providers/names.py]

delete: remove unused `LlmRequest.has_sampling_controls`. Replacement: nothing. [src/dr_providers/query/request.py]

shrink: collapse duplicate export barrels if `dr_providers.query` is not public API. Keep top-level `dr_providers` only. [src/dr_providers/query/__init__.py]

yagni: `OpenRouterProvider` is a 1-method subclass just filling constants; either keep `ApiProvider.openrouter()` or construct `ApiProvider` directly. [src/dr_providers/query/providers/openrouter.py]

net: -120 lines, -1 deps possible.
