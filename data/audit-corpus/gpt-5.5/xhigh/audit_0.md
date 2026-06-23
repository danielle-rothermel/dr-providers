~/drotherm/repos/dr-providers (lower-python-to-3.12)
↑73k ↓7.0k R216k CH97.1% $0.683 (sub) 14.1%/272k (auto)                       (openai-codex) gpt-5.5 • xhigh
done — 111 tok/s

yagni Generic provider/config/availability layer for one OpenRouter client. Collapse `ProviderTransport`/availability/config extras into concrete `OpenRouterProvider` + constants. [`src/dr_providers/query/transport.py`, `transport_config.py`, `providers/openrouter.py`]

yagni `RequestControls`/`ReasoningWarning` plumbing never emits warnings. Replace with one `reasoning_extra_body()` dict helper. [`src/dr_providers/query/reasoning.py`, `request.py`, `response.py`]

delete Placeholder marimo notebook and optional notebook extra. Recreate when there is real notebook content. [`nbs/hello_world.py`, `pyproject.toml`]

stdlib `tenacity` for one bounded HTTP retry. Replace with a small stdlib retry loop. [`src/dr_providers/query/transport.py`, `pyproject.toml`]

shrink Long hand-maintained sdist exclude list. Replace with a positive include/package list. [`pyproject.toml`]

delete Unused models/enums: `CallError`, `ControlRequestStyle`, `OpenRouterReasoningKey`. Nothing. [`src/dr_providers/query/response.py`, `src/dr_providers/names.py`]

yagni `LlmConfig` mirrors `LlmRequest` fields and has no caller. Build `LlmRequest` directly. [`src/dr_providers/config.py`, exports/tests/docs]

native `scripts/query_provider.py` only delegates to the module CLI. Use `python -m dr_providers.cli`. [`scripts/query_provider.py`, `scripts/live_test_llm_configs.sh`]

shrink Sampling convenience properties are single-use indirection. Inline `self.sampling` checks in `json_payload()`. [`src/dr_providers/query/request.py`]

net: ~-190 lines, -2 deps possible.
