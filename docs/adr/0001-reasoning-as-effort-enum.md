# Reasoning is a typed effort level, serialized per-config as field or object

The kernel's `LlmRequest.reasoning` was a free-form dict whose placement enum
(`TOP_LEVEL` vs `EXTRA_BODY`) produced byte-identical wire payloads — and that
nested-object shape is rejected by OpenAI Chat Completions and Gemini's
OpenAI-compat endpoint, which both take a flat `reasoning_effort` string
(verified against provider docs, 2026-07). We decided: reasoning becomes a
typed effort enum (`none/minimal/low/medium/high/xhigh`), and
`ReasoningRequestShape` becomes an honest wire-serialization declaration —
`EFFORT_FIELD` (OpenAI chat, Gemini compat), `REASONING_OBJECT` (OpenAI
Responses, OpenRouter), `NONE`. Invalid combinations fail at request build,
consistent with the kernel's no-silent-defaults stance.

## Considered Options

- **Richer typed spec (effort + budget + toggle)** — rejected: budgets are
  provider- and model-generation-specific (Gemini 2.5 `thinking_budget` vs
  3.x `thinking_level`, mutually exclusive with `reasoning_effort`), which
  would drag per-generation provider knowledge into the kernel's core types.
  Budgets and thinking configs go through `extra_body` (e.g.
  `extra_body.google.thinking_config`), which already reaches the wire intact.
- **Keep the free-form dict** — rejected: dict semantics stay
  provider-dependent and typos pass silently until the provider rejects them.
