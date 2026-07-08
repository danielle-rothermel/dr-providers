# Reasoning controls across providers

How "reasoning" / "effort" / "thinking" is requested differs per endpoint,
and the differences are wire-shape level, not just naming. This doc records
the verified matrix behind [ADR 0001](adr/0001-reasoning-as-effort-enum.md)
and how the kernel maps onto it.

## The kernel model

`LlmRequest.reasoning` takes a `ReasoningEffort`
(`none | minimal | low | medium | high | xhigh`). Each `ProviderConfig`
declares a `ReasoningRequestShape` — the wire serialization it can
transport:

| Shape | On the wire | Used by presets |
|---|---|---|
| `EFFORT_FIELD` | flat `"reasoning_effort": "<effort>"` | `openai_chat_config`, `gemini_chat_config` |
| `REASONING_OBJECT` | nested `"reasoning": {"effort": "<effort>"}` | `openai_responses_config`, `openrouter_chat_config` |
| `NONE` | reasoning cannot be transported; setting it raises `UnsupportedControlError` (unless `allow_unsupported_control_drop`) | custom configs |

Anything beyond an effort level (budgets, toggles, summaries, provider
thinking configs) is deliberately **not** modeled — it rides in
`extra_body`, which is merged inline into the payload. After the call, the
kernel checks the response and attaches a `reasoning_not_observed`
conformance warning if reasoning was requested but the response reports no
reasoning tokens.

## OpenAI

**Chat Completions** takes a flat `reasoning_effort` string; a nested
`reasoning` object is an unrecognized argument (400). Supported values:
`none`, `minimal`, `low`, `medium`, `high`, `xhigh`. Note that reasoning
models can reject other knobs — the live matrix caught `gpt-5-mini`
rejecting `temperature`.

**Responses** takes a top-level `reasoning` object:
`{"effort": "low", "summary": "auto"}`. Effort values as above; `summary`
and other subfields are `extra_body` territory for the kernel.

Sources: [OpenAI reasoning guide](https://developers.openai.com/api/docs/guides/reasoning),
[Chat Completions API reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

## Gemini (OpenAI-compat endpoint)

The compat endpoint (`https://generativelanguage.googleapis.com/v1beta/openai`)
accepts the same flat `reasoning_effort` string as OpenAI chat — there is
no nested `reasoning` object. Google maps effort onto its native thinking
machinery:

- **Gemini 2.5**: effort → `thinking_budget` (low=1,024, medium=8,192,
  high=24,576); `reasoning_effort: "none"` is allowed to disable thinking.
- **Gemini 3.x**: effort → `thinking_level` (`minimal`–`high`); reasoning
  cannot be disabled.

Precise budget control uses the native config via `extra_body` — this is
the confusing part, and it is generation-dependent:

```python
extra_body={
    "google": {
        "thinking_config": {
            # Gemini 2.5:  "thinking_budget": 8192,
            # Gemini 3.x:  "thinking_level": "low",
            "include_thoughts": True,
        }
    }
}
```

`thinking_budget` (2.5) and `thinking_level` (3.x) are mutually exclusive
with each other **and** with `reasoning_effort` — pick one mechanism per
request. The kernel keeps all of this out of its typed surface on purpose;
see ADR 0001.

Source: [Gemini OpenAI compatibility docs](https://ai.google.dev/gemini-api/docs/openai).

## OpenRouter

Takes a top-level `reasoning` object on chat completions, normalized across
its upstream providers: `{"effort": "low"}`, or alternatively
`{"max_tokens": N}` (budget-style models), `{"enabled": false}`,
`{"exclude": true}` (reason but don't return it). The kernel serializes
effort; the other subfields go through `extra_body` if needed — note
OpenRouter treats `effort` and `max_tokens` as alternatives, so don't send
both.

Source: [OpenRouter reasoning tokens docs](https://openrouter.ai/docs/use-cases/reasoning-tokens).

## Verifying

`uv run pytest -m live` runs one call per preset with
`reasoning=ReasoningEffort.LOW` against the real endpoints (skips per-case
without keys) and refreshes the recorded bodies in `data/wire-corpus/`.
Provider docs drift; the live matrix is the ground truth.
