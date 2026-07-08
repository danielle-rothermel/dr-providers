# dr-providers

Typed LLM provider-call kernel for OpenRouter, OpenAI, and Gemini: one
request, response, and failure vocabulary across providers. Requires
Python 3.12+.

## Ecosystem

dr-providers is the typed LLM-provider HTTP transport kernel, with an
optional `[serve]` FastAPI facade for localhost HTTP callers. Its neighboring
repos are dr-serialize, dr-graph, dr-platform, dr-code, whetstone-ai, and
unitbench. Package metadata shows no dependency on those neighbors; in-repo
notes identify whetstone-ai/dr-platform, dr-graph's graph runner, and
unitbench playgrounds as consumers.

## Install

```bash
pip install dr-providers
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add dr-providers
```

## Authentication

Set the API key env var for whichever provider(s) you call:

```bash
export OPENROUTER_API_KEY="sk-or-..."
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
```

## Quickstart

```python
from dr_providers import (
    HttpProvider,
    LlmRequest,
    MessageRole,
    PromptMessage,
    ReasoningEffort,
    openrouter_chat_config,
)

request = LlmRequest(
    provider_config=openrouter_chat_config(model="openai/gpt-4o-mini"),
    messages=(
        PromptMessage(role=MessageRole.USER, content="Say hello in one word."),
    ),
    reasoning=ReasoningEffort.LOW,
)

with HttpProvider() as provider:
    response = provider.complete(request)
    print(response.text)
```

`HttpProvider` is a context manager: it owns and closes its httpx
client on exit unless you inject your own (which is left open for you
to manage).

## Provider matrix

Presets in `dr_providers.config` fill in each provider's base URL, API
key env var, endpoint, and reasoning wire shape:

| Preset                     | Provider kind | Endpoint          | Reasoning wire shape                       |
| --------------------------- | -------------- | ----------------- | ------------------------------------------- |
| `openrouter_chat_config`    | `openrouter`   | chat completions  | `reasoning: {"effort": ...}` object         |
| `openai_chat_config`        | `openai`       | chat completions  | `reasoning_effort: ...` field               |
| `openai_responses_config`   | `openai`       | responses         | `reasoning: {"effort": ...}` object         |
| `gemini_chat_config`        | `gemini`       | chat completions  | `reasoning_effort: ...` field (OpenAI-compat endpoint) |

`ReasoningEffort` is a shared enum (`NONE`, `MINIMAL`, `LOW`, `MEDIUM`,
`HIGH`, `XHIGH`); each config's `reasoning_shape` determines how
`build_payload()` serializes it on the wire. For the full story — how
each provider actually accepts reasoning/effort/thinking, Gemini's
generation-dependent thinking configs, and links to the provider docs —
see [docs/reasoning-controls.md](docs/reasoning-controls.md).

## Testing with ScriptedProvider

`ScriptedProvider` implements the same `Provider` interface as
`HttpProvider` but scripts outcomes with no network:

```python
from dr_providers import ScriptedOutcome, ScriptedProvider, LlmRequest

provider = ScriptedProvider([ScriptedOutcome(text="scripted reply")])
response = provider.complete(request)
assert response.text == "scripted reply"
```

## Public API

Import stable symbols from the top-level package:

```python
from dr_providers import LlmRequest, HttpProvider, ReasoningEffort
```

See `dr_providers.__all__` for the full list. `HttpProvider` and
`TransportPolicy` load lazily so importing pure modules (config,
failures, request, response) never pulls in httpx.

## Serve facade

An optional FastAPI facade (the `[serve]` extra) exposes the kernel
over HTTP for non-Python callers:

```bash
uv run python -m dr_providers.serve serve
```

## Development

```bash
uv sync --frozen
uv run pre-commit install
uv run pre-commit run --all-files
```

### Live verification matrix

The default `uv run pytest` run is fully offline (`addopts = "-m 'not
live'"`). A `live`-marked matrix in `tests/live/test_live_matrix.py`
exercises the four presets against real provider endpoints:

```bash
uv run pytest -m live
```

Each case skips (not fails) when its API key env var
(`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`) is unset,
so this is safe to run without every provider configured. Successful
calls overwrite `data/wire-corpus/<provider_kind>_<endpoint_kind>.json`
with the raw response body; `tests/test_wire_corpus.py` re-parses
those bodies offline on every normal run.

### Audit corpus ground truth

This repo includes a small audit-output corpus and curated ground-truth
normalization artifacts under `data/audit-corpus/`. Regenerate the parsed audit
and analysis files with:

```bash
uv run python scripts/generate_audit_ground_truth.py \
  --corpus-dir data/audit-corpus \
  --output-dir data/audit-corpus/ground-truth
```
