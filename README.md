# dr-providers

Typed LLM provider-call kernel for OpenRouter, OpenAI, Gemini, and
Anthropic: one Provider Call Config, Request, Transport Policy, and
no-throw Transport Outcome vocabulary across providers. Requires
Python 3.12+.

## Ecosystem

dr-providers is the typed LLM-provider HTTP transport kernel, with an
optional `[serve]` FastAPI facade for localhost HTTP callers. It builds
Provider Call Config/Request Identity Documents and full Identity Hashes
through `dr-serialize`. Its neighboring repos are dr-serialize, dr-graph,
dr-platform, dr-code, whetstone-ai, and unitbench. Whetstone-ai /
dr-platform, dr-graph's graph runner, and unitbench playgrounds are
consumers.

The [vocabulary sheet](https://danielle-rothermel.github.io/dr-providers/)
(source: `.defs/vocab.html`) is the authoritative statement of the
provider-call transport contract this repo implements: the terms, the
guarantees, what is in and out of scope, and the mapping from each term
to the exported names.

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
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Quickstart

```python
from dr_providers import (
    ApiKeyEnv,
    GenerationControls,
    HttpProvider,
    MessageRole,
    ProviderBaseUrl,
    ProviderCallRequest,
    ProviderTransportPolicy,
    ProviderTransportResponse,
    PromptMessage,
    ReasoningEffort,
    Transcript,
    openrouter_chat_config,
)

# A Provider Call Config is a complete validated assignment of one
# Provider Call Definition; it carries a full SHA-256 Identity Hash.
config = openrouter_chat_config(
    model="openai/gpt-4o-mini",
    controls=GenerationControls(reasoning=ReasoningEffort.LOW),
)

# A Provider Call Request is one Config reference + one Transcript.
request = ProviderCallRequest(
    config=config,
    transcript=Transcript(
        messages=(
            PromptMessage(
                role=MessageRole.USER, content="Say hello in one word."
            ),
        )
    ),
)

# Transport policy (credentials, base URL, timeout, native retry) is
# separate and excluded from identity. Native retry defaults to zero.
policy = ProviderTransportPolicy(
    api_key_env=str(ApiKeyEnv.OPENROUTER),
    base_url=str(ProviderBaseUrl.OPENROUTER),
)

with HttpProvider(policy=policy) as provider:
    outcome = provider.complete(request)  # no-throw typed outcome
    if isinstance(outcome, ProviderTransportResponse):
        print(outcome.text)
```

`complete` returns a closed no-throw Provider Transport Outcome
(`ProviderTransportResponse | ProviderTransportFailure`); expected
outcomes never raise. `invoke` instead returns a stable
`ProviderInvocationEvidence` artifact binding the request + policy
identities to the outcome and the complete least-processed raw request
and success/failure bodies (authorization headers and credentials are
never persisted).

`HttpProvider` is a context manager: it owns and closes its httpx
client on exit unless you inject your own (which is left open for you
to manage).

## Provider matrix

Presets in `dr_providers.config` build a Provider Call Definition and
materialize its Config, fixing each provider's Model Route
`(provider, protocol, model)`, the token-limit parameter, and the
reasoning wire shape. Base URL and API key env var live on the separate
`ProviderTransportPolicy` (`DEFAULT_BASE_URLS` / `DEFAULT_API_KEY_ENVS`
map each provider kind to its defaults):

| Preset                       | Provider    | Protocol            | Reasoning wire shape                        |
| ---------------------------- | ----------- | ------------------- | ------------------------------------------- |
| `openrouter_chat_config`     | `openrouter`| `chat_completions`  | `reasoning: {"effort": ...}` object         |
| `openai_chat_config`         | `openai`    | `chat_completions`  | `reasoning_effort: ...` field               |
| `openai_responses_config`    | `openai`    | `responses`         | `reasoning: {"effort": ...}` object         |
| `gemini_chat_config`         | `gemini`    | `chat_completions`  | `reasoning_effort: ...` field (OpenAI-compat endpoint) |
| `anthropic_messages_config`  | `anthropic` | `anthropic_messages`| `reasoning: {"effort": ...}` object         |

Both the OpenAI-compatible / OpenRouter `chat_completions` path and the
Anthropic `anthropic_messages` path are first-class, each usable with a
custom base URL via the transport policy.

`ReasoningEffort` is a shared enum (`NONE`, `MINIMAL`, `LOW`, `MEDIUM`,
`HIGH`, `XHIGH`); each Definition's `reasoning_shape` constraint
determines how `build_payload()` serializes it on the wire. For the full
story — how
each provider actually accepts reasoning/effort/thinking, Gemini's
generation-dependent thinking configs, and links to the provider docs —
see [docs/reasoning-controls.md](docs/reasoning-controls.md).

OpenAI Responses bodies are normalized from wire `output[]` parts into text,
typed no-text failures, and content-free diagnostics. See
[docs/responses-normalization.md](docs/responses-normalization.md) for the
failure codes, privacy boundary, and schema evidence.

## Testing with ScriptedProvider

`ScriptedProvider` implements the same `Provider` interface as
`HttpProvider` but scripts outcomes with no network:

```python
from dr_providers import ScriptedOutcome, ScriptedProvider

provider = ScriptedProvider([ScriptedOutcome(text="scripted reply")])
outcome = provider.complete(request)
assert outcome.text == "scripted reply"
```

## Public API

Import stable symbols from the top-level package:

```python
from dr_providers import (
    ProviderCallConfig,
    ProviderCallRequest,
    ProviderTransportPolicy,
    HttpProvider,
    ReasoningEffort,
)
```

See `dr_providers.__all__` for the full list. `HttpProvider` loads
lazily so importing the pure modules (route, controls, definition,
config, request, response, outcome, policy, evidence) never pulls in
httpx.

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
(`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`,
`ANTHROPIC_API_KEY`) is unset, so this is safe to run without every
provider configured. Successful calls overwrite
`data/wire-corpus/<provider>_<protocol>.json` with the raw response
body; `tests/test_wire_corpus.py` re-parses those bodies offline on
every normal run.

### Audit corpus ground truth

This repo includes a small audit-output corpus and curated ground-truth
normalization artifacts under `data/audit-corpus/`. Regenerate the parsed audit
and analysis files with:

```bash
uv run python scripts/generate_audit_ground_truth.py \
  --corpus-dir data/audit-corpus \
  --output-dir data/audit-corpus/ground-truth
```
