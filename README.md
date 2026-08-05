# dr-providers

[![CI](https://github.com/danielle-rothermel/dr-providers/actions/workflows/ci.yml/badge.svg)](https://github.com/danielle-rothermel/dr-providers/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dr-providers.svg)](https://pypi.org/project/dr-providers/)

| [Repo Definitions](https://danielle-rothermel.github.io/dr-providers/) ([terms](https://github.com/danielle-rothermel/dr-providers/blob/main/.defs/terms.toml), [contracts](https://github.com/danielle-rothermel/dr-providers/blob/main/.defs/contracts.toml)) | [dr-serialize](https://github.com/danielle-rothermel/dr-serialize) |
| --- | --- |

**dr-providers makes LLM provider calls through explicit, typed contracts.**
It supports OpenRouter, OpenAI, Gemini, and Anthropic while keeping call
identity, provider translation, transport policy, and outcomes separate.

## Package map

| Package | Responsibility |
| --- | --- |
| `dr_providers.modeling` | Identity-bearing definitions, configs, requests, routes, controls, and transcripts |
| `dr_providers.translation` | Pure provider request-body construction and parsed-response translation |
| `dr_providers.transport` | Credentials, endpoints, timeout and native-retry policy, and HTTP execution |
| `dr_providers.outcomes` | Typed responses, expected failures, invocation evidence, and conformance warnings |
| `dr_providers.core` | Shared provider protocol and failure vocabulary |
| `dr_providers.surfaces.testing` | Deterministic `ScriptedProvider` for network-free tests |
| `dr_providers.surfaces.cli` | Optional `dr-providers` one-shot CLI |
| `dr_providers.surfaces.serve` | Optional localhost FastAPI facade |

The top-level `dr_providers` exports are the stable import surface. The
functional-area module paths make ownership discoverable but are not a second
public API to mirror in application imports.

## Install

dr-providers requires Python 3.12 or newer.

```bash
uv add dr-providers
```

Unless an API key is injected directly, real HTTP calls read the credential
selected by their transport policy:

| Provider | Environment variable |
| --- | --- |
| OpenRouter | `OPENROUTER_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Gemini | `GEMINI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |

## Python quickstart

This OpenAI example uses only names exported by `dr_providers`:

```python
from dr_providers import (
    GenerationControls,
    HttpProvider,
    MessageRole,
    PromptMessage,
    ProviderCallRequest,
    ProviderKind,
    Transcript,
    is_response,
    openai_responses_config,
    policy_for,
)

config = openai_responses_config(
    model="gpt-5-mini",
    controls=GenerationControls(token_limit=256),
)
request = ProviderCallRequest(
    config=config,
    transcript=Transcript(
        messages=(
            PromptMessage(
                role=MessageRole.USER,
                content="Say hello in one word.",
            ),
        )
    ),
)

with HttpProvider(policy=policy_for(ProviderKind.OPENAI)) as provider:
    outcome = provider.complete(request)

if is_response(outcome):
    print(outcome.text)
else:
    print(f"{outcome.code}: {outcome.message}")
```

Expected transport failures are returned as
`ProviderTransportFailure` values. Unexpected programming or infrastructure
errors can still raise.

## CLI and local server

Install and run the one-shot CLI:

```bash
uv add 'dr-providers[cli]'
uv run dr-providers --provider openai-responses \
  --model gpt-5-mini \
  --token-limit 256 \
  -m 'Say hello in one word.'
```

Install the serving extra and bind the FastAPI facade to localhost:

```bash
uv add 'dr-providers[serve]'
uv run python -m dr_providers.surfaces.serve serve --port 8322
```

## Outcome and evidence boundaries

`HttpProvider.complete()` returns a closed
`ProviderTransportResponse | ProviderTransportFailure` union for expected
transport results. The timeout plus a fixed five-second operational margin
bounds each native attempt's caller-visible wait; aggregate latency scales
with `native_retry_count + 1`. When a caller injects its own synchronous HTTP
client, a timed-out attempt can leave a daemon worker and socket lingering
until the caller-owned operation eventually ends.

`HttpProvider.invoke()` returns versioned serializable invocation evidence:
request and policy identity payloads, structured request metadata, the
constructed JSON request-body mapping, and the response body decoded as JSON
when possible or retained as text otherwise. It does not retain original HTTP
wire bytes. The standard `HttpProvider` path redacts known credential header
names; direct `ProviderHttpRequestEvidence` construction and deserialization
remain trusted-data paths. Evidence fields containing dictionaries remain
mutable after construction, so callers should serialize the snapshot before
sharing or persistence.

## Repository validation

The default suite is offline: pytest excludes tests marked `live`.

```bash
uv sync --locked --all-extras
uv run pre-commit install
scripts/pre-check.sh
uv build
```

Run the complete live matrix without changing the committed wire corpus:

```bash
uv run python scripts/run_live_matrix.py
```

Capturing and promoting replacement corpus data is a separate, deliberate
operation. It stages outside the repository, validates and redacts the
complete five-case capture, then updates `data/wire-corpus/`:

```bash
uv run python scripts/capture_live_corpus.py capture --promote
```
