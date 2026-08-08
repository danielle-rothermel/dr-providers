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
| `dr_providers.transport` | Credentials, endpoints, timeout policy, and one-invocation HTTP execution |
| `dr_providers.outcomes` | Typed responses, expected failures, invocation evidence, and conformance warnings |
| `dr_providers.lifecycle` | Invocation classification, serializable retry state, deterministic transitions, and terminal call results |
| `dr_providers.core` | Shared provider protocol and failure vocabulary |
| `dr_providers.surfaces.testing` | Deterministic `ScriptedProvider` for network-free tests |
| `dr_providers.surfaces.cli` | Optional `dr-providers` one-shot CLI |
| `dr_providers.surfaces.serve` | Optional localhost FastAPI facade |

The top-level `dr_providers` exports are the stable general import surface.
Lifecycle-specific contracts are exported by `dr_providers.lifecycle`; the
remaining functional-area module paths primarily make ownership discoverable.

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

This OpenAI example uses the stable general and lifecycle import surfaces:

```python
from threading import Event

from dr_providers import (
    GenerationControls,
    HttpProvider,
    MessageRole,
    PromptMessage,
    ProviderCallRequest,
    ProviderKind,
    Transcript,
    openai_responses_config,
    policy_for,
)
from dr_providers.lifecycle import (
    AcceptAllSemanticResponseClassifier,
    ProviderCallOutcomeKind,
    ProviderCallState,
    StandardProviderCallRetryPolicy,
    run_local_provider_call,
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

classifier = AcceptAllSemanticResponseClassifier()
state = ProviderCallState.initial(
    request=request,
    retry_policy=StandardProviderCallRetryPolicy(),
    classifier_identifier=classifier.identifier,
)
with HttpProvider(policy=policy_for(ProviderKind.OPENAI)) as provider:
    result = run_local_provider_call(
        provider=provider,
        state=state,
        classifier=classifier,
        cancellation=Event(),
    )

evidence = result.completed_invocations[-1].observation.evidence
if result.outcome.kind is ProviderCallOutcomeKind.ACCEPTED:
    assert evidence.response is not None
    print(evidence.response.text)
else:
    print(result.outcome)
```

Expected transport failures are retained in invocation evidence and classified
into the terminal `ProviderCallResult`. Unexpected programming or infrastructure
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

`HttpProvider.invoke()` makes at most one provider wire request and returns
versioned serializable `ProviderInvocationEvidence`. The evidence binds the
request identity hash and transport-policy identity to structured HTTP request
metadata and exactly one response or expected failure. The HTTP request
evidence is the sole owner of the constructed request-body mapping.

`run_local_provider_call()` classifies each invocation, applies the selected
serializable retry policy through the deterministic lifecycle transition, and
returns the complete ordered `ProviderCallResult`. The standard policy permits
at most two invocations with one one-second retry, only for contained transient
network/provider failures and contained transport timeouts. Uncontained
deadline expiration is terminal; when a caller injects its own synchronous HTTP
client, its daemon worker and socket may linger until that caller-owned
operation eventually ends.

Decoded response bodies are retained as JSON when possible or as text
otherwise; original HTTP wire bytes are not retained. The standard HTTP path
redacts known credential header names. Direct
`ProviderHttpRequestEvidence` construction and deserialization remain
trusted-data paths.

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
