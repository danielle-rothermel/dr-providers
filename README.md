# dr-providers

[![CI](https://github.com/danielle-rothermel/dr-providers/actions/workflows/ci.yml/badge.svg)](https://github.com/danielle-rothermel/dr-providers/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dr-providers.svg)](https://pypi.org/project/dr-providers/)

| [Repo Definitions](https://danielle-rothermel.github.io/dr-providers/) ([terms](https://github.com/danielle-rothermel/dr-providers/blob/main/.defs/terms.toml), [contracts](https://github.com/danielle-rothermel/dr-providers/blob/main/.defs/contracts.toml)) | [dr-serialize](https://github.com/danielle-rothermel/dr-serialize) | [dr-wire](https://github.com/danielle-rothermel/dr-wire) |
| --- | --- | --- |

**dr-providers makes LLM provider calls through explicit, typed contracts.**
It supports OpenRouter, OpenAI, Gemini, and Anthropic while keeping call
identity, provider translation, transport policy, and outcomes separate.

## Package map

| Package | Responsibility |
| --- | --- |
| `dr_providers.modeling` | Identity-bearing definitions, configs, requests, routes, controls, and transcripts |
| `dr_providers.translation` | Pure provider request-body construction and parsed-response translation |
| `dr_providers.transport` | Credentials, endpoints, timeout policy, and one-invocation HTTP execution over the [dr-wire](https://github.com/danielle-rothermel/dr-wire) bounded client |
| `dr_providers.outcomes` | Typed responses, expected failures, invocation evidence, and conformance warnings |
| `dr_providers.lifecycle` | Invocation classification, serializable retry state, deterministic transitions, and terminal call results |
| `dr_providers.core` | Shared provider protocol and failure vocabulary |
| `dr_providers.surfaces.testing` | Deterministic `ScriptedProvider` for network-free tests |
| `dr_providers.surfaces.cli` | Optional `dr-providers` one-shot CLI |

The top-level `dr_providers` exports are the stable general import surface.
Functional-area module paths primarily make ownership discoverable; they are
not a second compatibility surface.

[`.defs/terms.toml`](.defs/terms.toml) and its
[rendered defs site](https://danielle-rothermel.github.io/dr-providers/) are the
authoritative reference for that public surface: every export is mapped to a
term there and checked by `scripts/check_defs.py`. This README illustrates
common paths rather than enumerating them, so an export it does not mention is
supported, not unsupported.

[Potential future features](docs/future-features.md) records directions this
package deliberately does not build today.

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

This OpenAI example uses the stable package import surface:

```python
from threading import Event

from dr_providers import (
    AcceptAllSemanticResponseClassifier,
    GenerationControls,
    HttpProvider,
    MessageRole,
    PromptMessage,
    ProviderCallOutcomeKind,
    ProviderCallRequest,
    ProviderCallState,
    ProviderKind,
    StandardProviderCallRetryPolicy,
    Transcript,
    openai_responses_config,
    policy_for,
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
with HttpProvider(
    policy=policy_for(
        ProviderKind.OPENAI,
        timeout_seconds=120.0,
        connect_timeout_seconds=30.0,
        idle_timeout_seconds=90.0,
        max_connections=1,
        max_keepalive_connections=1,
        max_request_bytes=1024 * 1024,
        max_response_bytes=8 * 1024 * 1024,
    )
) as provider:
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

## CLI

Install and run the one-shot CLI:

```bash
uv add 'dr-providers[cli]'
uv run dr-providers --provider openai-responses \
  --model gpt-5-mini \
  --token-limit 256 \
  -m 'Say hello in one word.'

# Anthropic requires --token-limit:
uv run dr-providers --provider anthropic \
  --model claude-sonnet-4-6 \
  --token-limit 256 \
  -m 'Say hello in one word.'
```

## Outcome and evidence boundaries

`HttpProvider.invoke()` makes at most one provider wire request and returns
versioned serializable `ProviderInvocationEvidence`. The evidence binds the
request identity hash and transport-policy identity to structured HTTP request
metadata and exactly one response or expected failure. The HTTP request
evidence is the sole owner of the constructed request-body mapping.

`run_local_provider_call()` classifies each invocation, applies the selected
serializable retry policy through the deterministic lifecycle transition, and
returns the complete ordered `ProviderCallResult`. The standard policy permits exactly one invocation with no auto-retry.
Transient network/provider failures and contained transport timeouts are
terminal unless the caller selects an explicit custom retry policy. The
standard HTTP provider uses direct synchronous native phase timeouts, so it
observes a timeout only after the local HTTP operation has ended. It owns and
reuses one bounded client; a clean close stops offload admission, drains
offloaded work, stops invocation admission, drains active invocations, and
closes that client once. Invocation admission stays open to every caller, on any
thread, until the offload drain finishes. An exception escaping a drain wait,
such as a keyboard interrupt, aborts the close: the provider still becomes
terminal and releases the executor and client without joining workers, so no
later caller blocks, but the drain does not complete. Connect, write, and pool
phase timeouts and the
response-read idle timeout are each declared explicitly on transport policy and
do not bound the total wall-clock duration of a slow response that keeps
producing bytes.

Every `ProviderTransportPolicy` and `policy_for()` call must declare native
connect, write/pool, and response-read idle timeouts, connection-pool limits,
and request/response byte caps.
There are no library-wide implicit sizing defaults. One-shot examples in this
repository configure one open and one keep-alive connection because each run
admits a single invocation. A caller that shares one `HttpProvider` across
concurrent work must size both connection limits to its own maximum concurrent
`invoke()` calls.

`run_local_provider_call_async()` is the asynchronous entry point. It submits
the same synchronous driver to the provider's own executor through
`HttpProvider.offload()` and awaits the result, so the transport stays one
bounded synchronous client. That executor is created on first offload and sized
from `max_connections`, which also bounds the client connection pool, so thread
count and pool size cannot disagree. Cancelling the awaiting asyncio task does
not interrupt the offloaded call: the offloaded future is shielded, so
cancellation flows through the cancellation event, and a clean `close()` drains
admitted offloaded work. Offloaded work must not call `close()` or `offload()`
on the provider running it: closing from inside offloaded work waits on that
same work, and offloaded work blocking on a nested offload starves once every
worker is held that way.

`ProviderCallState`, `ProviderRetryInstruction`, and `ProviderCallResult` are
JSON-serializable handoff values. A durable consumer can persist the declared
next state and schedule the instruction's delay before invoking again; restoring
at that boundary produces the same terminal result as the uninterrupted local
driver. The local driver follows transition outputs and performs only its
declared cancellation-aware wait; the deterministic transition owns retry and
terminal decisions.

Cancellation is draining: it starts no successor and retains an active
invocation observation if that invocation completes. It does not promise remote
provider cancellation or prompt release of provider capacity. Lifecycle values
are neutral to storage and workflow runtimes. This package does not provide
durable persistence, workflow scheduling, global admission, or exactly-once
provider effects.

The exact encoded request body and decompressed response body are bounded by
identity-bearing transport policy limits. Complete in-limit response bodies are
retained as JSON when possible or as text otherwise; over-limit responses retain
no partial body. Failure summary messages are unbudgeted. Wire-path failures
retain the underlying exception traceback in invocation evidence; other
transport failures leave `traceback` unset. Original HTTP wire bytes are
not retained. The standard HTTP path redacts known credential header names.
Direct `ProviderHttpRequestEvidence` construction and deserialization remain
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
