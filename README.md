# dr-providers

[![CI](https://github.com/danielle-rothermel/dr-providers/actions/workflows/ci.yml/badge.svg)](https://github.com/danielle-rothermel/dr-providers/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dr-providers.svg)](https://pypi.org/project/dr-providers/)

| [Repo Definitions](https://danielle-rothermel.github.io/dr-providers/) ([terms](https://github.com/danielle-rothermel/dr-providers/blob/main/.defs/terms.toml), [contracts](https://github.com/danielle-rothermel/dr-providers/blob/main/.defs/contracts.toml)) | [dr-serialize](https://github.com/danielle-rothermel/dr-serialize) | [dr-wire](https://github.com/danielle-rothermel/dr-wire) |
| --- | --- | --- |

**dr-providers makes LLM provider calls through explicit, typed contracts.**
It executes HTTP calls for OpenRouter, OpenAI, Gemini, and Anthropic, and local
HuggingFace generation and continuation scoring through Transformers. Call
identity, model execution, transport policy, and outcomes remain separate.

## Package map

| Package | Responsibility |
| --- | --- |
| `dr_providers.modeling` | Identity-bearing definitions, configs, requests, routes, controls, and transcripts |
| `dr_providers.local` | Load-once causal-LM execution, serialized provider lifetime, and a torch-free backend seam |
| `dr_providers.translation` | Pure provider request-body construction and parsed-response translation |
| `dr_providers.transport` | Credentials, endpoints, timeout policy, and one-invocation HTTP execution over the [dr-wire](https://github.com/danielle-rothermel/dr-wire) bounded client |
| `dr_providers.outcomes` | Typed responses, expected failures, invocation evidence, and conformance warnings |
| `dr_providers.lifecycle` | Invocation classification, serializable retry state, deterministic transitions, and terminal call results |
| `dr_providers.core` | Shared provider protocol and failure vocabulary |
| `dr_providers.surfaces.testing` | Deterministic `ScriptedProvider` and `FakeLocalBackend` for network-free tests |
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
    ProviderGenerateRequest,
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
request = ProviderGenerateRequest(
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

`PromptRendering.ROLE_MESSAGES` is the default and preserves message roles.
Select `FLAT_TEXT` to concatenate every message's content in order, inserting
nothing, and send the result as one user message. Callers own whitespace;
system content and assistant-ending prefills become plain prompt text.
Transcript identity still retains the original roles; rendering participates
in definition identity and therefore changes config and request hashes.
An empty transcript produces an empty message list under `ROLE_MESSAGES`
and one user message with empty content under `FLAT_TEXT`; translation does
not validate whether a provider accepts empty input.

```python
from dr_providers import PromptRendering, openai_responses_config

config = openai_responses_config(
    model="gpt-5-mini",
    prompt_rendering=PromptRendering.FLAT_TEXT,
)
```

## Score continuations

`ProviderScoreRequest` carries one exact context string and a nonempty ordered
sequence of nonempty continuations. Empty context and duplicate continuations
are allowed. Definitions declare `supported_kinds`; all five HTTP presets
support only `ProviderCallKind.GENERATE`. Definitions used for scoring must
have empty `required_controls`: configurations enforce required controls, and
score requests reject assigned controls and nonempty body extensions.
Definition construction allows SCORE declarations with required controls, but
those definitions cannot produce a valid score request.

`ContinuationScore` reports total natural-log likelihood and token count for
the continuation tokens, excluding context tokens, plus `char_count` equal to
Python's `len()` of the original continuation string. Total and per-token log
probabilities must be finite and at most zero. Tokenization belongs to the
backend. `token_logprobs=True` requires per-token values for every score.
The evidence `build()` path checks score count, character counts, and required
per-token detail; direct evidence construction and deserialization are trusted
paths without request-dependent checks.

`LocalModelProvider` serves scores through Transformers, and `ScriptedProvider`
supports scores for offline testing. `HttpProvider.invoke()` raises `ValueError` for them
before admission or payload construction. Score responses classify as success
without invoking the semantic classifier. The shared lifecycle still requires
a classifier identifier and matching classifier object, although scoring does
not call its `classify()` method.

The local backend supplies ordered finite scores and validates correspondence
through the same evidence build path as the scripted provider.

```python
from threading import Event

from dr_providers import (
    AcceptAllSemanticResponseClassifier,
    ContinuationScore,
    ControlConstraints,
    ModelRoute,
    Protocol,
    ProviderCallDefinition,
    ProviderCallKind,
    ProviderCallState,
    ProviderKind,
    ProviderScoreRequest,
    ScriptedOutcome,
    ScriptedProvider,
    StandardProviderCallRetryPolicy,
    TokenLimitParameter,
    run_local_provider_call,
)

definition = ProviderCallDefinition(
    definition_id="example.score",
    route=ModelRoute(
        provider=ProviderKind.OPENAI,
        protocol=Protocol.CHAT_COMPLETIONS,
        model="scripted-model",
    ),
    supported_kinds=frozenset({ProviderCallKind.SCORE}),
    constraints=ControlConstraints(
        supported_controls=frozenset(),
        token_limit_parameter=TokenLimitParameter.MAX_COMPLETION_TOKENS,
    ),
)
request = ProviderScoreRequest(
    config=definition.materialize(),
    context="The capital of France is",
    continuations=(" Paris", " Lyon"),
    token_logprobs=True,
)
classifier = AcceptAllSemanticResponseClassifier()
state = ProviderCallState.initial(
    request=request,
    retry_policy=StandardProviderCallRetryPolicy(),
    classifier_identifier=classifier.identifier,
)
with ScriptedProvider([
    ScriptedOutcome(scores=(
        ContinuationScore(
            log_likelihood=-0.1, token_count=1, char_count=6,
            token_logprobs=(-0.1,),
        ),
        ContinuationScore(
            log_likelihood=-3.0, token_count=1, char_count=5,
            token_logprobs=(-3.0,),
        ),
    )),
]) as provider:
    result = run_local_provider_call(
        provider=provider, state=state, classifier=classifier,
        cancellation=Event(),
    )

response = result.completed_invocations[-1].observation.evidence.score_response
assert response is not None
print([score.log_likelihood for score in response.scores])  # [-0.1, -3.0]
```

The example uses an HTTP-shaped route as declaration data for the scripted
provider. That declaration does not make `HttpProvider` able to execute it.
`ProviderCallRequest` is a type alias; parse standalone serialized requests
with `TypeAdapter(ProviderCallRequest)` from Pydantic. Lifecycle models parse
the discriminated request automatically when restoring state or results.

## Local HuggingFace routes

`huggingface_config` declares a HuggingFace/Transformers route for generation
and continuation scoring. `LocalModelProvider` loads and runs that model;
`ScriptedProvider` can exercise both kinds without constructing an HTTP body.
The CLI remains HTTP generation only.

```python
from dr_providers import LocalDevice, LocalDtype, huggingface_config

config = huggingface_config(
    model="example/model",  # Hub repo ID or local path, retained verbatim
    revision="0123456789abcdef0123456789abcdef01234567",
    device=LocalDevice.CUDA,
    dtype=LocalDtype.BFLOAT16,
    batch_size=1,
    max_sequence_length=2048,
)
```

The definition's `local` spec is required for HuggingFace and forbidden for
HTTP services. Every spec field participates in definition identity and,
through references, config and request identity:

| Field | Declared meaning |
| --- | --- |
| `revision` | Nonempty revision selector; use a Hub commit ID to pin a snapshot |
| `device` | `cuda`, `mps`, or `cpu`; hardware availability is checked by the backend |
| `dtype` | `float32` or `bfloat16`; exact quantized compute rules need backend validation |
| `float32_matmul_precision` | `highest` (default) or `high`; `high` requires CUDA and permits reduced internal precision |
| `quantization` | `none` (default), `bitsandbytes_int8`, or `bitsandbytes_nf4`; quantization currently requires CUDA |
| `batch_size` | Positive declared batch size |
| `max_sequence_length` | Positive forward-input cap; backend refuses over-length inputs |
| `continuation_tokenization` | `lm_eval_encode_pair`: move trailing context whitespace to the continuation, jointly encode, then split at context token count |

Identity covers declarations, not immutable file contents. A branch or tag can
resolve differently later, and a local directory can change under the same
path. Resolved Hub commits belong in execution evidence when available; local
paths with no resolved commit do not establish artifact identity. Neither
request identity nor this evidence record promises reproducible execution.
The backend validates device availability and input lengths; model and dtype
support also depend on the installed runtime and hardware.

The preset supports temperature, top-p, token limit (`max_new_tokens`), and
seed. It has no required controls; reasoning, verbosity, and HTTP body
extensions are refused. Score requests require all controls to be unset.
For a local backend, `ROLE_MESSAGES` applies the tokenizer's chat template;
a tokenizer without one fails at load. `FLAT_TEXT` directly tokenizes the
separator-free concatenation of transcript contents. Scores ignore rendering
because they carry exact context and continuations rather than a transcript.
Role rendering always starts a new assistant response with
`add_generation_prompt=True`; assistant-prefill continuation is not supported.
Tool-role generation messages are refused with `ValueError`.

`LocalExecutionEvidence` records selected realized conditions: device and GPU
name, dtype, float32 matmul precision, quantization, resolved Hub commit when
available, torch and transformers versions, and whether a chat template and
automatic BOS insertion were applied to that invocation. Scoring never applies
a chat template; its empty-context prefix is distinct from automatic BOS insertion. Every field bears evidence identity, so a changed GPU
model or recorded library version changes that identity. This is a selected
conditions record, not an exhaustive environment fingerprint. It is mutually
exclusive with transport-policy identity or HTTP request evidence; neither
record kind is required for scripted invocations.

`wall_time_seconds` is an optional strict, finite, nonnegative duration.
It is persisted and restored but excluded from evidence identity; changing
only wall time also preserves the enclosing lifecycle result identity.

The following codes are module-level constants in `dr_providers.outcomes.models`:

| Failure code | Required recoverability | Invocation classification |
| --- | --- | --- |
| `local_out_of_memory` | `resource_exhaustion` | `resource_exhaustion` |
| `local_device_unavailable` | `permanent` | `permanent_provider_or_transport_failure` |
| `local_model_not_found` | `permanent` | `permanent_provider_or_transport_failure` |
| `local_sequence_too_long` | `permanent` | `permanent_provider_or_transport_failure` |
| `local_chat_template_missing` | `permanent` | `permanent_provider_or_transport_failure` |
| `local_non_finite_score` | `permanent` | `permanent_provider_or_transport_failure` |

Contradictory recoverability is refused. Out-of-memory is terminal under the
standard policy, which never retries any invocation. This does not assert that
a later identical invocation cannot succeed after memory pressure changes;
custom retry behavior is unchanged.

Local generation reports input tokens as `prompt_tokens`, generated tokens as
`completion_tokens`, and their sum as `total_tokens`. Local scoring reports the sum of unpadded forward-input lengths as
`prompt_tokens` and `total_tokens`, leaving `completion_tokens` unset. This
counts repeated context and any empty-context prefix for each row, excludes
the final target token (which is predicted but not fed into the model), and
excludes padding. It is a token-accounting convention, not a FLOP count. Local responses declare
`CostInfo(total_cost=0.0)`, meaning zero provider charge, rather than `None`;
this does not estimate hardware or electricity costs. Route-only quota identity
remains available but carries no quota meaning for local routes.

## Local HuggingFace backend

Install `dr-providers[local]` for Torch and Transformers, or
`dr-providers[local-cuda]` for CUDA quantization dependencies on Linux.
Importing `dr_providers`, constructing a fake backend, and using HTTP providers
load none of Torch, Transformers, or bitsandbytes.

```python
from dr_providers import (
    LocalDevice, LocalDtype, LocalModelProvider, PromptRendering,
    ProviderScoreRequest, huggingface_config,
)

config = huggingface_config(
    model="hf-internal-testing/tiny-random-gpt2",
    revision="71034c5d8bde858ff824298bdedc65515b97d2b9",
    device=LocalDevice.CPU,
    dtype=LocalDtype.FLOAT32,
    batch_size=2,
    max_sequence_length=512,
    prompt_rendering=PromptRendering.FLAT_TEXT,
)
with LocalModelProvider(config=config) as provider:
    evidence = provider.invoke(ProviderScoreRequest(
        config=config, context="The answer is", continuations=(" yes", " no"),
        token_logprobs=True,
    ))
    provider.release_memory()  # Release unused allocator cache between calls.
```

Construction eagerly loads one tokenizer and causal LM. Reuse the provider
for many calls with the same definition; controls may vary without reloading.
A different definition raises `ValueError`. Load failures raise directly;
expected invocation failures become evidence without tracebacks. Missing
runtime dependencies, incompatible model artifacts, authentication/network
errors, and unexpected exceptions propagate. A tokenizer without a chat
template fails at load for `ROLE_MESSAGES`; use `FLAT_TEXT` for base-model
scoring. Flat rendering concatenates contents without separators.

Scoring uses this fixed **lm-eval-derived causal encode-pair rule**:

1. Move all trailing context whitespace, as determined by `str.rstrip`, to
   the beginning of the continuation.
2. If context is now empty, use one BOS token as context, falling back to EOS,
   and encode the adjusted continuation. A tokenizer with neither ID is refused.
3. Otherwise encode both context and context-plus-continuation, then take
   continuation IDs by slicing the latter at the encoded context length.

Every encode disables automatic special tokens. `add_bos_token=False` records
that policy; the explicit empty-context prefix remains allowed. Token counts
use the resulting continuation IDs; character counts use Python `len()` of
the original caller continuation. Pairs yielding zero context or continuation
tokens raise `ValueError` rather than changing the tokenization rule. This
rule's empty-after-whitespace prefix handling is explicit; it is not a claim
of equivalence to every lm-eval/OLMES version. Caller-side OLMES equivalence
checks remain outside this package.

Scoring recomputes context for each row, batches up to the declared batch size,
right-pads with attention masks, and sums gathered float32 log probabilities.
A non-finite gathered value or total fails the whole request. Inputs longer
than the declared forward-input cap are refused without truncation. Generation
reserves at least one new token and caps output at remaining sequence space.
Empty tokenized generation prompts are refused. Recognized CUDA and MPS OOMs
release allocator cache and become `local_out_of_memory` evidence; they never
silently shrink batches or change retry policy.

Generation uses fixed single-sequence defaults rather than inheriting model
beam-search, penalty, or sampling settings. It is greedy unless temperature
is positive or top-p is set. Sampling defaults to temperature 1 and top-p 1,
with top-k filtering disabled. Thus temperature 0 plus an explicit top-p still
samples. Temperature must be nonnegative, top-p must be in `(0, 1]`, and token
limits must be positive; invalid local controls raise `ValueError` without
changing HTTP validation. EOS ends with `STOP`; exhausting the output cap ends
with `LENGTH`. Decoded text excludes the prompt and skips special tokens.

A supplied seed calls `torch.manual_seed` immediately before generation and
changes **process-global RNG state**. Without a seed, the backend does not
reseed. Matmul precision is also process-global: construction sets it, and an
invocation refuses a changed setting. Callers must coordinate other Torch
users and providers in the same process; per-provider serialization does not
isolate process-global state or guarantee reproducibility.

NF4 explicitly uses the requested dtype for its compute dtype. Quantized
weights load directly onto the selected current CUDA device; they are not moved
with an unconditional post-load `.to()`. CUDA quantization and MPS behavior
require hardware validation beyond the CPU integration suite.

The provider owns one lazy worker for `run_local_provider_call_async`; direct
invocations and `release_memory()` share its model-work lock. `close()` refuses
new work, waits for admitted offloads and active model work, then releases
weights. It is idempotent and supported through a context manager. Do not close
from its own worker or recursively await its single-worker async entry point.
`release_memory()` retains weights; `close()` drops them.

## CLI

Install and run the one-shot CLI:

```bash
uv add 'dr-providers[cli]'
uv run dr-providers --provider openai-responses \
  --model gpt-5-mini \
  --token-limit 256 \
  -m 'Say hello in one word.'

# Flatten system and user content; include any separator in the content:
uv run dr-providers --provider openai-responses \
  --model gpt-5-mini --prompt-rendering flat_text \
  --system 'Be brief. ' -m 'Say hello in one word.'

# Anthropic requires --token-limit:
uv run dr-providers --provider anthropic \
  --model claude-sonnet-4-6 \
  --token-limit 256 \
  -m 'Say hello in one word.'
```

## Outcome and evidence boundaries

Identity references use `definition_hash` in config identity payloads,
`config_hash` in request identity payloads, and `request_hash`,
`retry_policy_hash`, `call_hash`, and `evidence_hash` in the lifecycle and
evidence records that carry them. `provider_call_hash()` computes call
identity; models expose their own identity through `identity_hash`.

`HttpProvider.invoke()` makes at most one provider wire request and returns
versioned serializable `ProviderInvocationEvidence`. The evidence binds
request identity to the request kind and exactly one generate response, score
response, or expected failure. HTTP invocations record transport-policy
identity and structured HTTP request metadata; local execution evidence is an
alternative record kind. HTTP request evidence is the sole owner of the
constructed request-body mapping.

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

`ProviderTransportPolicy` and `policy_for()` bind only HTTP provider services
and refuse HuggingFace. HTTP body/path construction and response translation
refuse the transformers protocol.

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

The default suite is offline and torch-free: pytest excludes `live` and
`local_model` tests.

```bash
uv sync --locked --extra cli
uv run pre-commit install
scripts/pre-check.sh
uv build
```

Run the optional real CPU model suite and type-check the optional modules:

```bash
uv sync --locked --extra cli --extra local
uv run --locked --extra cli --extra local ty check --config 'src.exclude = []'
uv run --locked --extra cli --extra local pytest -q -m local_model
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
