# dr-providers

[![CI](https://github.com/danielle-rothermel/dr-providers/actions/workflows/ci.yml/badge.svg)](https://github.com/danielle-rothermel/dr-providers/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dr-providers.svg)](https://pypi.org/project/dr-providers/)

| [Repo Definitions](https://danielle-rothermel.github.io/dr-providers/) | [dr-serialize v0.1.1 (local checkout)](https://github.com/danielle-rothermel/dr-serialize) |
| --- | --- |

**dr-providers makes LLM provider calls through explicit, typed contracts.**
It supports OpenRouter, OpenAI, Gemini, and Anthropic and is organized into
these functional areas:

- **[Modeling](https://github.com/danielle-rothermel/dr-providers/tree/main/src/dr_providers/modeling)**
  builds validated, identity-hashed definitions, configs, and requests from
  provider routes, generation controls, and transcripts.
- **[Translation](https://github.com/danielle-rothermel/dr-providers/tree/main/src/dr_providers/translation)**
  maps the shared call model into provider request and response wire formats.
- **[Transport](https://github.com/danielle-rothermel/dr-providers/tree/main/src/dr_providers/transport)**
  owns credentials, endpoints, timeout and retry policy, and bounded HTTP
  execution.
- **[Outcomes](https://github.com/danielle-rothermel/dr-providers/tree/main/src/dr_providers/outcomes)**
  represents expected successes and failures as typed data and preserves
  raw HTTP evidence in versioned invocation records, with known credential
  header names redacted on the standard `HttpProvider` invocation path.
- **Infrastructure**
  - **[Core](https://github.com/danielle-rothermel/dr-providers/tree/main/src/dr_providers/core)**
    holds the shared provider protocol, failure vocabulary, and immutable-value
    primitives.
  - **[Surfaces](https://github.com/danielle-rothermel/dr-providers/tree/main/src/dr_providers/surfaces)**
    groups the user-facing and testing adapters:
    - **[Testing](https://github.com/danielle-rothermel/dr-providers/tree/main/src/dr_providers/surfaces/testing)**
      provides a deterministic scripted provider.
    - **[CLI](https://github.com/danielle-rothermel/dr-providers/tree/main/src/dr_providers/surfaces/cli)**
      provides the `dr-providers` command-line interface.
    - **[Serve](https://github.com/danielle-rothermel/dr-providers/tree/main/src/dr_providers/surfaces/serve)**
      provides an optional localhost HTTP facade.

The sketches below are intentionally abridged contract shapes. They omit
validation and provider-specific internals so the linked packages remain the
authoritative definitions.

## Modeling

Modeling describes a provider call without transport policy or credentials. A
definition fixes a route and its constraints, a config assigns controls, and a
request combines that config with an ordered transcript.

```python
class ProviderKind(StrEnum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


class Protocol(StrEnum):
    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"


class ModelRoute(BaseModel):
    provider: ProviderKind
    protocol: Protocol
    model: str
```

```python
class ReasoningEffort(StrEnum):
    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class GenerationControls(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    token_limit: int | None = None
    reasoning: ReasoningEffort | None = None


class Transcript(BaseModel):
    messages: tuple[PromptMessage, ...]
```

```python
class ProviderCallDefinition(BaseModel):
    route: ModelRoute
    constraints: ControlConstraints

    def materialize(
        self,
        *,
        controls: GenerationControls | None = None,
        extensions: ProviderBodyExtensions | None = None,
    ) -> ProviderCallConfig: ...


class ProviderCallConfig(BaseModel):
    definition: ProviderCallDefinition
    controls: GenerationControls
    extensions: ProviderBodyExtensions

    @property
    def identity_hash(self) -> str: ...


class ProviderCallRequest(BaseModel):
    config: ProviderCallConfig
    transcript: Transcript

    @property
    def identity_hash(self) -> str: ...
```

The Definition, Config, and Request payload models do not carry schema names
or versions. Their `IdentityDocument` envelopes are the sole owners of
`schema` and `schema_version`.

## Translation

Translation is the pure boundary between provider-independent call contracts
and provider-specific wire bodies. Dispatch follows the protocol carried by
the validated config rather than provider-specific branching in callers.

```python
def protocol_path(config: ProviderCallConfig) -> str: ...


def build_payload(
    request: ProviderCallRequest,
) -> dict[str, Any]: ...
```

```python
def parse_response(
    body: Mapping[str, Any],
    *,
    config: ProviderCallConfig,
) -> ProviderTransportOutcome: ...
```

## Transport

Transport owns operational execution policy and the bounded wire call. It
returns typed outcomes without deciding whether a successful generation is
semantically acceptable to a downstream application.

```python
class ProviderTransportPolicy(BaseModel):
    api_key_env: str
    base_url: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS
    native_retry_count: int = 0
```

```python
class Provider(typing.Protocol):
    def complete(
        self,
        request: ProviderCallRequest,
    ) -> ProviderTransportOutcome: ...


class HttpProvider:
    def complete(
        self,
        request: ProviderCallRequest,
    ) -> ProviderTransportOutcome: ...

    def invoke(
        self,
        request: ProviderCallRequest,
    ) -> ProviderInvocationEvidence: ...
```

## Outcomes

Outcomes form a closed no-throw union of successful transport responses and
expected transport failures. Invocation evidence binds that outcome to the
request, transport policy, and raw HTTP exchange; exactly one of its response
or failure fields is populated. Standard `HttpProvider` invocation redacts
known credential header names before binding the raw request to evidence.
Direct `RawHttpRequest` construction and deserialization are trusted-data paths:
they retain supplied headers without sanitizing them.

```python
class FailureClass(StrEnum):
    PERMANENT = "permanent"
    TRANSIENT = "transient"
    RATE_LIMITED = "rate_limited"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    UNKNOWN = "unknown"


class ProviderTransportResponse(BaseModel):
    text: str
    usage: TokenUsage | None = None
    cost: CostInfo | None = None
    warnings: tuple[ProviderTransportWarning, ...] = ()


class ProviderTransportFailure(BaseModel):
    failure_class: FailureClass
    code: str | None = None
    message: str
    retryable: bool


ProviderTransportOutcome = (
    ProviderTransportResponse | ProviderTransportFailure
)
```

```python
class ProviderInvocationEvidence(BaseModel):
    request_identity: Mapping[str, Any]
    policy_identity: Mapping[str, Any]
    raw_request: RawHttpRequest
    response: ProviderTransportResponse | None = None
    failure: ProviderTransportFailure | None = None
```

`ProviderInvocationEvidence` is likewise the bare payload model. Its stable
persistence form is an `IdentityDocument` envelope carrying `schema`,
`schema_version`, and that payload.

`base_url` is retained verbatim in policy identity and as the base of the raw
request URL captured in invocation evidence. Callers must not embed credentials
in it. Possible future hardening includes sanitizing at the `RawHttpRequest`
model boundary and separating or restricting wire URLs from URLs retained in
evidence.

```python
def conformance_warnings(
    request: ProviderCallRequest,
    response: ProviderTransportResponse,
) -> tuple[ProviderTransportWarning, ...]: ...
```
