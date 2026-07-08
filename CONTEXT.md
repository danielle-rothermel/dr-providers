# dr-providers

A typed LLM provider-call library: one request/response/failure vocabulary
across providers, with a local HTTP facade for browser-based tooling.

## Language

**Kernel**:
The provider-call domain model — request, response, config, failures,
conformance — and the sole public Python API.
_Avoid_: v0.2, query layer (the deleted 0.1.x lineage)

**Provider**:
Anything implementing the single-shot `complete(request) -> response`
interface, whether it talks to a network or replays scripted outcomes.
_Avoid_: client, backend

**Provider Config**:
A pure-data record describing how to reach one provider endpoint: base URL,
key env var, endpoint kind, and which controls it can transport.
_Avoid_: provider class, adapter

**Serve Facade**:
The localhost FastAPI surface over the kernel. Its HTTP contract (OpenAPI
schema) is the external API that unitbench's generated client consumes.
_Avoid_: server, backend API

**Control**:
A request knob (temperature, token limit, reasoning effort) that a provider
config either transports to the wire or rejects at build time.
_Avoid_: parameter, option, setting

**Reasoning Effort**:
The typed cross-provider reasoning level (none–xhigh). Provider-specific
thinking budgets are not efforts; they ride in `extra_body`.
_Avoid_: thinking budget, reasoning dict

**Throttle Identity**:
The stable string identifying a provider endpoint for record identity and
rate accounting (`provider:endpoint:model`, or an explicit `throttle_key`
override). Persisted in whetstone's record hashes — renaming is a migration.
_Avoid_: provider id, config key

**Conformance Warning**:
A post-response observation that the provider did not honor a requested
control (e.g. reasoning requested, none observed). Part of every Provider's
response contract; the caller decides fatality.
_Avoid_: validation error, soft error

**Wire Shape**:
A provider config's declaration of how a control serializes on the wire
(e.g. flat `reasoning_effort` field vs nested `reasoning` object).
_Avoid_: request shape, placement
