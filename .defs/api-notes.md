# dr-providers API naming proposals

Naming problems surfaced while maintaining `.defs/vocab.html`. **Proposals only — do not implement.**
Renames touch golden/fixture files and hash values only if a name feeds an identity payload; verify each blast radius before acting. Nothing here should be applied from a doc pass.

---

## 1. `Provider` protocol vs. the term Provider Service

- **Current name:** `Provider` — the injected-executor protocol (maps to the contract term *Provider Client*).
- **Problem:** The bare word "Provider" most naturally reads as the external service family (the contract's *Provider Service*: OpenRouter, OpenAI, Gemini, Anthropic), not the client protocol. The vocab has to spell out the divergence in the Provider Client row ("This package names that protocol `Provider`"). A reader seeing `Provider` in code cannot tell it means the client, not the service.
- **Proposed rename(s):**
  - `ProviderClient` for the protocol — frees the bare word `Provider` and aligns the export with the term name. Trade-off: `HttpProvider` / `ScriptedProvider` would want to become `HttpProviderClient` / `ScriptedProviderClient` for family consistency, widening the change.
  - Keep `Provider`, but document the collision as an intentional term/name divergence (already partly done in the vocab). Lowest churn; leaves the ambiguity in code.
- **Blast radius:** `provider.py` (protocol def), `transport.py`, `__init__.py` lazy `__getattr__`, every implementer (`HttpProvider`, `ScriptedProvider`), type annotations across the package, tests, and any docstring referencing the protocol. No identity/hash impact (protocol type is not serialized).

## 2. Failure taxonomy names imply semantic retry/classification authority the contract disclaims

- **Current names:** `FailureClass`, `RETRYABLE_FAILURE_CLASSES`, `RECOVERABLE_FAILURE_CLASSES`, `classify_status_code`, `failure_record`, and the retryable/recoverable flags.
- **Problem:** These read as semantic failure classification and retry-eligibility policy. The contract assigns that to the caller and lists "semantic failure taxonomy," "retry," and "backoff" under **Out of scope / Not owned**. Names asserting `RETRYABLE`/`RECOVERABLE` classes imply dr-providers holds retry authority at the transport boundary.
- **Proposed rename(s):**
  - `FailureClass` -> `TransportFailureClass`; `classify_status_code` -> `coarse_status_class` (or `transport_status_class`) — keeps the labeling plainly at the transport boundary.
  - `RETRYABLE_FAILURE_CLASSES` -> `RETRY_HINT_FAILURE_CLASSES` ("retry-hint," not "retryable") to signal advisory, not authoritative, and avoid implying retry eligibility.
  - Trade-off: longer names; `RETRYABLE`/`RECOVERABLE` may already be referenced by callers who read them as advisory. A rename removes an implied promise but breaks import sites.
- **Blast radius:** `failures.py`, `response.py`, `transport.py`, the frozensets and their consumers, `__init__.__all__`, and tests asserting on class names/membership. No identity/hash impact expected; confirm no failure-class name is embedded in a serialized invocation-evidence field before renaming.

## 3. `Llm*` prefix is inconsistent with the `Provider*` / `ProviderTransport*` family

- **Current name:** `LlmWarning` (and `Llm`-era language generally; `LlmResponse`/`LlmRequest` were retired to `ProviderTransport*`, but `LlmWarning` survives as an export).
- **Problem:** `Llm` is a migration-source spelling. Every sibling in the vocab uses the `Provider*` / `Transport*` family; `LlmWarning` is the lone holdout, so the conformance-warnings row breaks the naming pattern.
- **Proposed rename(s):** `ProviderWarning` or `TransportWarning`, matching `WarningSeverity` and the rest of the family. Trade-off: `LlmWarning` is a public export; renaming breaks import sites and any serialized warning payloads that carry a type tag.
- **Blast radius:** the conformance-warnings module, `conformance_warnings` / `with_conformance_warnings` producers, `__init__.__all__`, tests, and any golden evidence that records a warning type string (check before renaming — this could touch fixtures).

## 4. `ProviderFailure` vs. `ProviderTransportFailure` vs. `ProviderFailureError` — three near-homonyms for distinct concepts

- **Current names:**
  - `ProviderFailure` — the classification **record** held inside a transport failure.
  - `ProviderTransportFailure` — the typed expected transport-failure **outcome** (never raised).
  - `ProviderFailureError` — the raised **exception** for unexpected errors.
- **Problem:** Under-disambiguated cluster: three names one word apart for a record, a value-outcome, and an exception. A reader cannot infer which is raised vs. returned vs. embedded. The Exported Names note column currently carries the whole disambiguation load.
- **Proposed rename(s):** `ProviderFailure` -> `ProviderFailureRecord` to name the inner record explicitly and separate it from the outcome and the exception. Trade-off: `failure_record` (the builder) and `ProviderFailureError` (which wraps the record) would want matching updates; widens the change but sharpens the whole cluster.
- **Blast radius:** `failures.py`, `response.py`, `transport.py`, the error hierarchy, `failure_record`, `__init__.__all__`, tests. Verify no serialized field name is `provider_failure` before renaming (evidence/diagnostics).

## 5. `ResponsesDiagnostics.response_id_hash` — "hash" on a non-identity surface

- **Current name:** `ResponsesDiagnostics.response_id_hash` — a truncated, unsalted SHA-256 digest (`sha256(...).hexdigest()[:RESPONSE_ID_HASH_LENGTH]`), exported via the `ResponsesDiagnostics` type.
- **Problem:** The contract reserves "hash" for the identity-hash path (full 64-char). A `*_hash` name on a diagnostic surface invites mistaking a truncated correlator for an identity value, exactly the confusion the vocab warns against. It is documented as diagnostic-only, but the name still carries "hash."
- **Proposed rename(s):** `response_id_fingerprint` or `response_id_correlator` — keeps it visibly off the identity-hash path. Trade-off: "fingerprint"/"correlator" are slightly less literal about the underlying SHA-256; and `RESPONSE_ID_HASH_LENGTH` would want a matching rename.
- **Blast radius:** `response.py` / `outcome.py` (the method and the length constant), `ResponsesDiagnostics` consumers, tests. Method, not a top-level export, so `__all__` is unaffected — but confirm the field is not serialized into evidence under `response_id_hash` (would touch golden fixtures).

## 6. `*_config` presets hide the Definition-then-Config split

- **Current names:** `openrouter_chat_config`, `openai_chat_config`, `openai_responses_config`, `gemini_chat_config`, `anthropic_messages_config`.
- **Problem:** Each is named by protocol surface and returns a *Provider Call Config*, but internally builds a *Provider Call Definition* first. The `_config` suffix hides that a Definition is created and owned along the way. Whether that split should surface in the name is a judgment call.
- **Proposed rename(s):** Likely leave the names as-is — the return type (a config) is what the caller receives, and the Definition is an internal owner. If the split matters to callers, a paired `*_definition` builder could be exposed rather than renaming the config builders. Trade-off: exposing definitions widens the public surface for little caller benefit.
- **Blast radius (if changed):** `config` builder module, `__init__.__all__`, tests, and any docs/examples calling the presets. Recommendation: leave to the Exported Names note column; no rename.
