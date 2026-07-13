# Responses normalization

`parse_responses_body()` parses the OpenAI Responses wire body in one pass over
`output[]`. Matching the official SDK aggregation, it concatenates
`output_text` content parts only from `message` items, in wire order, and adds a
content-free `ResponsesDiagnostics` envelope to successful
`LlmResponse` values. Failures carry the same envelope in
`ProviderFailure.metadata["diagnostics"]`.

The envelope records only explicitly allowlisted response statuses and
incomplete reasons, allowlisted item and content-type counts, text and refusal
lengths, and a 16-character SHA-256 response-ID hash. Unknown enum values are
coalesced to `unknown`; unknown item and content types are counted under a
single `unknown` key. Provider error codes and echoed model names are omitted
because their value spaces are open-ended. It never copies prompt, output,
refusal, tool arguments, raw response IDs, or arbitrary provider-controlled
dictionary keys into failure metadata. Successful responses retain the
existing raw `provider_metadata` contract.

The response-ID digest supports correlation of high-entropy provider IDs. It
is truncated and unsalted, so it does not protect low-entropy values from
dictionary attacks and must not be used to sanitize arbitrary provider text.

No-text outcomes are permanent and typed:

| Code | Meaning |
| --- | --- |
| `response_refusal` | A refusal content part was present without output text. |
| `response_incomplete_no_text` | The response was incomplete without output text. |
| `response_failed` | The response status was `failed`. |
| `response_no_text` | A recognized response completed without output text. |
| `response_parse_error` | The body was malformed or structurally unrecognized. |

Any non-blank output text succeeds, including partial text on an incomplete
response. The status allowlist is `cancelled`, `completed`, `failed`,
`in_progress`, `incomplete`, and `queued`; incomplete reasons are
`max_output_tokens` and `content_filter`; item types are `message`,
`function_call`, and `reasoning`; and content-part types are `output_text` and
`refusal`. Unknown values remain observable only through the stable `unknown`
value or count category, so schema additions do not discard otherwise valid
text or expose their raw strings.

## Schema evidence and compatibility boundary

This contract was checked on 2026-07-13 against OpenAI's generated official
Python schema for the [Response object](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response.py)
and the [Responses API reference](https://platform.openai.com/docs/api-reference/responses/object).
Those sources define `output` as the wire list, `output_text` as an SDK
convenience aggregation, `output_text` and `refusal` message content parts, the
documented response statuses, and the currently documented incomplete reasons
`max_output_tokens` and `content_filter`.

Forward compatibility for unknown future type and reason strings is a local
policy, not a claim that OpenAI supports any inferred shape. Unknown objects are
counted without retaining their raw type strings, and their payloads are not
interpreted even when they have a list-valued `content` field. Content-part
counts describe only message content. A missing type, a non-list `output`, or
malformed message content remains `response_parse_error`.

## Whetstone rollout dependency

Whetstone needs a coordinated consumer change before these diagnostics can be
used end to end. Successful response diagnostics currently require propagation
through Whetstone's provider boundary because that boundary copies
`provider_metadata` but not `LlmResponse.diagnostics`. Failure diagnostics are
nested at `ProviderFailure.metadata["diagnostics"]`, while the current consumer
looks for response status directly in failure metadata. Whetstone must also map
the new typed outcome codes (`response_refusal`, `response_incomplete_no_text`,
`response_failed`, and `response_no_text`) instead of treating them as generic
provider failures. That rollout is intentionally not implemented in this
repository.
