# Responses normalization

`parse_responses_body()` parses the OpenAI Responses wire body in one pass over
`output[]`. Matching the official SDK aggregation, it concatenates
`output_text` content parts only from `message` items, in wire order, and adds a
content-free `ResponsesDiagnostics` envelope to successful
`LlmResponse` values. Failures carry the same envelope in
`ProviderFailure.metadata["diagnostics"]`.

The envelope records response status, incomplete reason, provider error code,
item and content-type counts, text and refusal lengths, a 16-character SHA-256
response-ID hash, and the model echoed by the provider. It never copies prompt,
output, refusal, tool arguments, or raw response IDs into failure metadata.
Successful responses retain the existing raw `provider_metadata` contract.

No-text outcomes are permanent and typed:

| Code | Meaning |
| --- | --- |
| `response_refusal` | A refusal content part was present without output text. |
| `response_incomplete_no_text` | The response was incomplete without output text. |
| `response_failed` | The response status was `failed`. |
| `response_no_text` | A recognized response completed without output text. |
| `response_parse_error` | The body was malformed or structurally unrecognized. |

Any non-blank output text succeeds, including partial text on an incomplete
response. Unknown item types, content-part types, statuses, and incomplete
reasons remain opaque strings in diagnostics so schema additions do not discard
otherwise valid text.

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
counted by item type but their payloads are not interpreted, even when they have
a list-valued `content` field. Content-part counts describe only message
content. A missing type, a non-list `output`, or malformed message content
remains `response_parse_error`.

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
