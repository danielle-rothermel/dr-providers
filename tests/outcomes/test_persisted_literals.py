"""Golden pins for every literal that lands in persisted payloads.

Consumers store ``ProviderCallResult.model_dump(mode="json")`` and
``ProviderInvocationEvidence.model_dump(mode="json")`` whole, so every enum
member value, failure ``code``, wire-format prefix, and schema version below
is a recorded-data format, not an implementation detail. Editing an expected
value in this module is a recorded-data format change: already-recorded
payloads keep the old literal, and identity hashes computed over the old
literal do not match the new one. Intentional changes update these
expectations in the same change that edits the literal.

The expectations are hand-written raw strings on purpose. Deriving them from
the enums or constants under test would make the pin restate whatever the
code currently says and catch nothing.

A literal defined in more than one module is pinned once per definition,
under an aliased import naming the module that defines it, so drift in the
producing copy fails here even when the consuming copy still agrees.
"""

from __future__ import annotations

from dr_providers.core.failures import RecoverabilityClass
from dr_providers.lifecycle.classifier import (
    ACCEPT_ALL_SEMANTIC_CLASSIFIER_IDENTIFIER,
    HTTP_STATUS_402_CODE,
)
from dr_providers.lifecycle.classifier import (
    INVALID_BASE_URL_CODE as CLASSIFIER_INVALID_BASE_URL_CODE,
)
from dr_providers.lifecycle.classifier import (
    MISSING_API_KEY_CODE as CLASSIFIER_MISSING_API_KEY_CODE,
)
from dr_providers.lifecycle.classifier import (
    MISSING_BASE_URL_CODE as CLASSIFIER_MISSING_BASE_URL_CODE,
)
from dr_providers.lifecycle.models import (
    COMPLETED_INVOCATION_OBSERVATION_SCHEMA,
    COMPLETED_INVOCATION_OBSERVATION_SCHEMA_VERSION,
    DECIDED_INVOCATION_RECORD_SCHEMA,
    DECIDED_INVOCATION_RECORD_SCHEMA_VERSION,
    PROVIDER_CALL_RESULT_SCHEMA,
    PROVIDER_CALL_RESULT_SCHEMA_VERSION,
    PROVIDER_CALL_SCHEMA,
    PROVIDER_CALL_SCHEMA_VERSION,
    PROVIDER_CALL_STATE_SCHEMA_VERSION,
    PROVIDER_RETRY_INSTRUCTION_SCHEMA_VERSION,
    ProviderRetryDelaySource,
)
from dr_providers.lifecycle.outcomes import (
    ProviderCallOutcomeKind,
    ProviderInvocationOutcome,
)
from dr_providers.lifecycle.policy import (
    PROVIDER_CALL_RETRY_POLICY_SCHEMA,
    PROVIDER_CALL_RETRY_POLICY_SCHEMA_VERSION,
    CustomProviderCallRetryPolicy,
    StandardProviderCallRetryPolicy,
)
from dr_providers.modeling.call import (
    PROVIDER_CALL_CONFIG_SCHEMA,
    PROVIDER_CALL_CONFIG_SCHEMA_VERSION,
    PROVIDER_CALL_DEFINITION_SCHEMA,
    PROVIDER_CALL_DEFINITION_SCHEMA_VERSION,
)
from dr_providers.modeling.controls import (
    ReasoningEffort,
    ReasoningRequestShape,
    RequestControl,
    TokenLimitParameter,
)
from dr_providers.modeling.request import (
    PROVIDER_CALL_REQUEST_SCHEMA,
    PROVIDER_CALL_REQUEST_SCHEMA_VERSION,
)
from dr_providers.modeling.route import Protocol, ProviderKind
from dr_providers.modeling.transcript import MessageRole
from dr_providers.outcomes.conformance import (
    MODEL_SUBSTITUTION_CODE,
    REASONING_NOT_OBSERVED_CODE,
)
from dr_providers.outcomes.evidence import (
    PROVIDER_INVOCATION_EVIDENCE_SCHEMA,
    PROVIDER_INVOCATION_EVIDENCE_SCHEMA_VERSION,
    SANITIZE_KEYS,
    ProviderRetryAfterHint,
    sanitize_headers,
)
from dr_providers.outcomes.models import (
    INVALID_JSON_CODE,
    POOL_TIMEOUT_CODE,
    STALLED_RESPONSE_CODE,
    TIMEOUT_CODE,
    TIMEOUT_CODES,
    ProviderStopReason,
    TransportTimeoutContainment,
    WarningSeverity,
)
from dr_providers.surfaces.testing.scripted import SCRIPTED_RESPONSE_ID_PREFIX
from dr_providers.translation.common import (
    PARSE_ERROR_CODE,
    PROVIDER_ERROR_ENVELOPE_CODE,
    RESPONSE_INCOMPLETE_NO_TEXT_CODE,
    RESPONSE_NO_TEXT_CODE,
)
from dr_providers.translation.responses import (
    RESPONSE_FAILED_CODE,
    RESPONSE_REFUSAL_CODE,
    RESPONSES_CONTENT_PART_TYPE_VALUES,
    RESPONSES_INCOMPLETE_REASON_VALUES,
    RESPONSES_OUTPUT_ITEM_TYPE_VALUES,
    RESPONSES_STATUS_VALUES,
    UNKNOWN_DIAGNOSTIC_CATEGORY,
)
from dr_providers.transport.http import (
    INVALID_BASE_URL_CODE as TRANSPORT_INVALID_BASE_URL_CODE,
)
from dr_providers.transport.http import (
    MISSING_API_KEY_CODE as TRANSPORT_MISSING_API_KEY_CODE,
)
from dr_providers.transport.http import (
    MISSING_BASE_URL_CODE as TRANSPORT_MISSING_BASE_URL_CODE,
)
from dr_providers.transport.http import (
    REDIRECT_STATUS_CODE_PREFIX,
    REQUEST_TOO_LARGE_CODE,
    RESPONSE_TOO_LARGE_CODE,
)
from dr_providers.transport.wire_failures import (
    HTTP_STATUS_CODE_PREFIX,
    REMOTE_PROTOCOL_ERROR_CODE,
    TRANSPORT_ERROR_CODE,
    TRANSPORT_PROTOCOL_ERROR_CODE,
)

EXPECTED_RECOVERABILITY_LITERALS = [
    "permanent",
    "transient",
    "rate_limited",
    "resource_exhaustion",
    "unknown",
]
EXPECTED_STOP_REASON_LITERALS = ["stop", "length", "content_filter"]
EXPECTED_TIMEOUT_CONTAINMENT_LITERALS = ["contained", "uncontained"]
EXPECTED_WARNING_SEVERITY_LITERALS = ["info", "warning", "critical"]
EXPECTED_PROVIDER_KIND_LITERALS = [
    "openrouter",
    "openai",
    "gemini",
    "anthropic",
]
EXPECTED_PROTOCOL_LITERALS = [
    "chat_completions",
    "responses",
    "anthropic_messages",
]
EXPECTED_MESSAGE_ROLE_LITERALS = ["system", "user", "assistant", "tool"]
EXPECTED_REQUEST_CONTROL_LITERALS = [
    "temperature",
    "top_p",
    "token_limit",
    "reasoning",
    "seed",
]
EXPECTED_TOKEN_LIMIT_PARAMETER_LITERALS = [
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
]
EXPECTED_REASONING_EFFORT_LITERALS = [
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
]
EXPECTED_REASONING_SHAPE_LITERALS = [
    "none",
    "effort_field",
    "reasoning_object",
]
EXPECTED_RETRY_DELAY_SOURCE_LITERALS = ["provider_call_retry_policy"]
EXPECTED_INVOCATION_OUTCOME_LITERALS = [
    "success",
    "empty_generation",
    "truncated_no_text",
    "missing_generation_text",
    "budget_exhausted",
    "missing_credential",
    "missing_transport_config",
    "never_sent",
    "malformed_response",
    "provider_rejection",
    "semantic_rejection",
    "permanent_provider_or_transport_failure",
    "transient_provider_or_network_failure",
    "rate_limiting",
    "resource_exhaustion",
    "contained_transport_timeout",
    "uncontained_deadline_expiration",
    "unknown_transport_failure",
]
EXPECTED_CALL_OUTCOME_KIND_LITERALS = [
    "accepted",
    "invocation_outcome",
    "draining_cancellation",
    "policy_exhaustion",
]


def test_recoverability_class_literals_are_pinned() -> None:
    assert [member.value for member in RecoverabilityClass] == (
        EXPECTED_RECOVERABILITY_LITERALS
    )
    assert len(RecoverabilityClass) == 5


def test_transport_outcome_literals_are_pinned() -> None:
    assert [member.value for member in ProviderStopReason] == (
        EXPECTED_STOP_REASON_LITERALS
    )
    assert len(ProviderStopReason) == 3
    assert [member.value for member in TransportTimeoutContainment] == (
        EXPECTED_TIMEOUT_CONTAINMENT_LITERALS
    )
    assert len(TransportTimeoutContainment) == 2
    assert [member.value for member in WarningSeverity] == (
        EXPECTED_WARNING_SEVERITY_LITERALS
    )
    assert len(WarningSeverity) == 3


def test_route_literals_are_pinned() -> None:
    assert [member.value for member in ProviderKind] == (
        EXPECTED_PROVIDER_KIND_LITERALS
    )
    assert len(ProviderKind) == 4
    assert [member.value for member in Protocol] == EXPECTED_PROTOCOL_LITERALS
    assert len(Protocol) == 3


def test_transcript_literals_are_pinned() -> None:
    assert [member.value for member in MessageRole] == (
        EXPECTED_MESSAGE_ROLE_LITERALS
    )
    assert len(MessageRole) == 4


def test_control_literals_are_pinned() -> None:
    assert [member.value for member in RequestControl] == (
        EXPECTED_REQUEST_CONTROL_LITERALS
    )
    assert len(RequestControl) == 5
    assert [member.value for member in TokenLimitParameter] == (
        EXPECTED_TOKEN_LIMIT_PARAMETER_LITERALS
    )
    assert len(TokenLimitParameter) == 3
    assert [member.value for member in ReasoningEffort] == (
        EXPECTED_REASONING_EFFORT_LITERALS
    )
    assert len(ReasoningEffort) == 6
    assert [member.value for member in ReasoningRequestShape] == (
        EXPECTED_REASONING_SHAPE_LITERALS
    )
    assert len(ReasoningRequestShape) == 3


def test_outcome_literals_are_pinned() -> None:
    assert [member.value for member in ProviderInvocationOutcome] == (
        EXPECTED_INVOCATION_OUTCOME_LITERALS
    )
    assert len(ProviderInvocationOutcome) == 18
    assert [member.value for member in ProviderCallOutcomeKind] == (
        EXPECTED_CALL_OUTCOME_KIND_LITERALS
    )
    assert len(ProviderCallOutcomeKind) == 4


def test_retry_delay_source_literals_are_pinned() -> None:
    assert [member.value for member in ProviderRetryDelaySource] == (
        EXPECTED_RETRY_DELAY_SOURCE_LITERALS
    )
    assert len(ProviderRetryDelaySource) == 1


def test_transport_failure_codes_are_pinned() -> None:
    assert TIMEOUT_CODE == "timeout"
    assert STALLED_RESPONSE_CODE == "stalled_response"
    assert POOL_TIMEOUT_CODE == "pool_timeout"
    assert sorted(TIMEOUT_CODES) == [
        "pool_timeout",
        "stalled_response",
        "timeout",
    ]
    assert INVALID_JSON_CODE == "invalid_response_json"
    assert REQUEST_TOO_LARGE_CODE == "request_body_too_large"
    assert RESPONSE_TOO_LARGE_CODE == "response_body_too_large"
    assert TRANSPORT_ERROR_CODE == "transport_error"
    assert TRANSPORT_PROTOCOL_ERROR_CODE == "transport_protocol_error"
    assert REMOTE_PROTOCOL_ERROR_CODE == "transport_remote_protocol_error"
    assert CLASSIFIER_MISSING_API_KEY_CODE == "missing_api_key"
    assert TRANSPORT_MISSING_API_KEY_CODE == "missing_api_key"
    assert CLASSIFIER_MISSING_BASE_URL_CODE == "missing_base_url"
    assert TRANSPORT_MISSING_BASE_URL_CODE == "missing_base_url"
    assert CLASSIFIER_INVALID_BASE_URL_CODE == "invalid_base_url"
    assert TRANSPORT_INVALID_BASE_URL_CODE == "invalid_base_url"


def test_protocol_failure_codes_are_pinned() -> None:
    assert PARSE_ERROR_CODE == "response_parse_error"
    assert RESPONSE_NO_TEXT_CODE == "response_no_text"
    assert RESPONSE_REFUSAL_CODE == "response_refusal"
    assert RESPONSE_INCOMPLETE_NO_TEXT_CODE == "response_incomplete_no_text"
    assert RESPONSE_FAILED_CODE == "response_failed"
    assert PROVIDER_ERROR_ENVELOPE_CODE == "provider_error_envelope"


def test_conformance_warning_codes_are_pinned() -> None:
    assert REASONING_NOT_OBSERVED_CODE == "reasoning_not_observed"
    assert MODEL_SUBSTITUTION_CODE == "model_substitution"


def test_http_status_code_format_is_pinned() -> None:
    assert HTTP_STATUS_CODE_PREFIX == "http_status_"
    assert f"{HTTP_STATUS_CODE_PREFIX}429" == "http_status_429"
    assert HTTP_STATUS_402_CODE == "http_status_402"
    assert REDIRECT_STATUS_CODE_PREFIX == "http_redirect_"
    assert f"{REDIRECT_STATUS_CODE_PREFIX}302" == "http_redirect_302"


def test_responses_diagnostic_literals_are_pinned() -> None:
    assert UNKNOWN_DIAGNOSTIC_CATEGORY == "unknown"
    assert sorted(RESPONSES_STATUS_VALUES) == [
        "cancelled",
        "completed",
        "failed",
        "in_progress",
        "incomplete",
        "queued",
    ]
    assert sorted(RESPONSES_INCOMPLETE_REASON_VALUES) == [
        "content_filter",
        "max_output_tokens",
    ]
    assert sorted(RESPONSES_OUTPUT_ITEM_TYPE_VALUES) == [
        "function_call",
        "message",
        "reasoning",
    ]
    assert sorted(RESPONSES_CONTENT_PART_TYPE_VALUES) == [
        "output_text",
        "refusal",
    ]


def test_discriminator_literals_are_pinned() -> None:
    assert StandardProviderCallRetryPolicy().model_dump(mode="json") == {
        "policy_type": "standard",
        "maximum_invocations": 1,
    }
    custom = CustomProviderCallRetryPolicy(
        maximum_invocations=2,
        eligible_outcomes=frozenset(),
        declared_delays_seconds=(0.0,),
    )
    assert custom.model_dump(mode="json")["policy_type"] == "custom"
    assert ProviderRetryAfterHint(kind="delta_seconds", value=1).model_dump(
        mode="json"
    ) == {"kind": "delta_seconds", "value": 1}
    assert ProviderRetryAfterHint(
        kind="http_date", value="Wed, 21 Oct 2015 07:28:00 GMT"
    ).model_dump(mode="json") == {
        "kind": "http_date",
        "value": "Wed, 21 Oct 2015 07:28:00 GMT",
    }


def test_evidence_redaction_literals_are_pinned() -> None:
    assert sanitize_headers({"Authorization": "secret"}) == {
        "Authorization": "<redacted>"
    }
    assert sorted(SANITIZE_KEYS) == [
        "api_base",
        "api_key",
        "authorization",
        "base_url",
        "model_list",
        "x-api-key",
        "x-goog-api-key",
    ]


def test_identifier_and_id_prefix_literals_are_pinned() -> None:
    assert ACCEPT_ALL_SEMANTIC_CLASSIFIER_IDENTIFIER.root == (
        "dr_providers.accept_all_semantic_response.v1"
    )
    assert SCRIPTED_RESPONSE_ID_PREFIX == "scripted-response"


def test_identity_schema_names_are_pinned() -> None:
    assert PROVIDER_CALL_DEFINITION_SCHEMA == (
        "dr_providers.provider_call_definition"
    )
    assert PROVIDER_CALL_CONFIG_SCHEMA == "dr_providers.provider_call_config"
    assert PROVIDER_CALL_REQUEST_SCHEMA == "dr_providers.provider_call_request"
    assert PROVIDER_CALL_RETRY_POLICY_SCHEMA == (
        "dr_providers.provider_call_retry_policy"
    )
    assert PROVIDER_INVOCATION_EVIDENCE_SCHEMA == (
        "dr_providers.provider_invocation_evidence"
    )
    assert COMPLETED_INVOCATION_OBSERVATION_SCHEMA == (
        "dr_providers.completed_invocation_observation"
    )
    assert DECIDED_INVOCATION_RECORD_SCHEMA == (
        "dr_providers.decided_invocation_record"
    )
    assert PROVIDER_CALL_SCHEMA == "dr_providers.provider_call"
    assert PROVIDER_CALL_RESULT_SCHEMA == "dr_providers.provider_call_result"


def test_persisted_schema_versions_are_pinned() -> None:
    assert PROVIDER_INVOCATION_EVIDENCE_SCHEMA_VERSION == 8
    assert PROVIDER_CALL_DEFINITION_SCHEMA_VERSION == 3
    assert PROVIDER_CALL_CONFIG_SCHEMA_VERSION == 1
    assert PROVIDER_CALL_REQUEST_SCHEMA_VERSION == 1
    assert PROVIDER_CALL_RETRY_POLICY_SCHEMA_VERSION == 1
    assert COMPLETED_INVOCATION_OBSERVATION_SCHEMA_VERSION == 1
    assert DECIDED_INVOCATION_RECORD_SCHEMA_VERSION == 1
    assert PROVIDER_CALL_SCHEMA_VERSION == 1
    assert PROVIDER_CALL_STATE_SCHEMA_VERSION == 1
    assert PROVIDER_CALL_RESULT_SCHEMA_VERSION == 1
    assert PROVIDER_RETRY_INSTRUCTION_SCHEMA_VERSION == 1
