from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Protocol, runtime_checkable

from pydantic import ConfigDict, Field, RootModel, StrictStr

from dr_providers.lifecycle.outcomes import ProviderInvocationOutcome

if TYPE_CHECKING:
    from dr_providers.outcomes.models import ProviderTransportResponse


class SemanticResponseClassifierIdentifier(
    RootModel[Annotated[StrictStr, Field(min_length=1)]]
):
    """Opaque caller-owned stable identifier for classifier behavior."""

    model_config = ConfigDict(frozen=True)


@runtime_checkable
class SemanticResponseClassifier(Protocol):
    @property
    def identifier(self) -> SemanticResponseClassifierIdentifier: ...

    def classify(
        self, response: ProviderTransportResponse
    ) -> ProviderInvocationOutcome: ...


def classify_semantic_response(
    classifier: SemanticResponseClassifier,
    response: ProviderTransportResponse,
) -> ProviderInvocationOutcome:
    """Apply the classifier while rejecting outcomes outside its boundary."""
    outcome = classifier.classify(response)
    allowed = {
        ProviderInvocationOutcome.SUCCESS,
        ProviderInvocationOutcome.SEMANTIC_REJECTION,
    }
    if outcome not in allowed:
        msg = (
            "semantic response classifier must return success or semantic "
            f"rejection, got {outcome!r}"
        )
        raise ValueError(msg)
    return outcome
