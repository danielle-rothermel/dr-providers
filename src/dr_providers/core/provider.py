from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from dr_providers.modeling.request import ProviderCallRequest
    from dr_providers.outcomes.evidence import ProviderInvocationEvidence


class Provider(Protocol):
    def invoke(
        self, request: ProviderCallRequest
    ) -> ProviderInvocationEvidence: ...
