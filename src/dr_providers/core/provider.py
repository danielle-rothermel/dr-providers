"""The Provider protocol: the single-shot provider call interface.

A pure module (no httpx) so importing the protocol never pulls in the
transport. Both ``ScriptedProvider`` and ``HttpProvider`` implement it:
``complete`` returns the closed no-throw Provider Transport Outcome.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from dr_providers.modeling.request import ProviderCallRequest
    from dr_providers.outcomes.models import ProviderTransportOutcome


class Provider(Protocol):
    """The single-shot provider call interface."""

    def complete(
        self, request: ProviderCallRequest
    ) -> ProviderTransportOutcome: ...
