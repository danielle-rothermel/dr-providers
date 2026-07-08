"""The Provider protocol: the single-shot provider call interface.

A pure module (no httpx) so importing the protocol never pulls in the
transport. Both ``FixtureProvider`` and ``HttpProvider`` implement it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from dr_providers.request import LlmRequest
    from dr_providers.response import LlmResponse


class Provider(Protocol):
    """The single-shot provider call interface."""

    def complete(self, request: LlmRequest) -> LlmResponse: ...
