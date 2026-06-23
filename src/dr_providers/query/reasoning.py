from __future__ import annotations

from typing import Any

from dr_providers.config import ReasoningSpec  # noqa: TC001


def _reasoning_extra_body(reasoning: ReasoningSpec | None) -> dict[str, Any]:
    if reasoning is None:
        return {}
    if reasoning.enabled is not None:
        payload: dict[str, Any] = {"enabled": reasoning.enabled}
    elif reasoning.effort is not None:
        payload = {"effort": reasoning.effort}
    else:
        payload = {"reasoning": True}
    return {"reasoning": payload}
