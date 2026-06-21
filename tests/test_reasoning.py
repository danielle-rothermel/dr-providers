from dr_providers import ReasoningSpec, RequestControls
from dr_providers.names import EffortLevel


def test_from_reasoning_none_returns_empty_controls() -> None:
    controls = RequestControls.from_reasoning(None)
    assert controls.extra_body == {}
    assert controls.warnings == []


def test_from_reasoning_enabled_flag() -> None:
    controls = RequestControls.from_reasoning(
        ReasoningSpec(enabled=True),
    )
    assert controls.extra_body == {"reasoning": {"enabled": True}}


def test_from_reasoning_effort() -> None:
    controls = RequestControls.from_reasoning(
        ReasoningSpec(effort=EffortLevel.HIGH),
    )
    assert controls.extra_body == {"reasoning": {"effort": EffortLevel.HIGH}}


def test_from_reasoning_default_payload() -> None:
    controls = RequestControls.from_reasoning(ReasoningSpec())
    assert controls.extra_body == {"reasoning": {"reasoning": True}}
