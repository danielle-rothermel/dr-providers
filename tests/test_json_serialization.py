"""Deterministic JSON serialization for unordered definition fields."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dr_providers import (
    ControlConstraints,
    ModelRoute,
    Protocol,
    ProviderCallDefinition,
    ProviderKind,
    RequestControl,
    TokenLimitParameter,
)

SUBPROCESS_WATCHDOG_SECONDS = 60
HASH_SEEDS = ("0", "1", "4", "4242")
SOURCE_ROOT = Path(__file__).parent.parent / "src"

EXPECTED_JSON_DUMP = {
    "schema_version": 1,
    "definition_id": "test.chat",
    "route": {
        "provider": "openai",
        "protocol": "chat_completions",
        "model": "m",
    },
    "constraints": {
        "supported_controls": [
            "reasoning",
            "temperature",
            "token_limit",
            "top_p",
        ],
        "token_limit_parameter": "max_completion_tokens",
        "reasoning_shape": "none",
        "allow_unsupported_control_drop": False,
    },
    "required_controls": ["temperature", "top_p"],
    "extension_keys": ["alpha", "middle", "zeta"],
}

EXPECTED_JSON_BYTES = (
    b'{"schema_version":1,"definition_id":"test.chat","route":'
    b'{"provider":"openai","protocol":"chat_completions","model":"m"},'
    b'"constraints":{"supported_controls":["reasoning","temperature",'
    b'"token_limit","top_p"],"token_limit_parameter":'
    b'"max_completion_tokens","reasoning_shape":"none",'
    b'"allow_unsupported_control_drop":false},"required_controls":'
    b'["temperature","top_p"],"extension_keys":'
    b'["alpha","middle","zeta"]}'
)

MODEL_DUMP_JSON_SCRIPT = """
import sys

from dr_providers import (
    ControlConstraints,
    ModelRoute,
    Protocol,
    ProviderCallDefinition,
    ProviderKind,
    RequestControl,
    TokenLimitParameter,
)

definition = ProviderCallDefinition(
    definition_id="test.chat",
    route=ModelRoute(
        provider=ProviderKind.OPENAI,
        protocol=Protocol.CHAT_COMPLETIONS,
        model="m",
    ),
    constraints=ControlConstraints(
        supported_controls=frozenset(RequestControl),
        token_limit_parameter=TokenLimitParameter.MAX_COMPLETION_TOKENS,
    ),
    required_controls=frozenset(
        {RequestControl.TOP_P, RequestControl.TEMPERATURE}
    ),
    extension_keys=frozenset({"zeta", "alpha", "middle"}),
)
sys.stdout.buffer.write(definition.model_dump_json().encode("utf-8"))
"""


def _definition() -> ProviderCallDefinition:
    return ProviderCallDefinition(
        definition_id="test.chat",
        route=ModelRoute(
            provider=ProviderKind.OPENAI,
            protocol=Protocol.CHAT_COMPLETIONS,
            model="m",
        ),
        constraints=ControlConstraints(
            supported_controls=frozenset(RequestControl),
            token_limit_parameter=(TokenLimitParameter.MAX_COMPLETION_TOKENS),
        ),
        required_controls=frozenset(
            {RequestControl.TOP_P, RequestControl.TEMPERATURE}
        ),
        extension_keys=frozenset({"zeta", "alpha", "middle"}),
    )


def test_definition_json_dump_orders_unordered_fields_exactly() -> None:
    assert _definition().model_dump(mode="json") == EXPECTED_JSON_DUMP


def test_definition_python_dump_preserves_frozensets() -> None:
    dumped = _definition().model_dump()
    supported_controls = dumped["constraints"]["supported_controls"]
    required_controls = dumped["required_controls"]
    extension_keys = dumped["extension_keys"]

    assert isinstance(supported_controls, frozenset)
    assert supported_controls == frozenset(RequestControl)
    assert all(
        isinstance(control, RequestControl) for control in supported_controls
    )
    assert isinstance(required_controls, frozenset)
    assert required_controls == frozenset(
        {RequestControl.TOP_P, RequestControl.TEMPERATURE}
    )
    assert all(
        isinstance(control, RequestControl) for control in required_controls
    )
    assert isinstance(extension_keys, frozenset)
    assert extension_keys == frozenset({"zeta", "alpha", "middle"})


def _model_dump_json_with_seed(seed: str) -> bytes:
    env = {
        **os.environ,
        "PYTHONHASHSEED": seed,
        "PYTHONPATH": str(SOURCE_ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", MODEL_DUMP_JSON_SCRIPT],
        capture_output=True,
        check=True,
        env=env,
        timeout=SUBPROCESS_WATCHDOG_SECONDS,
    )
    return completed.stdout


def test_model_dump_json_is_byte_identical_across_hash_seeds() -> None:
    outputs = {seed: _model_dump_json_with_seed(seed) for seed in HASH_SEEDS}

    assert outputs == dict.fromkeys(HASH_SEEDS, EXPECTED_JSON_BYTES)
    assert len(set(outputs.values())) == 1, outputs
