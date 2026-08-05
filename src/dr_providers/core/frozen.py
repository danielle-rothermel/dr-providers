"""Shared deep-immutability helpers for identity-bearing frozen models.

A ``_FrozenMap`` is a read-only, deep-copyable, pickle-safe mapping used to
make nested dict/list fields of otherwise-``frozen`` pydantic models actually
immutable, so a persisted structure (a Config's extra_body, an Evidence
record's redacted headers/identity payloads) can never drift from — or be
tampered into disagreeing with — the artifact it was serialized as.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class _FrozenMap(Mapping[str, Any]):
    """A read-only mapping that is deep-copyable and pickle-safe.

    ``types.MappingProxyType`` cannot be deep-copied on every CPython, and
    pydantic deep-copies field defaults, so a small explicit frozen mapping
    is used instead. Mutating item access raises, keeping the structure
    unable to drift from the owning model's persisted/identity form.
    """

    __slots__ = ("_data",)

    _data: dict[str, Any]

    def __init__(self, data: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_data", dict(data))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:  # pragma: no cover
        msg = "frozen mapping is immutable"
        raise TypeError(msg)

    def __iter__(self) -> Any:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"_FrozenMap({self._data!r})"

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenMap:
        return self

    def __copy__(self) -> _FrozenMap:
        return self


def _deep_freeze(value: Any) -> Any:
    """Recursively convert mappings/sequences into immutable equivalents.

    Mappings become read-only frozen maps and lists/tuples become tuples, so
    a nested structure cannot be mutated after construction and thus cannot
    desync from its owning model's persisted/identity form.
    """
    if isinstance(value, Mapping):
        return _FrozenMap({str(k): _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Recursively convert frozen maps/tuples back into plain dicts/lists."""
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value
