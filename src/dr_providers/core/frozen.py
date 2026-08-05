from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class _FrozenMap(Mapping[str, Any]):
    """Frozen mapping compatible with Pydantic's default deepcopy."""

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
    """Prevent nested mutation from desynchronizing persisted identity."""
    if isinstance(value, Mapping):
        return _FrozenMap({str(k): _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value
