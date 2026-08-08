from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Never


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


class _FrozenJsonDict(dict[str, Any]):
    """JSON-native immutable dictionary for identity-bearing model fields."""

    def _reject_mutation(self, *args: Any, **kwargs: Any) -> Never:
        del args, kwargs
        msg = "frozen JSON dictionary is immutable"
        raise TypeError(msg)

    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    clear = _reject_mutation
    pop = _reject_mutation
    popitem = _reject_mutation
    setdefault = _reject_mutation
    update = _reject_mutation
    __ior__ = _reject_mutation

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenJsonDict:
        del memo
        return self

    def __copy__(self) -> _FrozenJsonDict:
        return self


class _FrozenJsonList(list[Any]):
    """JSON-native immutable list for identity-bearing model fields."""

    def _reject_mutation(self, *args: Any, **kwargs: Any) -> Never:
        del args, kwargs
        msg = "frozen JSON list is immutable"
        raise TypeError(msg)

    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    append = _reject_mutation
    clear = _reject_mutation
    extend = _reject_mutation
    insert = _reject_mutation
    pop = _reject_mutation
    remove = _reject_mutation
    reverse = _reject_mutation
    sort = _reject_mutation
    __iadd__ = _reject_mutation
    __imul__ = _reject_mutation

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenJsonList:
        del memo
        return self

    def __copy__(self) -> _FrozenJsonList:
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


def _freeze_json(value: Any) -> Any:
    """Freeze strict-JSON containers while retaining ``json.dumps`` support."""
    if isinstance(value, Mapping):
        return _FrozenJsonDict(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return _FrozenJsonList(_freeze_json(item) for item in value)
    return value
