"""Process-wide reuse of the explicit template instantiations found in Metal source.

Project translation creates a fresh :class:`MetalPreprocessor` for every entry point, and each one scans the same
include-expanded translation unit for ``template [[host_name(...)]]`` declarations, instantiation macros and other
explicit instantiations. On a translation unit of several hundred kilobytes that scan costs seconds, and it is repeated
for every entry of the unit even though its result is identical.

The scan is a deterministic function of the source text and of the struct specializations the preprocessor has
materialized so far, which template-argument deduction consults. The memo key is a digest of exactly those inputs, so
a hit is only possible when the scan would return the same instantiations. Callers receive fresh copies, and failed
scans are not stored.
"""

import dataclasses
import hashlib
from collections import OrderedDict
from threading import Lock
from typing import Callable, List, Optional, Tuple, TypeVar

T = TypeVar("T")

MEMO_ENTRY_LIMIT = 16

# Preprocessor attributes, besides the source text, that the scan reads. The
# struct specializations steer template-argument deduction of declarations
# such as ``template void f(Box_float);``.
METAL_INSTANTIATION_STATE = (
    "_materialized_struct_specializations",
    "_materialized_struct_primary_templates",
    "group_non_type_template_substitutions",
)
# Methods compared by identity, so a replaced class-level implementation never
# reuses a result computed by an earlier one.
METAL_INSTANTIATION_GUARDS = (
    "_find_raw_mlx_kernel_instantiations",
    "_find_declared_mlx_kernel_instantiations",
    "_find_declared_template_instantiations",
    "_dedupe_template_instantiations",
    "_find_template_functions",
)

_MemoKey = Tuple[str, int, str]
_MemoEntry = Tuple[Tuple[object, ...], Tuple[object, ...]]
_memo: "OrderedDict[_MemoKey, _MemoEntry]" = OrderedDict()
_lock = Lock()
_stats = {"hits": 0, "misses": 0, "bypassed": 0}


class _Uncanonical(Exception):
    """A preprocessor state value has no order-independent canonical form."""


def _canonical(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted((_canonical(item) for item in value), key=repr)))
    if isinstance(value, dict):
        items = [(_canonical(key), _canonical(item)) for key, item in value.items()]
        return ("dict", tuple(sorted(items, key=repr)))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__qualname__,
            tuple(
                (item.name, _canonical(getattr(value, item.name)))
                for item in dataclasses.fields(value)
            ),
        )
    raise _Uncanonical(type(value).__qualname__)


def _state_digest(preprocessor: object, state_attributes: Tuple[str, ...]) -> str:
    state = tuple(
        (name, _canonical(getattr(preprocessor, name))) for name in state_attributes
    )
    return hashlib.sha256(repr(state).encode("utf-8")).hexdigest()


def _has_instance_overrides(preprocessor: object) -> bool:
    # An instance attribute that shadows a method (a test double, for example)
    # changes what the scan computes, so the shared memo must not answer it.
    cls = type(preprocessor)
    return any(callable(getattr(cls, name, None)) for name in vars(preprocessor))


def _detached(item: T) -> T:
    """A copy of a frozen dataclass whose container fields are copied too."""
    if not dataclasses.is_dataclass(item) or isinstance(item, type):
        raise TypeError(f"cannot detach {type(item).__qualname__}")
    containers = {}
    for field in dataclasses.fields(item):
        value = getattr(item, field.name)
        if isinstance(value, (list, dict, set)):
            containers[field.name] = type(value)(value)
    return dataclasses.replace(item, **containers) if containers else item


def memoized_instantiations(
    preprocessor: object,
    code: str,
    scan: Callable[[str], List[T]],
    *,
    exact_type: type,
    state_attributes: Tuple[str, ...] = METAL_INSTANTIATION_STATE,
    guard_methods: Tuple[str, ...] = METAL_INSTANTIATION_GUARDS,
) -> List[T]:
    """Return ``scan(code)``, reusing an earlier result for identical inputs.

    ``state_attributes`` names every preprocessor attribute the scan reads.
    Preprocessors of another type, or with instance-level method overrides,
    always scan, because the scan they run may differ from the memoized one.
    Results must be dataclasses; callers receive detached copies.
    """
    key: Optional[_MemoKey] = None
    if type(preprocessor) is exact_type and not _has_instance_overrides(preprocessor):
        try:
            state = _state_digest(preprocessor, state_attributes)
        except _Uncanonical:
            state = None
        if state is not None:
            digest = hashlib.sha256(code.encode("utf-8", "surrogatepass")).hexdigest()
            key = (digest, len(code), state)
    if key is None:
        with _lock:
            _stats["bypassed"] += 1
        return scan(code)
    guards = tuple(getattr(exact_type, name) for name in guard_methods)
    cached = None
    with _lock:
        entry = _memo.get(key)
        if entry is not None and all(
            old is new for old, new in zip(entry[0], guards, strict=True)
        ):
            _memo.move_to_end(key)
            _stats["hits"] += 1
            cached = entry[1]
    if cached is not None:
        return [_detached(item) for item in cached]
    result = scan(code)
    stored = tuple(_detached(item) for item in result)
    with _lock:
        _stats["misses"] += 1
        _memo[key] = (guards, stored)
        _memo.move_to_end(key)
        while len(_memo) > MEMO_ENTRY_LIMIT:
            _memo.popitem(last=False)
    return result


def memo_info() -> dict:
    """Hit, miss, bypass and size counters, for tests and profiling."""
    with _lock:
        return {**_stats, "size": len(_memo), "limit": MEMO_ENTRY_LIMIT}


def clear_memo() -> None:
    with _lock:
        _memo.clear()
        for name in _stats:
            _stats[name] = 0
