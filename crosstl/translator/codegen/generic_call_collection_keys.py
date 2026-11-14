"""Generic calls keep the instantiation that specialization collection deduced for them.

Collection deduces a call's type arguments from CrossGL types: ``bump(buffer_load(p, i))`` with ``p`` a
``RWStructuredBuffer<uint8>`` instantiates ``bump<uint8>``. Emission walks the same call again, but a generator
may by then describe its operands with the types it renders, which can be wider or spelled differently (HLSL stores
``uint8`` in ``uint`` and spells ``uint16`` as ``uint16_t``). Deducing again there named the ``bump<uint>``
instantiation, or cloned the primary for ``uint16_t``, so the explicit specializations ``bump<uint8>`` and
``bump<uint16>`` were silently skipped.

Collection visits every call node once, in one concrete function or one instantiation's clone, so its deduction
is the call's meaning. A call is identified by its callee name and the identity of its argument nodes; calls
without arguments are not remembered, because only explicit type arguments tell them apart.
"""

from __future__ import annotations

__all__ = [
    "collected_generic_call_key",
    "remember_collected_generic_call_key",
    "reset_collected_generic_call_keys",
]

_ATTRIBUTE = "generic_function_collected_call_keys"


def reset_collected_generic_call_keys(generator) -> None:
    setattr(generator, _ATTRIBUTE, {})


def _call_identity(func_name, args):
    if not func_name or not args:
        return None
    return func_name, tuple(id(arg) for arg in args)


def remember_collected_generic_call_key(generator, func_name, args, key) -> None:
    """Record the instantiation ``key`` collection deduced for the call ``func_name(args)``."""
    identity = _call_identity(func_name, args)
    if identity is None or key is None:
        return
    keys = getattr(generator, _ATTRIBUTE, None)
    if keys is None:
        keys = {}
        setattr(generator, _ATTRIBUTE, keys)
    # The argument nodes are kept with the key so their ids cannot be reused while the record lives.
    keys[identity] = (tuple(args), key)


def collected_generic_call_key(generator, func_name, args):
    """The instantiation collection deduced for this very call, if it is still specialized."""
    identity = _call_identity(func_name, args)
    if identity is None:
        return None
    record = (getattr(generator, _ATTRIBUTE, None) or {}).get(identity)
    if record is None:
        return None
    nodes, key = record
    if len(nodes) != len(args) or any(a is not b for a, b in zip(nodes, args)):
        return None
    specializations = getattr(generator, "generic_function_specializations", None) or {}
    return key if key in specializations else None
