"""GLSL: a shared array named whole as an array value.

The OpenGL generator registers every one-dimensional shared array as the root
of a workgroup pointer, so that a Metal ``threadgroup T *`` parameter can be
lowered to subscripts of that array. Naming such a pointer bare is refused,
because GLSL has no pointers. The shared array itself is an ordinary GLSL
array, though: passing it to a by-value array parameter copies it in (GLSL
4.60 section 6.1.1), as HLSL does for ``groupshared`` arrays. This module lets
exactly that root name through and leaves every pointer alias refused.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

_NAME_NODES = frozenset({"VariableNode", "IdentifierNode"})


def shared_array_value_root(codegen: Any, argument: Any, param_type: Any) -> str | None:
    """Return the shared array ``argument`` names when it fills an array parameter."""
    if isinstance(argument, str):
        name = argument
    elif type(argument).__name__ in _NAME_NODES:
        name = getattr(argument, "name", None)
    else:
        return None
    if not isinstance(name, str) or param_type is None:
        return None
    if not codegen.is_array_type(param_type):
        return None
    binding = codegen.current_glsl_workgroup_pointer_aliases.get(name)
    if not binding or binding.get("kind") != "workgroup-pointer":
        return None
    if binding.get("root") != name or binding.get("offset") not in (0, "0"):
        return None
    return name


@contextmanager
def allow_shared_array_value(codegen: Any, name: str | None) -> Iterator[None]:
    """Let ``name`` be emitted bare while the argument is generated."""
    previous = getattr(codegen, "glsl_shared_array_value_names", frozenset())
    if name is not None:
        codegen.glsl_shared_array_value_names = previous | {name}
    try:
        yield
    finally:
        codegen.glsl_shared_array_value_names = previous


def is_shared_array_value(codegen: Any, name: Any) -> bool:
    """Whether ``name`` is a shared array currently passed as an array value."""
    return name in getattr(codegen, "glsl_shared_array_value_names", frozenset())
