"""Non-uniform resource indices: ``NonUniformResourceIndex(i)`` and ``nonuniformEXT(i)``.

CrossGL accepts both spellings of the qualifier that marks an index into a resource array as not dynamically
uniform: HLSL's ``NonUniformResourceIndex`` (shader model 5.1, "Resource binding in HLSL") and GLSL's
``nonuniformEXT`` (GL_EXT_nonuniform_qualifier). Both return their argument unchanged; they only tell the compiler
that the descriptor the index selects may differ between invocations.

Each emitter spells the qualifier its own way, so :func:`normalize_non_uniform_indices` rewrites every call of
either spelling in place before emission:

* GLSL keeps ``nonuniformEXT`` and enables ``GL_EXT_nonuniform_qualifier``;
* HLSL keeps ``NonUniformResourceIndex`` (the DirectX emitter already maps the GLSL spelling);
* MSL has no such qualifier: a Metal array of textures, samplers or buffers may be indexed with any value (MSL 3.1
  section 2.12, "Arrays of textures, texture arrays, and samplers"), so the call is replaced by its argument, as
  SPIRV-Cross does for the ``NonUniform`` decoration.

A call with other than one argument is left as written for the emitter to reject. The rewrite is undone by
:func:`non_uniform_indices_spelled`, so one AST can be emitted for several targets in turn.
"""

from __future__ import annotations

from contextlib import contextmanager

from .. import ast as A
from .hub_types import call_name
from .rewrite import child_slots, get_child, set_child

__all__ = [
    "NON_UNIFORM_INDEX_SPELLINGS",
    "non_uniform_indices_spelled",
    "normalize_non_uniform_indices",
]

NON_UNIFORM_INDEX_SPELLINGS = frozenset({"NonUniformResourceIndex", "nonuniformEXT"})


def _rewritten(node, spelling):
    """The replacement of a one-argument non-uniform qualifier call, or ``None`` to keep ``node``."""
    if not isinstance(node, A.FunctionCallNode):
        return None
    name = call_name(node)
    arguments = list(getattr(node, "arguments", None) or [])
    if name not in NON_UNIFORM_INDEX_SPELLINGS or len(arguments) != 1:
        return None
    if spelling is None:
        return arguments[0]
    if name == spelling:
        return None
    renamed = A.FunctionCallNode(A.IdentifierNode(spelling), arguments)
    for attribute in ("source_location", "expression_type"):
        if hasattr(node, attribute):
            setattr(renamed, attribute, getattr(node, attribute))
    return renamed


def normalize_non_uniform_indices(node, spelling, _seen=None, _undo=None):
    """Spell every non-uniform qualifier call under ``node`` as ``spelling``; ``None`` drops the qualifier.

    ``node`` is rewritten in place. A dropped qualifier is replaced by its argument, which is rewritten too.
    Returns the ``(parent, field, key, original)`` slots that changed, outermost first.
    """
    if _seen is None:
        _seen, _undo = set(), []
    if id(node) in _seen:
        return _undo
    _seen.add(id(node))
    for field, key in list(child_slots(node)):
        child = get_child(node, field, key)
        replacement = _rewritten(child, spelling)
        if replacement is not None:
            _undo.append((node, field, key, child))
        while replacement is not None:
            set_child(node, field, key, replacement)
            child = replacement
            replacement = _rewritten(child, spelling)
        normalize_non_uniform_indices(child, spelling, _seen, _undo)
    return _undo


@contextmanager
def non_uniform_indices_spelled(ast, spelling):
    """Rewrite the non-uniform qualifiers of ``ast`` as ``spelling`` for the duration of the block."""
    undo = normalize_non_uniform_indices(ast, spelling)
    try:
        yield ast
    finally:
        for parent, field, key, original in reversed(undo):
            set_child(parent, field, key, original)
