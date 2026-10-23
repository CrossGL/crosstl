"""Non-uniform resource indices on targets that have no non-uniform qualifier.

HLSL marks a resource index that may differ between invocations with ``NonUniformResourceIndex(i)``, and GLSL with
``nonuniformEXT(i)`` (GL_EXT_nonuniform_qualifier). CrossGL accepts both spellings: SPIR-V lowers both to a
``NonUniform`` decoration, the HLSL generator emits both as ``NonUniformResourceIndex`` and the GLSL generator both
as ``nonuniformEXT``. Two targets have no such qualifier, and :func:`strip_non_uniform_qualifiers` replaces each call
with its operand for them, in place:

* MSL indexes arrays of textures, samplers and buffers with any integer expression and has no non-uniform
  qualifier;
* GLSL ES 3.00 (``webgl``) has neither the extension nor dynamically indexed resource arrays: its generator
  selects each resource with a constant index, so the qualifier has nothing left to mark.

The qualifier changes no value: ``a[NonUniformResourceIndex(i)]`` selects the same element as ``a[i]``. A program
that defines a function with either name keeps its calls.
"""

from __future__ import annotations

from .. import ast as A
from .hub_types import call_name, iter_functions
from .rewrite import child_slots, get_child, set_child

__all__ = ["NON_UNIFORM_INDEX_FUNCTIONS", "strip_non_uniform_qualifiers"]

NON_UNIFORM_INDEX_FUNCTIONS = frozenset({"NonUniformResourceIndex", "nonuniformEXT"})
TARGETS = frozenset({"metal", "webgl"})


def _rewrite(node, replace):
    for field, key in list(child_slots(node)):
        child = get_child(node, field, key)
        _rewrite(child, replace)
        replacement = replace(child)
        if replacement is not child:
            set_child(node, field, key, replacement)


def strip_non_uniform_qualifiers(program, target) -> int:
    """Replace the non-uniform index calls of ``program`` by their operands for ``target``; returns how many."""
    if str(target or "").lower() not in TARGETS or not isinstance(
        program, A.ShaderNode
    ):
        return 0
    functions = list(iter_functions(program))
    defined = {getattr(function, "name", None) for function in functions}
    names = NON_UNIFORM_INDEX_FUNCTIONS - defined
    changed = 0

    def replace(node):
        nonlocal changed
        if not isinstance(node, A.FunctionCallNode) or call_name(node) not in names:
            return node
        arguments = list(getattr(node, "arguments", None) or [])
        if len(arguments) != 1:
            return node
        changed += 1
        return arguments[0]

    for function in functions:
        body = getattr(function, "body", None)
        if body is not None:
            _rewrite(function, replace)
    return changed
