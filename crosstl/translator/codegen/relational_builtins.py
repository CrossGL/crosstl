"""GLSL's vector relational built-ins and componentwise ``!``, for the emitters that spell them as operators.

CrossGL's relational and equality operators are componentwise on vectors (ADR 0002): ``a < b`` on two ``vec3`` is a
``bvec3``, as in HLSL, MSL, WGSL, Slang and SPIR-V.  CrossGL spells its built-in functions as GLSL does, so it also
accepts GLSL 4.60 section 8.7's ``lessThan``, ``lessThanEqual``, ``greaterThan``, ``greaterThanEqual``, ``equal``,
``notEqual`` and ``not``.  Those names exist only in GLSL; every other emitter calls :func:`lower_relational_builtins`
first, which rewrites them into the operators they are defined as.  A user function of the same name is left alone,
and so is a call with the wrong number of arguments, which the emitter then refuses as an unknown call.

The GLSL emitter goes the other way: :func:`glsl_vector_logical_not` spells a componentwise ``!`` as ``not``.  MSL
and WGSL take ``!`` only on bool operands, so :func:`metal_logical_not_operand` and :func:`wgsl_logical_not_operand`
convert a numeric operand to bool first, componentwise, which is what ``!`` on it means in CrossGL and HLSL.
"""

from __future__ import annotations

import re

from .. import ast as A
from ..semantics.hub_types import call_name, iter_functions
from ..semantics.rewrite import child_slots, get_child, set_child
from .builtin_result_types import type_shape

RELATIONAL_BUILTIN_OPERATORS = {
    "lessThan": "<",
    "lessThanEqual": "<=",
    "greaterThan": ">",
    "greaterThanEqual": ">=",
    "equal": "==",
    "notEqual": "!=",
}
LOGICAL_NOT_BUILTIN = "not"


def lower_relational_builtins(program):
    """Rewrite GLSL relational built-in calls in ``program`` into CrossGL operators, in place; return the count."""
    if program is None:
        return 0
    user_functions = {
        getattr(function, "name", None) for function in iter_functions(program)
    }
    lowered = {
        name
        for name in (*RELATIONAL_BUILTIN_OPERATORS, LOGICAL_NOT_BUILTIN)
        if name not in user_functions
    }
    if not lowered:
        return 0
    return _RelationalBuiltinLowering(lowered).visit_children(program)


class _RelationalBuiltinLowering:
    def __init__(self, names):
        self.names = names
        self.visited = set()

    def visit_children(self, node):
        count = 0
        if id(node) in self.visited:
            return count
        self.visited.add(id(node))
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            count += self.visit_children(child)
            replacement = self.rewrite(child)
            if replacement is not child:
                set_child(node, field, key, replacement)
                count += 1
        return count

    def rewrite(self, node):
        if not isinstance(node, A.FunctionCallNode):
            return node
        name = call_name(node)
        if name not in self.names or getattr(node, "generic_args", None):
            return node
        arguments = list(getattr(node, "arguments", None) or [])
        if name == LOGICAL_NOT_BUILTIN:
            if len(arguments) != 1:
                return node
            replacement = A.UnaryOpNode("!", arguments[0])
        else:
            if len(arguments) != 2:
                return node
            replacement = A.BinaryOpNode(
                arguments[0], RELATIONAL_BUILTIN_OPERATORS[name], arguments[1]
            )
        _copy_metadata(node, replacement)
        return replacement


def _copy_metadata(original, replacement):
    for attribute in ("source_location", "parent"):
        value = getattr(original, attribute, None)
        if value is not None:
            setattr(replacement, attribute, value)
    annotations = getattr(original, "annotations", None)
    if isinstance(annotations, dict) and annotations:
        replacement.annotations = dict(annotations)


def glsl_vector_logical_not(emitter, operand):
    """GLSL for CrossGL ``!operand`` when ``operand`` is a vector, or ``None`` when it is a scalar or untyped.

    GLSL's ``!`` takes only a scalar ``bool``; its componentwise form is ``not(bvecN)`` (GLSL 4.60 section 8.7).  A
    numeric vector is first converted with ``bvecN(...)``, whose components are ``x != 0`` (section 5.4.1), so the
    result is ``x == 0`` componentwise, as CrossGL's ``!`` on a numeric vector is in HLSL, MSL and WGSL.
    """
    operand_type = emitter.glsl_source_expression_type(operand)
    info = emitter.glsl_value_type_info(operand_type)
    if info is None or info["width"] == 1:
        return None
    if info["family"] not in {"bool", "int", "uint", "float"}:
        return None
    generated = emitter.generate_expression(operand)
    if info["family"] == "bool":
        return f"not({generated})"
    return f"not(bvec{info['width']}({generated}))"


_WGSL_VECTOR = re.compile(r"vec([234])<(f16|f32|i32|u32|bool)>")
_WGSL_NUMERIC_SCALARS = frozenset({"f16", "f32", "i32", "u32"})


def metal_logical_not_operand(operand, operand_type):
    """``operand`` ready for MSL's ``!``: a numeric vector becomes ``boolN(operand)`` (MSL 4.1 section 2.13)."""
    shape = type_shape(operand_type)
    if shape is None or shape.matrix is not None or shape.width == 1:
        return operand
    if shape.scalar == "bool":
        return operand
    return f"bool{shape.width}({operand})"


def wgsl_logical_not_operand(operand, operand_type):
    """``operand`` ready for WGSL's ``!``, which takes only ``bool`` and ``vecN<bool>`` (WGSL section 8.6)."""
    type_name = re.sub(r"\s+", "", str(operand_type or ""))
    vector = _WGSL_VECTOR.fullmatch(type_name)
    if vector is not None and vector.group(2) != "bool":
        return f"vec{vector.group(1)}<bool>({operand})"
    if type_name in _WGSL_NUMERIC_SCALARS:
        return f"bool({operand})"
    return operand
