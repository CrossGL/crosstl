"""OpenCL C vector component selections that CrossGL spells differently.

Component selection (OpenCL C 3.0 section 6.3.7). Besides ``.xyzw``, OpenCL C selects components by number
(``v.s0``, ``v.S13``, with hexadecimal digits up to ``sF``) and by half (``.lo``, ``.hi``, ``.even``, ``.odd``).
CrossGL has only the letter spelling, and the importer used to copy the OpenCL suffix verbatim, which DXC rejected
with "invalid format for vector swizzle". The importer now writes each selection with letters:

- ``.sN...`` selects the components numbered ``N...``; 0 to 3 are ``x``, ``y``, ``z`` and ``w``.
- For a vector of ``n`` components, ``.lo`` and ``.hi`` are its lower and upper halves and ``.even`` and ``.odd``
  its even- and odd-numbered components; a 2-component vector's halves are single components. A 3-component
  vector acts as a 4-component vector whose ``w`` is undefined, so its ``.hi`` reads ``(z, undefined)`` and its
  ``.odd`` ``(y, undefined)``: CrossGL has no component for the undefined one, so the importer refuses a
  selection that includes it, as a value or as an assignment target.
- A selection of a selection, such as ``v.odd.hi``, becomes one swizzle of the vector, ``v.w``.
- OpenCL's 8- and 16-component vectors import as CrossGL arrays, so a single numbered component becomes an index
  (``v.s7`` is ``v[7]``); a selection of several of their components raises
  :class:`OpenCLVectorComponentUnsupportedError`.

The halves depend on the vector's width, so the importer rewrites them only where it knows the selected
expression's type; a member of a structure keeps its name.
"""

from __future__ import annotations

import re

from crosstl.backend.HIP.HipAst import (
    BinaryOpNode,
    FunctionCallNode,
    MemberAccessNode,
    TernaryOpNode,
    UnaryOpNode,
)
from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

from .conversions import CONVERSION_RE

__all__ = [
    "OpenCLVectorComponentMixin",
    "OpenCLVectorComponentUnsupportedError",
]

_LETTERS = "xyzw"
_LETTER_INDEX = {
    **{c: i for i, c in enumerate("xyzw")},
    **{c: i for i, c in enumerate("rgba")},
}
_NUMBERED = re.compile(r"[sS](?P<digits>[0-9a-fA-F]+)")
_HALVES = frozenset({"lo", "hi", "even", "odd"})
_CROSSGL_VECTOR = re.compile(r"vec(?P<lanes>[234])<(?P<scalar>\w+)>")
_CROSSGL_ARRAY_VECTOR = re.compile(r"array<(?P<scalar>\w+),\s*(?P<lanes>8|16)>")
_CROSSGL_SCALARS = frozenset(
    {"bool", "i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64", "f16", "f32", "f64"}
)
_VLOAD = re.compile(r"vload(?:a_)?(?:_half)?(?P<lanes>2|3|4|8|16)")
# OpenCL C built-ins whose result is the scalar type of their (vector) argument.
_SCALAR_RESULT_FUNCTIONS = frozenset(
    {"dot", "length", "distance", "fast_length", "fast_distance", "any", "all"}
)


class OpenCLVectorComponentUnsupportedError(TranslationDiagnostic):
    """An OpenCL C vector component selection that CrossGL cannot write."""

    code = CODE_PREFIX + "opencl-vector-component-unsupported"


def _half_indices(member, lanes):
    """Component numbers of ``.lo``/``.hi``/``.even``/``.odd`` on ``lanes`` components; None marks an undefined one."""
    width = 4 if lanes == 3 else lanes
    indices = list(range(width))
    if member == "lo":
        selected = indices[: width // 2]
    elif member == "hi":
        selected = indices[width // 2 :]
    elif member == "even":
        selected = indices[0::2]
    else:
        selected = indices[1::2]
    return [index if index < lanes else None for index in selected]


class OpenCLVectorComponentMixin:
    """Rewrites OpenCL C component selections and 4-component ``cross`` calls into CrossGL."""

    opencl_lvalue_target = None

    def visit_lvalue_expression(self, node):
        previous = self.opencl_lvalue_target
        self.opencl_lvalue_target = node
        try:
            return super().visit_lvalue_expression(node)
        finally:
            self.opencl_lvalue_target = previous

    def visit_MemberAccessNode(self, node):
        selection = self.opencl_vector_selection(node)
        if selection is None:
            return super().visit_MemberAccessNode(node)
        is_target = node is self.opencl_lvalue_target
        self.opencl_lvalue_target = None
        base, base_type, indices = selection
        return self.format_opencl_vector_selection(
            node.member, base, base_type, indices, is_target
        )

    def opencl_vector_selection(self, node):
        """``(base, base CrossGL type, component numbers)`` of a component selection to rewrite, else None.

        Only selections the importer must respell are returned: a numbered or half selection, or a selection of
        one. A plain ``.xyzw`` selection of a vector keeps the inherited spelling.
        """
        if not self.is_opencl_component_selection(node):
            return None
        resolved = self.resolve_opencl_vector_selection(node)
        if resolved is None:
            return None
        base, base_type, indices, respelled = resolved
        if not respelled:
            return None
        return base, base_type, indices

    def is_opencl_component_selection(self, node):
        return (
            isinstance(node, MemberAccessNode)
            and not getattr(node, "is_pointer", False)
            and isinstance(node.member, str)
            and self.infer_record_expression_definition(node.object) is None
        )

    def resolve_opencl_vector_selection(self, node):
        """Fold nested selections into ``(base, base type, component numbers, respelled)``, or None if unknown."""
        member = node.member
        inner = None
        if self.is_opencl_component_selection(node.object):
            inner = self.resolve_opencl_vector_selection(node.object)
        if inner is not None:
            base, base_type, outer_indices, respelled = inner
            lanes = len(outer_indices)
        else:
            base = node.object
            base_type = self.opencl_expression_crossgl_type(base)
            lanes = self.opencl_crossgl_lanes(base_type)
            outer_indices = None
            respelled = False
            if lanes is None or lanes < 2:
                # Not a vector the importer knows: the member keeps its spelling.
                return None
        numbered = _NUMBERED.fullmatch(member)
        if numbered is not None:
            local = [int(digit, 16) for digit in numbered.group("digits")]
            respelled = True
        elif member in _HALVES:
            local = _half_indices(member, lanes)
            respelled = True
        elif member and all(c in _LETTER_INDEX for c in member):
            local = [_LETTER_INDEX[c] for c in member]
        else:
            return None
        if any(index is not None and index >= lanes for index in local):
            if not respelled:
                return None
            raise OpenCLVectorComponentUnsupportedError(
                f"component selection '.{member}' reads past the {lanes} components of the vector it selects from"
            )
        if outer_indices is not None:
            local = [None if index is None else outer_indices[index] for index in local]
            respelled = True
        return base, base_type, local, respelled

    def format_opencl_vector_selection(
        self, member, base, base_type, indices, is_target
    ):
        base_lanes = self.opencl_crossgl_lanes(base_type)
        if base_lanes is not None and base_lanes > 4:
            if len(indices) == 1 and indices[0] is not None:
                return f"{self.visit(base)}[{indices[0]}]"
            raise OpenCLVectorComponentUnsupportedError(
                f"component selection '.{member}' selects several components of an OpenCL C vector of "
                f"{base_lanes} components, which CrossGL holds as an array and cannot swizzle"
            )
        if None in indices:
            # OpenCL C 3.0 section 6.3.7: '.hi' and '.odd' of a 3-component vector include its undefined w.
            raise OpenCLVectorComponentUnsupportedError(
                f"component selection '.{member}' of a 3-component vector includes the undefined w component, "
                "which CrossGL has no component for"
            )
        if is_target and len(set(indices)) != len(indices):
            raise OpenCLVectorComponentUnsupportedError(
                f"assignment to '.{member}', which selects a component more than once"
            )
        letters = "".join(_LETTERS[index] for index in indices)
        return f"{self.visit(base)}.{letters}"

    def opencl_crossgl_lanes(self, crossgl_type):
        """Component count of a CrossGL scalar or vector type (8 and 16 for array-held vectors), else None."""
        if not isinstance(crossgl_type, str):
            return None
        crossgl_type = crossgl_type.strip()
        match = _CROSSGL_VECTOR.fullmatch(crossgl_type)
        if match is not None:
            return int(match.group("lanes"))
        match = _CROSSGL_ARRAY_VECTOR.fullmatch(crossgl_type)
        if match is not None:
            return int(match.group("lanes"))
        if crossgl_type in _CROSSGL_SCALARS:
            return 1
        return None

    def opencl_crossgl_scalar(self, crossgl_type):
        if not isinstance(crossgl_type, str):
            return None
        crossgl_type = crossgl_type.strip()
        for pattern in (_CROSSGL_VECTOR, _CROSSGL_ARRAY_VECTOR):
            match = pattern.fullmatch(crossgl_type)
            if match is not None:
                return match.group("scalar")
        return crossgl_type if crossgl_type in _CROSSGL_SCALARS else None

    def opencl_vector_type_of(self, scalar, lanes):
        if scalar is None or lanes is None:
            return None
        if lanes == 1:
            return scalar
        if lanes in (2, 3, 4):
            return f"vec{lanes}<{scalar}>"
        return f"array<{scalar}, {lanes}>"

    def opencl_expression_crossgl_type(self, expression):
        """The CrossGL scalar or vector type of an OpenCL C expression, or None when the importer cannot tell."""
        if isinstance(
            expression, MemberAccessNode
        ) and self.is_opencl_component_selection(expression):
            resolved = self.resolve_opencl_vector_selection(expression)
            if resolved is not None:
                _, base_type, indices, _ = resolved
                return self.opencl_vector_type_of(
                    self.opencl_crossgl_scalar(base_type), len(indices)
                )
        if isinstance(expression, FunctionCallNode) and isinstance(
            expression.name, str
        ):
            return self.opencl_call_crossgl_type(expression)
        if isinstance(expression, BinaryOpNode):
            left = self.opencl_expression_crossgl_type(expression.left)
            right = self.opencl_expression_crossgl_type(expression.right)
            left_lanes = self.opencl_crossgl_lanes(left)
            right_lanes = self.opencl_crossgl_lanes(right)
            if left_lanes is not None and right_lanes is not None:
                return left if left_lanes >= right_lanes else right
            # A vector with an operand of unknown type (a literal): the declared-type inference decides.
        if isinstance(expression, TernaryOpNode):
            return self.opencl_expression_crossgl_type(
                expression.true_expr
            ) or self.opencl_expression_crossgl_type(expression.false_expr)
        if isinstance(expression, UnaryOpNode) and expression.op in {"-", "+", "~"}:
            return self.opencl_expression_crossgl_type(expression.operand)
        declared = self.expression_declared_type(expression)
        if not declared:
            return None
        return self.convert_hip_type_to_crossgl(self.strip_type_qualifiers(declared))

    def opencl_call_crossgl_type(self, call):
        name = call.name
        args = getattr(call, "args", None) or []
        if name in self.OPENCL_VECTOR_TYPE_MAPPING:
            return self.OPENCL_VECTOR_TYPE_MAPPING[name]
        conversion = CONVERSION_RE.fullmatch(name)
        if conversion is not None:
            return self.convert_hip_type_to_crossgl(
                conversion.group("type") + (conversion.group("lanes") or "")
            )
        vload = _VLOAD.fullmatch(name)
        if vload is not None and len(args) == 2:
            pointee = self.opencl_pointee_type(args[1])
            scalar = self.opencl_crossgl_scalar(
                self.convert_hip_type_to_crossgl(self.strip_type_qualifiers(pointee))
                if pointee
                else None
            )
            if name.startswith(("vload_half", "vloada_half")):
                scalar = "f32"
            return self.opencl_vector_type_of(scalar, int(vload.group("lanes")))
        declared = self.expression_declared_type(call)
        if declared:
            return self.convert_hip_type_to_crossgl(
                self.strip_type_qualifiers(declared)
            )
        if not args:
            return None
        argument_types = [self.opencl_expression_crossgl_type(arg) for arg in args]
        if name in _SCALAR_RESULT_FUNCTIONS:
            return self.opencl_crossgl_scalar(argument_types[0])
        if name in self.OPENCL_GENERIC_ARGUMENT_TYPE_FUNCTIONS:
            lanes = [self.opencl_crossgl_lanes(t) for t in argument_types]
            if None in lanes:
                return None
            return argument_types[lanes.index(max(lanes))]
        return None

    # OpenCL C math, common, geometric and integer built-ins whose result has the type of their widest argument.
    OPENCL_GENERIC_ARGUMENT_TYPE_FUNCTIONS = frozenset(
        {
            "abs",
            "acos",
            "acosh",
            "asin",
            "asinh",
            "atan",
            "atan2",
            "atanh",
            "cbrt",
            "ceil",
            "clamp",
            "copysign",
            "cos",
            "cosh",
            "cross",
            "degrees",
            "exp",
            "exp2",
            "exp10",
            "fabs",
            "fast_normalize",
            "fdim",
            "floor",
            "fma",
            "fmax",
            "fmin",
            "fmod",
            "hypot",
            "log",
            "log2",
            "log10",
            "mad",
            "max",
            "min",
            "mix",
            "native_cos",
            "native_exp",
            "native_log",
            "native_recip",
            "native_rsqrt",
            "native_sin",
            "native_sqrt",
            "normalize",
            "pow",
            "pown",
            "powr",
            "radians",
            "rint",
            "round",
            "rsqrt",
            "sign",
            "sin",
            "sinh",
            "smoothstep",
            "sqrt",
            "step",
            "tan",
            "tanh",
            "trunc",
        }
    )
