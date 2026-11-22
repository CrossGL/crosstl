"""OpenCL C math built-ins that write a second result through a pointer (OpenCL C 3.0 section 6.15.2).

``fract(x, iptr)``, ``frexp(x, exp)``, ``modf(x, iptr)``, ``sincos(x, cosval)`` and ``remquo(x, y, quo)`` return
one result and store another through their last argument. CrossGL spells such a result as an ``out`` argument (see
``crosstl/translator/codegen/c_pointer_math.py``), so the importer passes the value the pointer addresses: ``&x``
becomes ``x``, ``p + i`` becomes ``p[i]`` and any other pointer ``p`` becomes ``*p``. The call copied verbatim passed
``&x`` where CrossGL expects an assignable value, and DXC rejected the HLSL with "operator is not supported".

- ``frexp`` and ``modf`` have the meaning of CrossGL's ``frexp(x, out e)`` and ``modf(x, out i)``, and ``remquo``
  that of CrossGL's C-family ``remquo(x, y, out q)``, so the call keeps its name.
- ``sincos`` returns the sine and stores the cosine, while CrossGL's ``sincos(x, out s, out c)`` returns nothing.
  The importer declares a CrossGL helper per argument type that evaluates the cosine and the sine the way it imports
  OpenCL's ``cos`` and ``sin``.
- ``fract`` returns ``fmin(x - floor(x), 0x1.fffffep-1f)`` and stores ``floor(x)``, with the edge cases of OpenCL C
  3.0 section 7.5.1: a NaN returns the NaN, an infinity returns a zero of its sign, and a zero returns itself. It is
  not CrossGL's one-operand ``fract``, which is ``x - floor(x)`` and reaches 1.0 for tiny negative operands, so the
  importer declares a helper per argument type that computes the OpenCL result component by component.

A pointer whose pointee type the importer cannot find, a cast pointer, or a helper type of 8 or 16 lanes, which
CrossGL has no vector for, raises :class:`OpenCLPointerResultUnsupportedError`.
"""

from __future__ import annotations

import re

from crosstl.backend.HIP.HipAst import (
    ArrayAccessNode,
    BinaryOpNode,
    CastNode,
    MemberAccessNode,
    UnaryOpNode,
)
from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = ["OpenCLPointerResultMixin", "OpenCLPointerResultUnsupportedError"]

# Built-in name -> (argument count, position of the pointer argument).
POINTER_RESULT_FUNCTIONS = {
    "fract": (2, 1),
    "frexp": (2, 1),
    "modf": (2, 1),
    "remquo": (3, 2),
    "sincos": (2, 1),
}
# Built-ins whose CrossGL function has the same name and an ``out`` argument in the pointer's place.
_SAME_NAME_FUNCTIONS = frozenset({"frexp", "modf", "remquo"})
_HELPER_FUNCTIONS = frozenset({"fract", "sincos"})
_FLOAT_TYPE = re.compile(
    r"(?:vec(?P<lanes>[234])<(?P<vector>f16|f32|f64)>|(?P<scalar>f16|f32|f64))"
)
_COMPONENTS = ("x", "y", "z", "w")
# The largest value below 1.0 in each format, written out exactly: 1 - 2^-11, 1 - 2^-24 and 1 - 2^-53.
_BELOW_ONE = {
    "f16": "0.99951171875",
    "f32": "0.999999940395355224609375",
    "f64": "0.99999999999999988897769753748434595763683319091796875",
}


class OpenCLPointerResultUnsupportedError(TranslationDiagnostic):
    """An OpenCL C built-in that stores a result through a pointer CrossGL cannot address or type."""

    code = CODE_PREFIX + "opencl-pointer-result-unsupported"


def _float_type_parts(crossgl_type):
    """``(scalar type, lane count)`` of a CrossGL floating-point scalar or vector type, or ``None``."""
    match = _FLOAT_TYPE.fullmatch(str(crossgl_type or "").replace(" ", ""))
    if match is None:
        return None
    if match.group("scalar"):
        return match.group("scalar"), 1
    return match.group("vector"), int(match.group("lanes"))


class OpenCLPointerResultMixin:
    """Import the OpenCL C built-ins of :data:`POINTER_RESULT_FUNCTIONS` with CrossGL ``out`` arguments."""

    def reset_opencl_pointer_results(self):
        # Helper key -> (name, lines), in the order the importer first needs them.
        self.opencl_pointer_result_helpers = {}

    def format_opencl_pointer_result_call(self, func_name, args):
        """The CrossGL call for one of :data:`POINTER_RESULT_FUNCTIONS`, or ``None`` for any other call."""
        entry = POINTER_RESULT_FUNCTIONS.get(func_name)
        if (
            entry is None
            or len(args) != entry[0]
            or self.is_user_defined_function(func_name)
        ):
            return None
        position = entry[1]
        pointer = args[position]
        rendered = [self.visit(arg) for arg in args[:position]]
        rendered.append(self.visit(self.opencl_pointed_value(func_name, pointer)))
        if func_name in _SAME_NAME_FUNCTIONS:
            return f"{func_name}({', '.join(rendered)})"
        crossgl_type = self.opencl_pointer_result_type(func_name, pointer)
        helper = self.require_opencl_pointer_result_helper(func_name, crossgl_type)
        return f"{helper}({', '.join(rendered)})"

    def opencl_pointed_value(self, func_name, pointer):
        """The assignable value ``pointer`` addresses, as an AST node the converter renders."""
        if isinstance(pointer, CastNode):
            raise OpenCLPointerResultUnsupportedError(
                f"{func_name} stores its result through a cast pointer; CrossGL out arguments cannot "
                "reinterpret the value they write"
            )
        if isinstance(pointer, UnaryOpNode) and pointer.op == "&":
            return pointer.operand
        base, offset = self.opencl_pointer_base_and_offset(pointer)
        if offset is not None:
            return ArrayAccessNode(base, offset)
        return UnaryOpNode("*", pointer)

    def opencl_pointer_base_and_offset(self, pointer):
        """``(p, i)`` for a pointer written ``p + i``, ``i + p`` or a sum of several offsets, else ``(pointer, None)``."""
        if isinstance(pointer, BinaryOpNode) and pointer.op == "+":
            for base_side, offset_side in (
                (pointer.left, pointer.right),
                (pointer.right, pointer.left),
            ):
                if self.opencl_pointee_type(base_side) is None:
                    continue
                base, offset = self.opencl_pointer_base_and_offset(base_side)
                if offset is None:
                    return base, offset_side
                return base, BinaryOpNode(offset, "+", offset_side)
        return pointer, None

    def opencl_pointee_type(self, pointer):
        """The OpenCL C type of the value ``pointer`` addresses, or ``None`` when the importer cannot tell."""
        if isinstance(pointer, UnaryOpNode) and pointer.op == "&":
            return self.expression_declared_type(pointer.operand) or None
        if isinstance(pointer, BinaryOpNode) and pointer.op == "+":
            return self.opencl_pointee_type(pointer.left) or self.opencl_pointee_type(
                pointer.right
            )
        if isinstance(pointer, (str, ArrayAccessNode, MemberAccessNode)):
            return (
                self.indexed_expression_type(self.expression_declared_type(pointer))
                or None
            )
        return None

    def opencl_pointer_result_type(self, func_name, pointer):
        """The CrossGL type of the value ``pointer`` addresses, refusing one the importer cannot find."""
        pointee = self.opencl_pointee_type(pointer)
        if not pointee:
            raise OpenCLPointerResultUnsupportedError(
                f"{func_name}: the type of the value its pointer argument addresses is not known"
            )
        return self.convert_hip_type_to_crossgl(self.strip_type_qualifiers(pointee))

    def require_opencl_pointer_result_helper(self, func_name, crossgl_type):
        """The name of the CrossGL helper for ``func_name`` on ``crossgl_type``, declaring it on first use."""
        parts = _float_type_parts(crossgl_type)
        if parts is None:
            raise OpenCLPointerResultUnsupportedError(
                f"{func_name} on {crossgl_type}: CrossGL has no floating-point scalar or vector of 2, 3 or 4 "
                "components of that type to declare the helper with"
            )
        scalar, lanes = parts
        key = (func_name, scalar, lanes)
        known = self.opencl_pointer_result_helpers.get(key)
        if known is not None:
            return known[0]
        if func_name == "fract" and lanes > 1:
            # The vector helper calls the scalar one, which must be declared first.
            self.require_opencl_pointer_result_helper(func_name, scalar)
        stem = (
            f"cl_{func_name}_{scalar}"
            if lanes == 1
            else f"cl_{func_name}_vec{lanes}_{scalar}"
        )
        name = self.opencl_unused_helper_name(stem)
        if func_name == "fract":
            lines = self.opencl_fract_helper_lines(name, scalar, lanes)
        else:
            lines = self.opencl_sincos_helper_lines(name, crossgl_type)
        self.opencl_pointer_result_helpers[key] = (name, lines)
        return name

    def opencl_unused_helper_name(self, stem):
        taken = set(getattr(self, "source_identifier_names", ()) or ())
        taken.update(name for name, _ in self.opencl_pointer_result_helpers.values())
        name, suffix = stem, 1
        while name in taken:
            name = f"{stem}_{suffix}"
            suffix += 1
        return name

    def opencl_fract_helper_lines(self, name, scalar, lanes):
        if lanes == 1:
            return [
                f"// OpenCL C fract(x, iptr) on {scalar} (OpenCL C 3.0 sections 6.15.2.1 and 7.5.1).",
                f"{scalar} {name}({scalar} x, out {scalar} whole) {{",
                f"    var floor_x: {scalar} = floor(x);",
                "    whole = floor_x;",
                "    if (isnan(x)) {",
                "        return x;",
                "    }",
                "    if (isinf(x)) {",
                "        return (x > 0.0) ? 0.0 : -0.0;",
                "    }",
                "    if (x == 0.0) {",
                "        return x;",
                "    }",
                f"    return min(x - floor_x, {_BELOW_ONE[scalar]});",
                "}",
            ]
        scalar_helper = self.opencl_pointer_result_helpers[("fract", scalar, 1)][0]
        vector = f"vec{lanes}<{scalar}>"
        lines = [
            f"// OpenCL C fract(x, iptr) on {vector}, component by component.",
            f"{vector} {name}({vector} x, out {vector} whole) {{",
            f"    var result: {vector};",
            f"    var part: {scalar};",
        ]
        for component in _COMPONENTS[:lanes]:
            lines.append(
                f"    result.{component} = {scalar_helper}(x.{component}, part);"
            )
            lines.append(f"    whole.{component} = part;")
        lines.extend(["    return result;", "}"])
        return lines

    def opencl_sincos_helper_lines(self, name, crossgl_type):
        sine = self.convert_hip_builtin_function("sin")
        cosine = self.convert_hip_builtin_function("cos")
        return [
            f"// OpenCL C sincos(x, cosval) on {crossgl_type}: returns the sine and stores the cosine.",
            f"{crossgl_type} {name}({crossgl_type} x, out {crossgl_type} cosine) {{",
            f"    cosine = {cosine}(x);",
            f"    return {sine}(x);",
            "}",
        ]

    def opencl_pointer_result_helper_text(self):
        """The CrossGL declarations of every helper the program needs, or ``""``."""
        lines = []
        for _, helper_lines in self.opencl_pointer_result_helpers.values():
            lines.extend(helper_lines)
            lines.append("")
        return "\n".join(lines)
