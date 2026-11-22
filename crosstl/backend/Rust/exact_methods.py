"""Rust methods whose CrossGL namesake rounds or signs differently, written with the operations Rust computes.

- Rust's ``f32::fract`` and glam's vector ``fract`` are ``x - x.trunc()`` (core ``f32::fract``; glam 0.34
  ``src/f32/vec3.rs``), where CrossGL's ``fract`` is ``x - floor(x)``, which glam names ``fract_gl``.
- ``rem_euclid`` is the Euclidean remainder, which is never negative. core computes ``r = x % y`` and returns
  ``r + |y|`` when ``r < 0`` (core ``f32::rem_euclid``, ``i32::rem_euclid``; glam applies it per element), where
  CrossGL's ``mod`` takes the sign of ``y``. For an unsigned integer it is ``x % y``.
- ``div_euclid`` is the quotient that goes with it: core computes ``q = x / y`` (truncated, ``trunc(x / y)`` for a
  float) and returns ``q - 1`` when ``x % y < 0`` and ``y > 0``, ``q + 1`` when ``x % y < 0`` and ``y < 0``
  (core ``f32::div_euclid``, ``i32::div_euclid``). For an unsigned integer it is ``x / y``.
- glam's ``x.smoothstep(edge0, edge1)`` interpolates its receiver; CrossGL's ``smoothstep`` takes the edges first.

The lowerings repeat an operand, so they need operands without side effects; any other operand is refused, and so
is a ``rem_euclid`` whose operand type the importer cannot tell.
"""

from __future__ import annotations

import re

from .RustAst import BinaryOpNode, CastNode, FunctionCallNode, UnaryOpNode

EXACT_METHODS = frozenset(
    {"fract", "fract_gl", "rem_euclid", "div_euclid", "smoothstep"}
)

_VECTOR = re.compile(r"^(vec|ivec|uvec|dvec)([234])$")
_FLOAT_SCALARS = {"float", "double", "half"}
_SIGNED_SCALARS = {"int", "int8_t", "int16_t", "int64_t"}
_UNSIGNED_SCALARS = {"uint", "uint8_t", "uint16_t", "uint64_t"}
_VECTOR_KINDS = {"vec": "float", "dvec": "float", "ivec": "signed", "uvec": "unsigned"}
_COMPONENTS = "xyzw"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_GLAM_CONSTRUCTOR = re.compile(
    r"^(?:(?:spirv_std::)?glam::)?(?:[DIU]|I64|U64)?Vec[234]A?::(?:new|splat)$"
)


class RustExactMethodMixin:
    """Members of ``RustToCrossGLConverter`` for :data:`EXACT_METHODS`."""

    def format_exact_method_call(
        self, method_name, obj, args, arg_nodes, receiver_type, receiver_node
    ):
        """The CrossGL of ``obj.method_name(args)`` for a method in :data:`EXACT_METHODS`, or ``None``."""
        if method_name == "fract_gl" and not args:
            return f"fract({obj})"
        if method_name == "fract" and not args:
            if not self._exact_operands_are_pure(obj, receiver_node, arg_nodes):
                return self._refuse_exact_method(method_name)
            return f"({obj} - trunc({obj}))"
        if method_name == "smoothstep" and len(args) == 2:
            if not self._exact_operands_are_pure(obj, receiver_node, arg_nodes):
                return self._refuse_exact_method(method_name)
            return f"smoothstep({args[0]}, {args[1]}, {obj})"
        if method_name in _EUCLIDEAN and len(args) == 1:
            return self._format_euclidean(
                method_name, obj, args[0], arg_nodes, receiver_type, receiver_node
            )
        return None

    def _format_euclidean(
        self, method_name, obj, divisor, arg_nodes, receiver_type, receiver_node
    ):
        shape = self._exact_numeric_shape(receiver_type)
        if shape is None and arg_nodes:
            shape = self._exact_numeric_shape(self.infer_value_type(arg_nodes[0]))
        if shape is None:
            return self._refuse_exact_method(
                method_name, "whose operand type is unknown"
            )
        kind, width, constructor = shape
        element, unsigned_operator = _EUCLIDEAN[method_name]
        if kind == "unsigned":
            return f"({obj} {unsigned_operator} {divisor})"
        if not self._exact_operands_are_pure(obj, receiver_node, arg_nodes):
            return self._refuse_exact_method(method_name)
        if width == 1:
            return element(kind, obj, divisor)
        elements = ", ".join(
            element(kind, f"{obj}.{c}", f"{divisor}.{c}") for c in _COMPONENTS[:width]
        )
        return f"{constructor}({elements})"

    def _exact_numeric_shape(self, value_type):
        """``(kind, width, constructor)`` of a numeric scalar or vector type, ``kind`` float, signed or unsigned."""
        if not isinstance(value_type, str) or not value_type:
            return None
        mapped = self.map_type(self.normalize_receiver_type(value_type))
        if not isinstance(mapped, str):
            return None
        mapped = mapped.strip()
        if mapped in _FLOAT_SCALARS:
            return "float", 1, mapped
        if mapped in _SIGNED_SCALARS:
            return "signed", 1, mapped
        if mapped in _UNSIGNED_SCALARS:
            return "unsigned", 1, mapped
        match = _VECTOR.match(mapped)
        if match is None:
            return None
        return _VECTOR_KINDS[match.group(1)], int(match.group(2)), mapped

    def _exact_operands_are_pure(self, obj, receiver_node, arg_nodes):
        if receiver_node is None:
            if not _IDENTIFIER.match(obj):
                return False
        elif not self._exact_operand_is_pure(receiver_node):
            return False
        return all(self._exact_operand_is_pure(node) for node in arg_nodes)

    def _exact_operand_is_pure(self, node):
        """Whether evaluating ``node`` twice computes the same value and changes nothing.

        glam's vector constructors (``Vec3::new``, ``Vec3::splat``) are pure as well as what
        ``expression_has_side_effects`` accepts.
        """
        if isinstance(node, FunctionCallNode) and isinstance(node.name, str):
            if not _GLAM_CONSTRUCTOR.match(
                self.resolve_imported_module_path(node.name)
            ):
                return False
            return all(self._exact_operand_is_pure(arg) for arg in node.args)
        if isinstance(node, BinaryOpNode):
            return self._exact_operand_is_pure(
                node.left
            ) and self._exact_operand_is_pure(node.right)
        if isinstance(node, (UnaryOpNode, CastNode)):
            return self._exact_operand_is_pure(
                node.operand if isinstance(node, UnaryOpNode) else node.expression
            )
        return not self.expression_has_side_effects(node)

    def _refuse_exact_method(
        self, method_name, detail="with an operand that has side effects"
    ):
        return self.refuse_import(
            f"'{method_name}' {detail}: the importer writes Rust's {method_name} by repeating its operands",
            placeholder="0",
        )


def _euclidean_remainder(kind, x, y):
    """Rust's ``r = x % y; if r < 0 { r + |y| } else { r }`` for one float or signed integer element."""
    remainder = f"fmod({x}, {y})" if kind == "float" else f"({x} % {y})"
    zero = "0.0" if kind == "float" else "0"
    return f"(({remainder} < {zero}) ? ({remainder} + abs({y})) : {remainder})"


def _euclidean_quotient(kind, x, y):
    """Rust's ``q = x / y; if x % y < 0 { if y > 0 { q - 1 } else { q + 1 } } else { q }`` for one element."""
    if kind == "float":
        quotient, remainder, zero, one = (
            f"trunc({x} / {y})",
            f"fmod({x}, {y})",
            "0.0",
            "1.0",
        )
    else:
        quotient, remainder, zero, one = f"({x} / {y})", f"({x} % {y})", "0", "1"
    return (
        f"(({remainder} < {zero}) ? (({y} > {zero}) ? ({quotient} - {one}) : ({quotient} + {one}))"
        f" : {quotient})"
    )


_EUCLIDEAN = {
    "rem_euclid": (_euclidean_remainder, "%"),
    "div_euclid": (_euclidean_quotient, "/"),
}
