"""Metal values of 8- and 16-bit integer vectors.

The Metal emitter keeps a CrossGL ``u16vec2`` value in a 32-bit carrier register (``uint2``), while struct fields
and buffer elements keep the native storage type (``ushort2``). Two consequences need explicit code:

- Metal does not convert between vector types implicitly, so a native storage read that meets a carrier in an
  operator or a conditional is widened to the carrier first.
- CrossGL integer promotion applies to scalars only; vector arithmetic keeps its narrow type and wraps at that width
  (MSL section 2.2 and OpenCL C 6.3 do not promote vectors, which is the rule ``semantics/narrow.py`` legalizes for
  the other targets). A carrier does not wrap by itself, so an operation whose CrossGL result is a narrow vector is
  reduced to the native width and widened again.

Conditionals whose arms are vectors of different component types take the usual arithmetic conversions, as in
CrossGL, GLSL and HLSL; Metal would reject them.
"""

from ..ast import ArrayAccessNode, LiteralNode, MemberAccessNode, UnaryOpNode

__all__ = ["MetalNarrowVectorMixin"]

# Operators whose result can leave the operand range (``semantics/narrow.py`` uses the same set).
_RANGE_CHANGING_BINARY = frozenset({"+", "-", "*", "/", "%", "<<"})
_RANGE_CHANGING_UNARY = frozenset({"-", "~"})
_COMPONENT_RANKS = {"bool": 0, "int": 1, "uint": 2, "half": 3, "float": 4}


def _wrapped_by(rendered, callee):
    """Whether ``rendered`` is exactly one call of ``callee``."""
    prefix = f"{callee}("
    if not rendered.startswith(prefix) or not rendered.endswith(")"):
        return False
    depth = 0
    for index, character in enumerate(rendered[len(prefix) - 1 :]):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return index == len(rendered) - len(prefix)
    return False


class MetalNarrowVectorMixin:
    """Narrow integer vector conversions of ``MetalCodeGen``."""

    def metal_narrow_vector_types(self, value_type):
        """``(carrier, native)`` Metal types of an 8- or 16-bit integer vector, else ``None``."""
        if value_type is None:
            return None
        native = self.metal_native_narrow_bitcast_storage_type(value_type)
        if native is None or native[-1:] not in {"2", "3", "4"}:
            return None
        carrier = self.map_type(value_type)
        if not isinstance(carrier, str) or carrier == native:
            return None
        return carrier, native

    def metal_narrow_shape(self, value_type):
        """``(native scalar, lanes)`` of an 8- or 16-bit integer scalar or vector type."""
        if value_type is None:
            return None
        native = self.metal_native_narrow_bitcast_storage_type(value_type)
        if native is None:
            return None
        if native[-1:] in {"2", "3", "4"}:
            return native[:-1], int(native[-1])
        return native, 1

    def metal_narrow_vector_operand(self, expr, rendered):
        """Widen a field or element read of native narrow vector storage to its carrier."""
        if not isinstance(expr, (MemberAccessNode, ArrayAccessNode)):
            return rendered
        types = self.metal_narrow_vector_types(self.expression_result_type(expr))
        if types is None or _wrapped_by(rendered, types[0]):
            return rendered
        return f"{types[0]}({rendered})"

    def metal_narrow_vector_result_types(self, expr, operator):
        """``(carrier, native)`` when the CrossGL result of ``expr`` is a narrow vector that may leave its range."""
        if isinstance(expr, UnaryOpNode):
            if operator not in _RANGE_CHANGING_UNARY:
                return None
            return self.metal_narrow_vector_types(
                self.expression_result_type(expr.operand)
            )
        if operator not in _RANGE_CHANGING_BINARY:
            return None
        operand_types = [
            self.expression_result_type(operand) for operand in (expr.left, expr.right)
        ]
        shapes = [self.metal_narrow_shape(value_type) for value_type in operand_types]
        if shapes[0] is None and shapes[1] is None:
            return None
        if None in shapes:
            # Only a literal adopts the narrow type of the other operand.
            other = expr.right if shapes[1] is None else expr.left
            if not isinstance(other, LiteralNode):
                return None
        known = [shape for shape in shapes if shape is not None]
        if len({scalar for scalar, _lanes in known}) != 1:
            return None
        lanes = {count for _scalar, count in known}
        if len(lanes) > 1 and 1 not in lanes:
            return None
        result_lanes = max(lanes)
        if result_lanes == 1:
            # Scalars promote to int before arithmetic.
            return None
        vector_type = next(
            value_type
            for value_type, shape in zip(operand_types, shapes)
            if shape is not None and shape[1] == result_lanes
        )
        return self.metal_narrow_vector_types(vector_type)

    def metal_narrow_vector_wrap(self, expr, operator, rendered):
        """Reduce a narrow vector result to its native width, keeping the carrier type."""
        types = self.metal_narrow_vector_result_types(expr, operator)
        if types is None:
            return rendered
        carrier, native = types
        return f"{carrier}({native}({rendered}))"

    @staticmethod
    def metal_narrow_is_normalized(rendered, carrier, native):
        """Whether ``rendered`` already is ``carrier(native(...))``."""
        return _wrapped_by(rendered, carrier) and _wrapped_by(
            rendered[len(carrier) + 1 : -1], native
        )

    def metal_conditional_common_type(self, true_type, false_type):
        """Common Metal vector type of conditional arms with different component types, else ``None``."""
        if true_type is None or false_type is None:
            return None
        mapped = [self.map_type(true_type), self.map_type(false_type)]
        if not all(isinstance(value, str) for value in mapped):
            return None
        widths = [self.vector_value_width(value) for value in mapped]
        if None in widths:
            # A scalar arm is splat by Metal itself.
            return None
        if widths[0] != widths[1]:
            return None
        components = [self.vector_component_type(value) for value in mapped]
        if any(component not in _COMPONENT_RANKS for component in components):
            return None
        narrow = [self.metal_narrow_shape(value) for value in (true_type, false_type)]
        if components[0] == components[1] and narrow[0] == narrow[1]:
            return None
        common = max(components, key=_COMPONENT_RANKS.get)
        if common == "bool":
            common = "int"
        return f"{common}{widths[0]}"

    def metal_conditional_arms(self, expr, true_code, false_code):
        """Conditional arms Metal accepts: native narrow reads widened, mixed components converted."""
        true_code = self.metal_narrow_vector_operand(expr.true_expr, true_code)
        false_code = self.metal_narrow_vector_operand(expr.false_expr, false_code)
        true_type = self.expression_result_type(expr.true_expr)
        false_type = self.expression_result_type(expr.false_expr)
        common = self.metal_conditional_common_type(true_type, false_type)
        if common is None:
            return true_code, false_code
        if self.map_type(true_type) != common:
            true_code = f"{common}({true_code})"
        if self.map_type(false_type) != common:
            false_code = f"{common}({false_code})"
        return true_code, false_code

    def metal_conditional_result_type(self, true_type, false_type):
        """Result type of a conditional whose arms need a common vector type, else ``None``."""
        common = self.metal_conditional_common_type(true_type, false_type)
        if common is None:
            return None
        narrow = [self.metal_narrow_shape(value) for value in (true_type, false_type)]
        if narrow[0] is not None and narrow[0] == narrow[1]:
            return true_type
        return common
