"""Expression values in the WGSL generator: negative integer literals, matrix array literals, local declaration types, vector splats and narrowing, ternary conversions, literals and array lengths."""

import re

from ..ast import (
    ArrayLiteralNode,
    ArrayType,
    LiteralNode,
    MemberAccessNode,
    NamedType,
    PointerType,
    UnaryOpNode,
)
from ..refusals import WGSLRefusal
from .wgsl_array_constructors import wgsl_array_element


class WGSLExpressionValueMixin:
    """Expression value members of ``WGSLCodeGen``."""

    def concrete_negative_integer_literal(self, expr, rendered):
        """``-1i`` for a negative integer literal ``expr`` rendered as ``-1``, else ``rendered``.

        WGSL reads ``-1`` as an abstract integer, and ``u32(-1)`` converts its value, which ``u32`` cannot
        represent, so naga rejects it. ``u32(-1i)`` converts an ``i32`` and keeps its bits, 4294967295, as GLSL's
        ``uint(-1)`` does.
        """
        operand = expr
        if isinstance(expr, UnaryOpNode) and expr.operator == "-":
            operand = expr.operand
        elif not isinstance(expr, LiteralNode):
            return rendered
        value = getattr(operand, "value", None)
        if (
            isinstance(operand, LiteralNode)
            and isinstance(value, int)
            and not isinstance(value, bool)
            and (operand is not expr or value < 0)
            and re.fullmatch(r"\(?-\d+\)?", rendered)
        ):
            return f"{rendered.strip('()')}i"
        return rendered

    def generate_matrix_array_literal(self, expr, target_type):
        """``array(...)`` for an array literal of matrices that gives every element, else ``None``.

        A scalar element, as in ``mat2 m[2] = mat2[2](x, 1.0);`` or ``{x, 1.0}``, is the diagonal
        matrix it stands for (wgsl_array_element), as in a short initializer.
        """
        if not isinstance(expr, ArrayLiteralNode) or not isinstance(
            target_type, ArrayType
        ):
            return None
        element_type = target_type.element_type
        if self.matrix_shape(element_type) is None:
            return None
        values = ", ".join(
            wgsl_array_element(self, value, element_type) for value in expr.elements
        )
        return f"array({values})"

    def local_variable_declaration_type(self, node):
        declared_type = getattr(node, "var_type", None)
        if not self.is_inferred_local_declaration_type(declared_type):
            return declared_type
        initializer = getattr(node, "initial_value", None)
        inferred_type = (
            self.expression_type(initializer) if initializer is not None else None
        )
        if inferred_type is not None:
            return inferred_type
        raise WGSLRefusal(
            "WGSL target cannot infer a concrete type for local variable "
            f"{getattr(node, 'name', '<unnamed>')} declared as void"
        )

    def is_inferred_local_declaration_type(self, vtype):
        return self.is_void_local_declaration_type(vtype) or self.is_auto_type(vtype)

    def is_void_local_declaration_type(self, vtype):
        if vtype is None:
            return True
        try:
            return self.type_name_string(vtype) == "void"
        except ValueError:
            return False

    def is_auto_type(self, vtype):
        if isinstance(vtype, NamedType) and not vtype.generic_args:
            return str(vtype.name).lower() == "auto"
        if isinstance(vtype, str):
            return vtype.strip().lower() == "auto"
        return False

    def generate_vector_narrowing_conversion(self, target_type, expr):
        return self.vector_narrowing_expression(
            self.generate_expression(expr), target_type, self.expression_type(expr)
        )

    def vector_splat_expression(self, rendered, target_type, source_type, expr=None):
        """A scalar stored to a vector, which HLSL, GLSL, MSL and CrossGL replicate into every component.

        WGSL has no implicit scalar-to-vector conversion, so the value becomes ``vecN<T>(T(value))``.
        A negative integer literal ``expr`` stored to a ``vecN<u32>`` is written ``u32(-1i)``, as for a
        scalar ``u32`` (see ``concrete_negative_integer_literal``).
        """
        target_shape = self.vector_shape(target_type)
        if target_shape is None or source_type is None:
            return None
        try:
            source_scalar = self.type_name_string(source_type)
        except ValueError:
            return None
        if source_scalar not in {"bool", "f16", "f32", "i32", "u32"}:
            return None
        element = target_shape[0]
        if element == "u32" and source_scalar == "i32" and expr is not None:
            rendered = self.concrete_negative_integer_literal(expr, rendered)
        value = rendered if source_scalar == element else f"{element}({rendered})"
        return f"{self.type_name_string(target_type)}({value})"

    def vector_narrowing_expression(self, rendered, target_type, source_type):
        target_shape = self.vector_shape(target_type)
        source_shape = self.vector_shape(source_type)
        if target_shape is None or source_shape is None:
            return None
        target_element, target_size = target_shape
        source_element, source_size = source_shape
        if source_size <= target_size:
            return None
        components = "xyzw"[:target_size]
        narrowed = f"{rendered}.{components}"
        if target_element == source_element:
            return narrowed
        return f"{self.type_name_string(target_type)}({narrowed})"

    def vector_element_conversion(self, rendered, target_type, source_type):
        """A vector stored to a vector of the same width and another element type, else ``None``.

        HLSL and CrossGL convert ``int2 r = sign(f)`` or ``uint2 r = countbits(i)`` implicitly; WGSL has no
        implicit conversion between ``vecN<i32>``, ``vecN<u32>``, ``vecN<f32>`` and ``vecN<bool>``, so the value
        becomes the value conversion ``vecN<T>(value)`` (WGSL section 17.1.2), which converts each component as
        the scalar conversion does.
        """
        target_shape = self.vector_shape(target_type)
        source_shape = self.vector_shape(source_type)
        if target_shape is None or source_shape is None:
            return None
        elements = {"bool", "f16", "f32", "i32", "u32"}
        if target_shape[1] != source_shape[1] or target_shape[0] == source_shape[0]:
            return None
        if target_shape[0] not in elements or source_shape[0] not in elements:
            return None
        return f"{self.type_name_string(target_type)}({rendered})"

    def normalized_type_name(self, vtype):
        try:
            return self.type_name_string(vtype) if vtype is not None else None
        except ValueError:
            return None

    ternary_scalar_ranks = {"i32": 0, "u32": 1, "f32": 2}

    def ternary_conversion_target(self, branch_type, other_type):
        """Return the type a ternary branch converts to, or None.

        GLSL 4.60 §5.8 lets the conditional operator apply the implicit
        conversions of §4.1.10 to the operands: `int` to `uint` or `float`,
        and `uint` to `float`, per component for vectors. WGSL has no implicit
        conversion and `select` requires both operands to share a type, so the
        conversion has to be written out.
        """
        branch_name = self.normalized_type_name(branch_type)
        other_name = self.normalized_type_name(other_type)
        branch_shape = self.vector_shape(branch_name)
        other_shape = self.vector_shape(other_name)
        if branch_shape is None:
            branch_rank = self.ternary_scalar_ranks.get(branch_name)
            other_rank = self.ternary_scalar_ranks.get(other_name)
            if branch_rank is None or other_rank is None or branch_rank >= other_rank:
                return None
            return other_name
        if other_shape is None or other_shape[1] != branch_shape[1]:
            return None
        branch_rank = self.ternary_scalar_ranks.get(branch_shape[0])
        other_rank = self.ternary_scalar_ranks.get(other_shape[0])
        if branch_rank is None or other_rank is None or branch_rank >= other_rank:
            return None
        return f"vec{other_shape[1]}<{other_shape[0]}>"

    def ternary_result_type(self, true_type, false_type):
        true_name = self.normalized_type_name(true_type)
        false_name = self.normalized_type_name(false_type)
        if true_name == false_name:
            return true_name
        if true_name is None or false_name is None:
            # An unknown branch keeps the declaration unresolved, as the
            # `_common_value_type` rule did before the conditional got a
            # conversion rule. Typing it from the known branch would drop the
            # check on the other one and could emit invalid WGSL.
            return None
        target = self.ternary_conversion_target(true_name, false_name)
        if target is not None:
            return target
        target = self.ternary_conversion_target(false_name, true_name)
        if target is not None:
            return target
        # No conversion makes the branches match, so leave the declaration
        # unresolved and let the emitter refuse it.
        return None

    def is_integer_literal_expression(self, expr):
        if isinstance(expr, UnaryOpNode) and expr.operator in {"+", "-"}:
            expr = expr.operand
        if not isinstance(expr, LiteralNode):
            return False
        return self.normalized_type_name(getattr(expr, "literal_type", None)) == "i32"

    def convert_ternary_int_branch(self, rendered, expr, target):
        # An integer literal is an abstract value in WGSL, and `u32(-3)` is a
        # shader-creation error because the value is not representable. Typing
        # the literal first makes the conversion a runtime one, which wraps the
        # way GLSL's int -> uint conversion does. Conversions to `float` accept
        # the literal unchanged.
        if target == "u32" and self.is_integer_literal_expression(expr):
            return f"{target}({rendered}i)"
        return f"{target}({rendered})"

    def generate_ternary_expression(self, expr):
        true_type = self.expression_type(expr.true_expr)
        false_type = self.expression_type(expr.false_expr)
        true_rendered = self.generate_expression(expr.true_expr)
        false_rendered = self.generate_expression(expr.false_expr)
        if true_type is not None and false_type is not None:
            target = self.ternary_conversion_target(true_type, false_type)
            if target is not None:
                true_rendered = self.convert_ternary_int_branch(
                    true_rendered, expr.true_expr, target
                )
            else:
                target = self.ternary_conversion_target(false_type, true_type)
                if target is not None:
                    false_rendered = self.convert_ternary_int_branch(
                        false_rendered, expr.false_expr, target
                    )
        return (
            f"select({false_rendered}, {true_rendered}, "
            f"{self.generate_expression(expr.condition)})"
        )

    def generate_literal(self, node):
        value = node.value
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return ""
        text = str(value)
        literal_type = self.type_name_string(getattr(node, "literal_type", ""))
        if literal_type == "f32" and re.fullmatch(r"[-+]?\d+", text):
            return f"{text}.0"
        # An unsigned literal keeps its type: an abstract integer would become i32 in
        # `select(0, 1, c)` or `1 << n` and mismatch a u32 operand (WGSL section 6.1.2).
        if literal_type == "u32" and re.fullmatch(r"\d+", text):
            return f"{text}u"
        return text

    def generate_workgroup_size_literal(self):
        values = self._current_workgroup_size or ("1", "1", "1")
        return (
            "vec3<u32>(" + ", ".join(self.u32_literal(value) for value in values) + ")"
        )

    def u32_literal(self, value):
        text = str(value)
        if re.fullmatch(r"\d+", text):
            return f"{text}u"
        return text

    def generate_array_length_call(self, node):
        """GLSL ``array.length()`` (GLSL 4.60 section 4.1.9) as an i32: ``arrayLength`` or the declared size."""
        function = node.function
        if not (
            isinstance(function, MemberAccessNode)
            and function.member == "length"
            and not node.arguments
        ):
            return None
        array_type = self.expression_type(function.object_expr)
        if isinstance(array_type, PointerType):
            # A `buffer float*` global is declared as a storage array, so its
            # length is the runtime-sized array's length, just like an
            # unsized ArrayType. A helper parameter of that type is already a
            # pointer to the array and is passed to arrayLength unchanged.
            pointer = self.pointer_argument_expression(function.object_expr)
            return f"i32(arrayLength({pointer}))"
        if isinstance(array_type, NamedType):
            # HLSL StructuredBuffer / RWStructuredBuffer globals are emitted as
            # runtime-sized storage arrays, so their length is arrayLength, the
            # same as an unsized ArrayType. They used to be copied as `.length()`,
            # which naga rejects (#3582).
            if str(array_type.name).lower() in self.STRUCTURED_BUFFER_TYPE_NAMES:
                return (
                    "i32(arrayLength(&"
                    f"{self.generate_expression(function.object_expr)}))"
                )
        if not isinstance(array_type, ArrayType):
            return None
        if array_type.size is None:
            return (
                f"i32(arrayLength(&{self.generate_expression(function.object_expr)}))"
            )
        return f"i32({self.generate_expression(array_type.size)})"

    def specialize_generic_function_call(self, node):
        specializer = self._generic_function_specializer
        if specializer is None:
            return node
        return specializer.rewrite_call(node)
