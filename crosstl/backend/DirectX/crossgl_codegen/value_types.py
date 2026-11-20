"""Value types in the HLSL to CrossGL converter: swizzles and scalar splats, member and expression raw types, resource element types and wide-integer arithmetic."""

import re

from crosstl.translator.arithmetic_conversions import (
    ArithmeticScalarKind,
    source_integer_shape,
)

from ...comma_expressions import (
    comma_operands,
    is_binary_comma,
)
from ...common_ast import (
    ArrayAccessNode,
    CastNode,
)
from ..DirectxAst import (
    AssignmentNode,
    BinaryOpNode,
    FunctionCallNode,
    IntegerLiteral,
    MemberAccessNode,
    TernaryOpNode,
    UnaryOpNode,
    VectorConstructorNode,
)


class ValueTypeMixin:
    """Value type members of ``HLSLToCrossGLConverter``."""

    def vector_swizzle_raw_type(self, type_name, swizzle):
        if not type_name or not isinstance(swizzle, str):
            return None
        component_sets = ("xyzw", "rgba")
        component_positions = None
        for components in component_sets:
            if all(component in components for component in swizzle):
                component_positions = [
                    components.index(component) for component in swizzle
                ]
                break
        if component_positions is None:
            return None

        base = str(type_name).strip()
        scalar_prefixes = (
            "uint64_t",
            "int64_t",
            "min16float",
            "min10float",
            "min16uint",
            "min16int",
            "min12int",
            "float16_t",
            "uint16_t",
            "int16_t",
            "double",
            "float",
            "half",
            "fixed",
            "uint",
            "int",
            "bool",
        )
        for scalar in scalar_prefixes:
            for width in range(2, 5):
                if base != f"{scalar}{width}":
                    continue
                if max(component_positions) >= width:
                    continue
                if len(swizzle) == 1:
                    return scalar
                if len(swizzle) <= 4:
                    return f"{scalar}{len(swizzle)}"
        return None

    def scalar_splat_swizzle_components(self, member):
        if not isinstance(member, str) or not 1 <= len(member) <= 4:
            return None
        if all(component == "x" for component in member):
            return member
        if all(component == "r" for component in member):
            return member
        return None

    def scalar_splat_constructor_from_expected_type(
        self, expected_type, component_count
    ):
        if expected_type is None:
            return None
        mapped_type = self.map_type(expected_type)
        mapped_base = str(mapped_type).strip()
        crossgl_vectors = (
            "vec",
            "dvec",
            "ivec",
            "uvec",
            "bvec",
            "f16vec",
            "i16vec",
            "u16vec",
        )
        if any(
            mapped_base == f"{prefix}{component_count}" for prefix in crossgl_vectors
        ):
            return mapped_base

        expected_base = self.canonical_composite_type(str(expected_type).strip())
        hlsl_vector_type = self.vector_swizzle_raw_type(
            expected_base, "x" * component_count
        )
        if hlsl_vector_type is None:
            return None
        return self.map_type(hlsl_vector_type)

    def scalar_splat_constructor_from_scalar_type(self, scalar_type, component_count):
        if scalar_type is None:
            return None
        scalar_name = self.map_type(scalar_type)
        scalar_vectors = {
            "float": "vec",
            "double": "dvec",
            "int": "ivec",
            "uint": "uvec",
            "bool": "bvec",
            "float16": "f16vec",
            "int16": "i16vec",
            "uint16": "u16vec",
        }
        vector_prefix = scalar_vectors.get(str(scalar_name).strip())
        if vector_prefix is None:
            return None
        return f"{vector_prefix}{component_count}"

    def literal_raw_type(self, expr):
        if isinstance(expr, IntegerLiteral):
            return expr.source_type
        if isinstance(expr, bool):
            return "bool"
        if isinstance(expr, float):
            return "float"
        if isinstance(expr, int):
            return "int"
        return None

    def scalar_splat_swizzle_expression(
        self, expr, rendered_object, expected_type=None
    ):
        components = self.scalar_splat_swizzle_components(getattr(expr, "member", None))
        if components is None:
            return None

        object_expr = getattr(expr, "object", None)
        object_type = self.expression_value_raw_type(
            object_expr
        ) or self.literal_raw_type(object_expr)
        if object_type is None:
            return None
        if self.vector_swizzle_raw_type(object_type, components) is not None:
            return None

        component_count = len(components)
        if component_count == 1:
            return None

        constructor_type = self.scalar_splat_constructor_from_expected_type(
            expected_type, component_count
        ) or self.scalar_splat_constructor_from_scalar_type(
            object_type, component_count
        )
        if constructor_type is None:
            return None
        return f"{constructor_type}({rendered_object})"

    def member_owner_type(self, object_type):
        """The struct whose members ``value.member`` names, for a value of ``object_type``."""
        object_base = str(object_type).strip()
        if "[" in object_base:
            object_base = object_base.split("[", 1)[0]
        wrapper, _, element = object_base.partition("<")
        if wrapper.strip() in {"ConstantBuffer", "TextureBuffer"} and element.endswith(
            ">"
        ):
            # ConstantBuffer<T> and TextureBuffer<T> name T's members directly.
            object_base = element[:-1].strip()
        return object_base

    def member_access_raw_type(self, object_type, member):
        if not object_type:
            return None

        object_base = self.member_owner_type(object_type)

        struct_members = self.struct_member_types.get(object_base)
        if struct_members and member in struct_members:
            return struct_members[member]

        return self.vector_swizzle_raw_type(object_base, member)

    def expression_raw_type(self, expr):
        if isinstance(expr, MemberAccessNode):
            object_type = self.expression_raw_type(expr.object)
            member_type = self.member_access_raw_type(object_type, expr.member)
            return member_type if member_type is not None else object_type
        if isinstance(expr, ArrayAccessNode):
            return self.expression_raw_type(expr.array)

        name = self.expression_base_name(expr)
        if not name:
            return None
        context_type = self.struct_method_context_name_type(name)
        if context_type is not None:
            return context_type
        raw_type = self.current_variable_types.get(name)
        if raw_type is None:
            raw_type = self.global_variable_types.get(name)
        return raw_type

    def expression_resource_array_dims(self, expr):
        member_array_dims = self.expression_struct_member_array_dims(expr)
        if member_array_dims is not None:
            return member_array_dims
        name = self.expression_base_name(expr)
        if not name:
            return 0
        if name in self.current_resource_array_dims:
            return self.current_resource_array_dims[name]
        return self.global_resource_array_dims.get(name, 0)

    def expression_struct_member_array_dims(self, expr):
        while isinstance(expr, ArrayAccessNode):
            expr = expr.array
        if not isinstance(expr, MemberAccessNode):
            return None
        owner_base = self.raw_type_base(
            self.member_owner_type(self.expression_raw_type(expr.object))
        )
        dims = self.struct_member_array_dims.get(owner_base, {}).get(expr.member)
        if dims is None and isinstance(
            expr.object, (ArrayAccessNode, MemberAccessNode)
        ):
            # ``buffer[i].member``: the declared type is the buffer's, the struct is its element's.
            owner_base = self.raw_type_base(
                self.member_owner_type(self.expression_value_raw_type(expr.object))
            )
            dims = self.struct_member_array_dims.get(owner_base, {}).get(expr.member)
        return dims

    def raw_type_base(self, type_name):
        if not type_name:
            return ""
        base = str(type_name).strip()
        if "<" in base and base.endswith(">"):
            base = base.split("<", 1)[0]
        return base

    def numeric_type_family(self, type_name):
        base = self.raw_type_base(type_name)
        if not base:
            return None
        if base.startswith(("uint", "min16uint")):
            return "uint"
        if base.startswith(("int", "min16int", "min12int")):
            return "int"
        if base.startswith(("float", "half", "fixed", "min16float", "min10float")):
            return "float"
        return None

    def resource_element_raw_type(self, type_name):
        base = self.raw_type_base(type_name)
        if not base:
            return None
        if not (
            self.is_buffer_resource_type(type_name)
            or base in self.structured_buffer_types
            or base.startswith(
                ("Texture", "RWTexture", "RasterizerOrderedTexture", "FeedbackTexture")
            )
        ):
            return None
        return self.first_template_argument(type_name)

    def first_template_argument(self, type_name):
        if not type_name:
            return None
        text = str(type_name).strip()
        if "<" not in text or not text.endswith(">"):
            return None
        generic_type = text.split("<", 1)[1][:-1].strip()
        args = self.split_generic_arguments(generic_type)
        if not args:
            return None
        return self.canonical_composite_type(args[0])

    def vector_element_raw_type(self, type_name):
        base = self.canonical_composite_type(str(type_name or "").strip())
        match = re.fullmatch(
            r"(min16float|min10float|min16uint|min16int|min12int|float16_t|"
            r"uint16_t|int16_t|uint64_t|int64_t|double|float|half|fixed|uint|int|bool)[2-4]",
            base,
        )
        if match:
            return match.group(1)
        return None

    def indexed_value_raw_type(self, type_name):
        return (
            self.resource_element_raw_type(type_name)
            or self.vector_element_raw_type(type_name)
            or self.matrix_row_raw_type(type_name)
        )

    def byte_address_load_raw_type(self, member):
        templated_type = self.first_template_argument(member)
        if templated_type is not None:
            return templated_type
        return {
            "Load": "uint",
            "Load2": "uint2",
            "Load3": "uint3",
            "Load4": "uint4",
        }.get(member)

    def expression_value_raw_type(self, expr):
        if is_binary_comma(expr):
            return self.expression_value_raw_type(comma_operands(expr)[-1])
        if isinstance(expr, BinaryOpNode):
            # Keep long left-associated sums iterative, as in their renderer.
            nodes = []
            current = expr
            while isinstance(current, BinaryOpNode):
                nodes.append(current)
                current = current.left
            result = self.expression_value_raw_type(current)
            for node in reversed(nodes):
                result = self.integer_expression_raw_type(
                    result,
                    self.expression_value_raw_type(node.right),
                    self.binary_operator_text(node.op),
                )
            return result
        if isinstance(expr, TernaryOpNode):
            return self.integer_expression_raw_type(
                self.expression_value_raw_type(expr.true_expr),
                self.expression_value_raw_type(expr.false_expr),
                "?:",
            )
        if isinstance(expr, AssignmentNode):
            return self.expression_value_raw_type(expr.left)
        if isinstance(expr, UnaryOpNode):
            return (
                "bool"
                if self.binary_operator_text(expr.op) == "!"
                else self.expression_value_raw_type(expr.operand)
            )
        if isinstance(expr, MemberAccessNode):
            object_type = self.expression_value_raw_type(expr.object)
            member_type = self.member_access_raw_type(object_type, expr.member)
            return member_type if member_type is not None else object_type
        if isinstance(expr, ArrayAccessNode):
            array_type = self.expression_value_raw_type(expr.array)
            depth = 1
            base = expr.array
            while isinstance(base, ArrayAccessNode):
                depth += 1
                base = base.array
            if depth <= self.expression_resource_array_dims(expr):
                return array_type
            value_type = self.indexed_value_raw_type(array_type)
            return value_type if value_type is not None else array_type
        if isinstance(expr, FunctionCallNode):
            return self.function_call_value_raw_type(expr)
        if isinstance(expr, VectorConstructorNode):
            return getattr(expr, "type_name", None)
        if isinstance(expr, CastNode):
            return getattr(expr, "target_type", None)
        literal_type = self.literal_raw_type(expr)
        if literal_type is not None:
            return literal_type
        return self.expression_raw_type(expr)

    def resolve_integer_alias(self, type_name):
        name = self.canonical_composite_type(str(type_name or ""))
        seen = set()
        while name in self.source_type_aliases and name not in seen:
            seen.add(name)
            name = self.canonical_composite_type(self.source_type_aliases[name])
        return name

    def integer_expression_raw_type(self, left_type, right_type, operator):
        if operator not in {
            "+",
            "-",
            "*",
            "/",
            "%",
            "&",
            "|",
            "^",
            "<<",
            ">>",
            "<",
            "<=",
            ">",
            ">=",
            "==",
            "!=",
            "?:",
        }:
            return None
        left = source_integer_shape(self.resolve_integer_alias(left_type))
        right = source_integer_shape(self.resolve_integer_alias(right_type))
        if left is None or right is None:
            return None
        widths = [s[2] for s in (left, right) if s[2] > 1]
        lanes = min(widths) if widths else 1
        if operator in {"<", "<=", ">", ">=", "==", "!="}:
            base = "bool"
        elif operator in {"<<", ">>"}:
            base = self.integer_scalar_raw_type(left[0], max(32, left[1]))
        else:
            # Preserve the native DXC conversion contract, including its unsigned
            # preference across widths, before entering the C-family IR.
            kind = (
                ArithmeticScalarKind.UNSIGNED_INTEGER
                if any(
                    s[0] == ArithmeticScalarKind.UNSIGNED_INTEGER for s in (left, right)
                )
                else ArithmeticScalarKind.SIGNED_INTEGER
            )
            base = self.integer_scalar_raw_type(kind, max(32, left[1], right[1]))
        return base + (str(lanes) if lanes > 1 else "")

    def integer_scalar_raw_type(self, kind, bits):
        prefix = "uint" if kind == ArithmeticScalarKind.UNSIGNED_INTEGER else "int"
        return prefix + ("64_t" if bits == 64 else "")

    def wide_integer_common_type(self, left_type, right_type, operator):
        shapes = [
            source_integer_shape(self.resolve_integer_alias(t))
            for t in (left_type, right_type)
        ]
        if any(s is None for s in shapes) or not any(s[1] == 64 for s in shapes):
            return None
        if operator in {"<<", ">>"}:
            return None
        operation = "+" if operator in {"<", "<=", ">", ">=", "==", "!="} else operator
        return self.integer_expression_raw_type(left_type, right_type, operation)

    def convert_integer_operand(self, rendered, source_type, common_type):
        source_shape = source_integer_shape(self.resolve_integer_alias(source_type))
        common_shape = source_integer_shape(common_type)
        if source_shape[2] == 1 and common_shape[2] > 1:
            common_type = self.integer_scalar_raw_type(*common_shape[:2])
            common_shape = source_integer_shape(common_type)
        if source_shape == common_shape:
            return rendered
        return f"{self.map_type(common_type)}({rendered})"

    def wide_integer_binary_expression(self, expr, is_main=False):
        operator = self.binary_operator_text(expr.op)
        left_type = self.expression_value_raw_type(expr.left)
        right_type = self.expression_value_raw_type(expr.right)
        common = self.wide_integer_common_type(left_type, right_type, operator)
        if common is None:
            return None
        left = self.maybe_parenthesize(
            expr.left, self.generate_expression(expr.left, is_main)
        )
        right = self.maybe_parenthesize(
            expr.right, self.generate_expression(expr.right, is_main)
        )
        left = self.convert_integer_operand(left, left_type, common)
        right = self.convert_integer_operand(right, right_type, common)
        return f"{left} {operator} {right}"

    def integer_expression_has_effects(self, expr):
        pending = [expr]
        while pending:
            node = pending.pop()
            if isinstance(node, AssignmentNode):
                return True
            if isinstance(node, UnaryOpNode) and self.binary_operator_text(node.op) in {
                "++",
                "--",
            }:
                return True
            if isinstance(node, FunctionCallNode):
                if (
                    not isinstance(node.name, str)
                    or source_integer_shape(self.resolve_integer_alias(node.name))
                    is None
                ):
                    return True
            pending.extend(self.iter_ast_children(node))
        return False

    def generate_integer_compound_helpers(self):
        return "".join(
            f"    {target} {name}(inout {target} target, {value} value) {{\n"
            f"        target = {target}({left} {operator} {right});\n"
            "        return target;\n    }\n"
            for (
                target,
                value,
                operator,
                left,
                right,
            ), name in self.integer_compound_helpers.items()
        )

    def bitcast_intrinsic_expression(self, func_name, original_args, rendered_args):
        if len(original_args) != 1 or func_name not in {"asfloat", "asint", "asuint"}:
            return None

        source_family = self.numeric_type_family(
            self.expression_value_raw_type(original_args[0])
        )
        if func_name == "asfloat":
            if source_family == "uint":
                return f"uintBitsToFloat({rendered_args[0]})"
            if source_family == "int":
                return f"intBitsToFloat({rendered_args[0]})"
            if source_family == "float":
                return rendered_args[0]
        elif func_name == "asint" and source_family == "float":
            return f"floatBitsToInt({rendered_args[0]})"
        elif func_name == "asuint" and source_family == "float":
            return f"floatBitsToUint({rendered_args[0]})"
        return None
