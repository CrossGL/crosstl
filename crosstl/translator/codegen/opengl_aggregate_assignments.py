"""Aggregate initializers and mesh outputs in the GLSL generator: aggregate contracts, vector constructors, mesh output targets and counts, and boolean compound lvalues."""

import re
from copy import deepcopy

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    ConstructorNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    MeshOpNode,
    SwizzleNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from ..refusals import OpenGLRefusal
from ..validation import expression_debug_name
from .array_utils import split_array_type_suffix
from .builtin_variable_types import crossgl_builtin_variable_type
from .glsl_draw_parameters import DRAW_PARAMETERS
from .glsl_hlsl_matrix import emulated_matrix_shape
from .glsl_instance_index import INSTANCE_INDEX
from .glsl_mesh_builtin_values import glsl_mesh_builtin_value
from .glsl_narrow_integer_literals import widened_integer_literal
from .opengl_values import OpenGLTemplateTypeError


class OpenGLAggregateAssignmentMixin:
    """Aggregate assignment members of ``GLSLCodeGen``."""

    def glsl_aggregate_initializer_contract(self, expr, expected_type):
        expected_type = self.type_name_string(expected_type)
        if not expected_type:
            self.glsl_aggregate_error(
                expr,
                expected_type=None,
                reason="expected-type-unresolved",
            )

        mapped_type = self.map_type(expected_type)
        _base_type, array_suffix = split_array_type_suffix(mapped_type)
        if array_suffix:
            expected_count, element_types = self.glsl_fixed_array_aggregate_contract(
                expr,
                mapped_type,
            )
            constructor = self.glsl_fixed_array_constructor_type(
                mapped_type, expected_count
            )
            return constructor, expected_count, element_types

        member_types = self.struct_member_types.get(mapped_type)
        if member_types is not None:
            element_types = list(member_types.values())
            return mapped_type, len(element_types), element_types

        if self.is_vector_value_type(mapped_type):
            width = int(mapped_type[-1])
            component_type = self.vector_component_type(mapped_type)
            return mapped_type, width, [component_type] * width

        self.glsl_aggregate_error(
            expr,
            expected_type=mapped_type,
            reason="non-aggregate-destination-type",
            detail="the destination is not a struct, fixed array, vector, or matrix",
        )

    def glsl_is_zero_aggregate_initializer(self, expr):
        elements = list(getattr(expr, "elements", []) or [])
        if len(elements) != 1 or not isinstance(elements[0], LiteralNode):
            return False
        value = elements[0].value
        return not isinstance(value, bool) and value in {0, 0.0}

    def glsl_is_value_initializable_aggregate_type(self, expected_type):
        mapped_type = self.map_type(expected_type)
        _base_type, array_suffix = split_array_type_suffix(mapped_type)
        return bool(
            array_suffix
            or self.glsl_source_type_identifier_name(mapped_type)
            in self.struct_member_types
            or self.is_vector_value_type(mapped_type)
            or self.glsl_matrix_dimensions(mapped_type) is not None
        )

    def glsl_value_initialized_expression(self, expr, expected_type):
        union = self.union_storage.construct(expr, expected_type, [])
        if union is not None:
            return union
        mapped_type = self.map_type(expected_type)
        _base_type, array_suffix = split_array_type_suffix(mapped_type)
        if array_suffix:
            extent, element_types = self.glsl_fixed_array_aggregate_contract(
                expr,
                mapped_type,
            )
            elements = [
                self.glsl_value_initialized_expression(expr, element_type)
                for element_type in element_types
            ]
            constructor = self.glsl_fixed_array_constructor_type(mapped_type, extent)
            return f"{constructor}({', '.join(elements)})"

        source_mapped_type = self.glsl_source_type_identifier_name(mapped_type)
        member_types = self.struct_member_types.get(source_mapped_type)
        if member_types is not None:
            elements = [
                self.glsl_value_initialized_expression(expr, member_type)
                for member_type in member_types.values()
            ]
            if not elements and source_mapped_type in self.structs_by_name:
                return f"{mapped_type}(0)"
            return f"{mapped_type}({', '.join(elements)})"

        if (
            self.is_scalar_value_type(mapped_type)
            or self.is_vector_value_type(mapped_type)
            or self.glsl_matrix_dimensions(mapped_type) is not None
        ):
            return self.zero_value_expression(mapped_type)

        self.glsl_aggregate_error(
            expr,
            expected_type=mapped_type,
            reason="value-initialization-unsupported",
            detail="the omitted element type has no OpenGL value initializer",
        )

    def generate_glsl_aggregate_initializer(self, expr):
        expected_type = self.type_name_string(self.current_expression_expected_type)
        union = self.union_storage.construct(expr, expected_type, expr.elements)
        if union is not None:
            return union
        if expected_type:
            mapped_type = self.map_type(expected_type)
            if self.glsl_is_zero_aggregate_initializer(
                expr
            ) and self.glsl_is_value_initializable_aggregate_type(mapped_type):
                return self.glsl_value_initialized_expression(expr, mapped_type)
            vector_initializer = self.generate_glsl_vector_aggregate_initializer(
                expr,
                mapped_type,
            )
            if vector_initializer is not None:
                return vector_initializer
            matrix_initializer = self.generate_glsl_matrix_aggregate_initializer(
                expr,
                mapped_type,
            )
            if matrix_initializer is not None:
                return matrix_initializer

        constructor, expected_count, element_types = (
            self.glsl_aggregate_initializer_contract(
                expr,
                self.current_expression_expected_type,
            )
        )
        elements = list(getattr(expr, "elements", []) or [])
        if len(elements) > expected_count:
            self.glsl_aggregate_error(
                expr,
                expected_type=constructor,
                reason="element-count-mismatch",
                expected_element_count=expected_count,
            )
        rendered = [
            widened_integer_literal(self, element, element_types[index])
            or self.generate_expression_with_expected(element, element_types[index])
            for index, element in enumerate(elements)
        ]
        rendered.extend(
            self.glsl_value_initialized_expression(expr, element_types[index])
            for index in range(len(elements), expected_count)
        )
        if not rendered and self.glsl_struct_node(constructor) is not None:
            return f"{constructor}(0)"
        return f"{constructor}({', '.join(rendered)})"

    def signed_compute_builtin_member_cast(self, object_name, member_name):
        expected_type = self.map_type(self.current_expression_expected_type)
        if expected_type != "int":
            return None
        if object_name not in self.UNSIGNED_VECTOR_COMPUTE_BUILTINS:
            return None
        if member_name not in {"x", "y", "z"}:
            return None
        return f"int({object_name}.{member_name})"

    def is_scalar_value_type(self, vtype):
        vtype = self.type_name_string(vtype)
        if not vtype:
            return False
        return self.map_type(vtype) in {
            "float",
            "double",
            "int",
            "uint",
            "int64_t",
            "uint64_t",
            "bool",
        }

    def is_vector_value_type(self, vtype):
        vtype = self.type_name_string(vtype)
        if not vtype:
            return False
        return self.map_type(vtype) in {
            "vec2",
            "vec3",
            "vec4",
            "dvec2",
            "dvec3",
            "dvec4",
            "ivec2",
            "ivec3",
            "ivec4",
            "uvec2",
            "uvec3",
            "uvec4",
            "i64vec2",
            "i64vec3",
            "i64vec4",
            "u64vec2",
            "u64vec3",
            "u64vec4",
            "bvec2",
            "bvec3",
            "bvec4",
        }

    def is_matrix_value_type(self, vtype):
        vtype = self.type_name_string(vtype)
        if not vtype:
            return False
        return self.map_type(vtype).startswith(("mat", "dmat"))

    def type_contains_opaque_resource(self, vtype, seen=None):
        type_name = self.type_name_string(vtype)
        if not type_name:
            return False

        base_type, _array_suffix = split_array_type_suffix(str(type_name))
        if self.is_sampler_type(base_type):
            return True
        mapped_resource_type = self.map_resource_type_with_format(base_type)
        if self.is_opaque_resource_type(mapped_resource_type):
            return True
        if self.is_structured_buffer_type(base_type):
            return True

        mapped_type = self.map_type(base_type)
        source_mapped_type = self.glsl_source_type_identifier_name(mapped_type)
        seen = set(seen or set())
        if source_mapped_type in seen:
            return False
        seen.add(source_mapped_type)

        struct = self.structs_by_name.get(source_mapped_type)
        if struct is None:
            return False
        for member_type in self.struct_member_types.get(
            source_mapped_type, {}
        ).values():
            if self.type_contains_opaque_resource(member_type, seen):
                return True
        return False

    def vector_component_type(self, vtype):
        mapped_type = self.map_type(vtype)
        if mapped_type.startswith("dvec"):
            return "double"
        if mapped_type.startswith("uvec"):
            return "uint"
        if mapped_type.startswith("ivec"):
            return "int"
        if mapped_type.startswith("u64vec"):
            return "uint64_t"
        if mapped_type.startswith("i64vec"):
            return "int64_t"
        if mapped_type.startswith("bvec"):
            return "bool"
        if mapped_type.startswith("vec"):
            return "float"
        return None

    def glsl_materialized_generic_vector_constructor_type(self, name):
        if not isinstance(name, str) or "_u" not in name:
            return None
        decoded = name
        for encoded, character in (
            ("_u3c", "<"),
            ("_u2c", ","),
            ("_u3e", ">"),
            ("_u20", " "),
        ):
            decoded = decoded.replace(encoded, character)
        if (
            re.fullmatch(
                r"(?:metal::)?vec\s*<\s*[^,>]+\s*,\s*[234]\s*>",
                decoded.strip(),
            )
            is None
        ):
            return None
        return self.map_type(decoded)

    def glsl_constructor_type(self, func_name):
        if emulated_matrix_shape(func_name) is not None:
            return self.map_type(func_name)
        constructor = self.glsl_materialized_generic_vector_constructor_type(func_name)
        if constructor is None and isinstance(func_name, str):
            try:
                constructor = self.map_type(func_name)
            except OpenGLTemplateTypeError:
                # Short capitalized callee names (``Foo``, ``DE``) look like
                # template parameters to the type mapper, but a name that is
                # not a concrete type cannot be a builtin constructor.
                constructor = None
        if constructor in {
            "float",
            "double",
            "int",
            "uint",
            "int64_t",
            "uint64_t",
            "bool",
            "vec2",
            "vec3",
            "vec4",
            "dvec2",
            "dvec3",
            "dvec4",
            "ivec2",
            "ivec3",
            "ivec4",
            "uvec2",
            "uvec3",
            "uvec4",
            "i64vec2",
            "i64vec3",
            "i64vec4",
            "u64vec2",
            "u64vec3",
            "u64vec4",
            "bvec2",
            "bvec3",
            "bvec4",
            "mat2",
            "mat3",
            "mat4",
            "mat2x2",
            "mat2x3",
            "mat2x4",
            "mat3x2",
            "mat3x3",
            "mat3x4",
            "mat4x2",
            "mat4x3",
            "mat4x4",
            "dmat2",
            "dmat3",
            "dmat4",
            "dmat2x2",
            "dmat2x3",
            "dmat2x4",
            "dmat3x2",
            "dmat3x3",
            "dmat3x4",
            "dmat4x2",
            "dmat4x3",
            "dmat4x4",
        }:
            return constructor
        if func_name in getattr(self, "structs_by_name", {}):
            return constructor
        if func_name in getattr(self, "struct_member_types", {}):
            return constructor
        if func_name in getattr(self, "enum_struct_type_names", set()):
            return constructor
        if constructor in self.glsl_type_identifier_names.values():
            return constructor
        return None

    def glsl_builtin_identifier_result_type(self, name):
        if not name:
            return None
        if name == INSTANCE_INDEX or name in DRAW_PARAMETERS:
            return crossgl_builtin_variable_type(name)
        return crossgl_builtin_variable_type(self.glsl_target_builtin_identifier(name))

    def glsl_bool_vector_type(self, vector_type):
        mapped_type = self.map_type(vector_type)
        if self.vector_component_type(mapped_type) is None:
            return None
        if mapped_type.endswith(("2", "3", "4")):
            return f"bvec{mapped_type[-1]}"
        return None

    def texture_size_result_type(self, texture_arg):
        texture_type = self.resource_base_type(self.texture_resource_type(texture_arg))
        if not texture_type:
            return None
        if texture_type.endswith(("1D", "1DShadow", "Buffer")):
            return "int"
        if texture_type.endswith(
            (
                "3D",
                "2DArray",
                "2DArrayShadow",
                "2DMSArray",
                "CubeArray",
                "CubeArrayShadow",
            )
        ):
            return "ivec3"
        return "ivec2"

    def expected_float_vector_expression_type(self):
        expected_type = self.map_type(self.current_expression_expected_type)
        if expected_type in {"vec2", "vec3", "vec4"}:
            return expected_type
        return None

    def cast_integer_vector_expression_for_expected_float(
        self, expression, expression_type, expected_type
    ):
        mapped_type = self.map_type(expression_type)
        if mapped_type == self.glsl_integer_vector_type_for_float(expected_type):
            return f"{expected_type}({expression})"
        return expression

    def glsl_integer_vector_type_for_float(self, vector_type):
        if vector_type in {"vec2", "vec3", "vec4"}:
            return f"ivec{vector_type[-1]}"
        return None

    def glsl_relational_vector_type(self, left_type, right_type):
        if self.is_vector_value_type(left_type):
            return self.map_type(left_type)
        if self.is_vector_value_type(right_type):
            return self.map_type(right_type)
        return None

    def is_numeric_literal_expression(self, expr):
        if isinstance(expr, (int, float)) and not isinstance(expr, bool):
            return True
        if hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            value = getattr(expr, "value", None)
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        return False

    def glsl_vector_relational_operand(self, expr, expr_type, vector_type):
        generated = self.generate_expression(expr)
        if self.is_vector_value_type(expr_type):
            return generated
        is_scalar = self.is_scalar_numeric_type(expr_type)
        is_literal = self.is_numeric_literal_expression(expr)
        if is_scalar or is_literal:
            return f"{vector_type}({generated})"
        return generated

    def generate_vector_relational_expression(
        self, left_expr, operator, right_expr, *, source_node=None
    ):
        function_name = self.GLSL_VECTOR_RELATIONAL_FUNCTIONS.get(operator)
        if function_name is None:
            return None

        left_type = self.glsl_source_expression_type(left_expr)
        right_type = self.glsl_source_expression_type(right_expr)
        vector_type = self.glsl_relational_vector_type(left_type, right_type)
        if vector_type is None:
            return None

        plan = self.glsl_arithmetic_conversion_plan(
            left_type,
            right_type,
            operator,
            source_node=source_node,
            fail_closed=True,
        )
        if plan is None:
            left = self.glsl_vector_relational_operand(
                left_expr, left_type, vector_type
            )
            right = self.glsl_vector_relational_operand(
                right_expr, right_type, vector_type
            )
        else:
            left = self.generate_expression_with_expected(
                left_expr, plan.left_target_type
            )
            right = self.generate_expression_with_expected(
                right_expr, plan.right_target_type
            )
        return f"{function_name}({left}, {right})"

    def validate_dispatch_mesh_count_argument(self, arg, index):
        axis = ("x", "y", "z")[index]
        arg_type = self.expression_result_type(arg)
        if arg_type is not None:
            mapped_type = self.map_type(arg_type)
            if not self.is_scalar_integer_type(mapped_type):
                raise OpenGLRefusal(
                    f"GLSL DispatchMesh {axis} group count requires a "
                    "non-negative scalar integer value, "
                    f"got {mapped_type}"
                )

        literal_count = self.literal_int_value(arg, self.literal_int_constants)
        if literal_count is None:
            return
        if (
            not isinstance(literal_count, int)
            or isinstance(literal_count, bool)
            or literal_count < 0
        ):
            raise OpenGLRefusal(
                f"GLSL DispatchMesh {axis} group count requires a "
                "non-negative scalar integer value, got "
                f"{self.glsl_attribute_argument_source(arg)}"
            )

    def validate_dispatch_mesh_count_arguments(self, args):
        for index, arg in enumerate(args[:3]):
            self.validate_dispatch_mesh_count_argument(arg, index)

    def validate_dispatch_mesh_arguments(self, args):
        if len(args) not in {3, 4}:
            raise OpenGLRefusal(
                "GLSL DispatchMesh requires three group counts and optional "
                "payload argument"
            )
        self.validate_dispatch_mesh_count_arguments(args)

    def generate_mesh_intrinsic_statement(self, node, indent=0):
        if not isinstance(node, MeshOpNode):
            return None
        if node.operation != "DispatchMesh":
            return None

        if self.current_stage_entry_type not in {"task", "amplification", "object"}:
            return None

        args = getattr(node, "arguments", []) or []
        self.validate_dispatch_mesh_arguments(args)

        indent_str = "    " * indent
        dispatch_args = ", ".join(self.generate_expression(arg) for arg in args[:3])
        statements = []

        if len(args) == 4:
            payload_expr = args[3]
            target = self.glsl_dispatch_mesh_payload_target(payload_expr)
            if self.expression_name(payload_expr) != target["name"]:
                payload_value = self.generate_expression_with_expected(
                    payload_expr, target["raw_type"]
                )
                statements.append(f"{indent_str}{target['name']} = {payload_value};")

        statements.append(f"{indent_str}EmitMeshTasksEXT({dispatch_args});")
        return "\n".join(statements) + "\n"

    def mesh_output_role_count_limit(self, roles):
        for role in roles:
            for info in self.current_mesh_output_parameters.values():
                if info["role"] == role and info.get("count_value") is not None:
                    return info["count_value"], f"@{role} array size"

        for role in roles:
            limit = self.current_mesh_output_count_limits.get(role)
            if limit is not None:
                return limit, self.mesh_output_role_count_attribute(role)
        return None, None

    def validate_mesh_output_count_argument(self, args, index, roles, label):
        arg = args[index]
        count_type = self.expression_result_type(arg)
        if count_type is not None:
            mapped_type = self.map_type(count_type)
            if not self.is_scalar_integer_type(mapped_type):
                raise OpenGLRefusal(
                    "GLSL mesh SetMeshOutputCounts "
                    f"{label} count requires a non-negative scalar integer "
                    f"value, got {mapped_type}"
                )

        value = self.literal_int_value(arg, self.literal_int_constants)
        if value is None:
            return
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise OpenGLRefusal(
                "GLSL mesh SetMeshOutputCounts "
                f"{label} count requires a non-negative scalar integer value, "
                f"got {self.glsl_attribute_argument_source(arg)}"
            )
        limit, limit_label = self.mesh_output_role_count_limit(roles)
        if limit is None or value <= limit:
            return
        raise OpenGLRefusal(
            "GLSL mesh SetMeshOutputCounts "
            f"{label} count {value} exceeds {limit_label} {limit}"
        )

    def validate_mesh_output_index_expression(self, index_expr, context, limit):
        index_type = self.expression_result_type(index_expr)
        if index_type is not None:
            mapped_type = self.map_type(index_type)
            if not self.is_scalar_integer_type(mapped_type):
                raise OpenGLRefusal(
                    f"GLSL mesh {context} index requires a scalar integer value, "
                    f"got {mapped_type}"
                )

        literal_index = self.literal_int_value(index_expr, self.literal_int_constants)
        if literal_index is None or limit is None:
            return
        if literal_index < 0 or literal_index >= limit:
            raise OpenGLRefusal(
                f"GLSL mesh {context} index {literal_index} out of range; "
                f"valid range is 0..{limit - 1}"
            )

    def validate_mesh_output_helper_index(self, index_expr, roles, context):
        limit, _ = self.mesh_output_role_count_limit(roles)
        self.validate_mesh_output_index_expression(index_expr, context, limit)

    def generate_mesh_output_counts_call(self, func_name, args):
        if func_name != "SetMeshOutputCounts":
            return None
        if self.current_stage_entry_type != "mesh":
            return None
        if len(args) != 2:
            raise OpenGLRefusal(
                "GLSL mesh SetMeshOutputCounts requires exactly two arguments "
                "(vertex count, primitive count)"
            )
        self.validate_mesh_output_count_argument(args, 0, ("vertices",), "vertex")
        self.validate_mesh_output_count_argument(
            args, 1, ("primitives", "indices"), "primitive"
        )
        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        return f"SetMeshOutputsEXT({generated_args})"

    def generate_mesh_output_helper_call_statement(self, node, indent=0):
        if not isinstance(node, FunctionCallNode):
            return None
        if self.current_stage_entry_type != "mesh":
            return None

        func_name = self.function_call_name(node)
        args = getattr(node, "arguments", getattr(node, "args", [])) or []
        indent_str = "    " * indent
        helper = self.mesh_output_helper_call_components(func_name, args)
        if helper is None:
            return None

        if helper["kind"] == "comment":
            return f"{indent_str}{helper['text']};\n"
        if helper["kind"] == "vertex":
            return (
                f"{indent_str}gl_MeshVerticesEXT[{helper['index']}].gl_Position = "
                f"{helper['value']};\n"
            )
        if helper["kind"] == "primitive":
            return (
                f"{indent_str}{helper['builtin']}[{helper['index']}] = "
                f"{helper['value']};\n"
            )
        return None

    def mesh_output_helper_call_components(self, func_name, args):
        if func_name == "SetVertex":
            if len(args) != 2:
                raise OpenGLRefusal(
                    "GLSL mesh SetVertex helper requires exactly two arguments "
                    "(vertex index, position)"
                )
            self.validate_mesh_output_helper_index(
                args[0], ("vertices",), "SetVertex vertex"
            )
            index = self.generate_expression(args[0])
            value = self.glsl_mesh_vertex_helper_value(args[1])
            return {"kind": "vertex", "index": index, "value": value}

        if func_name == "SetPrimitive":
            if len(args) != 2:
                raise OpenGLRefusal(
                    "GLSL mesh SetPrimitive helper requires exactly two arguments "
                    "(primitive index, primitive indices)"
                )
            index_builtin = self.mesh_primitive_index_builtin(
                self.current_mesh_output_topology
            )
            if index_builtin is None:
                return {
                    "kind": "comment",
                    "text": self.refuse_lowering(
                        "mesh SetPrimitive: requires a declared output topology"
                    ),
                }
            self.validate_mesh_output_helper_index(
                args[0], ("primitives", "indices"), "SetPrimitive primitive"
            )
            index = self.generate_expression(args[0])
            value = self.glsl_mesh_primitive_helper_value(args[1])
            return {
                "kind": "primitive",
                "builtin": index_builtin,
                "index": index,
                "value": value,
            }

        return None

    def reject_mesh_output_helper_expression_context(self, func_name, args):
        if self.current_stage_entry_type != "mesh":
            return
        if func_name not in {"SetVertex", "SetPrimitive"}:
            return
        self.mesh_output_helper_call_components(func_name, args)
        raise OpenGLRefusal(
            f"GLSL mesh {func_name} helper can only be used as a statement"
        )

    def glsl_mesh_vertex_helper_value(self, expr):
        result_type = self.expression_result_type(expr)
        value_type = self.map_type(result_type) if result_type is not None else None
        if value_type is not None and value_type not in {"vec3", "vec4"}:
            raise OpenGLRefusal(
                f"GLSL mesh SetVertex position requires vec3 or vec4, got {value_type}"
            )

        value = self.generate_expression_with_expected(expr, None)
        if value_type == "vec3":
            return f"vec4({value}, 1.0)"
        return value

    def glsl_mesh_primitive_helper_value(self, expr):
        expected_type = {
            "points": "uint",
            "lines": "uvec2",
            "triangles": "uvec3",
        }.get(self.current_mesh_output_topology)

        result_type = self.expression_result_type(expr)
        value_type = self.map_type(result_type) if result_type is not None else None
        if expected_type and value_type is not None and value_type != expected_type:
            raise OpenGLRefusal(
                "GLSL mesh SetPrimitive for "
                f"{self.current_mesh_output_topology} topology requires "
                f"{expected_type} primitive indices, got {value_type}"
            )

        return self.generate_expression_with_expected(expr, expected_type)

    def generate_mesh_output_assignment_statement(self, node, indent=0):
        if (
            not isinstance(node, AssignmentNode)
            or not self.current_mesh_output_parameters
        ):
            return None

        left_node = getattr(node, "target", getattr(node, "left", None))
        right_node = getattr(node, "value", getattr(node, "right", None))
        op = self.map_operator(getattr(node, "operator", getattr(node, "op", "=")))
        indent_str = "    " * indent

        target = self.mesh_output_assignment_target(left_node)
        if target is not None:
            value_type = self.expression_result_type(left_node)
            right = self.generate_expression_with_expected(right_node, value_type)
            right = glsl_mesh_builtin_value(self, target, right, value_type)
            return f"{indent_str}{target} {op} {right};\n"

        expanded_targets = self.mesh_output_whole_assignment_targets(left_node)
        if expanded_targets is None or op != "=":
            return None

        access = self.mesh_output_array_access(left_node)
        constructor_values = (
            self.mesh_output_constructor_assignment_values(access[0], right_node)
            if access is not None
            else None
        )
        if constructor_values is None and access is not None:
            self.validate_mesh_output_whole_assignment_value(access[0], right_node)
        right = self.generate_expression_with_expected(
            right_node, self.expression_result_type(left_node)
        )
        code = ""
        for target, member_name, member_type in expanded_targets:
            value_expr = (
                constructor_values.get(member_name)
                if constructor_values is not None
                else None
            )
            if value_expr is None:
                value = f"{right}.{member_name}"
            else:
                value = self.generate_expression_with_expected(value_expr, member_type)
            value = glsl_mesh_builtin_value(self, target, value, member_type)
            code += f"{indent_str}{target} = {value};\n"
        return code

    def validate_mesh_output_whole_assignment_value(self, info, expr):
        expected_type = self.type_name_string(info.get("element_type"))
        if expected_type is None:
            return

        if self.mesh_output_constructor_for_type(expr, expected_type):
            raise OpenGLRefusal(
                "GLSL mesh output "
                f"@{info['role']} '{info['name']}' assignment requires a "
                f"complete {expected_type} constructor value"
            )

        result_type = self.type_name_string(self.expression_result_type(expr))
        if result_type is None:
            return

        expected_label = self.map_type(expected_type)
        result_label = self.map_type(result_type)
        if result_label != expected_label:
            raise OpenGLRefusal(
                "GLSL mesh output "
                f"@{info['role']} '{info['name']}' assignment requires "
                f"{expected_label} value, got {result_label}"
            )

    def mesh_output_constructor_for_type(self, expr, struct_name):
        if isinstance(expr, FunctionCallNode):
            return self.function_call_name(expr) == struct_name
        if not isinstance(expr, ConstructorNode):
            return False
        constructor_type = self.type_name_string(
            getattr(expr, "constructor_type", None)
        )
        return constructor_type == struct_name

    def mesh_output_constructor_assignment_values(self, info, expr):
        struct = self.structs_by_name.get(info["element_type"])
        if struct is None:
            return None

        members = getattr(struct, "members", []) or []
        field_values = self.mesh_output_constructor_field_values(
            expr, info["element_type"], members
        )
        if field_values is None:
            return None
        return {member.name: value_expr for member, value_expr in field_values}

    def mesh_output_constructor_field_values(self, expr, struct_name, members):
        if isinstance(expr, FunctionCallNode):
            if self.function_call_name(expr) != struct_name:
                return None
            args = getattr(expr, "arguments", getattr(expr, "args", [])) or []
            if len(args) != len(members):
                return None
            return list(zip(members, args))

        if not isinstance(expr, ConstructorNode):
            return None

        constructor_type = self.type_name_string(
            getattr(expr, "constructor_type", None)
        )
        if constructor_type != struct_name:
            return None

        args = list(getattr(expr, "arguments", []) or [])
        named_args = dict(getattr(expr, "named_arguments", {}) or {})
        if len(args) > len(members):
            return None

        values = []
        for index, member in enumerate(members):
            if index < len(args):
                values.append((member, args[index]))
                continue
            if member.name not in named_args:
                return None
            values.append((member, named_args[member.name]))
        return values

    def mesh_output_assignment_target(self, node):
        if isinstance(node, MemberAccessNode):
            access = self.mesh_output_array_access(getattr(node, "object", None))
            if access is None:
                return None
            info, index = access
            if info["role"] not in {"vertices", "primitives"}:
                return None
            return self.mesh_output_member_target(info, index, str(node.member))

        access = self.mesh_output_array_access(node)
        if access is None:
            return None
        info, index = access
        if info["role"] != "indices":
            return None
        index_builtin = info.get("index_builtin")
        return f"{index_builtin}[{index}]" if index_builtin else None

    def mesh_output_whole_assignment_targets(self, node):
        access = self.mesh_output_array_access(node)
        if access is None:
            return None
        info, index = access
        if info["role"] not in {"vertices", "primitives"}:
            return None
        targets = []
        for member_name, member_info in info["members"].items():
            target = self.mesh_output_member_target(info, index, member_name)
            if target is None:
                return None
            targets.append((target, member_name, member_info["type"]))
        return targets

    def mesh_output_array_access(self, node):
        if not isinstance(node, ArrayAccessNode):
            return None
        array_expr = getattr(node, "array", getattr(node, "array_expr", None))
        name = self.expression_name(array_expr)
        info = self.current_mesh_output_parameters.get(name)
        if info is None:
            return None
        index_expr = getattr(node, "index", getattr(node, "index_expr", None))
        limit, _ = self.mesh_output_index_limit(info)
        self.validate_mesh_output_index_expression(
            index_expr, f"output @{info['role']} '{info['name']}'", limit
        )
        return info, self.generate_expression(index_expr)

    def mesh_output_index_limit(self, info):
        if info.get("count_value") is not None:
            return info["count_value"], f"@{info['role']} array size"
        limit = self.current_mesh_output_count_limits.get(info["role"])
        if limit is not None:
            return limit, self.mesh_output_role_count_attribute(info["role"])
        return None, None

    def mesh_output_member_target(self, info, index, member_name):
        member_info = info["members"].get(member_name)
        if member_info is None:
            return None

        semantic = member_info["semantic"]
        if info["role"] == "vertices":
            if self.is_vertex_builtin_output(semantic):
                return f"gl_MeshVerticesEXT[{index}].{semantic}"
            return f"{member_info['output_name']}[{index}]"

        if info["role"] == "primitives":
            if self.is_mesh_primitive_builtin(semantic):
                return f"gl_MeshPrimitivesEXT[{index}].{semantic}"
            return f"{member_info['output_name']}[{index}]"

        return None

    def glsl_tessellation_factor_assignment_expected_type(self, target):
        target_kind = self.glsl_tessellation_factor_assignment_target_kind(target)
        if target_kind is None:
            return None

        builtin_name, is_component, index_expr = target_kind
        if self.current_stage_entry_type not in {None, "tessellation_control"}:
            raise OpenGLRefusal(
                "GLSL tessellation factor assignments are only valid in "
                "tessellation_control stages"
            )
        if not is_component:
            raise OpenGLRefusal(
                f"GLSL tessellation factor {builtin_name} assignment requires "
                "an indexed scalar component target"
            )
        self.validate_glsl_tessellation_factor_assignment_index(
            builtin_name, index_expr
        )
        return "float"

    def glsl_tessellation_factor_assignment_target_kind(self, target):
        if isinstance(target, ArrayAccessNode):
            array_expr = getattr(target, "array", getattr(target, "array_expr", None))
            builtin_name = self.expression_name(array_expr)
            if builtin_name in self.GLSL_TESSELLATION_FACTOR_BUILTINS:
                index_expr = getattr(
                    target, "index", getattr(target, "index_expr", None)
                )
                return (
                    builtin_name,
                    not isinstance(array_expr, ArrayAccessNode),
                    index_expr,
                )
            return None

        builtin_name = self.expression_name(target)
        if builtin_name in self.GLSL_TESSELLATION_FACTOR_BUILTINS:
            return builtin_name, False, None
        return None

    def validate_glsl_tessellation_factor_assignment_index(
        self, builtin_name, index_expr
    ):
        index_type = self.expression_result_type(index_expr)
        if index_type is not None:
            mapped_type = self.map_type(index_type)
            if not self.is_scalar_integer_type(mapped_type):
                raise OpenGLRefusal(
                    f"GLSL tessellation factor {builtin_name} component index "
                    "requires a scalar integer value, "
                    f"got {mapped_type}"
                )

        literal_index = self.literal_int_value(index_expr, self.literal_int_constants)
        if literal_index is None:
            return

        component_count = self.GLSL_TESSELLATION_FACTOR_COMPONENT_COUNTS[builtin_name]
        if literal_index < 0 or literal_index >= component_count:
            raise OpenGLRefusal(
                f"GLSL tessellation factor {builtin_name} component index "
                f"{literal_index} out of range; valid range is "
                f"0..{component_count - 1}"
            )

    def validate_glsl_tessellation_factor_assignment_value(self, target, value):
        builtin_name = self.expression_name(target)
        value_type = self.expression_result_type(value)
        if value_type is None:
            return
        mapped_type = self.map_type(value_type)
        if not self.is_scalar_numeric_type(mapped_type):
            raise OpenGLRefusal(
                f"GLSL tessellation factor {builtin_name} component assignment "
                f"requires a scalar numeric value, got {mapped_type}"
            )

    def glsl_side_effect_free_expression(self, expression):
        if expression is None or isinstance(expression, LiteralNode):
            return True
        if isinstance(expression, VariableNode) or (
            hasattr(expression, "__class__")
            and "IdentifierNode" in str(expression.__class__)
        ):
            return True
        if isinstance(expression, MemberAccessNode):
            return self.glsl_side_effect_free_expression(expression.object)
        if isinstance(expression, ArrayAccessNode):
            return self.glsl_side_effect_free_expression(
                expression.array
            ) and self.glsl_side_effect_free_expression(expression.index)
        if isinstance(expression, BinaryOpNode):
            return self.glsl_side_effect_free_expression(
                expression.left
            ) and self.glsl_side_effect_free_expression(expression.right)
        if isinstance(expression, UnaryOpNode):
            operator = self.map_operator(expression.op)
            return operator not in {
                "++",
                "--",
            } and self.glsl_side_effect_free_expression(expression.operand)
        if isinstance(expression, TernaryOpNode):
            return all(
                self.glsl_side_effect_free_expression(part)
                for part in (
                    expression.condition,
                    expression.true_expr,
                    expression.false_expr,
                )
            )
        if isinstance(expression, FunctionCallNode):
            constructor = self.glsl_constructor_type(
                self.function_call_name(expression)
            )
            if constructor is None:
                return False
            return all(
                self.glsl_side_effect_free_expression(argument)
                for argument in (
                    getattr(expression, "arguments", getattr(expression, "args", []))
                    or []
                )
            )
        if isinstance(expression, ConstructorNode):
            return all(
                self.glsl_side_effect_free_expression(argument)
                for argument in list(getattr(expression, "arguments", []) or []) + list(
                    (getattr(expression, "named_arguments", {}) or {}).values()
                )
            )
        return False

    def glsl_stable_update_target(self, expression):
        if isinstance(expression, VariableNode) or (
            hasattr(expression, "__class__")
            and "IdentifierNode" in str(expression.__class__)
        ):
            return True
        if (
            isinstance(expression, UnaryOpNode)
            and self.map_operator(expression.op) == "*"
        ):
            return (
                self.generate_glsl_private_pointer_view_access(
                    expression.operand,
                    access_expression=expression,
                )
                is not None
            )
        if isinstance(expression, MemberAccessNode):
            return self.glsl_stable_update_target(expression.object)
        if isinstance(expression, ArrayAccessNode):
            return self.glsl_stable_update_target(
                expression.array
            ) and self.glsl_side_effect_free_expression(expression.index)
        return False

    def glsl_boolean_compound_assignment_error(
        self,
        node,
        target,
        operator,
        target_type,
        right_type,
        *,
        reason,
        detail,
        operation_type=None,
    ):
        raise self.GLSL_BOOLEAN_COMPOUND_ASSIGNMENT_ERROR(
            f"{self.GLSL_TARGET_DISPLAY_NAME} cannot lower boolean compound "
            f"assignment '{operator}': {detail}",
            operator=operator,
            target=expression_debug_name(target),
            target_type=self.type_name_string(target_type),
            right_type=self.type_name_string(right_type),
            operation_type=operation_type,
            context="compound assignment",
            reason=reason,
            source_location=getattr(node, "source_location", None),
        )

    def glsl_boolean_compound_temporary(self, base_name, vtype):
        name = self.glsl_synthetic_local_identifier(base_name)
        self.local_variable_types[name] = vtype
        self.local_variable_source_types[name] = vtype
        return name

    def glsl_stabilize_boolean_compound_lvalue(
        self,
        node,
        target,
        operator,
        target_type,
        right_type,
    ):
        def reject(reason, detail):
            self.glsl_boolean_compound_assignment_error(
                node,
                target,
                operator,
                target_type,
                right_type,
                reason=reason,
                detail=detail,
            )

        prefix = []
        stabilized_target = deepcopy(target)

        def stabilize(expression):
            if isinstance(expression, (str, IdentifierNode, VariableNode)):
                return expression

            if isinstance(expression, MemberAccessNode):
                owner = stabilize(
                    getattr(
                        expression, "object_expr", getattr(expression, "object", None)
                    )
                )
                expression.object_expr = owner
                expression.object = owner
                return expression

            if isinstance(expression, SwizzleNode):
                expression.vector_expr = stabilize(expression.vector_expr)
                return expression

            if isinstance(expression, ArrayAccessNode):
                array = stabilize(
                    getattr(
                        expression, "array_expr", getattr(expression, "array", None)
                    )
                )
                expression.array_expr = array
                expression.array = array
                index = getattr(
                    expression, "index_expr", getattr(expression, "index", None)
                )
                index_info = self.glsl_value_type_info(
                    self.glsl_source_expression_type(index)
                )
                static_index = (
                    isinstance(index, int)
                    and not isinstance(index, bool)
                    or isinstance(index, LiteralNode)
                    and index_info is not None
                    and index_info["family"] in {"int", "uint"}
                    and index_info["width"] == 1
                    and self.literal_int_value(index, {}) is not None
                )
                if static_index:
                    return expression

                if (
                    index_info is None
                    or index_info["family"] not in {"int", "uint"}
                    or index_info["width"] != 1
                ):
                    reject(
                        "unresolved-index-type",
                        (
                            "each dynamic lvalue index must resolve to a scalar "
                            "integer type before it can be evaluated once"
                        ),
                    )
                promoted_index_type = self.glsl_promoted_integer_type(index_info)
                if promoted_index_type is None:
                    reject(
                        "unresolved-index-type",
                        "the dynamic lvalue index has no representable promotion",
                    )
                rendered_index = self.generate_expression_with_expected(
                    index, promoted_index_type
                )
                index_name = self.glsl_boolean_compound_temporary(
                    "cgl_bool_compound_index", promoted_index_type
                )
                prefix.append(f"{promoted_index_type} {index_name} = {rendered_index}")
                stable_index = IdentifierNode(
                    index_name,
                    source_location=getattr(index, "source_location", None),
                )
                expression.index_expr = stable_index
                expression.index = stable_index
                return expression

            reject(
                "unstable-assignment-target",
                (
                    "only identifier, member, swizzle, and indexed lvalues can be "
                    "stabilized without changing evaluation semantics"
                ),
            )

        return prefix, stabilize(stabilized_target)
