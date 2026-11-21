"""Rust member, index and identifier values, operand promotion plans, literals and identifiers."""

import re

from ..ast import (
    ArrayAccessNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    VariableNode,
)
from ..float_limit_constants import float_limit_constant
from ..standard_constants import standard_math_constant
from ..refusals import RustRefusal
from .char_literals import char_literal_code
from .rust_narrow_integers import (
    RUST_EXACT_WIDTH_INTEGER_TYPES,
    RUST_SCALAR_RANKS,
    promoted_operand_type,
)


class RustValueMixin:
    """Value and promotion members of ``RustCodeGen``."""

    def rust_temp_binding_type_hint(self, expr):
        func_name = (
            self.function_call_name(getattr(expr, "function", None))
            if isinstance(expr, FunctionCallNode)
            else None
        )
        if func_name is None:
            return None

        mapped_name = self.mapped_function_name(
            func_name,
            len(getattr(expr, "arguments", getattr(expr, "args", [])) or []),
            getattr(expr, "arguments", getattr(expr, "args", [])) or [],
        )
        if mapped_name not in {"texture_size", "texture_size_lod", "image_size"}:
            return None

        result_type = self.expression_result_type(expr)
        if result_type is None:
            return None
        return self.map_type(result_type)

    def rust_member_base_expression(self, generated_expr):
        generated_expr = str(generated_expr)
        if generated_expr.startswith("*") and not generated_expr.startswith("(*"):
            return f"({generated_expr})"
        return generated_expr

    def generate_member_access_expression(self, expr):
        access = self.glsl_buffer_block_access(expr)
        if access == "writeonly" and self.assignment_lhs_depth == 0:
            return self.unsupported_glsl_buffer_block_expression(
                "load",
                "a writeonly block cannot be read",
                self.expression_result_type(expr),
            )

        obj_expr = getattr(expr, "object_expr", getattr(expr, "object", ""))
        member = getattr(expr, "member", "")
        self.member_object_depth += 1
        try:
            obj = self.generate_expression(obj_expr)
        finally:
            self.member_object_depth -= 1
        obj = self.lazy_static_object_expression(obj_expr, obj)

        swizzle_components = self.member_swizzle_components(expr)
        if swizzle_components is None:
            access = f"{obj}.{self.rust_identifier(member)}"
            if self.should_clone_member_access_value(expr):
                return f"{access}.clone()"
            return access

        if len(swizzle_components) == 1:
            return f"{obj}.{swizzle_components[0]}"

        rust_type = self.swizzle_constructor_type(obj_expr, len(swizzle_components))
        if not self.is_repeat_safe_expression(obj_expr):
            temp_name = self.next_swizzle_temp_name()
            args = ", ".join(
                f"{temp_name}.{component}" for component in swizzle_components
            )
            return (
                f"{{ let {temp_name} = {obj}; "
                f"{self.rust_constructor_path(rust_type)}::new({args}) }}"
            )

        args = ", ".join(f"{obj}.{component}" for component in swizzle_components)
        return f"{self.rust_constructor_path(rust_type)}::new({args})"

    def should_clone_member_access_value(self, expr):
        if (
            self.assignment_lhs_depth > 0
            or self.member_object_depth > 0
            or self.array_object_depth > 0
        ):
            return False

        member_type = self.expression_result_type(expr)
        if member_type is None:
            return False

        return not self.type_is_copy_derivable(
            member_type,
            self.current_generic_param_names,
        )

    def should_clone_array_access_value(self, expr):
        if (
            self.assignment_lhs_depth > 0
            or self.member_object_depth > 0
            or self.array_object_depth > 0
        ):
            return False

        element_type = self.expression_result_type(expr)
        if element_type is None:
            return False

        return not self.type_is_copy_derivable(
            element_type,
            self.current_generic_param_names,
        )

    def generate_identifier_value_expression(self, name):
        name = self.lambda_capture_alias_name(name) or name
        builtin_value = self.rust_builtin_value_expression(name)
        if builtin_value is not None and name not in self.variable_types:
            return builtin_value
        constant_value = self.rust_predefined_constant_expression(name)
        if constant_value is not None:
            return constant_value
        value = self.lazy_static_identifier_expression(name)
        if self.should_clone_identifier_value(name):
            return self.clone_value_expression(value)
        return value

    def rust_predefined_constant_expression(self, name):
        """C and Metal predefined constants (INFINITY, FLT_MAX, M_PI_F, ...) that the program does not declare.

        Rust spells the binary32 limits as associated constants of f32, which are exact, and the math constants as
        suffixed decimal literals with the source's own digits.
        """
        if not isinstance(name, str) or name in self.variable_types:
            return None
        if name in getattr(self, "global_variable_types", {}) or name in getattr(
            self, "local_variable_names", ()
        ):
            return None
        limit = float_limit_constant(name)
        if limit is not None:
            return {
                "INFINITY": "f32::INFINITY",
                "HUGE_VALF": "f32::INFINITY",
                "NAN": "f32::NAN",
                "MAXFLOAT": "f32::MAX",
                "FLT_MAX": "f32::MAX",
                "FLT_MIN": "f32::MIN_POSITIVE",
                "FLT_EPSILON": "f32::EPSILON",
            }.get(name)
        constant = standard_math_constant(name)
        if constant is not None:
            suffix = "f64" if constant.scalar_type == "double" else "f32"
            return f"{constant.decimal}{suffix}"
        return None

    def rust_builtin_value_expression(self, name):
        return {
            "gl_WorkGroupID": "workgroup_id()",
            "gl_LocalInvocationID": "local_invocation_id()",
            "gl_GlobalInvocationID": "global_invocation_id()",
            "gl_LocalInvocationIndex": "local_invocation_index()",
            "gl_NumWorkGroups": "num_workgroups()",
            "gl_SubgroupInvocationID": "subgroup_invocation_id()",
            "gl_SubgroupSize": "subgroup_size()",
            "gl_SubgroupID": "subgroup_id()",
            "gl_NumSubgroups": "num_subgroups()",
            "SV_GroupID": "workgroup_id()",
            "SV_GroupThreadID": "local_invocation_id()",
            "SV_DispatchThreadID": "global_invocation_id()",
            "SV_GroupIndex": "local_invocation_index()",
            "gl_LaunchIDEXT": "ray_launch_id()",
            "gl_LaunchSizeEXT": "ray_launch_size()",
            "gl_HitTEXT": "ray_hit_t()",
            "gl_HitKindEXT": "ray_hit_kind()",
            "gl_WorldRayOriginEXT": "ray_world_origin()",
            "gl_WorldRayDirectionEXT": "ray_world_direction()",
            "gl_ObjectRayOriginEXT": "ray_object_origin()",
            "gl_ObjectRayDirectionEXT": "ray_object_direction()",
            "gl_RayTminEXT": "ray_t_min()",
            "gl_IncomingRayFlagsEXT": "ray_incoming_flags()",
            "gl_InstanceCustomIndexEXT": "ray_instance_custom_index()",
            "gl_GeometryIndexEXT": "ray_geometry_index()",
        }.get(name)

    def rust_builtin_value_type(self, name):
        return {
            "gl_WorkGroupID": "uvec3",
            "gl_LocalInvocationID": "uvec3",
            "gl_GlobalInvocationID": "uvec3",
            "gl_LocalInvocationIndex": "uint",
            "gl_NumWorkGroups": "uvec3",
            "gl_SubgroupInvocationID": "uint",
            "gl_SubgroupSize": "uint",
            "gl_SubgroupID": "uint",
            "gl_NumSubgroups": "uint",
            "SV_GroupID": "uvec3",
            "SV_GroupThreadID": "uvec3",
            "SV_DispatchThreadID": "uvec3",
            "SV_GroupIndex": "uint",
            "gl_LaunchIDEXT": "uvec3",
            "gl_LaunchSizeEXT": "uvec3",
            "gl_HitTEXT": "float",
            "gl_HitKindEXT": "uint",
            "gl_WorldRayOriginEXT": "vec3",
            "gl_WorldRayDirectionEXT": "vec3",
            "gl_ObjectRayOriginEXT": "vec3",
            "gl_ObjectRayDirectionEXT": "vec3",
            "gl_RayTminEXT": "float",
            "gl_IncomingRayFlagsEXT": "uint",
            "gl_InstanceCustomIndexEXT": "uint",
            "gl_GeometryIndexEXT": "uint",
        }.get(name)

    def is_default_constructible_ray_runtime_type(self, rust_type):
        return str(rust_type) in {
            "BuiltInTriangleIntersectionAttributes",
            "RayDesc",
            "RayQuery",
            "RayTracingAccelerationStructure",
        }

    def should_clone_identifier_value(self, name):
        if (
            self.assignment_lhs_depth > 0
            or self.member_object_depth > 0
            or self.array_object_depth > 0
        ):
            return False

        value_type = self.variable_types.get(name)
        if value_type is None:
            return False

        return not self.type_is_copy_derivable(
            value_type,
            self.current_generic_param_names,
        )

    def clone_value_expression(self, value):
        if isinstance(value, str) and value.startswith("*"):
            return f"({value}).clone()"
        return f"{value}.clone()"

    def swizzle_constructor_type(self, obj_expr, component_count):
        object_type = self.expression_result_type(obj_expr)
        vector_info = self.vector_type_info(object_type)
        component_type = vector_info["component_type"]
        result_type = self.vector_type_for_components(component_type, component_count)
        return self.map_type(result_type)

    def next_swizzle_temp_name(self):
        name = f"__cgl_swizzle_{self.swizzle_temp_counter}"
        self.swizzle_temp_counter += 1
        return name

    def next_vector_arg_temp_name(self):
        name = f"__cgl_vec_arg_{self.vector_arg_temp_counter}"
        self.vector_arg_temp_counter += 1
        return name

    def next_matrix_arg_temp_name(self):
        name = f"__cgl_mat_arg_{self.matrix_arg_temp_counter}"
        self.matrix_arg_temp_counter += 1
        return name

    def next_lambda_capture_temp_name(self, root):
        safe_root = re.sub(r"[^0-9A-Za-z_]+", "_", str(root)).strip("_")
        if not safe_root:
            safe_root = "capture"
        name = f"__cgl_capture_{safe_root}_{self.lambda_capture_temp_counter}"
        self.lambda_capture_temp_counter += 1
        return name

    def is_repeat_safe_expression(self, expr):
        if isinstance(expr, (IdentifierNode, VariableNode, LiteralNode)):
            return True
        if isinstance(expr, str):
            return True
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
            return self.is_repeat_safe_expression(object_expr)
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
            index_expr = getattr(expr, "index_expr", getattr(expr, "index", None))
            return self.is_repeat_safe_expression(
                array_expr
            ) and self.is_repeat_safe_expression(index_expr)
        return False

    def generate_array_index_expression(self, index_expr):
        index = self.generate_expression(index_expr)
        if self.is_usize_compatible_index(index_expr, index):
            return index
        return f"{index} as usize"

    def is_usize_compatible_index(self, index_expr, generated_index):
        if isinstance(index_expr, LiteralNode) and isinstance(index_expr.value, int):
            return index_expr.value >= 0
        if isinstance(index_expr, int):
            return index_expr >= 0
        if isinstance(index_expr, str) and index_expr.isdigit():
            return True
        return generated_index.endswith(" as usize")

    def member_swizzle_components(self, expr):
        if not isinstance(expr, MemberAccessNode):
            return None

        object_expr = getattr(expr, "object_expr", getattr(expr, "object", None))
        if not self.vector_type_info(self.expression_result_type(object_expr)):
            return None

        component_aliases = {
            "x": "x",
            "y": "y",
            "z": "z",
            "w": "w",
            "r": "x",
            "g": "y",
            "b": "z",
            "a": "w",
        }
        member = getattr(expr, "member", "")
        components = [component_aliases.get(component) for component in member]
        if not components or any(component is None for component in components):
            return None
        return components

    def scalar_constructor_type(self, func_name):
        if func_name in RUST_EXACT_WIDTH_INTEGER_TYPES:
            return RUST_EXACT_WIDTH_INTEGER_TYPES[func_name]
        scalar_types = {
            "i8": "i8",
            "u8": "u8",
            "bool": "bool",
            "char": "i8",
            "short": "i16",
            "ushort": "u16",
            "int": "i32",
            "uint": "u32",
            "long": "i64",
            "ulong": "u64",
            "float": "f32",
            "double": "f64",
            "half": "f16",
            "i16": "i16",
            "u16": "u16",
            "i32": "i32",
            "u32": "u32",
            "i64": "i64",
            "u64": "u64",
            "f16": "f16",
            "f32": "f32",
            "f64": "f64",
        }
        return scalar_types.get(func_name)

    def binary_scalar_result_type(self, left_type, right_type, operator=None):
        comparison_ops = {"<", ">", "<=", ">=", "==", "!="}
        logical_ops = {"&&", "||"}
        if operator in comparison_ops or operator in logical_ops:
            return "bool"

        return self.promoted_binary_operand_type(left_type, right_type, operator)

    def binary_scalar_operand_type(self, left_type, right_type, operator=None):
        if operator in {"&&", "||"}:
            return None
        return self.promoted_binary_operand_type(left_type, right_type, operator)

    def promoted_binary_operand_type(self, left_type, right_type, operator):
        """The type both operands compute in: integer promotion, then the higher-ranked of the two."""
        left = self.normalize_scalar_type(left_type)
        right = self.normalize_scalar_type(right_type)
        if left is None or right is None:
            return None
        return self.promoted_scalar_type(
            promoted_operand_type(left, operator),
            promoted_operand_type(right, operator),
        )

    def binary_composite_operand_type(self, left_type, right_type, operator=None):
        arithmetic_ops = {"+", "-", "*", "/", "%"}
        bitwise_ops = {"&", "|", "^", "<<", ">>"}

        vector_type = self.promoted_vector_type(left_type, right_type)
        if vector_type is not None:
            vector_info = self.vector_type_info(vector_type)
            component_type = self.normalize_scalar_type(vector_info["component_type"])
            if operator in arithmetic_ops:
                return vector_type
            if operator in bitwise_ops and component_type in {
                "i16",
                "u16",
                "i32",
                "u32",
                "i64",
                "u64",
            }:
                return vector_type
            return None

        matrix_type = self.promoted_matrix_type(left_type, right_type)
        if matrix_type is not None and operator in {"+", "-", "*", "/"}:
            return matrix_type

        vector_scalar_type = self.promoted_vector_scalar_type(left_type, right_type)
        if vector_scalar_type is not None:
            vector_info = self.vector_type_info(vector_scalar_type)
            component_type = self.normalize_scalar_type(vector_info["component_type"])
            if operator in arithmetic_ops:
                return vector_scalar_type
            if operator in bitwise_ops and component_type in {
                "i16",
                "u16",
                "i32",
                "u32",
                "i64",
                "u64",
            }:
                return vector_scalar_type
            return None

        matrix_scalar_type = self.promoted_matrix_scalar_type(left_type, right_type)
        if matrix_scalar_type is not None and operator in {"+", "-", "*", "/"}:
            return matrix_scalar_type
        return None

    def bool_vector_logical_plan(self, left_type, right_type, operator=None):
        if operator not in {"&&", "||"}:
            return None

        left_vector = self.vector_type_info(left_type)
        right_vector = self.vector_type_info(right_type)
        left_scalar = self.normalize_scalar_type(left_type)
        right_scalar = self.normalize_scalar_type(right_type)

        if left_vector is not None and left_vector["component_type"] != "bool":
            return None
        if right_vector is not None and right_vector["component_type"] != "bool":
            return None

        if left_vector is not None and right_vector is not None:
            if left_vector["size"] != right_vector["size"]:
                return None
            size = left_vector["size"]
        elif left_vector is not None and right_scalar == "bool":
            size = left_vector["size"]
        elif right_vector is not None and left_scalar == "bool":
            size = right_vector["size"]
        else:
            return None

        result_type = self.vector_type_for_components("bool", size)
        if result_type is None:
            return None
        return {"size": size, "result_type": result_type}

    def bool_vector_ternary_plan(
        self, condition_expr, true_expr, false_expr, target_type=None
    ):
        condition_info = self.vector_type_info(
            self.expression_result_type(condition_expr)
        )
        if condition_info is None or condition_info["component_type"] != "bool":
            return None

        size = condition_info["size"]
        true_type = self.expression_result_type(true_expr)
        false_type = self.expression_result_type(false_expr)
        result_type = self.bool_vector_ternary_result_type(
            true_type, false_type, size, target_type
        )
        result_info = self.vector_type_info(result_type)
        if result_info is None or result_info["size"] != size:
            return None
        return {
            "size": size,
            "result_type": result_type,
            "component_type": result_info["component_type"],
        }

    def bool_vector_ternary_result_type(
        self, true_type, false_type, condition_size, target_type=None
    ):
        target_info = self.vector_type_info(target_type)
        if target_info is not None and target_info["size"] == condition_size:
            return target_type

        true_info = self.vector_type_info(true_type)
        false_info = self.vector_type_info(false_type)
        true_scalar = self.normalize_scalar_type(true_type)
        false_scalar = self.normalize_scalar_type(false_type)

        if true_info is not None and true_info["size"] == condition_size:
            if false_info is not None and false_info["size"] == condition_size:
                return self.promoted_vector_type(true_type, false_type)
            if false_scalar is not None:
                return self.vector_type_for_promoted_scalar(true_info, false_scalar)
            return true_type

        if false_info is not None and false_info["size"] == condition_size:
            if true_scalar is not None:
                return self.vector_type_for_promoted_scalar(false_info, true_scalar)
            return false_type

        if true_scalar is None or false_scalar is None:
            return None
        component_type = self.promoted_scalar_type(true_scalar, false_scalar)
        if component_type is None:
            return None
        return self.vector_type_for_components(component_type, condition_size)

    def vector_comparison_plan(self, left_type, right_type, operator=None):
        if operator not in {"<", ">", "<=", ">=", "==", "!="}:
            return None

        left_vector = self.vector_type_info(left_type)
        right_vector = self.vector_type_info(right_type)
        left_scalar = self.normalize_scalar_type(left_type)
        right_scalar = self.normalize_scalar_type(right_type)

        if left_vector is not None and right_vector is not None:
            if left_vector["size"] != right_vector["size"]:
                return None
            component_type = self.promoted_scalar_type(
                left_vector["component_type"], right_vector["component_type"]
            )
            size = left_vector["size"]
        elif left_vector is not None and right_scalar is not None:
            component_type = self.promoted_scalar_type(
                left_vector["component_type"], right_scalar
            )
            size = left_vector["size"]
        elif right_vector is not None and left_scalar is not None:
            component_type = self.promoted_scalar_type(
                left_scalar, right_vector["component_type"]
            )
            size = right_vector["size"]
        else:
            return None

        if component_type is None:
            return None
        component_type = self.normalize_scalar_type(component_type)
        if component_type == "bool" and operator not in {"==", "!="}:
            return None

        result_type = self.vector_type_for_components("bool", size)
        if result_type is None:
            return None
        return {
            "component_type": component_type,
            "size": size,
            "result_type": result_type,
        }

    def vector_arithmetic_plan(self, left_type, right_type, operator=None):
        if operator not in {"+", "-", "*", "/", "%"}:
            return None

        left_vector = self.vector_type_info(left_type)
        right_vector = self.vector_type_info(right_type)
        left_scalar = self.normalize_scalar_type(left_type)
        right_scalar = self.normalize_scalar_type(right_type)

        if left_vector is not None and right_vector is not None:
            if left_vector["size"] != right_vector["size"]:
                return None
            component_type = self.promoted_scalar_type(
                left_vector["component_type"], right_vector["component_type"]
            )
            size = left_vector["size"]
            operand_shape = "vector_vector"
        elif left_vector is not None and right_scalar is not None:
            component_type = self.promoted_scalar_type(
                left_vector["component_type"], right_scalar
            )
            size = left_vector["size"]
            operand_shape = "vector_scalar"
        elif right_vector is not None and left_scalar is not None:
            component_type = self.promoted_scalar_type(
                left_scalar, right_vector["component_type"]
            )
            size = right_vector["size"]
            operand_shape = "scalar_vector"
        else:
            return None

        component_type = self.normalize_scalar_type(component_type)
        if component_type is None or component_type == "bool":
            return None

        result_type = self.vector_type_for_components(component_type, size)
        if result_type is None:
            return None

        return {
            "component_type": component_type,
            "operand_shape": operand_shape,
            "operator": operator,
            "result_type": result_type,
            "size": size,
        }

    def rust_vector_arithmetic_requires_lanes(self, plan):
        """Return whether Rust output should avoid relying on vector operator traits."""
        if plan["operand_shape"] != "vector_vector":
            return True

        if plan["size"] == 2:
            return True

        # The current standalone Rust contract only defines Vec4 addition.
        if plan["size"] == 4:
            return plan["operator"] != "+"

        if plan["size"] == 3:
            return plan["operator"] in {"/", "%"}

        return True

    def matrix_multiply_plan(self, left_type, right_type, operator=None):
        if operator != "*":
            return None

        left_matrix = self.matrix_type_info(left_type)
        right_matrix = self.matrix_type_info(right_type)
        if left_matrix is None or right_matrix is None:
            return None
        if left_matrix["columns"] != right_matrix["rows"]:
            return None

        component_type = self.promoted_scalar_type(
            left_matrix["component_type"], right_matrix["component_type"]
        )
        if component_type is None:
            return None

        result_type = self.matrix_type_for_dimensions(
            component_type,
            right_matrix["columns"],
            left_matrix["rows"],
        )
        if result_type is None:
            return None

        return {
            "component_type": component_type,
            "left_matrix": left_matrix,
            "result_columns": right_matrix["columns"],
            "result_rows": left_matrix["rows"],
            "result_type": result_type,
            "right_matrix": right_matrix,
            "shared_size": left_matrix["columns"],
        }

    def rust_matrix_vector_requires_lanes(self, plan):
        if not plan["matrix_on_left"]:
            return True

        matrix_info = self.matrix_type_info(plan["left_target_type"])
        vector_info = self.vector_type_info(plan["right_target_type"])
        if matrix_info is None or vector_info is None:
            return True

        component_type = self.normalize_scalar_type(matrix_info["component_type"])
        return not (
            component_type == "f32"
            and matrix_info["columns"] == matrix_info["rows"]
            and matrix_info["columns"] in {3, 4}
            and vector_info["size"] == matrix_info["columns"]
        )

    def binary_matrix_vector_plan(
        self, left_type, right_type, operator=None, target_type=None
    ):
        if operator != "*":
            return None

        left_matrix = self.matrix_type_info(left_type)
        right_matrix = self.matrix_type_info(right_type)
        left_vector = self.vector_type_info(left_type)
        right_vector = self.vector_type_info(right_type)

        if left_matrix is not None and right_vector is not None:
            if right_vector["size"] != left_matrix["columns"]:
                return None
            return self.build_matrix_vector_plan(
                left_matrix,
                right_vector,
                result_size=left_matrix["rows"],
                target_type=target_type,
                matrix_on_left=True,
            )

        if left_vector is not None and right_matrix is not None:
            if left_vector["size"] != right_matrix["rows"]:
                return None
            return self.build_matrix_vector_plan(
                right_matrix,
                left_vector,
                result_size=right_matrix["columns"],
                target_type=target_type,
                matrix_on_left=False,
            )

        return None

    def build_matrix_vector_plan(
        self, matrix_info, vector_info, result_size, target_type, matrix_on_left
    ):
        component_type = self.promoted_scalar_type(
            matrix_info["component_type"], vector_info["component_type"]
        )
        if component_type is None:
            return None

        operation_component_type = self.promoted_component_with_target(
            component_type, target_type, result_size
        )
        result_type = self.vector_type_for_components(
            operation_component_type, result_size
        )
        matrix_type = self.matrix_type_for_dimensions(
            operation_component_type, matrix_info["columns"], matrix_info["rows"]
        )
        vector_type = self.vector_type_for_components(
            operation_component_type, vector_info["size"]
        )
        if result_type is None or matrix_type is None or vector_type is None:
            return None

        if matrix_on_left:
            left_target_type = matrix_type
            right_target_type = vector_type
        else:
            left_target_type = vector_type
            right_target_type = matrix_type

        return {
            "result_type": result_type,
            "left_target_type": left_target_type,
            "matrix_on_left": matrix_on_left,
            "right_target_type": right_target_type,
        }

    def promoted_component_with_target(self, component_type, target_type, result_size):
        target_info = self.vector_type_info(target_type)
        if target_info is None or target_info["size"] != result_size:
            return component_type

        target_component_type = self.normalize_scalar_type(
            target_info["component_type"]
        )
        promoted_type = self.promoted_scalar_type(component_type, target_component_type)
        if promoted_type == target_component_type:
            return promoted_type
        return component_type

    def promoted_vector_scalar_type(self, left_type, right_type):
        left_vector = self.vector_type_info(left_type)
        right_vector = self.vector_type_info(right_type)
        left_scalar = self.normalize_scalar_type(left_type)
        right_scalar = self.normalize_scalar_type(right_type)

        if left_vector is not None and right_scalar is not None:
            return self.vector_type_for_promoted_scalar(left_vector, right_scalar)
        if right_vector is not None and left_scalar is not None:
            return self.vector_type_for_promoted_scalar(right_vector, left_scalar)
        return None

    def vector_type_for_promoted_scalar(self, vector_info, scalar_type):
        component_type = self.promoted_scalar_type(
            vector_info["component_type"], scalar_type
        )
        if component_type is None:
            return None
        return self.vector_type_for_components(component_type, vector_info["size"])

    def promoted_matrix_scalar_type(self, left_type, right_type):
        left_matrix = self.matrix_type_info(left_type)
        right_matrix = self.matrix_type_info(right_type)
        left_scalar = self.normalize_scalar_type(left_type)
        right_scalar = self.normalize_scalar_type(right_type)

        if left_matrix is not None and right_scalar is not None:
            return self.matrix_type_for_promoted_scalar(left_matrix, right_scalar)
        if right_matrix is not None and left_scalar is not None:
            return self.matrix_type_for_promoted_scalar(right_matrix, left_scalar)
        return None

    def matrix_type_for_promoted_scalar(self, matrix_info, scalar_type):
        component_type = self.promoted_scalar_type(
            matrix_info["component_type"], scalar_type
        )
        if component_type is None:
            return None
        return self.matrix_type_for_dimensions(
            component_type, matrix_info["columns"], matrix_info["rows"]
        )

    def promoted_scalar_type(self, left_type, right_type):
        left = self.normalize_scalar_type(left_type)
        right = self.normalize_scalar_type(right_type)
        if left is None or right is None:
            return None

        ranks = RUST_SCALAR_RANKS
        return left if ranks[left] >= ranks[right] else right

    def promoted_vector_type(self, left_type, right_type):
        left_info = self.vector_type_info(left_type)
        right_info = self.vector_type_info(right_type)
        if left_info is None or right_info is None:
            return None
        if left_info["size"] != right_info["size"]:
            return None

        component_type = self.promoted_scalar_type(
            left_info["component_type"], right_info["component_type"]
        )
        if component_type is None:
            return None
        return self.vector_type_for_components(component_type, left_info["size"])

    def promoted_matrix_type(self, left_type, right_type):
        left_info = self.matrix_type_info(left_type)
        right_info = self.matrix_type_info(right_type)
        if left_info is None or right_info is None:
            return None
        if (
            left_info["columns"] != right_info["columns"]
            or left_info["rows"] != right_info["rows"]
        ):
            return None

        component_type = self.promoted_scalar_type(
            left_info["component_type"], right_info["component_type"]
        )
        if component_type is None:
            return None
        return self.matrix_type_for_dimensions(
            component_type, left_info["columns"], left_info["rows"]
        )

    def normalize_binary_scalar_operand(
        self, expr, generated_expr, source_type, target_type
    ):
        source_type = self.normalize_scalar_type(source_type)
        if source_type is None or source_type == target_type:
            return generated_expr

        if self.is_integer_literal_expression(expr):
            if target_type in {"f32", "f64"}:
                return f"{self.integer_literal_expression_value(expr)}.0"
            if target_type == "f16":
                return f"({generated_expr} as f16)"
            return generated_expr

        if source_type == "bool" and target_type in {"f16", "f32", "f64"}:
            return f"(({generated_expr} as u32) as {target_type})"
        return f"({generated_expr} as {target_type})"

    def is_integer_literal_expression(self, expr):
        if isinstance(expr, int) and not isinstance(expr, bool):
            return True
        return (
            isinstance(expr, LiteralNode)
            and isinstance(expr.value, int)
            and not isinstance(expr.value, bool)
        )

    def integer_literal_expression_value(self, expr):
        return expr.value if isinstance(expr, LiteralNode) else expr

    def normalize_scalar_type(self, type_name):
        if type_name is None:
            return None
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)

        aliases = {
            "bool": "bool",
            "char": "i8",
            "short": "i16",
            "ushort": "u16",
            "int": "i32",
            "uint": "u32",
            "long": "i64",
            "ulong": "u64",
            "int64_t": "i64",
            "uint64_t": "u64",
            "half": "f16",
            "float": "f32",
            "double": "f64",
            "i8": "i8",
            "u8": "u8",
            "i16": "i16",
            "u16": "u16",
            "i32": "i32",
            "u32": "u32",
            "i64": "i64",
            "u64": "u64",
            "f16": "f16",
            "f32": "f32",
            "f64": "f64",
            **RUST_EXACT_WIDTH_INTEGER_TYPES,
        }
        return aliases.get(type_name)

    def format_literal(self, value, literal_type=None):
        if isinstance(value, bool):
            return "true" if value else "false"
        if literal_type == "bool" and isinstance(value, str):
            lower_value = value.lower()
            if lower_value in {"true", "false"}:
                return lower_value
        if literal_type == "char":
            return f"{char_literal_code(value, RustRefusal, 'Rust')}i8"
        if isinstance(value, str):
            escaped = self.escape_literal(value, quote='"')
            return f'"{escaped}"'
        if isinstance(value, int) and not -(2**31) <= value < 2**31:
            # An unsuffixed integer literal defaults to i32 where context does not fix its type, as in a cast
            # operand, and rustc rejects one out of i32 range (overflowing_literals).
            suffix = {
                "uint": "u32",
                "int64": "i64",
                "int64_t": "i64",
                "uint64": "u64",
                "uint64_t": "u64",
            }.get(str(literal_type))
            if suffix is None:
                suffix = "u32" if 0 <= value < 2**32 else ("i64" if value < 2**63 else "u64")
            return f"{value}{suffix}"
        return str(value)

    def register_variable_type(self, name, type_name, scope="local"):
        if not name or type_name is None:
            return
        if hasattr(type_name, "name") or hasattr(type_name, "element_type"):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)
        self.variable_types[name] = type_name
        if scope == "local":
            self.local_variable_names.add(name)

    def lazy_static_identifier_expression(self, name):
        if self.is_static_reference(name):
            symbol = self.static_symbol_name(name)
            if self.is_lazy_static_reference(name):
                return f"{symbol}()"
            pointer_parts = self.pointer_type_parts(
                self.map_type(self.variable_types.get(name))
            )
            if pointer_parts is not None:
                is_mutable, pointee_type = pointer_parts
                pointer_keyword = "mut" if is_mutable else "const"
                return f"({symbol} as *{pointer_keyword} {pointee_type})"
            return symbol
        return self.rust_identifier(name)

    def lazy_static_object_expression(self, expr, generated_expr):
        name = self.get_expression_name(expr)
        if name is not None and self.is_lazy_static_reference(name):
            return f"({generated_expr})"
        return generated_expr

    def is_static_reference(self, name):
        return (
            name in self.static_variable_names and name not in self.local_variable_names
        )

    def is_lazy_static_reference(self, name):
        return name in self.lazy_static_names and name not in self.local_variable_names

    def static_symbol_name(self, name):
        symbol = self.static_symbol_names.get(name)
        if symbol is None:
            # collect_static_variable_names names every static before generation; a static it missed would be
            # declared and referenced under different names.
            return self.refuse_lowering(
                f"static {name}: not collected before generation", str(name)
            )
        return symbol

    def rust_identifier(self, name):
        """Return a Rust-safe value/field identifier for source-level names."""
        name = str(name)
        if name in {"self", "Self", "super", "crate"}:
            return f"{name}_"
        if name in self.rust_reserved_identifiers():
            return f"{name}_"
        return name

    def rust_callable_name(self, name):
        if isinstance(name, str) and name.isidentifier():
            return self.rust_identifier(name)
        return self.rust_turbofish_callable_name(name)

    def rust_turbofish_callable_name(self, name):
        """Restore the turbofish on a callee that carries explicit type arguments.

        CrossGL has no ``::``, so the Rust importer writes ``f::<T>(x)`` into
        CrossGL as ``f<T>(x)``.  Emitted back verbatim, rustc reads the ``<`` as
        a comparison and the generated crate does not parse, so the ``::`` goes
        back in front of the type arguments here.  A path that already carries
        one, and anything whose head is not a ``::``-separated path of plain
        identifiers, is left exactly as it was.
        """
        if not isinstance(name, str):
            return name
        head, separator, type_arguments = name.partition("<")
        if not separator:
            return name
        segments = head.split("::")
        if not all(segment.isidentifier() for segment in segments):
            return name
        return f"{head}::<{self.rust_explicit_type_arguments(type_arguments)}"

    def rust_explicit_type_arguments(self, type_arguments):
        """Spell the CrossGL type arguments ``float, vec3>`` of a call as Rust types.

        ``type_arguments`` is the text after the call's ``<``, closing ``>``
        included.  Each top-level argument goes through ``map_type``, so
        ``pick<float>`` becomes ``pick::<f32>`` rather than naming ``float``,
        which Rust does not have.  Text that is not a closed argument list is
        returned unchanged.
        """
        if not type_arguments.endswith(">"):
            return type_arguments
        arguments, depth, start = [], 0, 0
        body = type_arguments[:-1]
        for index, character in enumerate(body):
            if character in "<[(":
                depth += 1
            elif character in ">])":
                depth -= 1
            elif character == "," and depth == 0:
                arguments.append(body[start:index])
                start = index + 1
        if depth != 0:
            return type_arguments
        arguments.append(body[start:])
        mapped = [self.map_type(argument.strip()) for argument in arguments]
        return ", ".join(mapped) + ">"

    def rust_reserved_identifiers(self):
        return {
            "abstract",
            "as",
            "async",
            "await",
            "become",
            "box",
            "break",
            "const",
            "continue",
            "do",
            "dyn",
            "else",
            "enum",
            "extern",
            "false",
            "final",
            "fn",
            "for",
            "gen",
            "if",
            "impl",
            "in",
            "let",
            "loop",
            "macro",
            "match",
            "mod",
            "move",
            "mut",
            "override",
            "priv",
            "pub",
            "ref",
            "return",
            "static",
            "struct",
            "trait",
            "true",
            "try",
            "type",
            "typeof",
            "union",
            "unsafe",
            "unsized",
            "use",
            "virtual",
            "where",
            "while",
            "yield",
        }
