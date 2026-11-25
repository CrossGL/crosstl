"""Rust expression dispatch, inline intrinsics, bit casts and imported math functions."""

from ..ast import (
    ArrayLiteralNode,
    AssignmentNode,
    BlockNode,
    ConstructorNode,
    LambdaNode,
    MatchNode,
    MemberAccessNode,
    MeshOpNode,
    RayQueryOpNode,
    RayTracingOpNode,
    TextureNode,
    TextureOpNode,
    VariableNode,
    WaveOpNode,
)
from .assignment_values import is_reevaluable_place
from .comma_expressions import (
    is_comma_expression,
    refuse_comma_expression,
)
from .rust_exact_lookups import rust_foreign_call_reason
from .rust_gpu_builtins import generate_rust_gpu_builtin_call
from .rust_gpu_geometry import generate_rust_geometry_call
from .rust_gpu_images import generate_rust_gpu_image_call
from .rust_gpu_mesh import generate_rust_mesh_call
from .rust_memory_atomics import generate_rust_memory_atomic_call
from .shader_clock_refusals import (
    SHADER_CLOCK_BUILTINS,
    refuse_shader_clock_call,
)


class RustExpressionMixin:
    """Expression members of ``RustCodeGen``."""

    def generate_expression(self, expr):
        """Render a CrossGL expression as Rust expression syntax."""
        if expr is None:
            return ""
        elif is_comma_expression(expr):
            return refuse_comma_expression("Rust", expr)
        elif isinstance(expr, str):
            return expr
        elif isinstance(expr, (int, float, bool)):
            if isinstance(expr, bool):
                return "true" if expr else "false"
            return str(expr)
        elif hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            if hasattr(expr, "value"):
                literal_type = getattr(
                    getattr(expr, "literal_type", None), "name", None
                )
                return self.format_literal(expr.value, literal_type)
            return str(expr)
        elif hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            name = getattr(expr, "name", str(expr))
            return self.generate_identifier_value_expression(name)
        elif isinstance(expr, VariableNode):
            if hasattr(expr, "name"):
                return self.generate_identifier_value_expression(expr.name)
            else:
                return str(expr)
        elif hasattr(expr, "__class__") and "BinaryOp" in str(expr.__class__):
            return self.generate_binary_expression(expr)
        elif isinstance(expr, AssignmentNode):
            return self.generate_assignment_value(expr)
        elif isinstance(expr, ArrayLiteralNode):
            return self.generate_array_literal_expression(expr)
        elif isinstance(expr, ConstructorNode):
            return self.generate_constructor_expression(expr)
        elif isinstance(expr, TextureNode):
            return self.generate_texture_node_expression(expr)
        elif isinstance(expr, TextureOpNode):
            return self.generate_texture_op_node_expression(expr)
        elif isinstance(expr, BlockNode):
            return self.generate_block_expression(expr)
        elif isinstance(expr, RayTracingOpNode):
            return self.generate_ray_tracing_op_expression(expr)
        elif isinstance(expr, RayQueryOpNode):
            return self.generate_ray_query_op_expression(expr)
        elif isinstance(expr, MeshOpNode):
            return generate_rust_mesh_call(self, expr) or self.refuse_lowering(
                f"function: '{expr.operation}' has no rust-gpu lowering for these operands",
                "()",
            )
        elif isinstance(expr, LambdaNode):
            return self.generate_lambda_node_expression(expr)
        elif isinstance(expr, MatchNode):
            return self.generate_match_expression(expr)
        elif hasattr(expr, "__class__") and "UnaryOp" in str(expr.__class__):
            return self.generate_unary_expression(expr)
        elif hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", ""))
            index_expr = getattr(expr, "index_expr", getattr(expr, "index", ""))
            self.array_object_depth += 1
            try:
                array = self.generate_expression(array_expr)
            finally:
                self.array_object_depth -= 1
            array = self.lazy_static_object_expression(array_expr, array)
            component = self.generate_component_access(expr, array_expr, array)
            if component is not None:
                return component
            index = self.generate_array_index_expression(index_expr)
            padded = self.rust_gpu_padded_slice_read(array_expr, array, index)
            if padded is not None:
                return padded
            if (
                self.pointer_type_parts(
                    self.map_type(self.expression_result_type(array_expr))
                )
                is not None
            ):
                return f"unsafe {{ *({array}).add({index}) }}"
            access = f"{array}[{index}]"
            if self.should_clone_array_access_value(expr):
                return self.clone_value_expression(access)
            return access
        elif hasattr(expr, "__class__") and "FunctionCall" in str(expr.__class__):
            func_expr = getattr(expr, "function", getattr(expr, "name", "unknown"))
            func_name = None
            if getattr(func_expr, "name", None):
                func_name = func_expr.name
                callee = func_name
            elif isinstance(func_expr, str):
                func_name = func_expr
                callee = func_expr
            else:
                callee = self.generate_expression(func_expr)
            if (
                func_name in SHADER_CLOCK_BUILTINS
                and not self.is_user_defined_function(func_name)
            ):
                return refuse_shader_clock_call("Rust", func_name)
            args = getattr(expr, "arguments", getattr(expr, "args", []))

            if isinstance(func_expr, MemberAccessNode):
                texture_call = self.generate_texture_member_function_call(
                    func_expr,
                    args,
                )
                if texture_call is not None:
                    return texture_call
                slice_length = self.generate_rust_gpu_slice_length(func_expr, args)
                if slice_length is not None:
                    return slice_length

            qualified_method_call = self.generate_qualified_generic_trait_method_call(
                func_expr,
                args,
            )
            if qualified_method_call is not None:
                return qualified_method_call

            if func_name == "lambda":
                return self.generate_lambda_expression(args)

            user_function = self.rust_user_function_of_call(func_name)
            if user_function is not None:
                args_str = ", ".join(
                    self.generate_user_function_call_args(user_function, args)
                )
                return f"{self.rust_callable_name(callee)}({args_str})"

            memory_atomic = generate_rust_memory_atomic_call(self, func_name, args)
            if memory_atomic is not None:
                return memory_atomic
            image_call = generate_rust_gpu_image_call(self, func_name, args)
            if image_call is not None:
                return image_call
            rust_gpu_call = (
                generate_rust_gpu_builtin_call(self, func_name, args)
                or generate_rust_geometry_call(self, func_name, args)
                or generate_rust_mesh_call(self, expr)
            )
            if rust_gpu_call is not None:
                return rust_gpu_call
            if not self.is_rust_gpu_constructor_call(
                func_name, args
            ) and not self.is_rust_program_callable(func_name or callee):
                return self.refuse_lowering(
                    f"function: '{func_name or callee}' has no rust-gpu lowering for these operands",
                    "()",
                )

            ray_tracing_call = self.generate_ray_tracing_call_expression(
                func_name,
                args,
            )
            if ray_tracing_call is not None:
                return ray_tracing_call

            enum_variant = self.generate_enum_variant_call_with_typed_args(
                func_name,
                args,
            )
            if enum_variant is not None:
                return enum_variant

            struct_constructor = self.generate_struct_new_call_with_typed_args(
                func_name,
                args,
            )
            if struct_constructor is not None:
                return struct_constructor

            if func_name == "mix" and len(args) == 3:
                bool_mix = self.generate_bool_mix_expression(args)
                if bool_mix is not None:
                    return bool_mix

            bitcast_alias = self.generate_bitcast_alias_call(func_name, args)
            if bitcast_alias is not None:
                return bitcast_alias

            bitcast_intrinsic = self.generate_bitcast_intrinsic_call(func_name, args)
            if bitcast_intrinsic is not None:
                return bitcast_intrinsic

            original_func_name = func_name
            func_name = self.mapped_function_name(func_name, len(args), args)
            cuda_shuffle_call = self.generate_cuda_shuffle_intrinsic_call(
                original_func_name, args
            )
            if cuda_shuffle_call is not None:
                return cuda_shuffle_call
            self.validate_rust_texture_helper_call(func_name, args)
            if func_name == "saturate" and len(args) == 1:
                arg = self.generate_expression(args[0])
                clamp_name = self.rust_imported_math_callable_name("clamp")
                return f"{clamp_name}({arg}, 0.0, 1.0)"

            reciprocal_call = self.generate_reciprocal_call(func_name, args)
            if reciprocal_call is not None:
                return reciprocal_call

            scalar_mul_add = self.generate_scalar_mul_add_call(func_name, args)
            if scalar_mul_add is not None:
                return scalar_mul_add

            dot_call = self.generate_inline_dot_call(func_name, args)
            if dot_call is not None:
                return dot_call

            vector_intrinsic = self.generate_inline_vector_intrinsic_call(
                func_name,
                args,
            )
            if vector_intrinsic is not None:
                return vector_intrinsic

            generic_intrinsic = self.generate_generic_intrinsic_call(func_name, args)
            if generic_intrinsic is not None:
                return generic_intrinsic

            scalar_cast = self.generate_scalar_constructor_call(func_name, args)
            if scalar_cast is not None:
                return scalar_cast

            vector_info = self.vector_type_info(func_name)
            if vector_info:
                return self.generate_vector_constructor_call(
                    func_name, vector_info, args
                )

            matrix_func_name = self.normalize_hlsl_matrix_type_name(func_name)
            if matrix_func_name in [
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
            ]:
                return self.generate_matrix_constructor_call(matrix_func_name, args)

            args_str = ", ".join(self.generate_expression(arg) for arg in args)
            callable_name = self.rust_imported_math_callable_name(func_name or callee)
            call = f"{self.rust_callable_name(callable_name)}({args_str})"
            foreign_call = rust_foreign_call_reason(self, original_func_name, func_name)
            if foreign_call is not None:
                return self.refuse_lowering(foreign_call, call)
            return call
        elif hasattr(expr, "__class__") and "MemberAccess" in str(expr.__class__):
            return self.generate_member_access_expression(expr)
        elif hasattr(expr, "__class__") and "TernaryOp" in str(expr.__class__):
            return self.generate_ternary_expression(expr)
        elif isinstance(expr, WaveOpNode):
            return self.generate_wave_op_expression(expr)
        else:
            return str(expr)

    def generate_unary_expression(self, expr):
        """Write a CrossGL unary operation as a Rust expression."""
        operand_expr = getattr(expr, "operand", "")
        raw_op = getattr(expr, "operator", getattr(expr, "op", "+"))
        op = self.map_operator(raw_op)
        if op in ["++", "--"]:
            return self.generate_increment_value(expr, operand_expr, op)
        if op == "~":
            operand = self.generate_expression(operand_expr)
            return f"(!{operand})"
        if op == "+":
            # Rust has no unary plus; +x is x.
            return f"({self.generate_expression(operand_expr)})"
        bool_vector_not = self.generate_bool_vector_not_expression(operand_expr, op)
        if bool_vector_not is not None:
            return bool_vector_not
        operand = self.generate_expression(operand_expr)
        if op == "-":
            operand_type = self.expression_result_type(operand_expr)
            vector = self.vector_type_info(operand_type)
            component = (
                vector["component_type"]
                if vector is not None
                else self.normalize_scalar_type(operand_type)
            )
            if component in {"u8", "u16", "u32", "u64", "uint"}:
                # Rust has no unary minus on unsigned types; C negation is modulo 2^n.
                if vector is not None:
                    return (
                        f"{self.map_type(operand_type)}::ZERO.wrapping_sub({operand})"
                    )
                return f"({operand}).wrapping_neg()"
        if op == "!":
            operand_type = self.normalize_scalar_type(
                self.expression_result_type(operand_expr)
            )
            if operand_type is not None and operand_type != "bool":
                zero_literal = "0.0" if operand_type in {"f16", "f32", "f64"} else "0"
                return f"({operand} == {zero_literal})"
        return f"({op}{operand})"

    def rust_user_function_of_call(self, func_name):
        """The program function a call names, also when it carries explicit type arguments (``pick<mat4>``)."""
        if self.is_user_defined_function(func_name):
            return func_name
        if isinstance(func_name, str) and func_name.endswith(">"):
            head = func_name.partition("<")[0]
            if head and self.is_user_defined_function(head):
                return head
        return None

    def is_rust_program_callable(self, callee):
        """Whether ``callee`` names something the program defines: a local closure or a callable parameter, or an
        associated function of a generic parameter or of a trait, struct or enum the program declares.
        """
        if not isinstance(callee, str) or not callee:
            return False
        if callee in self.variable_types or callee in self.local_variable_names:
            return True
        owner = callee.split("::", 1)[0] if "::" in callee else None
        if owner is None:
            return False
        if owner in self.current_generic_param_names:
            return True
        return owner in getattr(self, "rust_program_type_names", ())

    def is_rust_gpu_constructor_call(self, func_name, args):
        """Whether ``func_name(args)`` builds a value: a scalar conversion, a glam vector or matrix, a struct or an
        enum variant. Any other call that is neither a program function nor a rust-gpu built-in is refused.
        """
        if not isinstance(func_name, str):
            return False
        if self.enum_variant_call_info(func_name) is not None:
            return True
        if self.struct_constructor_call_type_name(func_name) is not None:
            return True
        if self.scalar_constructor_type(func_name) is not None:
            return True
        if self.vector_type_info(func_name) is not None:
            return True
        return self.matrix_type_info(func_name) is not None

    def generate_increment_update(self, operand_expr, op):
        """``x += 1`` or ``x -= 1``: a ``++``/``--`` whose value is not used."""
        component_store = self.expand_dynamic_component_write(
            operand_expr, lambda: self.generate_increment_update(operand_expr, op)
        )
        if component_store is not None:
            return component_store
        operand = self.generate_expression(operand_expr)
        assignment_op = "+=" if op == "++" else "-="
        increment = self.rust_increment_literal(
            self.expression_result_type(operand_expr)
        )
        return f"{operand} {assignment_op} {increment}"

    def generate_increment_value(self, expr, operand_expr, op):
        """A ``++``/``--`` used as a value: a block that updates the place and yields the new or prior value.

        A Rust compound assignment has type ``()``, so ``++x`` is ``{ x += 1; x }`` and ``x++`` reads the place
        into a local first.
        """
        update = self.generate_increment_update(operand_expr, op)
        if not is_reevaluable_place(operand_expr):
            return self.refuse_lowering(
                "increment used as a value: its operand has side effects, so the "
                "value cannot be read back",
                update,
            )
        place = self.generate_expression(operand_expr)
        if getattr(expr, "is_postfix", getattr(expr, "postfix", False)):
            return f"{{ let crosstl_prior = {place}; {update}; crosstl_prior }}"
        return f"{{ {update}; {place} }}"

    def rust_increment_literal(self, operand_type):
        scalar_type = self.normalize_scalar_type(operand_type)
        if scalar_type in {"f16", "f32", "f64"}:
            return "1.0"
        return "1"

    def generate_inline_dot_call(self, func_name, args):
        if func_name != "dot" or len(args or []) != 2:
            return None

        left_type = self.expression_result_type(args[0])
        right_type = self.expression_result_type(args[1])
        left_info = self.vector_type_info(left_type)
        right_info = self.vector_type_info(right_type)
        if left_info is None or right_info is None:
            return None
        if left_info["size"] != right_info["size"]:
            return None

        # The standalone Rust contract already has Vec3 dot; other vector widths
        # need concrete lane math instead of a runtime overload.
        if left_info["size"] == 3:
            return None

        component_type = self.promoted_scalar_type(
            left_info["component_type"],
            right_info["component_type"],
        )
        if component_type is None:
            return None

        temp_bindings = []
        left_lanes = self.vector_argument_lane_expressions(
            args[0],
            left_info,
            temp_bindings,
            component_type,
        )
        right_lanes = self.vector_argument_lane_expressions(
            args[1],
            right_info,
            temp_bindings,
            component_type,
        )
        products = [
            f"({left_lane} * {right_lane})"
            for left_lane, right_lane in zip(left_lanes, right_lanes)
        ]
        expression = " + ".join(products)
        if not temp_bindings:
            return expression

        bindings = " ".join(
            self.generate_temp_binding_statement(binding) for binding in temp_bindings
        )
        return f"{{ {bindings} {expression} }}"

    def generate_inline_vector_intrinsic_call(self, func_name, args):
        if len(args or []) != 1 or func_name not in {"length", "normalize"}:
            return None

        value_type = self.expression_result_type(args[0])
        value_info = self.vector_type_info(value_type)
        if value_info is None:
            return None

        # The standalone Rust contract already has Vec3 length/normalize.
        if value_info["size"] == 3:
            return None

        component_type = value_info["component_type"]
        temp_bindings = []
        value_expr = self.generate_expression(args[0])
        temp_name = self.next_vector_arg_temp_name()
        type_hint = self.map_type(value_type)
        temp_bindings.append((temp_name, value_expr, type_hint))

        components = ("x", "y", "z", "w")[: value_info["size"]]
        length_terms = [
            f"({temp_name}.{component} * {temp_name}.{component})"
            for component in components
        ]
        length_expr = f"({self.sum_rust_terms(length_terms)}).sqrt()"
        if func_name == "length":
            bindings = " ".join(
                self.generate_temp_binding_statement(binding)
                for binding in temp_bindings
            )
            return f"{{ {bindings} {length_expr} }}"

        length_name = self.next_vector_arg_temp_name()
        temp_bindings.append((length_name, length_expr, self.map_type(component_type)))
        lanes = [
            f"({temp_name}.{component} / {length_name})" for component in components
        ]
        return self.generate_constructor_call(
            self.map_type(value_type),
            lanes,
            temp_bindings,
        )

    def generate_generic_intrinsic_call(self, func_name, args):
        if func_name != "sqrt" or len(args) != 1:
            return None

        argument_type = self.expression_result_type(args[0])
        if argument_type not in self.current_generic_param_names:
            return None

        self.required_generic_math_traits.add("CglSqrt")
        return f"CglSqrt::cgl_sqrt({self.generate_expression(args[0])})"

    def generate_bitcast_alias_call(self, func_name, args):
        if func_name not in {"asfloat", "asint", "asuint"} or len(args or []) != 1:
            return None

        arg = args[0]
        arg_type = self.expression_result_type(arg)
        target_component_type = {
            "asfloat": "float",
            "asint": "int",
            "asuint": "uint",
        }[func_name]
        target_type = self.bitcast_value_type(arg_type, target_component_type)
        if target_type is None:
            return None
        source_component_type = self.bitcast_argument_component_type(arg_type)
        if source_component_type not in {"float", "int", "uint"}:
            return None

        helper_name = self.bitcast_alias_helper_name(func_name, arg_type)
        vector_info = self.vector_type_info(arg_type)
        if helper_name is not None:
            return self.generate_bitcast_helper_call(
                arg,
                target_type,
                vector_info,
                helper_name,
            )

        if vector_info is not None:
            temp_bindings = []
            lanes = self.vector_argument_lane_expressions(
                arg,
                vector_info,
                temp_bindings,
                target_component_type,
            )
            return self.generate_constructor_call(
                self.map_type(target_type),
                lanes[: vector_info["size"]],
                temp_bindings,
            )

        arg_expr = self.generate_expression(arg)
        return self.normalize_scalar_assignment_value(
            arg,
            arg_expr,
            arg_type,
            target_component_type,
        )

    def bitcast_alias_helper_name(self, func_name, arg_type):
        source_component_type = self.bitcast_argument_component_type(arg_type)
        if func_name == "asfloat":
            return {
                "int": "int_bits_to_float",
                "uint": "uint_bits_to_float",
            }.get(source_component_type)
        if func_name == "asint" and source_component_type == "float":
            return "float_bits_to_int"
        if func_name == "asuint" and source_component_type == "float":
            return "float_bits_to_uint"
        return None

    def generate_bitcast_intrinsic_call(self, func_name, args):
        if len(args or []) != 1:
            return None

        target_component_type = {
            "doubleBitsToLong": "ulong",
            "floatBitsToInt": "int",
            "floatBitsToUint": "uint",
            "intBitsToFloat": "float",
            "longBitsToDouble": "double",
            "uintBitsToFloat": "float",
        }.get(func_name)
        if target_component_type is None:
            return None

        arg = args[0]
        arg_type = self.expression_result_type(arg)
        source_component_type = self.bitcast_argument_component_type(arg_type)
        expected_source_component_type = {
            "doubleBitsToLong": "double",
            "floatBitsToInt": "float",
            "floatBitsToUint": "float",
            "intBitsToFloat": "int",
            "longBitsToDouble": "ulong",
            "uintBitsToFloat": "uint",
        }[func_name]
        if source_component_type != expected_source_component_type:
            return None

        arg_expr = self.generate_expression(arg)
        if func_name == "doubleBitsToLong":
            return f"{arg_expr}.to_bits()"
        if func_name == "longBitsToDouble":
            return f"f64::from_bits({arg_expr})"

        target_type = self.bitcast_value_type(arg_type, target_component_type)
        if target_type is None:
            return None

        return self.generate_bitcast_helper_call(
            arg,
            target_type,
            self.vector_type_info(arg_type),
            self.function_map[func_name],
        )

    def generate_bitcast_helper_call(
        self,
        arg,
        target_type,
        vector_info,
        helper_name,
    ):
        callee = self.rust_imported_math_callable_name(helper_name)
        callable_name = self.rust_callable_name(callee)

        # The standalone Rust contract has scalar and Vec3 bitcast helpers.
        # Emit Vec2/Vec4 bitcasts lane-wise so those widths do not require
        # additional runtime trait implementations.
        if vector_info is None or vector_info["size"] == 3:
            return f"{callable_name}({self.generate_expression(arg)})"

        temp_bindings = []
        lanes = self.bitcast_vector_helper_lane_expressions(
            arg,
            vector_info,
            temp_bindings,
            callable_name,
        )
        return self.generate_constructor_call(
            self.map_type(target_type),
            lanes,
            temp_bindings,
        )

    def bitcast_vector_helper_lane_expressions(
        self,
        arg,
        vector_info,
        temp_bindings,
        callable_name,
    ):
        swizzle_components = self.member_swizzle_components(arg)
        if swizzle_components is not None:
            object_expr = getattr(arg, "object_expr", getattr(arg, "object", None))
            object_value = self.generate_vector_lane_source(
                object_expr,
                self.generate_expression(object_expr),
                temp_bindings,
            )
            object_value = self.rust_member_base_expression(object_value)
            return [
                f"{callable_name}({object_value}.{component})"
                for component in swizzle_components
            ]

        arg_expr = self.generate_expression(arg)
        arg_expr = self.generate_vector_lane_source(arg, arg_expr, temp_bindings)
        arg_expr = self.rust_member_base_expression(arg_expr)
        components = ("x", "y", "z", "w")[: vector_info["size"]]
        return [f"{callable_name}({arg_expr}.{component})" for component in components]

    def bitcast_argument_component_type(self, type_name):
        vector_info = self.vector_type_info(type_name)
        if vector_info is not None:
            return vector_info["component_type"]

        scalar_type = self.normalize_scalar_type(type_name)
        return self.scalar_type_for_type_constructor(scalar_type)

    def rust_imported_math_callable_name(self, func_name):
        helper_module = self.rust_imported_helper_module(func_name)
        if (
            isinstance(func_name, str)
            and helper_module is not None
            and self.rust_function_name_is_shadowed(func_name)
        ):
            return f"{helper_module}::{func_name}"
        return func_name

    def rust_imported_helper_module(self, func_name):
        if func_name in self.rust_imported_math_function_names():
            return "math"
        if self.is_rust_texture_helper_name(func_name):
            return "gpu"
        if str(func_name).startswith(("image_", "buffer_", "subgroup_")):
            return "gpu"
        return None

    def rust_function_name_is_shadowed(self, func_name):
        return (
            func_name in self.local_variable_names
            or func_name in self.user_function_names
        )

    def rust_runtime_function_names(self):
        """Functions of the runtime ``math`` module and the names ``function_map`` renames calls to."""
        names = self.__dict__.get("_rust_runtime_function_names")
        if names is None:
            names = frozenset(self.rust_imported_math_function_names()) | frozenset(
                name for name in self.function_map.values() if isinstance(name, str)
            )
            self.__dict__["_rust_runtime_function_names"] = names
        return names

    def rust_imported_math_function_names(self):
        return {
            "abs",
            "acos",
            "all",
            "any",
            "asin",
            "atan",
            "atan2",
            "ceil",
            "clamp",
            "cos",
            "cosh",
            "cross",
            "degrees",
            "determinant",
            "dfdx",
            "dfdx_coarse",
            "dfdx_fine",
            "dfdy",
            "dfdy_coarse",
            "dfdy_fine",
            "distance",
            "dot",
            "equal",
            "exp",
            "exp2",
            "faceforward",
            "find_lsb",
            "find_msb",
            "floor",
            "float_bits_to_int",
            "float_bits_to_uint",
            "fract",
            "fwidth",
            "fwidth_coarse",
            "fwidth_fine",
            "greater_than",
            "greater_than_equal",
            "inverse",
            "int_bits_to_float",
            "isfinite",
            "isinf",
            "isnan",
            "ldexp",
            "length",
            "lerp",
            "less_than",
            "less_than_equal",
            "log",
            "log2",
            "matrix_comp_mult",
            "max",
            "min",
            "modulo",
            "normalize",
            "not_equal",
            "outer_product",
            "pack_double_2x32",
            "pack_half_2x16",
            "pack_snorm_2x16",
            "pack_snorm_4x8",
            "pack_unorm_2x16",
            "pack_unorm_4x8",
            "pow",
            "radians",
            "reflect",
            "refract",
            "round",
            "round_even",
            "rsqrt",
            "sign",
            "sin",
            "sinh",
            "smoothstep",
            "sqrt",
            "step",
            "tan",
            "tanh",
            "transpose",
            "trunc",
            "uint_bits_to_float",
            "unpack_double_2x32",
            "unpack_half_2x16",
            "unpack_snorm_2x16",
            "unpack_snorm_4x8",
            "unpack_unorm_2x16",
            "unpack_unorm_4x8",
        }

    def generate_cuda_shuffle_intrinsic_call(self, func_name, args):
        if func_name != "__shfl_down_sync" or len(args or []) < 3:
            return None
        value = self.generate_expression(args[1])
        delta = self.generate_expression(args[2])
        return f"subgroup_shuffle_down({value}, {delta})"
