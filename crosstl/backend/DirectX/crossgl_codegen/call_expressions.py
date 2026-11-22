"""CrossGL rendering of HLSL function-call and member-access expressions: intrinsic, constructor and method calls, resource member access and swizzles."""

from crosstl.translator.arithmetic_conversions import (
    source_integer_shape,
)

from ..byte_address_methods import (
    byte_address_method_expression,
)
from ..DirectxParser import (
    AssignmentNode,
    BinaryOpNode,
    FunctionCallNode,
    MemberAccessNode,
    TernaryOpNode,
    TextureSampleNode,
    UnaryOpNode,
)
from ..function_templates import (
    templated_call_name,
)
from ..helper_lanes import (
    helper_lanes_always_false,
)
from ..select_overload_error import (
    HLSLSelectOverloadError,
)


class CallExpressionMixin:
    """Function-call and member-access members of ``HLSLToCrossGLConverter.generate_expression``."""

    def generate_function_call_expression(
        self,
        expr,
        is_main,
    ):
        """Render a function-call expression: intrinsics, constructors, resource methods and user functions."""
        narrow_texel = not self.read_raw_texel_method_result
        self.read_raw_texel_method_result = False
        struct_method_call = self.struct_method_call_expression(expr, is_main)
        if struct_method_call is not None:
            return struct_method_call
        if isinstance(expr.name, MemberAccessNode):
            obj = self.generate_expression(expr.name.object, is_main)
            member = expr.name.member
            method_member = self.templated_method_base(member)
            rendered_args = [
                self.generate_expression(arg, is_main) for arg in expr.args
            ]
            args = ", ".join(rendered_args)
            resource_type = self.expression_raw_type(expr.name.object)
            resource_base = self.raw_type_base(resource_type)
            byte_address_access = byte_address_method_expression(
                self, member, obj, rendered_args, expr.args, resource_type
            )
            if byte_address_access is not None:
                return byte_address_access
            if (
                method_member == "GatherCmpRed"
                and len(rendered_args) in {7, 8}
                and resource_base in {"Texture2D", "Texture2DArray"}
            ):
                call = self.texture_gather_compare_red_offsets_expression(
                    obj, rendered_args[:7]
                )
                if len(rendered_args) == 8:
                    self.resource_method_diagnostic(
                        method_member, {"dropped_parameters": ["status output"]}
                    )
                return call
            descriptor = self.resource_method_descriptor(
                method_member, len(expr.args), resource_type
            )
            if descriptor:
                descriptor = self.refine_texture_load_status_descriptor(
                    method_member, expr.args, descriptor
                )
                byte_address_atomic = self.byte_address_atomic_method_expression(
                    obj, rendered_args, descriptor, expr.args, is_main
                )
                if byte_address_atomic is not None:
                    call = byte_address_atomic
                elif descriptor.get("fallback_expression") is not None:
                    call = descriptor["fallback_expression"]
                else:
                    method_args = self.resource_method_arguments(
                        obj,
                        method_member,
                        rendered_args,
                        descriptor,
                        expr.args,
                        is_main,
                    )
                    call = f"{descriptor['function']}({', '.join(method_args)})"
                    result_component = descriptor.get("result_component")
                    if result_component:
                        call = f"{call}{result_component}"
                    elif (
                        narrow_texel
                        and descriptor["function"] in self.TEXEL_VALUE_FUNCTIONS
                    ):
                        call += self.texel_narrowing_suffix(resource_type)
                self.resource_method_diagnostic(method_member, descriptor)
                return call
            call = f"{obj}.{member}({args})"
            if (
                self.hlsl_matrix_rows_are_columns()
                and self.raytracing_matrix_call_raw_type(expr) is not None
            ):
                return f"transpose({call})"
            return call

        func_name = (
            expr.name
            if isinstance(expr.name, str)
            else self.generate_expression(expr.name, is_main)
        )
        func_name = self.normalize_hlsl_intrinsic_name(func_name)
        if func_name == "IsHelperLane" and not expr.args:
            if helper_lanes_always_false(
                self, getattr(self, "current_source_function", None)
            ):
                return "false"
            return "gl_HelperInvocation"
        if func_name == "sizeof":
            sizeof_value = self.generate_sizeof_expression(expr.args)
            if sizeof_value is not None:
                return sizeof_value
        if func_name == "CheckAccessFullyMapped":
            return self.refuse_import(
                "DirectX tiled-resource status check CheckAccessFullyMapped: "
                "CrossGL has no residency status, so the mapped result is unknown",
                "true",
            )
        interlocked_image_atomic = self.interlocked_storage_image_atomic_expression(
            func_name, expr.args, is_main
        )
        if interlocked_image_atomic is not None:
            return interlocked_image_atomic
        interlocked_buffer_atomic = self.interlocked_typed_buffer_atomic_expression(
            func_name, expr.args, is_main
        )
        if interlocked_buffer_atomic is not None:
            return interlocked_buffer_atomic
        if func_name in self.interlocked_map:
            rendered_args = [
                self.generate_without_storage_index_lowering(arg, is_main)
                for arg in expr.args
            ]
        else:
            rendered_args = [
                self.generate_expression(arg, is_main, expected_type=param_type)
                for arg, param_type in zip(
                    expr.args, self.source_parameter_types(func_name, expr.args)
                )
            ]
        legacy_texture_call = self.legacy_texture_function_call(
            func_name, expr.args, rendered_args, is_main
        )
        if legacy_texture_call is not None:
            return legacy_texture_call
        bitcast_call = self.bitcast_intrinsic_expression(
            func_name, expr.args, rendered_args
        )
        if bitcast_call is not None:
            return bitcast_call
        if func_name == "select" and len(rendered_args) == 3:
            if func_name in self.source_function_names:
                raise HLSLSelectOverloadError()
            condition, true_value, false_value = rendered_args
            return f"select({false_value}, {true_value}, {condition})"
        if func_name not in self.source_function_names:
            rendered_args = self.truncate_intrinsic_vector_arguments(
                func_name, expr.args, rendered_args
            )
        args = ", ".join(rendered_args)
        matrix_constructor = self.matrix_constructor_expression(
            func_name, expr.args, is_main
        )
        if matrix_constructor is not None:
            return matrix_constructor
        if func_name == "mul" and len(expr.args) == 2:
            return self.mul_call_expression(expr.args, rendered_args)
        if func_name == "mad" and len(expr.args) == 3:
            left = self.maybe_parenthesize(expr.args[0], rendered_args[0])
            right = self.maybe_parenthesize(expr.args[1], rendered_args[1])
            addend = self.maybe_parenthesize(expr.args[2], rendered_args[2])
            return f"(({left} * {right}) + {addend})"
        if func_name == "dst" and len(expr.args) == 2:
            src0 = self.maybe_parenthesize(expr.args[0], rendered_args[0])
            src1 = self.maybe_parenthesize(expr.args[1], rendered_args[1])
            return f"vec4(1.0, {src0}.y * {src1}.y, {src0}.z, {src1}.w)"
        if func_name == "saturate":
            if expr.args:
                return f"clamp({self.generate_expression(expr.args[0], is_main)}, 0.0, 1.0)"
            return "clamp(0.0, 0.0, 1.0)"
        if func_name == "rcp" and len(expr.args) == 1:
            value = self.maybe_parenthesize(expr.args[0], rendered_args[0])
            return f"(1.0 / {value})"
        if (
            func_name in self.REVOLUTION_REDUCED_INTRINSICS
            and len(expr.args) == 1
            and func_name not in self.source_function_names
        ):
            return self.trigonometric_call(func_name, expr.args[0], rendered_args[0])
        precise_call = self.precise_intrinsic_call(func_name, expr.args, rendered_args)
        if precise_call is not None:
            return precise_call
        if func_name != "round" or not self.source_function_accepts_argument_count(
            func_name, len(expr.args)
        ):
            func_name = self.crossgl_intrinsic_name(func_name)
        if func_name in self.interlocked_map:
            func_name = self.interlocked_map[func_name]
        integer_shape = source_integer_shape(self.resolve_integer_alias(func_name))
        if (
            integer_shape is not None
            and integer_shape[1] == 64
            and func_name not in self.source_function_names
        ):
            func_name = self.map_type(self.resolve_integer_alias(func_name))
        if isinstance(func_name, str) and "<" in func_name:
            func_name = templated_call_name(self, func_name)
        func_name = self.render_function_identifier(func_name)
        call = f"{func_name}({args})"
        if (
            self.hlsl_matrix_rows_are_columns()
            and self.raytracing_matrix_call_raw_type(expr) is not None
        ):
            return f"transpose({call})"
        return call

    def generate_member_access_expression(
        self,
        expr,
        is_main,
        expected_type,
    ):
        """Render a member-access expression: struct members, swizzles and resource members."""
        alias = self.stage_struct_parameter_member_alias(expr)
        if alias is not None:
            return self.render_identifier(alias)
        matrix_swizzle = self.generate_matrix_swizzle(expr, is_main)
        if matrix_swizzle is not None:
            return matrix_swizzle
        obj = None
        if self.texel_swizzle_reads_element_components(expr.object, expr.member):
            # Swizzle the four-component texel directly; the selected
            # components all exist in the declared element type.
            obj = self.generate_texel_access(expr.object, is_main, narrow=False)
            if obj is None and isinstance(
                expr.object, (FunctionCallNode, TextureSampleNode)
            ):
                self.read_raw_texel_method_result = True
        if obj is None:
            obj = self.generate_expression(expr.object, is_main)
        self.read_raw_texel_method_result = False
        scalar_splat = self.scalar_splat_swizzle_expression(expr, obj, expected_type)
        if scalar_splat is not None:
            return scalar_splat
        matrix_member = self.matrix_member_expression(expr, is_main)
        if matrix_member is not None:
            return matrix_member
        if isinstance(expr.object, (int, float)):
            obj = f"({obj})"
        elif isinstance(
            expr.object, (BinaryOpNode, TernaryOpNode, UnaryOpNode, AssignmentNode)
        ):
            # Member access binds tighter than any operator.
            obj = f"({obj})"
        return f"{obj}.{expr.member}"
