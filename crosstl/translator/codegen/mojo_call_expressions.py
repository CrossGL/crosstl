"""Mojo rendering of CrossGL function-call expressions."""

from ..ast import (
    MemberAccessNode,
)
from ..refusals import (
    MojoRefusal,
)
from .generic_function_utils import (
    generate_numeric_trait_method_call,
    generate_static_generic_numeric_call,
    generic_function_call_name,
    reject_unresolved_generic_function_call,
)
from .lowering_refusals import (
    refuse_unsupported,
)
from .mojo_bit_functions import (
    generate_mojo_bit_function_call,
)
from .mojo_min_max import generate_mojo_min_max
from .mojo_numeric_functions import (
    generate_mojo_numeric_function_call,
)
from .mojo_tables import (
    MOJO_GENERIC_TEXTURE_BUILTINS,
    MOJO_IMAGE_ATOMIC_BUILTINS,
    MOJO_MATH_HELPERS,
    MOJO_MATRIX_TYPES,
    MOJO_SCALAR_DTYPES,
    MOJO_SYNC_BUILTINS,
)
from .shader_clock_refusals import (
    SHADER_CLOCK_BUILTINS,
    refuse_shader_clock_call,
)


class MojoCallExpressionMixin:
    """Function-call member of ``MojoCodeGen.generate_expression``."""

    def generate_mojo_function_call_expression(
        self,
        expr,
        target_type,
        target_context,
    ):
        """Render a function call."""
        func_expr = getattr(expr, "function", None)
        if func_expr is None:
            func_expr = expr.name
        func_name = None
        if isinstance(func_expr, MemberAccessNode):
            member_call = self.generate_member_function_call(func_expr, expr.args)
            if member_call is not None:
                return member_call
            callee = self.generate_expression(func_expr)
        elif hasattr(func_expr, "name") and getattr(func_expr, "name", None):
            # It's an IdentifierNode, extract the name
            func_name = func_expr.name
            callee = func_name
        elif isinstance(func_expr, str):
            func_name = func_expr
            callee = func_expr
        else:
            callee = self.generate_expression(func_expr)

        if func_name in SHADER_CLOCK_BUILTINS and not self.is_user_defined_function(
            func_name
        ):
            return refuse_shader_clock_call("Mojo", func_name)

        numeric_trait_call = generate_numeric_trait_method_call(
            self,
            func_expr,
            expr.args,
        )
        if numeric_trait_call is not None:
            return numeric_trait_call

        static_generic_call = generate_static_generic_numeric_call(self, func_name)
        if static_generic_call is not None:
            return static_generic_call

        if func_name == "lambda":
            lambda_expr = self.generate_lambda_expression(expr.args)
            if lambda_expr is not None:
                return lambda_expr
            # A lambda whose parameters or body the helper materialization cannot type has no Mojo spelling:
            # the CrossGL text of it is not a Mojo expression.
            refuse_unsupported(
                "Mojo",
                "lambda",
                "lambda whose parameter list or body cannot be materialized as a typed Mojo helper function",
                "lambda",
            )

        enum_constructor = self.generate_enum_variant_constructor_call(
            func_name, expr.args, target_type
        )
        if enum_constructor is not None:
            return enum_constructor

        specialized_func_name = generic_function_call_name(
            self,
            func_name,
            expr.args,
        )
        if specialized_func_name is not None:
            func_name = specialized_func_name
            callee = specialized_func_name
        else:
            reject_unresolved_generic_function_call(self, func_name, "Mojo")

        if self.is_user_defined_function(func_name):
            args = self.generate_user_function_call_arguments(func_name, expr.args)
            call_name = (
                self.mojo_identifier(func_name) if func_name is not None else callee
            )
            return f"{call_name}({args})"

        reduction = self.generate_boolean_vector_reduction(func_name, expr.args)
        if reduction is not None:
            return reduction

        buffer_object_call = self.generate_mojo_buffer_object_call(func_name, expr.args)
        if buffer_object_call is not None:
            return buffer_object_call

        ray_tracing_call = self.generate_ray_tracing_call_expression(
            func_name, expr.args
        )
        if ray_tracing_call is not None:
            return ray_tracing_call

        if func_name in {"fract", "frac"}:
            return self.generate_fract_call(expr.args)
        if (
            func_name == "roundEven"
            and len(expr.args) == 1
            and not self.is_user_defined_function(func_name)
        ):
            return self.generate_round_even_call(expr.args[0])
        if func_name == "mod":
            return self.generate_mod_call(expr.args)
        if func_name == "saturate":
            saturate_call = self.generate_saturate_call(expr.args)
            if saturate_call is not None:
                return saturate_call
        if func_name in {"degrees", "radians"}:
            return self.generate_angle_conversion_call(func_name, expr.args)
        if func_name == "mix":
            bool_mix_call = self.generate_bool_mix_call(expr.args)
            if bool_mix_call is not None:
                return bool_mix_call
        if func_name in {"min", "max", "clamp"}:
            min_max_call = generate_mojo_min_max(self, func_name, expr.args)
            if min_max_call is not None:
                return min_max_call
        if func_name == "smoothstep":
            smoothstep_call = self.generate_smoothstep_call(expr.args, target_type)
            if smoothstep_call is not None:
                return smoothstep_call
        if func_name == "atan" and len(expr.args) == 2:
            args = ", ".join(self.generate_expression(arg) for arg in expr.args)
            return f"atan2({args})"
        if func_name == "texture":
            return self.generate_texture_call(expr.args, "sample")
        if func_name == "textureLod":
            return self.generate_texture_call(expr.args, "sample_lod")
        if func_name == "textureGrad":
            return self.generate_texture_call(expr.args, "sample_grad")
        if func_name == "textureSize":
            return self.generate_resource_size_call(expr.args, "texture_size")
        if func_name == "textureQueryLevels":
            return self.generate_resource_query_levels_call(expr.args)
        if func_name == "textureSamples":
            return self.generate_resource_sample_count_call(
                expr.args, "texture_samples"
            )
        if func_name == "imageSamples":
            return self.generate_resource_sample_count_call(expr.args, "image_samples")
        if func_name == "texelFetch":
            return self.generate_texel_fetch_call(expr.args)
        if func_name == "texelFetchOffset":
            return self.generate_texel_fetch_call(expr.args, with_offset=True)
        if func_name == "imageSize":
            return self.generate_resource_size_call(expr.args, "image_size")
        if func_name == "imageLoad":
            return self.generate_image_load_call(expr.args)
        if func_name == "imageStore":
            return self.generate_image_store_call(expr.args)
        if func_name == "buffer_load":
            return self.generate_buffer_load_call(expr.args)
        if func_name == "buffer_store":
            return self.generate_buffer_store_call(expr.args)
        if func_name == "buffer_append":
            return self.generate_buffer_append_call(expr.args)
        if func_name == "buffer_consume":
            return self.generate_buffer_consume_call(expr.args)
        if func_name == "buffer_dimensions":
            return self.generate_buffer_dimensions_call(expr.args)
        if func_name in {"buffer_increment_counter", "buffer_decrement_counter"}:
            raise MojoRefusal(
                f"Unsupported Mojo buffer counter call {func_name}; "
                "structured-buffer counters are not available in the Mojo "
                "placeholder backend"
            )
        reinterpret_name = self.mojo_reinterpret_builtin_name(func_name)
        if reinterpret_name is not None:
            return self.generate_reinterpret_call(reinterpret_name, expr.args)
        if func_name in MOJO_GENERIC_TEXTURE_BUILTINS:
            helper_base, return_kind = MOJO_GENERIC_TEXTURE_BUILTINS[func_name]
            return self.generate_resource_builtin_call(
                expr.args, helper_base, return_kind
            )
        scalar_atomic = self.generate_scalar_atomic_call(func_name, expr.args)
        if scalar_atomic is not None:
            return scalar_atomic
        if func_name in MOJO_IMAGE_ATOMIC_BUILTINS:
            return self.generate_image_atomic_call(
                expr.args, MOJO_IMAGE_ATOMIC_BUILTINS[func_name]
            )
        if (
            func_name in MOJO_SYNC_BUILTINS
            and not expr.args
            and not self.is_user_defined_function(func_name)
        ):
            return self.mojo_sync_call(MOJO_SYNC_BUILTINS[func_name])
        if func_name == "__shfl_down_sync":
            return self.generate_cuda_shuffle_down_call(expr.args)
        if not self.is_user_defined_function(func_name):
            if func_name == "__match_any_sync":
                return self.generate_cuda_match_any_call(expr.args)
            barrier = None if expr.args else self.mojo_subgroup_barrier_call(func_name)
            if barrier is not None:
                return barrier
        bit_function = generate_mojo_bit_function_call(self, func_name, expr.args)
        if bit_function is not None:
            return bit_function
        numeric_function = generate_mojo_numeric_function_call(
            self, func_name, expr.args, target_type
        )
        if numeric_function is not None:
            return numeric_function
        split_builtin = self.generate_mojo_split_builtin_call(func_name, expr.args)
        if split_builtin is not None:
            return split_builtin
        if isinstance(func_name, str) and self.is_array_type_name(func_name):
            return self.generate_mojo_array_constructor_call(func_name, expr.args)

        if func_name in self.function_map:
            func_name = self.function_map[func_name]
        # Otherwise a program function or a Mojo library function, spelled as it is; a built-in function of
        # another language left in the output is refused by generate.
        if func_name in MOJO_MATH_HELPERS:
            self.required_math_helpers.add(func_name)
        if func_name in self.scalar_constructor_map:
            scalar_constructor_name = func_name
            mapped_constructor_name = self.scalar_constructor_map[func_name]
            target_dtype = MOJO_SCALAR_DTYPES.get(
                self.type_name(scalar_constructor_name)
            )
            if target_dtype is not None and len(expr.args) == 1:
                arg = self.generate_scalar_constructor_argument(
                    expr.args[0],
                    target_dtype,
                    f"scalar constructor {scalar_constructor_name} argument 1",
                )
                return f"{mapped_constructor_name}({arg})"
            func_name = mapped_constructor_name

        vector_constructor_name = self.normalize_generic_vector_type_name(func_name)
        if vector_constructor_name in self.vector_constructor_info:
            return self.generate_vector_constructor(vector_constructor_name, expr.args)

        if func_name in MOJO_MATRIX_TYPES:
            return self.generate_matrix_constructor(func_name, expr.args)

        if self.generic_struct_mapped_type(func_name) is not None:
            return self.generate_struct_function_constructor_call(
                expr, func_name, target_context, target_type
            )

        if func_name in self.struct_types:
            return self.generate_struct_function_constructor_call(
                expr, func_name, target_context, target_type
            )

        args = ", ".join(self.generate_expression(arg) for arg in expr.args)
        call_name = self.mojo_identifier(func_name) if func_name is not None else callee
        return f"{call_name}({args})"

    def generate_mojo_array_constructor_call(self, type_name, args):
        """``float[3](a, b, c)`` as a StaticTuple of the element type, one argument per element."""
        element_type, size = self.parse_array_type_name(type_name)
        size = self.format_array_size(size)
        try:
            element_count = int(size) if size is not None else len(args)
        except (TypeError, ValueError):
            element_count = None
        if element_count != len(args):
            raise MojoRefusal(
                f"Mojo array constructor {type_name} has {len(args)} arguments, and Mojo needs one per element"
            )
        elements = [
            self.generate_array_literal_element(
                arg, element_type, f"array constructor element {index}"
            )
            for index, arg in enumerate(args, start=1)
        ]
        array_type = self.array_storage_type(element_type, element_count)
        return f"{array_type}({', '.join(elements)})"
