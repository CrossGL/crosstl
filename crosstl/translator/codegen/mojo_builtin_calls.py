"""Built-in calls in the Mojo generator: texture operations, synchronization, atomics, wave and ray operations, vector and matrix constructors, and math built-ins."""

import re

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    MemberAccessNode,
    UnaryOpNode,
    VariableNode,
)
from ..refusals import MojoRefusal
from .lowering_refusals import refuse_unsupported
from .mojo_exact_lookups import MOJO_RAY_TRACING_INTRINSICS
from .mojo_tables import (
    MOJO_ATOMIC_OP_ALIASES,
    MOJO_DTYPE_INFO,
    MOJO_FLOAT_DTYPES,
    MOJO_GENERIC_TEXTURE_BUILTINS,
    MOJO_INTEGER_DTYPES,
    MOJO_MAPPED_INTEGER_TYPES,
    MOJO_MATRIX_TYPES,
    MOJO_SCALAR_ATOMIC_OPS,
    MOJO_SCALAR_ATOMIC_TYPES,
    MOJO_SCALAR_DTYPES,
    MOJO_SYNC_BUILTINS,
    MOJO_TYPED_BUFFER_LOAD_RESOURCE_TYPES,
    MOJO_WAVE_BOOL_RESULT_OPS,
    MOJO_WAVE_OP_ARITIES,
    MOJO_WAVE_UINT_RESULT_OPS,
    MOJO_WAVE_UVEC4_RESULT_OPS,
    ROUND_EVEN_FLOAT_DTYPES,
)


class MojoBuiltinCallMixin:
    """Built-in-call members of ``MojoCodeGen``."""

    def generate_texture_operation_call(
        self, operation, args, argument_count, fetch_args=None
    ):
        resource_args = list(fetch_args if fetch_args is not None else args)

        if operation in {"Sample", "sample", "texture"}:
            if argument_count >= 2:
                return self.generate_resource_builtin_call(
                    args, "sample_offset", "vec4"
                )
            return self.generate_texture_call(args, "sample")
        if operation in {"SampleLevel", "SampleLOD", "sample_lod", "textureLod"}:
            if argument_count >= 3:
                return self.generate_resource_builtin_call(
                    args, "sample_lod_offset", "vec4"
                )
            return self.generate_texture_call(args, "sample_lod")
        if operation in {"SampleGrad", "sample_grad", "textureGrad"}:
            if argument_count >= 4:
                return self.generate_resource_builtin_call(
                    args, "sample_grad_offset", "vec4"
                )
            return self.generate_texture_call(args, "sample_grad")
        if operation in {"SampleBias", "sample_bias"}:
            if argument_count >= 3:
                return self.generate_resource_builtin_call(
                    args, "sample_bias_offset", "vec4"
                )
            return self.generate_resource_builtin_call(args, "sample_bias", "vec4")

        resource_builtin = {
            "SampleCmp": (
                (
                    "texture_compare_offset"
                    if argument_count >= 3
                    else "texture_compare"
                ),
                "float",
            ),
            "SampleCmpLevel": (
                (
                    "texture_compare_lod_offset"
                    if argument_count >= 4
                    else "texture_compare_lod"
                ),
                "float",
            ),
            "SampleCmpLevelZero": (
                (
                    "texture_compare_offset"
                    if argument_count >= 3
                    else "texture_compare"
                ),
                "float",
            ),
            "SampleCmpGrad": (
                (
                    "texture_compare_grad_offset"
                    if argument_count >= 5
                    else "texture_compare_grad"
                ),
                "float",
            ),
            "Gather": (
                ("texture_gather_offset" if argument_count >= 2 else "texture_gather"),
                "vec4",
            ),
            "GatherRed": (
                ("texture_gather_offset" if argument_count >= 2 else "texture_gather"),
                "vec4",
            ),
            "GatherGreen": (
                ("texture_gather_offset" if argument_count >= 2 else "texture_gather"),
                "vec4",
            ),
            "GatherBlue": (
                ("texture_gather_offset" if argument_count >= 2 else "texture_gather"),
                "vec4",
            ),
            "GatherAlpha": (
                ("texture_gather_offset" if argument_count >= 2 else "texture_gather"),
                "vec4",
            ),
            "GatherCmp": (
                (
                    "texture_gather_compare_offset"
                    if argument_count >= 3
                    else "texture_gather_compare"
                ),
                "vec4",
            ),
            "GatherCmpRed": (
                (
                    "texture_gather_compare_offset"
                    if argument_count >= 3
                    else "texture_gather_compare"
                ),
                "vec4",
            ),
            "GatherCmpGreen": (
                (
                    "texture_gather_compare_offset"
                    if argument_count >= 3
                    else "texture_gather_compare"
                ),
                "vec4",
            ),
            "GatherCmpBlue": (
                (
                    "texture_gather_compare_offset"
                    if argument_count >= 3
                    else "texture_gather_compare"
                ),
                "vec4",
            ),
            "GatherCmpAlpha": (
                (
                    "texture_gather_compare_offset"
                    if argument_count >= 3
                    else "texture_gather_compare"
                ),
                "vec4",
            ),
        }.get(operation)
        if resource_builtin is not None:
            helper_base, return_kind = resource_builtin
            return self.generate_resource_builtin_call(args, helper_base, return_kind)

        if operation in MOJO_GENERIC_TEXTURE_BUILTINS:
            helper_base, return_kind = MOJO_GENERIC_TEXTURE_BUILTINS[operation]
            return self.generate_resource_builtin_call(args, helper_base, return_kind)
        atomic_helper = MOJO_ATOMIC_OP_ALIASES.get(operation)
        if atomic_helper is not None and resource_args:
            resource_type = self.map_type(self.expression_result_type(resource_args[0]))
            if self.is_image_resource_type(resource_type):
                return self.generate_image_atomic_call(resource_args, atomic_helper)
        if operation == "texelFetchOffset":
            return self.generate_texel_fetch_call(resource_args, with_offset=True)
        if operation in {"Load", "texelFetch"}:
            fetch_args = list(resource_args)
            if fetch_args:
                resource_type = self.map_type(
                    self.expression_result_type(fetch_args[0])
                )
                if self.is_image_resource_type(resource_type):
                    return self.generate_image_load_call(fetch_args)
            if len(fetch_args) == 2:
                fetch_args.append(0)
            return self.generate_texel_fetch_call(fetch_args)
        if operation in {"Store", "imageStore"}:
            return self.generate_image_store_call(resource_args)
        if operation in {"GetDimensions", "textureSize"}:
            helper_name = "texture_size"
            if resource_args:
                resource_type = self.map_type(
                    self.expression_result_type(resource_args[0])
                )
                if self.is_image_resource_type(resource_type):
                    helper_name = "image_size"
            return self.generate_resource_size_call(resource_args, helper_name)
        if operation == "textureQueryLevels":
            return self.generate_resource_query_levels_call(resource_args)
        if operation == "textureSamples":
            return self.generate_resource_sample_count_call(
                resource_args, "texture_samples"
            )
        if operation == "imageSamples":
            return self.generate_resource_sample_count_call(
                resource_args, "image_samples"
            )

        raise MojoRefusal(f"Unsupported Mojo texture operation {operation}")

    def generate_texture_op_node(self, expr):
        operation = getattr(expr, "operation", "")
        args = self.texture_op_args(expr)
        return self.generate_texture_operation_call(
            operation,
            args,
            len(expr.arguments),
            fetch_args=self.texture_op_args(expr, include_sampler=False),
        )

    def generate_sync_node(self, stmt, indent):
        sync_type = getattr(stmt, "sync_type", "")
        if getattr(stmt, "arguments", None):
            raise MojoRefusal(
                "Invalid Mojo synchronization operation "
                f"{sync_type}; arguments are not supported"
            )

        helper_name = MOJO_SYNC_BUILTINS.get(sync_type)
        if helper_name is None:
            raise MojoRefusal(f"Invalid Mojo synchronization operation {sync_type}")
        indent_str = "    " * indent
        return f"{indent_str}{self.mojo_sync_call(helper_name)}\n"

    def mojo_sync_call(self, mojo_name):
        """The Mojo GPU synchronization call ``mojo_name()``, noting the import threadfence needs."""
        if mojo_name == "threadfence":
            self.requires_threadfence_import = True
        return f"{mojo_name}()"

    def generate_atomic_op_node(self, expr):
        operation = getattr(expr, "operation", "")
        helper_base = MOJO_ATOMIC_OP_ALIASES.get(operation)
        if helper_base is None:
            raise MojoRefusal(f"Unsupported Mojo atomic operation {operation}")
        scalar_atomic = self.generate_scalar_atomic_call(
            operation, [expr.target, *getattr(expr, "arguments", [])]
        )
        if scalar_atomic is not None:
            return scalar_atomic
        return self.generate_image_atomic_call(
            [expr.target, *getattr(expr, "arguments", [])], helper_base
        )

    def generate_scalar_atomic_call(self, func_name, args):
        operation = MOJO_SCALAR_ATOMIC_OPS.get(func_name)
        if operation is None or not args:
            return None

        target = args[0]
        target_type = self.type_name(self.expression_result_type(target))
        if target_type is None:
            return None
        if self.buffer_resource_info(target_type) is not None:
            return None
        mapped_target_type = self.map_type(target_type)
        if self.is_image_resource_type(mapped_target_type):
            return None
        if self.is_texture_resource_type(mapped_target_type):
            return None
        if target_type not in MOJO_SCALAR_ATOMIC_TYPES:
            raise MojoRefusal(
                f"Unsupported {func_name} for Mojo codegen; "
                f"scalar integer target required: {target_type}"
            )
        if not self.is_assignable_atomic_target(target):
            raise MojoRefusal(
                f"Unsupported {func_name} for Mojo codegen; assignable target required"
            )
        expected_args = 3 if operation == "compare_exchange" else 2
        if len(args) != expected_args:
            raise MojoRefusal(
                f"Unsupported {func_name} for Mojo codegen; expected "
                f"{expected_args} argument(s), got {len(args)}"
            )
        if not self.expression_prelude_stack:
            raise MojoRefusal(
                f"Unsupported {func_name} for Mojo codegen; "
                "scalar atomic lowering requires statement context"
            )

        self.validate_resource_read_write_access(target, func_name)
        target_expr = self.generate_expression(target)
        temp_name = self.next_scalar_atomic_temp_name()
        mojo_type = self.map_type(target_type)
        context = self.expression_prelude_stack[-1]
        indent_str = "    " * context["indent"]
        context["lines"].append(
            f"{indent_str}var {temp_name}: {mojo_type} = {target_expr}\n"
        )

        if operation == "compare_exchange":
            compare_expr = self.generate_expression(
                args[1], target_type, f"{func_name} compare"
            )
            value_expr = self.generate_expression(
                args[2], target_type, f"{func_name} value"
            )
            context["lines"].append(f"{indent_str}if {temp_name} == {compare_expr}:\n")
            context["lines"].append(f"{indent_str}    {target_expr} = {value_expr}\n")
            return temp_name

        value_expr = self.generate_expression(
            args[1], target_type, f"{func_name} value"
        )
        if operation == "exchange":
            update_expr = value_expr
        elif operation == "add":
            update_expr = f"({temp_name} + {value_expr})"
        elif operation == "min":
            update_expr = f"min({temp_name}, {value_expr})"
        elif operation == "max":
            update_expr = f"max({temp_name}, {value_expr})"
        elif operation == "and":
            update_expr = f"({temp_name} & {value_expr})"
        elif operation == "or":
            update_expr = f"({temp_name} | {value_expr})"
        elif operation == "xor":
            update_expr = f"({temp_name} ^ {value_expr})"
        else:
            raise MojoRefusal(f"Unsupported Mojo scalar atomic operation {func_name}")

        context["lines"].append(f"{indent_str}{target_expr} = {update_expr}\n")
        return temp_name

    def is_assignable_atomic_target(self, target):
        return isinstance(
            target, (ArrayAccessNode, MemberAccessNode, VariableNode)
        ) or (hasattr(target, "__class__") and "Identifier" in str(target.__class__))

    def builtin_variable_result_type(self, name):
        if name in {
            "SV_DispatchThreadID",
            "SV_GroupID",
            "SV_GroupThreadID",
            "gl_GlobalInvocationID",
            "gl_LocalInvocationID",
            "gl_WorkGroupID",
            "gl_NumWorkGroups",
            "gl_WorkGroupSize",
        }:
            return "uvec3"
        if name in {"SV_GroupIndex", "SV_InstanceID", "SV_VertexID"}:
            return "uint"
        if name in {
            "gl_LocalInvocationIndex",
            "gl_NumSubgroups",
            "gl_SubgroupID",
            "gl_SubgroupInvocationID",
            "gl_SubgroupSize",
        }:
            return "uint"
        if name in {"gl_InstanceID", "gl_VertexID"}:
            return "int"
        if name in {"SV_IsFrontFace", "gl_FrontFacing"}:
            return "bool"
        if name in {"SV_Depth", "gl_FragDepth"}:
            return "float"
        return self.variable_types.get(self.map_semantic(name), "vec4")

    def generate_wave_op(self, expr):
        operation = getattr(expr, "operation", "wave intrinsic")
        args = list(getattr(expr, "arguments", []) or [])
        expected_count = MOJO_WAVE_OP_ARITIES.get(operation)
        if expected_count is None:
            raise MojoRefusal(f"Unsupported Mojo wave intrinsic {operation}")
        if len(args) != expected_count:
            raise MojoRefusal(
                f"Mojo wave intrinsic {operation} requires {expected_count} "
                f"argument{'s' if expected_count != 1 else ''}, got {len(args)}"
            )

        raise MojoRefusal(
            f"Mojo wave intrinsic {operation}: the Mojo generator has no lowering to "
            "Mojo's warp operations, and a single-lane stand-in computes another "
            "result whenever a wave has more than one active lane"
        )

    def generate_cuda_shuffle_down_call(self, args):
        if len(args) != 3:
            raise MojoRefusal(
                f"Mojo __shfl_down_sync requires 3 arguments, got {len(args)}"
            )
        raise MojoRefusal(
            "Mojo __shfl_down_sync: the Mojo generator has no lowering to Mojo's warp "
            "shuffles, and returning the lane's own value computes another result"
        )

    def mojo_wave_result_type(self, operation, args):
        if operation in MOJO_WAVE_UINT_RESULT_OPS:
            return "uint"
        if operation in MOJO_WAVE_BOOL_RESULT_OPS:
            return "bool"
        if operation in MOJO_WAVE_UVEC4_RESULT_OPS:
            return "uvec4"
        if args:
            return self.expression_result_type(args[0]) or "uint"
        return "uint"

    def generate_ray_tracing_op(self, expr):
        return self.generate_ray_tracing_call_expression(
            getattr(expr, "operation", "ray tracing intrinsic"),
            getattr(expr, "arguments", []),
        )

    def generate_ray_tracing_call_expression(self, operation, args):
        intrinsic = MOJO_RAY_TRACING_INTRINSICS.get(operation)
        if intrinsic is None:
            return None

        helper_name, argument_counts = intrinsic
        if len(args or []) not in argument_counts:
            counts = " or ".join(str(count) for count in argument_counts)
            return refuse_unsupported(
                "Mojo",
                f"{operation} with {len(args or [])} arguments",
                f"the Mojo runtime helper {helper_name} takes {counts}",
                f"{helper_name}()",
            )
        self.required_ray_helpers.add(helper_name)
        generated_args = [self.generate_expression(arg) for arg in args or []]
        if operation == "TraceRay" and len(generated_args) == 11:
            ray_desc = (
                "RayDesc("
                f"{generated_args[6]}, {generated_args[7]}, "
                f"{generated_args[8]}, {generated_args[9]}"
                ")"
            )
            generated_args = generated_args[:6] + [ray_desc, generated_args[10]]
        elif operation == "ReportHit" and len(generated_args) == 2:
            generated_args.append("False")
        return f"{helper_name}({', '.join(generated_args)})"

    def generate_ray_query_op(self, expr):
        operation = getattr(expr, "operation", "ray query method")
        helper_name = f"ray_query_{self.mojo_snake_case_name(operation)}"
        self.required_ray_helpers.add(helper_name)
        args = [self.generate_expression(getattr(expr, "query_expr", None))]
        args.extend(
            self.generate_expression(arg)
            for arg in getattr(expr, "arguments", []) or []
        )
        return f"{helper_name}({', '.join(args)})"

    def mojo_snake_case_name(self, name):
        name = str(name)
        name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
        return name.replace("__", "_").strip("_").lower()

    def generate_vector_constructor(self, func_name, args):
        if len(args) == 1:
            self.validate_image_result_target_shape(
                args[0], func_name, f"vector constructor {func_name} argument 1"
            )

        helper_call = self.generate_constructor_helper_call(func_name, args)
        if helper_call is not None:
            return helper_call

        dtype, source_width, storage_width, pad_literal = self.vector_constructor_info[
            func_name
        ]
        mojo_type = f"SIMD[{dtype}, {storage_width}]"
        emitted_args = []
        uses_scalar_splat_constructor = False

        if len(args) == 1:
            arg = args[0]
            arg_components = self.vector_components_for_expression(arg, dtype)
            if arg_components is not None:
                emitted_args.extend(arg_components[:source_width])
            elif source_width == 3:
                arg_expr = self.generate_constructor_scalar_expression(
                    arg, dtype, f"vector constructor {func_name} argument 1"
                )
                if self.is_duplicate_sensitive_expression(arg):
                    helper_name = self.vec3_splat_helper_name(dtype)
                    self.required_splat_helpers.add(dtype)
                    return f"{helper_name}({arg_expr})"
                emitted_args.extend([arg_expr] * source_width)
            else:
                emitted_args.append(
                    self.generate_constructor_scalar_expression(
                        arg, dtype, f"vector constructor {func_name} argument 1"
                    )
                )
                uses_scalar_splat_constructor = True
        else:
            for index, arg in enumerate(args, start=1):
                arg_components = self.vector_components_for_expression(arg, dtype)
                if arg_components is not None:
                    emitted_args.extend(arg_components)
                else:
                    emitted_args.append(
                        self.generate_constructor_scalar_expression(
                            arg,
                            dtype,
                            f"vector constructor {func_name} argument {index}",
                        )
                    )

        if len(emitted_args) > source_width:
            emitted_args = emitted_args[:source_width]
        elif len(emitted_args) < source_width and not uses_scalar_splat_constructor:
            emitted_args.extend(
                MOJO_DTYPE_INFO[dtype][2]
                for _ in range(source_width - len(emitted_args))
            )

        if source_width == 3 and len(emitted_args) == 3:
            emitted_args.append(pad_literal)

        if uses_scalar_splat_constructor and len(emitted_args) == 1:
            return self.scalar_splat_expression(dtype, storage_width, emitted_args[0])

        return f"{mojo_type}({', '.join(emitted_args)})"

    def scalar_splat_expression(self, dtype, storage_width, scalar_expr):
        """Splat ``scalar_expr`` into every lane of a ``SIMD[dtype, width]``.

        Mojo's one-argument SIMD constructor only splats a ``Scalar[dtype]``.
        ``Bool`` is not ``Scalar[DType.bool]``, so a boolean splat must use the
        explicit ``fill`` keyword.
        """
        if dtype == "DType.bool":
            return f"SIMD[{dtype}, {storage_width}](fill={scalar_expr})"
        return f"SIMD[{dtype}, {storage_width}]({scalar_expr})"

    def generate_matrix_constructor(self, func_name, args):
        dtype, columns, rows = MOJO_MATRIX_TYPES[func_name]
        matrix_key = (dtype, columns, rows)
        self.required_matrix_types.add(matrix_key)

        if len(args) == 1:
            source_matrix_info = self.matrix_type_info(
                self.expression_result_type(args[0])
            )
            if source_matrix_info is not None:
                source_dtype, source_columns, source_rows = source_matrix_info
                if (
                    source_dtype == dtype
                    and source_columns >= columns
                    and source_rows >= rows
                ):
                    source = self.generate_expression(args[0])
                    if source_matrix_info == matrix_key:
                        return source
                    matrix_type = self.matrix_type_name(dtype, columns, rows)
                    storage_rows = self.matrix_storage_rows(rows)
                    column_type = f"SIMD[{dtype}, {storage_rows}]"
                    pad_literal = self.matrix_zero_literal(dtype)
                    column_args = []
                    for column in range(columns):
                        components = [
                            f"{source}.c{column}[{row}]" for row in range(rows)
                        ]
                        if rows == 3:
                            components.append(pad_literal)
                        column_args.append(f"{column_type}({', '.join(components)})")
                    return f"{matrix_type}({', '.join(column_args)})"

        helper_call = self.generate_matrix_constructor_helper_call(
            dtype, columns, rows, args
        )
        if helper_call is not None:
            return helper_call

        if len(args) == 1 and self.is_duplicate_sensitive_expression(args[0]):
            arg_type = self.constructor_scalar_expression_target_type(args[0], dtype)
            if self.vector_type_info(arg_type) is None:
                scalar = self.generate_constructor_scalar_expression(
                    args[0], dtype, f"matrix constructor {func_name} argument 1"
                )
                key = (dtype, columns, rows)
                self.required_matrix_diagonal_helpers.add(key)
                return f"{self.matrix_diagonal_helper_name(*key)}({scalar})"

        component_count = columns * rows
        components = []
        for index, arg in enumerate(args, start=1):
            arg_components = self.vector_components_for_expression(arg, dtype)
            if arg_components is not None:
                components.extend(arg_components)
            else:
                components.append(
                    self.generate_constructor_scalar_expression(
                        arg, dtype, f"matrix constructor {func_name} argument {index}"
                    )
                )

        if len(args) == 1 and len(components) == 1:
            scalar = components[0]
            components = [
                scalar if column == row else self.matrix_zero_literal(dtype)
                for column in range(columns)
                for row in range(rows)
            ]

        if len(components) > component_count:
            components = components[:component_count]
        elif len(components) < component_count:
            components.extend(
                self.matrix_zero_literal(dtype)
                for _ in range(component_count - len(components))
            )

        matrix_type = self.matrix_type_name(dtype, columns, rows)
        column_args = []
        storage_rows = self.matrix_storage_rows(rows)
        pad_literal = self.matrix_zero_literal(dtype)
        for column in range(columns):
            start = column * rows
            column_components = components[start : start + rows]
            if rows == 3:
                column_components.append(pad_literal)
            column_type = f"SIMD[{dtype}, {storage_rows}]"
            column_args.append(f"{column_type}({', '.join(column_components)})")

        return f"{matrix_type}({', '.join(column_args)})"

    def generate_fract_call(self, args):
        if not args:
            return "fract()"

        arg = args[0]
        arg_expr = self.generate_expression(arg)
        arg_type = self.expression_result_type(arg)
        supported_float_dtypes = MOJO_FLOAT_DTYPES | {"DType.float16"}
        vector_info = self.vector_type_info(arg_type)
        if vector_info is not None:
            dtype, source_width, storage_width, _ = vector_info
            if dtype in supported_float_dtypes:
                self.required_fract_helpers.add(
                    ("vector", dtype, source_width, storage_width)
                )
                helper_name = self.fract_vector_helper_name(
                    dtype, source_width, storage_width
                )
                return f"{helper_name}({arg_expr})"

        dtype = self.expression_mojo_dtype(arg) or "DType.float32"
        if dtype not in supported_float_dtypes:
            dtype = "DType.float32"
        self.required_fract_helpers.add(("scalar", dtype, 1, 1))
        return f"{self.fract_scalar_helper_name(dtype)}({arg_expr})"

    def round_even_operand_type(self, arg):
        """Source type of a ``round_even`` operand, including typed buffer elements."""
        arg_type = self.expression_result_type(arg)
        if arg_type is None and isinstance(arg, ArrayAccessNode):
            info = self.buffer_resource_info(self.expression_result_type(arg.array))
            if info is not None and info[0] in MOJO_TYPED_BUFFER_LOAD_RESOURCE_TYPES:
                arg_type = info[1]
        return arg_type

    def round_even_operand_dtypes(self, arg):
        """Return ``(source dtype, dtype the helper rounds in, source type)``.

        The helper has overloads for float16, float32 and float64 elements;
        integer operands round as float32, as HLSL ``round`` converts them.
        """
        arg_type = self.round_even_operand_type(arg)
        vector_info = self.vector_type_info(arg_type)
        if vector_info is not None:
            source_dtype = vector_info[0]
        else:
            source_dtype = MOJO_SCALAR_DTYPES.get(self.type_name(arg_type))
        if source_dtype is None or source_dtype in ROUND_EVEN_FLOAT_DTYPES:
            return source_dtype, source_dtype, arg_type
        return source_dtype, "DType.float32", arg_type

    def is_scalar_literal_operand(self, arg):
        """A numeric literal, optionally under unary ``+`` or ``-``."""
        while isinstance(arg, UnaryOpNode) and self.map_operator(arg.op) in {
            "+",
            "-",
        }:
            arg = arg.operand
        return self.is_literal_expression(arg) and not isinstance(arg, ArrayLiteralNode)

    def generate_round_even_call(self, arg):
        self.required_math_helpers.add("round_even")
        arg_expr = self.generate_expression(arg)
        source_dtype, target_dtype, _ = self.round_even_operand_dtypes(arg)
        if self.is_scalar_literal_operand(arg):
            # A literal converts to every overload, and an integer literal has no
            # cast method, so it is spelled in the scalar type it rounds in.
            scalar_type = ROUND_EVEN_FLOAT_DTYPES.get(target_dtype, "Float32")
            return f"round_even({scalar_type}({arg_expr}))"
        if source_dtype != target_dtype:
            arg_expr = f"({arg_expr}).cast[{target_dtype}]()"
        return f"round_even({arg_expr})"

    def round_even_result_type(self, arg):
        source_dtype, target_dtype, arg_type = self.round_even_operand_dtypes(arg)
        if source_dtype == target_dtype:
            return arg_type or "float"
        vector_info = self.vector_type_info(arg_type)
        if vector_info is not None:
            return self.vector_type_name_for_dtype_width(target_dtype, vector_info[1])
        return "float"

    def generate_mod_call(self, args):
        generated_args = [self.generate_expression(arg) for arg in args]
        if len(generated_args) != 2:
            self.required_math_helpers.add("fmod")
            return f"fmod({', '.join(generated_args)})"

        left_type = self.expression_result_type(args[0])
        right_type = self.expression_result_type(args[1])
        if self.is_scalar_integer_type(left_type) and (
            right_type is None or self.is_scalar_integer_type(right_type)
        ):
            return f"({generated_args[0]} % {generated_args[1]})"

        left_info = self.vector_type_info(left_type)
        right_info = self.vector_type_info(right_type)
        if left_info is not None or right_info is not None:
            mod_call = self.generate_vector_mod_call(
                generated_args, left_info, right_info
            )
            if mod_call is not None:
                return mod_call

        left_dtype = self.expression_mojo_dtype(args[0])
        right_dtype = self.expression_mojo_dtype(args[1])
        dtype = left_dtype if left_dtype in MOJO_FLOAT_DTYPES else right_dtype
        if dtype in MOJO_FLOAT_DTYPES:
            self.required_mod_helpers.add((dtype, "ss", 1, 1))
            helper_name = self.mod_helper_name(dtype, "ss", 1, 1)
            return f"{helper_name}({generated_args[0]}, {generated_args[1]})"

        self.required_math_helpers.add("fmod")
        return f"fmod({generated_args[0]}, {generated_args[1]})"

    def generate_vector_mod_call(self, generated_args, left_info, right_info):
        if left_info is not None and right_info is not None:
            if left_info[:3] != right_info[:3]:
                return None
            dtype, source_width, storage_width, _ = left_info
            kind = "vv"
        elif left_info is not None:
            dtype, source_width, storage_width, _ = left_info
            kind = "vs"
        else:
            dtype, source_width, storage_width, _ = right_info
            kind = "sv"

        if dtype not in MOJO_FLOAT_DTYPES:
            return None

        self.required_mod_helpers.add((dtype, "ss", 1, 1))
        self.required_mod_helpers.add((dtype, kind, source_width, storage_width))
        helper_name = self.mod_helper_name(dtype, kind, source_width, storage_width)
        return f"{helper_name}({generated_args[0]}, {generated_args[1]})"

    def generate_saturate_call(self, args):
        if len(args) != 1:
            return None

        supported_float_dtypes = MOJO_FLOAT_DTYPES | {"DType.float16"}
        arg_type = self.expression_result_type(args[0])
        vector_info = self.vector_type_info(arg_type)
        if vector_info is not None:
            dtype, source_width, storage_width, _ = vector_info
            if dtype in MOJO_INTEGER_DTYPES:
                # saturate keeps its argument's type (semantics.hub_types), so an integer vector clamps to 0 and
                # 1 per component, through SIMD's clamp method, which no generated free helper shadows.
                return f"({self.generate_expression(args[0])}).clamp(0, 1)"
            if dtype not in supported_float_dtypes:
                return None
            arg_expr = self.generate_expression(args[0])
            self.required_saturate_helpers.add(
                ("vector", dtype, source_width, storage_width)
            )
            helper_name = self.saturate_vector_helper_name(
                dtype, source_width, storage_width
            )
            return f"{helper_name}({arg_expr})"

        matrix_info = self.matrix_type_info(arg_type)
        if matrix_info is not None:
            dtype, columns, rows = matrix_info
            if dtype in MOJO_INTEGER_DTYPES:
                return refuse_unsupported(
                    "Mojo",
                    "function",
                    "saturate of an integer matrix: the Mojo matrix runtime has no integer clamp",
                    "saturate",
                )
            if dtype not in supported_float_dtypes:
                return None
            arg_expr = self.generate_expression(args[0])
            self.required_matrix_types.add((dtype, columns, rows))
            self.required_saturate_helpers.add(("matrix", dtype, columns, rows))
            helper_name = self.saturate_matrix_helper_name(dtype, columns, rows)
            return f"{helper_name}({arg_expr})"

        arg_dtype = self.expression_mojo_dtype(args[0])
        if arg_dtype in MOJO_INTEGER_DTYPES:
            # An integer scalar is a one-lane SIMD value in Mojo, with the same clamp method.
            return f"({self.generate_expression(args[0])}).clamp(0, 1)"
        if arg_dtype not in supported_float_dtypes:
            return None

        arg_expr = self.generate_expression(args[0])
        self.required_saturate_helpers.add(("scalar", arg_dtype, 1, 1))
        return f"{self.saturate_scalar_helper_name(arg_dtype)}({arg_expr})"

    def generate_angle_conversion_call(self, func_name, args):
        if len(args) != 1:
            generated_args = ", ".join(self.generate_expression(arg) for arg in args)
            return f"{func_name}({generated_args})"

        factor = {
            "degrees": "57.29577951308232",
            "radians": "0.017453292519943295",
        }[func_name]
        arg_expr = self.generate_expression(args[0])
        return f"({arg_expr} * {factor})"

    def generate_bool_mix_call(self, args):
        if len(args) != 3:
            return None

        factor_type = self.expression_result_type(args[2])
        factor_info = self.vector_type_info(factor_type)
        if factor_info is not None:
            if factor_info[0] != "DType.bool":
                return None
            return self.generate_bool_vector_select_expression(
                args[2], args[1], args[0]
            )

        if self.expression_mojo_dtype(args[2]) != "DType.bool":
            return None

        condition = self.generate_expression(args[2])
        true_value = self.generate_expression(args[1])
        false_value = self.generate_expression(args[0])
        return f"({true_value} if {condition} else {false_value})"

    def generate_min_max_call(self, func_name, args, target_type=None):
        if len(args) != 2:
            return None

        left_type = self.expression_result_type(args[0])
        right_type = self.expression_result_type(args[1])
        left_info = self.vector_type_info(left_type)
        right_info = self.vector_type_info(right_type)
        if left_info is None and right_info is None:
            return self.generate_scalar_min_max_call(
                func_name, args, target_type, left_type, right_type
            )
        if left_info is not None and left_info[0] != "DType.float32":
            return None
        if right_info is not None and right_info[0] != "DType.float32":
            return None

        self.required_math_helpers.add(func_name)
        left = self.generate_expression(args[0])
        right = self.generate_expression(args[1])
        if left_info is not None and right_info is None:
            right = f"Float32({right})"
        elif left_info is None and right_info is not None:
            left = f"Float32({left})"
        return f"{func_name}({left}, {right})"

    def generate_scalar_min_max_call(
        self, func_name, args, target_type, left_type, right_type
    ):
        left = self.generate_expression(args[0])
        right = self.generate_expression(args[1])
        comparator = ">=" if func_name == "max" else "<="

        if self.is_scalar_integer_type(target_type) or (
            self.is_scalar_integer_type(left_type)
            and self.is_scalar_integer_type(right_type)
        ):
            scalar_type = self.scalar_integer_mojo_type(
                target_type, left_type, right_type
            )
            self.required_math_helpers.add(func_name)
            return f"{func_name}({scalar_type}({left}), {scalar_type}({right}))"

        if (
            self.is_float_scalar_type(target_type)
            or self.is_float_scalar_type(left_type)
            or self.is_float_scalar_type(right_type)
        ):
            self.required_math_helpers.add(func_name)
            return f"{func_name}(Float32({left}), Float32({right}))"

        return None

    def scalar_integer_mojo_type(self, *type_names):
        for type_name in type_names:
            if not self.is_scalar_integer_type(type_name):
                continue
            mapped_type = self.map_type(type_name)
            if mapped_type in MOJO_MAPPED_INTEGER_TYPES:
                return mapped_type
        return "Int32"

    def generate_clamp_call(self, args):
        if len(args) != 3:
            return None

        value_type = self.expression_result_type(args[0])
        value_info = self.vector_type_info(value_type)
        if value_info is None:
            return None
        if value_info[0] != "DType.float32":
            return None

        min_type = self.expression_result_type(args[1])
        max_type = self.expression_result_type(args[2])
        min_info = self.vector_type_info(min_type)
        max_info = self.vector_type_info(max_type)
        if min_info is not None and min_info[:3] != value_info[:3]:
            return None
        if max_info is not None and max_info[:3] != value_info[:3]:
            return None
        if (min_info is None) != (max_info is None):
            return None

        self.required_math_helpers.add("clamp")
        value = self.generate_expression(args[0])
        min_value = self.generate_expression(args[1])
        max_value = self.generate_expression(args[2])
        if min_info is None and max_info is None:
            min_value = f"Float32({min_value})"
            max_value = f"Float32({max_value})"
        return f"clamp({value}, {min_value}, {max_value})"

    def generate_smoothstep_call(self, args, target_type=None):
        if len(args) != 3:
            return None

        arg_types = [self.expression_result_type(arg) for arg in args]
        if any(self.vector_type_info(arg_type) is not None for arg_type in arg_types):
            return None
        if not (
            self.is_float_scalar_type(target_type)
            or any(self.is_float_scalar_type(arg_type) for arg_type in arg_types)
        ):
            return None

        self.required_math_helpers.add("smoothstep")
        rendered_args = [self.generate_expression(arg) for arg in args]
        return "smoothstep({})".format(
            ", ".join(f"Float32({arg})" for arg in rendered_args)
        )

    def generate_texture_call(self, args, helper_name):
        if not args:
            return f"{helper_name}()"

        validation_args, sampler_arg = self.split_sampler_arg(args)
        texture_type = self.expression_result_type(validation_args[0])
        mapped_type = self.map_type(texture_type)
        self.validate_texture_sample_args(validation_args, helper_name, mapped_type)
        if self.resource_sample_coord_type(mapped_type) is not None:
            if sampler_arg is None:
                if helper_name == "sample":
                    self.required_resource_sample_types.add(mapped_type)
                elif helper_name == "sample_lod":
                    self.required_resource_lod_types.add(mapped_type)
                elif helper_name == "sample_grad":
                    self.required_resource_grad_types.add(mapped_type)
            elif helper_name == "sample":
                self.required_resource_sample_sampler_types.add(mapped_type)
            elif helper_name == "sample_lod":
                self.required_resource_lod_sampler_types.add(mapped_type)
            elif helper_name == "sample_grad":
                self.required_resource_grad_sampler_types.add(mapped_type)

        generated_args = ", ".join(self.generate_expression(arg) for arg in args)
        if sampler_arg is not None:
            helper_name = f"{helper_name}_sampler"
        return f"{helper_name}({generated_args})"

    def validate_texture_sample_args(self, args, helper_name, resource_type):
        if not self.is_texture_resource_type(resource_type):
            raise MojoRefusal(
                f"Unsupported {helper_name} for Mojo codegen; "
                f"texture resource required: {resource_type}"
            )
        if self.resource_sample_coord_type(resource_type) is None:
            required = (
                "non-multisample texture required"
                if self.is_multisample_resource_type(resource_type)
                else "sampleable texture required"
            )
            raise MojoRefusal(
                f"Unsupported {helper_name} for Mojo codegen; "
                f"{required}: {resource_type}"
            )

        expected_counts = {"sample": 2, "sample_lod": 3, "sample_grad": 4}
        expected_count = expected_counts.get(helper_name)
        if expected_count is not None and len(args) != expected_count:
            argument_count = expected_count - 1
            raise MojoRefusal(
                f"Unsupported {helper_name} for Mojo codegen; expected "
                f"{argument_count} argument(s) after texture resource, got "
                f"{len(args) - 1}"
            )

        expected_coord_type = self.resource_sample_coord_type(resource_type)
        if len(args) >= 2:
            self.validate_resource_argument_type(
                args[1], expected_coord_type, helper_name, resource_type, "coordinate"
            )
        if helper_name == "sample_lod" and len(args) >= 3:
            self.validate_resource_argument_type(
                args[2], "Float32", helper_name, resource_type, "lod"
            )
        if helper_name == "sample_grad":
            if len(args) >= 3:
                self.validate_resource_argument_type(
                    args[2],
                    expected_coord_type,
                    helper_name,
                    resource_type,
                    "ddx",
                )
            if len(args) >= 4:
                self.validate_resource_argument_type(
                    args[3],
                    expected_coord_type,
                    helper_name,
                    resource_type,
                    "ddy",
                )

    def strip_split_sampler_arg(self, args):
        stripped_args, _ = self.split_sampler_arg(args)
        return stripped_args

    def split_sampler_arg(self, args):
        if len(args) < 3:
            return args, None
        sampler_type = self.map_type(self.expression_result_type(args[1]))
        if sampler_type == "Sampler":
            return [args[0], *args[2:]], args[1]
        return args, None
