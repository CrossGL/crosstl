"""Wave intrinsics in the HLSL generator: results, shuffles and fills, argument and component shapes, buffer and resource indices, scalar bitcasts and interpolation."""

from ..ast import (
    ArrayAccessNode,
    IdentifierNode,
    MemberAccessNode,
    VariableNode,
)
from ..refusals import DirectXRefusal
from ..wave_operations import (
    WAVE_EXTENDED_OPERATIONS,
    wave_extended_result_type,
)
from .array_utils import split_array_type_suffix
from .buffer_length import BufferLengthUnsupportedError
from .hlsl_alias_offsets import (
    hlsl_alias_offset_for_source,
    hlsl_source_identifier_name,
)
from .hlsl_pointer_buffer_calls import hlsl_pointer_buffer_call


class DirectXWaveIntrinsicMixin:
    """Wave intrinsic members of ``HLSLCodeGen``."""

    def hlsl_bfloat16_bitcast_expression(
        self,
        target_type,
        source_type,
        argument_code,
        *,
        operation,
        source_location=None,
    ):
        vector_bitcast = self.hlsl_bfloat16_vector_bitcast_expression(
            target_type,
            source_type,
            argument_code,
            operation=operation,
            source_location=source_location,
        )
        if vector_bitcast is not None:
            return vector_bitcast
        target_is_bfloat = self.is_hlsl_bfloat16_type(target_type)
        source_is_bfloat = self.is_hlsl_bfloat16_type(source_type)
        target_name = str(self.type_name_string(target_type) or "").strip()
        source_name = str(self.type_name_string(source_type) or "").strip()
        unsigned_16_bit_names = {
            "min16uint",
            "u16",
            "uint16",
            "uint16_t",
            "unsigned short",
            "ushort",
        }
        target_is_uint16 = target_name in unsigned_16_bit_names
        source_is_uint16 = source_name in unsigned_16_bit_names
        if not (target_is_bfloat or source_is_bfloat):
            return None
        if target_is_bfloat and source_is_bfloat:
            return argument_code
        if target_is_bfloat and source_is_uint16:
            self.require_hlsl_bfloat16_helper("from_uint16")
            uint16_type = self.map_type(source_type)
            return f"__crossgl_bfloat16_from_uint16({uint16_type}({argument_code}))"
        if target_is_uint16 and source_is_bfloat:
            self.require_hlsl_bfloat16_helper("to_uint16")
            return f"__crossgl_bfloat16_to_uint16(uint({argument_code}))"

        raise self.directx_bfloat16_unsupported(
            "DirectX exact bfloat16 bitcasts require matching bfloat16 and "
            f"16-bit unsigned payloads, got '{source_name}' to '{target_name}'",
            operation=operation,
            source_type=source_type,
            reason="bitcast-width-or-type-mismatch",
            source_location=source_location,
        )

    def generate_hlsl_metal_as_type_call(self, func_name, args):
        source_target_type = self.hlsl_metal_as_type_source_target_type(func_name)
        if source_target_type is None or len(args) != 1:
            return None

        argument = args[0]
        argument_code = self.generate_expression_with_expected(argument, None)
        source_type = self.expression_result_type(argument)
        bfloat_bitcast = self.hlsl_bfloat16_bitcast_expression(
            source_target_type,
            source_type,
            argument_code,
            operation=str(func_name),
            source_location=getattr(argument, "source_location", None),
        )
        if bfloat_bitcast is not None:
            return bfloat_bitcast
        target_type = self.map_type(source_target_type)
        explicit_bitcast = self.hlsl_explicit_bitcast_expression(
            source_target_type, source_type, argument_code
        )
        if explicit_bitcast is not None:
            return explicit_bitcast
        return self.hlsl_bitcast_or_cast_expression(
            target_type,
            argument_code,
            source_type,
        )

    def generate_hlsl_bitcast_call(self, func_name, args, *, argument_code=None):
        if (
            not func_name
            or func_name in getattr(self, "function_return_types", {})
            or func_name not in self.HLSL_BITCAST_FUNCTION_TARGETS
            or len(args) != 1
        ):
            return None

        target_type = self.hlsl_bitcast_result_type(func_name, args)
        if target_type is None:
            return None
        argument = args[0]
        if argument_code is None:
            argument_code = self.generate_expression_with_expected(argument, None)
        source_type = self.expression_result_type(argument)
        if func_name == "asuint" and self.is_hlsl_bfloat16_type(source_type):
            self.require_hlsl_bfloat16_helper("to_uint16")
            return f"uint(__crossgl_bfloat16_to_uint16(uint({argument_code})))"
        if self.is_hlsl_bfloat16_type(source_type):
            raise self.directx_bfloat16_unsupported(
                "DirectX cannot apply 32-bit HLSL bitcast intrinsic "
                f"'{func_name}' to a 16-bit bfloat16 payload",
                operation=func_name,
                source_type=source_type,
                reason="bitcast-width-mismatch",
                source_location=getattr(argument, "source_location", None),
            )
        intrinsic = self.HLSL_BITCAST_INTRINSICS[func_name]
        return self.hlsl_bitcast_or_cast_expression(
            target_type,
            argument_code,
            source_type,
            intrinsic=intrinsic,
        )

    def hlsl_bitcast_or_cast_expression(
        self,
        target_type,
        argument_code,
        source_type=None,
        *,
        intrinsic=None,
    ):
        mapped_target = self.map_type(target_type)
        mapped_source = self.map_type(source_type) if source_type else None
        if mapped_source == mapped_target:
            return argument_code

        source_base = self.hlsl_scalar_or_vector_component_type(mapped_source)
        target_base = self.hlsl_scalar_or_vector_component_type(mapped_target)
        if source_base in {"half", "min16float", "min10float"}:
            return f"{mapped_target}({argument_code})"
        if source_base == target_base and source_base is not None:
            return f"{mapped_target}({argument_code})"
        if intrinsic in {"asuint", "asint"} and source_base in {
            "int",
            "uint",
            "int16_t",
            "uint16_t",
            "min16int",
            "min16uint",
            "min12int",
            "uint64_t",
            "int64_t",
        }:
            return f"{mapped_target}({argument_code})"
        if intrinsic in {"asuint", "asint"} and (
            source_base is None
            and self.hlsl_integer_wave_read_expression(argument_code)
        ):
            return f"{mapped_target}({argument_code})"
        if intrinsic is None and target_base in {
            "half",
            "min16float",
            "min10float",
        }:
            return f"{mapped_target}({argument_code})"
        if intrinsic:
            return f"{intrinsic}({argument_code})"
        return f"{mapped_target}({argument_code})"

    def hlsl_scalar_or_vector_component_type(self, type_name):
        if not type_name:
            return None
        mapped = self.map_type(type_name)
        for component in (
            "min16float",
            "min10float",
            "uint16_t",
            "int16_t",
            "min16uint",
            "min16int",
            "min12int",
            "uint64_t",
            "int64_t",
            "float",
            "half",
            "double",
            "uint",
            "int",
            "bool",
        ):
            if mapped == component:
                return component
            if mapped.startswith(component):
                suffix = mapped[len(component) :]
                if suffix in {"2", "3", "4"}:
                    return component
        return None

    def hlsl_integer_wave_read_expression(self, expression):
        if not isinstance(expression, str):
            return False
        return any(
            token in expression
            for token in (
                "WaveReadLaneAt(uint(",
                "WaveReadLaneAt(int(",
                "WaveReadLaneAt(min16uint(",
                "WaveReadLaneAt(min16int(",
            )
        )

    def hlsl_countbits_result_type(self, func_name, args):
        if func_name not in {"bitCount", "countbits"} or func_name in getattr(
            self, "function_return_types", {}
        ):
            return None

        source_type = self.map_type(
            (self.expression_result_type(args[0]) if args else None) or "uint"
        )
        for component in ("uint", "int"):
            if source_type == component:
                return "uint"
            if source_type.startswith(component):
                suffix = source_type[len(component) :]
                if suffix in {"2", "3", "4"}:
                    return f"uint{suffix}"
        return "uint"

    def hlsl_reversebits_result_type(self, func_name, args):
        if func_name not in {"bitfieldReverse", "reversebits"}:
            return None
        if func_name in getattr(self, "function_return_types", {}):
            return None

        source_type = self.map_type(
            (self.expression_result_type(args[0]) if args else None) or "uint"
        )
        for component in ("uint", "int"):
            if source_type == component:
                return component
            if source_type.startswith(component):
                suffix = source_type[len(component) :]
                if suffix in {"2", "3", "4"}:
                    return f"{component}{suffix}"
        return "uint"

    def hlsl_firstbit_result_type(self, func_name, args):
        if func_name not in {"findLSB", "findMSB", "firstbitlow", "firstbithigh"}:
            return None
        if func_name in getattr(self, "function_return_types", {}):
            return None

        source_type = self.map_type(
            (self.expression_result_type(args[0]) if args else None) or "int"
        )
        for component in ("int", "uint"):
            if source_type == component:
                return component
            if source_type.startswith(component):
                suffix = source_type[len(component) :]
                if suffix in {"2", "3", "4"}:
                    return f"{component}{suffix}"
        return "int"

    def hlsl_function_name_is_shadowed(self, func_name):
        return func_name in self.local_variable_types or func_name in getattr(
            self, "function_return_types", {}
        )

    def interpolation_function_call(self, func_name, args):
        if not func_name or func_name in getattr(self, "function_return_types", {}):
            return None

        intrinsic = self.interpolation_intrinsic_name(func_name)
        if intrinsic is None:
            return None

        expected_args = 1 if func_name == "interpolateAtCentroid" else 2
        if len(args) != expected_args:
            raise DirectXRefusal(
                f"DirectX interpolation builtin '{func_name}' requires "
                f"{expected_args} argument(s), got {len(args)}"
            )

        args_str = ", ".join(self.generate_expression(arg) for arg in args)
        return f"{intrinsic}({args_str})"

    def interpolation_intrinsic_name(self, func_name):
        return self.HLSL_INTERPOLATION_INTRINSICS.get(func_name)

    def generate_wave_op_expression(self, expr):
        # A predicated reduction already ran outside its guard; its source
        # expression reads that result.
        override = getattr(self, "hlsl_software_subgroup_result_overrides", {}).get(
            id(expr)
        )
        if override is not None:
            return override
        operation = getattr(expr, "operation", "")
        return self.generate_hlsl_wave_intrinsic_call(operation, expr.arguments)

    def validate_directx_wave_intrinsic_profile(self, operation):
        if self.target_profile not in {"dx10", "dx11"}:
            return

        raise DirectXRefusal(
            f"DirectX profile {self.target_profile} does not support wave intrinsic "
            f"'{operation}'; "
            "wave intrinsics require DirectX 12 / Shader Model 6.0"
        )

    def hlsl_wave_shuffle_payload_type(self, value_type, mapped_value_type):
        """The expected type that moves a shuffled value's register unchanged.

        Exact bfloat16 values travel as their bits in a ``uint``. Generating
        them for ``uint`` would convert the number, so ``simd_shuffle_down``
        and ``simd_shuffle_and_fill_up`` of MLX's bfloat16 scan values moved
        ``uint(float(x))``; the source type keeps the bits, as
        ``WaveReadLaneAt`` already does.
        """
        if value_type is not None and self.is_hlsl_bfloat16_type(value_type):
            return value_type
        return mapped_value_type

    def generate_hlsl_wave_shuffle_and_fill_up_call(self, operation, args):
        value_type = self.expression_result_type(args[0])
        mapped_value_type = self.map_type(
            value_type or self.current_expression_expected_type
        )
        if not mapped_value_type:
            raise DirectXRefusal(
                f"DirectX wave intrinsic '{operation}' requires a statically "
                "known value type"
            )
        payload_type = self.hlsl_wave_shuffle_payload_type(
            value_type, mapped_value_type
        )
        value = self.generate_expression_with_expected(args[0], payload_type)
        fill = self.generate_expression_with_expected(args[1], payload_type)
        delta = self.generate_expression_with_expected(args[2], "uint")
        delta_type = self.map_type(self.expression_result_type(args[2]))
        if delta_type in {"int16_t", "uint16_t", "min16int", "min16uint"}:
            delta = f"uint({delta})"
        if len(args) == 4:
            modulo = self.generate_expression_with_expected(args[3], "uint")
            modulo_type = self.map_type(self.expression_result_type(args[3]))
            if modulo_type in {"int16_t", "uint16_t", "min16int", "min16uint"}:
                modulo = f"uint({modulo})"
        else:
            modulo = (
                f"{self.software_subgroup_width}u"
                if self.software_subgroup_width is not None
                else "WaveGetLaneCount()"
            )
        if self.software_subgroup_width is not None:
            result = self.hlsl_software_subgroup_call(
                operation,
                value_type,
                value,
                delta,
                fill_expression=fill,
                modulo_expression=modulo,
            )
            return self.hlsl_wave_shuffle_result_conversion(operation, args, result)
        self.required_hlsl_wave_shuffle_and_fill_up_types.add(mapped_value_type)
        helper_name = self.hlsl_wave_shuffle_and_fill_up_helper_name(mapped_value_type)
        result = f"{helper_name}({value}, {fill}, {delta}, {modulo})"
        return self.hlsl_wave_shuffle_result_conversion(operation, args, result)

    def hlsl_wave_shuffle_bool_conversion(self, operation, args):
        if operation not in self.HLSL_WAVE_SHUFFLE_VALUE_INTRINSICS or not args:
            return False
        expected_type = self.map_type(
            self.type_name_string(self.current_expression_expected_type)
        )
        actual_type = self.map_type(
            self.hlsl_wave_intrinsic_return_type(operation, args)
        )
        return expected_type == "bool" and actual_type in {
            "int",
            "int16_t",
            "min16int",
            "min12int",
            "uint",
            "uint16_t",
            "min16uint",
            "int64_t",
            "uint64_t",
        }

    def hlsl_wave_shuffle_result_conversion(self, operation, args, expression):
        if self.hlsl_wave_shuffle_bool_conversion(operation, args):
            return f"(({expression}) != 0)"
        return expression

    def hlsl_wave_intrinsic_return_type(self, operation, args):
        if operation in WAVE_EXTENDED_OPERATIONS:
            return wave_extended_result_type(
                operation, self.expression_result_type(args[0]) if args else None
            )
        if operation in self.HLSL_WAVE_UINT_RESULT_INTRINSICS:
            return "uint"
        if operation == "WaveActiveAllEqual" and args:
            return self.hlsl_wave_all_equal_return_type(args[0])
        if operation in self.HLSL_WAVE_BOOL_RESULT_INTRINSICS:
            return "bool"
        if operation in self.HLSL_WAVE_UINT4_RESULT_INTRINSICS:
            return "uint4"
        if operation in self.HLSL_WAVE_VALUE_RESULT_INTRINSICS and args:
            return self.expression_result_type(args[0])
        return None

    def hlsl_wave_argument_base_type(self, argument):
        argument_type = self.expression_result_type(argument)
        if argument_type is None:
            return None, "", None

        mapped_type = self.map_type(argument_type)
        base_type, array_suffix = split_array_type_suffix(mapped_type)
        return base_type, array_suffix, mapped_type

    def hlsl_wave_value_component_type(self, argument):
        base_type, array_suffix, mapped_type = self.hlsl_wave_argument_base_type(
            argument
        )
        if mapped_type is None:
            return None, None
        if array_suffix:
            return None, mapped_type
        if base_type in self.HLSL_WAVE_NUMERIC_COMPONENT_TYPES or base_type == "bool":
            return base_type, mapped_type
        if self.is_vector_value_type(base_type):
            return self.vector_component_type(base_type), mapped_type
        return None, mapped_type

    def hlsl_wave_basic_value_component_type(self, argument):
        component_type, mapped_type = self.hlsl_wave_value_component_type(argument)
        if component_type is not None or mapped_type is None:
            return component_type, mapped_type

        base_type, array_suffix, mapped_type = self.hlsl_wave_argument_base_type(
            argument
        )
        if array_suffix:
            return None, mapped_type
        matrix_shape = self.hlsl_matrix_shape(base_type)
        if matrix_shape:
            return matrix_shape[0], mapped_type
        return None, mapped_type

    def hlsl_wave_all_equal_return_type(self, argument):
        base_type, array_suffix, mapped_type = self.hlsl_wave_argument_base_type(
            argument
        )
        if mapped_type is None or array_suffix:
            return "bool"
        if self.is_scalar_value_type(base_type):
            return "bool"
        if self.is_vector_value_type(base_type):
            size = self.map_type(base_type)[-1:]
            if size in {"2", "3", "4"}:
                return f"bool{size}"
        matrix_shape = self.hlsl_matrix_shape(base_type)
        if matrix_shape:
            _, rows, columns = matrix_shape
            return f"bool{rows}x{columns}"
        return "bool"

    def hlsl_result_component_type(self, vtype):
        mapped_type = self.map_type(vtype)
        if self.is_scalar_value_type(mapped_type):
            return mapped_type
        if self.is_vector_value_type(mapped_type):
            return self.vector_component_type(mapped_type)
        matrix_shape = self.hlsl_matrix_shape(mapped_type)
        if matrix_shape:
            return matrix_shape[0]
        return None

    def hlsl_wave_result_context_matches(self, actual_type, expected_type):
        if not self.hlsl_value_shape_matches(actual_type, expected_type):
            return False

        actual_component = self.hlsl_result_component_type(actual_type)
        expected_component = self.hlsl_result_component_type(expected_type)
        if actual_component == "bool" or expected_component == "bool":
            return actual_component == expected_component
        return True

    def hlsl_wave_operand_expected_type(self, operation, index, argument):
        """The type context in which one wave intrinsic operand is generated.

        Only the value operand of a value-preserving intrinsic shares the
        call's result context. Lane indices are ``uint``, partition masks are
        ``uint4``, and the operands of ballots, bit counts, votes and matches
        keep their own type, so a ``WaveGetLaneIndex()`` inside the lane index
        of a ``float4`` ``WaveReadLaneAt`` is checked against ``uint``.
        """
        if index == 0 and operation in self.HLSL_WAVE_VALUE_RESULT_INTRINSICS:
            return self.current_expression_expected_type
        if index == 1 and operation in {"WaveReadLaneAt", "QuadReadLaneAt"}:
            return "uint"
        if index == 1 and operation.startswith("WaveMultiPrefix"):
            return "uint4"
        return self.expression_result_type(argument)

    def generate_hlsl_wave_operands(self, operation, args):
        rendered = []
        for index, argument in enumerate(args):
            if index == 0 and operation in self.HLSL_WAVE_VALUE_RESULT_INTRINSICS:
                rendered.append(self.generate_expression(argument))
                continue
            rendered.append(
                self.generate_expression_with_expected(
                    argument,
                    self.hlsl_wave_operand_expected_type(operation, index, argument),
                )
            )
        return ", ".join(rendered)

    def validate_hlsl_wave_intrinsic_result_context(self, operation, args):
        expected_type = self.type_name_string(self.current_expression_expected_type)
        if not expected_type or expected_type == "void":
            return

        actual_type = self.hlsl_wave_intrinsic_return_type(operation, args)
        if not actual_type:
            return
        if self.hlsl_wave_result_context_matches(actual_type, expected_type):
            return
        if self.hlsl_wave_shuffle_bool_conversion(operation, args):
            return

        raise DirectXRefusal(
            f"DirectX wave intrinsic '{operation}' requires "
            f"{self.hlsl_value_type_display(actual_type)} result context, got "
            f"{self.hlsl_value_type_display(expected_type)}"
        )

    def validate_hlsl_wave_scalar_bool_argument(self, operation, argument, role):
        base_type, array_suffix, mapped_type = self.hlsl_wave_argument_base_type(
            argument
        )
        if mapped_type is None:
            return
        if array_suffix or base_type != "bool":
            raise DirectXRefusal(
                f"DirectX wave intrinsic '{operation}' {role} argument must be "
                f"scalar bool, got {mapped_type}"
            )

    def validate_hlsl_wave_scalar_int_uint_argument(self, operation, argument, role):
        base_type, array_suffix, mapped_type = self.hlsl_wave_argument_base_type(
            argument
        )
        if mapped_type is None:
            return
        # Accept any scalar integer width for the lane index. Metal's
        # simd_shuffle/simd_broadcast take a ushort lane (-> min16uint), which
        # HLSL implicitly widens to the uint laneIndex of WaveReadLaneAt; only
        # non-integer or vector lane arguments are rejected.
        if array_suffix or base_type not in self.HLSL_WAVE_INTEGER_COMPONENT_TYPES:
            raise DirectXRefusal(
                f"DirectX wave intrinsic '{operation}' {role} argument must be "
                f"scalar int or uint, got {mapped_type}"
            )

    def validate_hlsl_quad_lane_index_range(self, operation, argument):
        lane_index = self.literal_int_value(
            argument, self.hlsl_current_visible_int_constants()
        )
        if lane_index is None:
            return
        if not 0 <= lane_index <= 3:
            raise DirectXRefusal(
                f"DirectX wave intrinsic '{operation}' quad lane index must be "
                f"in the range 0 to 3, got {lane_index}"
            )

    def validate_hlsl_wave_uint4_argument(self, operation, argument, role):
        base_type, array_suffix, mapped_type = self.hlsl_wave_argument_base_type(
            argument
        )
        if mapped_type is None:
            return
        if array_suffix or base_type != "uint4":
            raise DirectXRefusal(
                f"DirectX wave intrinsic '{operation}' {role} argument must be "
                f"uint4, got {mapped_type}"
            )

    def validate_hlsl_wave_value_argument(
        self,
        operation,
        argument,
        role,
        allowed_components,
        description,
        include_matrices=False,
    ):
        if include_matrices:
            component_type, mapped_type = self.hlsl_wave_basic_value_component_type(
                argument
            )
        else:
            component_type, mapped_type = self.hlsl_wave_value_component_type(argument)
        if mapped_type is None:
            return
        if component_type not in allowed_components:
            raise DirectXRefusal(
                f"DirectX wave intrinsic '{operation}' {role} argument must be "
                f"{description}, got {mapped_type}"
            )

    def validate_hlsl_wave_shuffle_and_fill_arguments(self, operation, args):
        for argument, role in ((args[0], "value"), (args[1], "fill")):
            self.validate_hlsl_wave_value_argument(
                operation,
                argument,
                role,
                self.HLSL_WAVE_BASIC_COMPONENT_TYPES,
                "a basic scalar or vector",
            )

        value_type = self.type_name_string(self.expression_result_type(args[0]))
        fill_type = self.type_name_string(self.expression_result_type(args[1]))
        if (
            value_type
            and fill_type
            and self.map_type(value_type) != self.map_type(fill_type)
        ):
            raise DirectXRefusal(
                f"DirectX wave intrinsic '{operation}' value and fill arguments "
                "must have matching types, got "
                f"{self.map_type(value_type)} and {self.map_type(fill_type)}"
            )

        self.validate_hlsl_wave_scalar_int_uint_argument(operation, args[2], "delta")
        if len(args) == 4:
            self.validate_hlsl_wave_scalar_int_uint_argument(
                operation, args[3], "modulo"
            )

    def validate_hlsl_wave_intrinsic_arguments(self, operation, args):
        if operation in self.HLSL_WAVE_BOOL_ARGUMENT_INTRINSICS:
            self.validate_hlsl_wave_scalar_bool_argument(
                operation, args[0], "predicate"
            )
        elif operation in self.HLSL_WAVE_NUMERIC_VALUE_INTRINSICS:
            self.validate_hlsl_wave_value_argument(
                operation,
                args[0],
                "value",
                self.HLSL_WAVE_NUMERIC_COMPONENT_TYPES,
                "numeric scalar or vector",
            )
        elif operation in self.HLSL_WAVE_INTEGER_VALUE_INTRINSICS:
            self.validate_hlsl_wave_value_argument(
                operation,
                args[0],
                "value",
                self.HLSL_WAVE_INTEGER_COMPONENT_TYPES,
                "integer scalar or vector",
            )
        elif operation == "WaveActiveAllEqual":
            self.validate_hlsl_wave_value_argument(
                operation,
                args[0],
                "value",
                self.HLSL_WAVE_BASIC_COMPONENT_TYPES,
                "basic scalar, vector, or matrix",
                include_matrices=True,
            )
        elif operation == "WaveMatch":
            self.validate_hlsl_wave_value_argument(
                operation,
                args[0],
                "value",
                self.HLSL_WAVE_BASIC_COMPONENT_TYPES,
                "primitive scalar, vector, or matrix",
                include_matrices=True,
            )
        elif operation in self.HLSL_WAVE_SHUFFLE_AND_FILL_INTRINSICS:
            self.validate_hlsl_wave_shuffle_and_fill_arguments(operation, args)
        elif operation in self.HLSL_WAVE_LANE_READ_VALUE_INTRINSICS:
            relative_shuffle_with_defined_fallback = (
                operation in self.HLSL_WAVE_RELATIVE_SHUFFLE_OPERATORS
                and self.relative_wave_shuffle_out_of_range == "self"
            )
            allowed_components = (
                self.HLSL_WAVE_BASIC_COMPONENT_TYPES
                if relative_shuffle_with_defined_fallback
                else self.HLSL_WAVE_NUMERIC_COMPONENT_TYPES
            )
            description = (
                "basic scalar or vector"
                if relative_shuffle_with_defined_fallback
                else "numeric scalar or vector"
            )
            self.validate_hlsl_wave_value_argument(
                operation,
                args[0],
                "value",
                allowed_components,
                description,
            )

        if operation == "WaveReadLaneAt":
            self.validate_hlsl_wave_scalar_int_uint_argument(
                operation, args[1], "lane index"
            )
        elif operation == "QuadReadLaneAt":
            self.validate_hlsl_wave_scalar_int_uint_argument(
                operation, args[1], "quad lane index"
            )
            self.validate_hlsl_quad_lane_index_range(operation, args[1])

        if operation.startswith("WaveMultiPrefix"):
            self.validate_hlsl_wave_uint4_argument(operation, args[1], "partition mask")

    def hlsl_buffer_helper_resource_type(self, expr):
        vtype = self.type_name_string(self.expression_result_type(expr))
        if not vtype:
            return None
        resource_type = self.resource_base_type(vtype)
        resource_name = self.hlsl_resource_type_name(resource_type)
        if resource_name in {
            "Buffer",
            "StructuredBuffer",
            "ByteAddressBuffer",
            "RWBuffer",
            "RWStructuredBuffer",
            "RWByteAddressBuffer",
            "RasterizerOrderedBuffer",
            "RasterizerOrderedStructuredBuffer",
            "RasterizerOrderedByteAddressBuffer",
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
        }:
            return resource_type
        return None

    def hlsl_byte_address_vector_helper_width(self, func_name):
        if not isinstance(func_name, str):
            return None
        for prefix in ("buffer_load", "buffer_store"):
            if not func_name.startswith(prefix):
                continue
            suffix = func_name[len(prefix) :]
            if suffix in {"2", "3", "4"}:
                return int(suffix)
        return None

    def hlsl_buffer_helper_names(self):
        return {
            "buffer_append",
            "buffer_consume",
            "buffer_decrement_counter",
            "buffer_dimensions",
            "buffer_increment_counter",
            "buffer_load",
            "buffer_load2",
            "buffer_load3",
            "buffer_load4",
            "buffer_store",
            "buffer_store2",
            "buffer_store3",
            "buffer_store4",
        }

    def hlsl_resource_pointer_offset_for_name(self, resource_name, source=None):
        pointer_offset = self.current_hlsl_resource_pointer_offsets.get(resource_name)
        if pointer_offset is not None:
            return pointer_offset
        # A pointer alias renders as its root resource, so the rendered name
        # cannot tell `row[i]` from `root[i]`. A source identifier can: only
        # an alias, never its root, carries the alias offset.
        source_name = hlsl_source_identifier_name(source)
        if source_name is not None:
            return hlsl_alias_offset_for_source(self, source_name)
        binding = self.current_hlsl_resource_pointer_aliases.get(resource_name)
        if binding is not None:
            if binding.get("kind") == "workgroup-pointer" and not binding.get(
                "is_pointer_alias"
            ):
                return None
            return binding.get("offset")
        for alias_binding in self.current_hlsl_resource_pointer_aliases.values():
            if alias_binding.get("root") == resource_name:
                if alias_binding.get("kind") == "workgroup-pointer":
                    continue
                return alias_binding.get("offset")
        return None

    def hlsl_resource_pointer_offset_index(self, resource_name, index, source=None):
        pointer_offset = self.hlsl_resource_pointer_offset_for_name(
            resource_name, source
        )
        if pointer_offset:
            return f"({pointer_offset} + {index})"
        return index

    def hlsl_resource_index_expression(
        self,
        resource_expr,
        index_expr,
        rendered_index,
        *,
        rendered_resource=None,
        resource_type=None,
    ):
        resource_type = resource_type or self.hlsl_buffer_helper_resource_type(
            resource_expr
        )
        resource_name = self.hlsl_resource_type_name(resource_type)
        if resource_name is None:
            return rendered_index

        source_type = self.map_type(self.expression_result_type(index_expr))
        has_pointer_offset = (
            rendered_resource is not None
            and self.hlsl_resource_pointer_offset_for_name(
                rendered_resource, resource_expr
            )
            is not None
        )
        needs_cast = has_pointer_offset or source_type in {"int64_t", "uint64_t"}
        if not needs_cast:
            return rendered_index

        if resource_name in {
            "ByteAddressBuffer",
            "RWByteAddressBuffer",
            "RasterizerOrderedByteAddressBuffer",
        }:
            return f"int({rendered_index})"
        if resource_name in {
            "Buffer",
            "StructuredBuffer",
            "RWBuffer",
            "RWStructuredBuffer",
            "RasterizerOrderedBuffer",
            "RasterizerOrderedStructuredBuffer",
        }:
            return f"uint({rendered_index})"
        return rendered_index

    def hlsl_value_shape_label(self, vtype):
        type_name = self.type_name_string(vtype)
        if not type_name:
            return None
        mapped_type = self.map_type(type_name)
        if self.is_scalar_value_type(mapped_type):
            return "scalar"
        if self.is_vector_value_type(mapped_type) and mapped_type[-1:] in {
            "2",
            "3",
            "4",
        }:
            return f"vector{mapped_type[-1]}"
        matrix_shape = self.hlsl_matrix_shape(mapped_type)
        if matrix_shape:
            _, rows, columns = matrix_shape
            return f"matrix{rows}x{columns}"
        return mapped_type

    def hlsl_value_shape_matches(self, expected_type, actual_type):
        expected_shape = self.hlsl_value_shape_label(expected_type)
        actual_shape = self.hlsl_value_shape_label(actual_type)
        if not expected_shape or not actual_shape:
            return True
        shaped_values = {"scalar", "vector2", "vector3", "vector4"}
        if expected_shape in shaped_values or actual_shape in shaped_values:
            return expected_shape == actual_shape
        return self.map_type(expected_type) == self.map_type(actual_type)

    def hlsl_value_type_display(self, vtype):
        type_name = self.type_name_string(vtype)
        if not type_name:
            return "unknown"
        return self.map_type(type_name)

    def validate_hlsl_buffer_consume_result_shape(self, resource_type):
        expected_type = self.type_name_string(self.current_expression_expected_type)
        if not expected_type or expected_type == "void":
            return
        element_type = self.hlsl_typed_buffer_element_type(
            resource_type, {"ConsumeStructuredBuffer"}
        )
        if element_type is None:
            return
        if self.hlsl_value_shape_matches(element_type, expected_type):
            return
        raise DirectXRefusal(
            "DirectX buffer helper 'buffer_consume' result requires target matching "
            "ConsumeStructuredBuffer element type "
            f"{self.hlsl_value_type_display(element_type)}, got "
            f"{self.hlsl_value_type_display(expected_type)}"
        )

    def validate_buffer_call_access(self, func_name, args):
        if func_name not in self.hlsl_buffer_helper_names():
            return

        helper_arg_counts = {
            "buffer_append": 2,
            "buffer_consume": 1,
            "buffer_increment_counter": 1,
            "buffer_decrement_counter": 1,
        }
        expected_args = helper_arg_counts.get(func_name)
        if expected_args is not None and len(args) != expected_args:
            raise DirectXRefusal(
                f"DirectX buffer helper '{func_name}' requires "
                f"{expected_args} argument(s), got {len(args)}"
            )

        if not args:
            return

        resource_type = self.hlsl_buffer_helper_resource_type(args[0])
        if resource_type is None:
            return

        resource_name = self.hlsl_resource_type_name(resource_type)
        indexed_read_helpers = {
            "buffer_load",
            "buffer_load2",
            "buffer_load3",
            "buffer_load4",
        }
        indexed_write_helpers = {
            "buffer_store",
            "buffer_store2",
            "buffer_store3",
            "buffer_store4",
        }
        indexed_read_resources = {
            "Buffer",
            "StructuredBuffer",
            "ByteAddressBuffer",
            "RWBuffer",
            "RWStructuredBuffer",
            "RWByteAddressBuffer",
            "RasterizerOrderedBuffer",
            "RasterizerOrderedStructuredBuffer",
            "RasterizerOrderedByteAddressBuffer",
        }
        indexed_write_resources = {
            "RWBuffer",
            "RWStructuredBuffer",
            "RWByteAddressBuffer",
            "RasterizerOrderedBuffer",
            "RasterizerOrderedStructuredBuffer",
            "RasterizerOrderedByteAddressBuffer",
        }

        if (
            func_name in indexed_read_helpers
            and resource_name not in indexed_read_resources
        ):
            raise DirectXRefusal(
                f"DirectX buffer helper '{func_name}' requires a resource with "
                f"indexed read support, got {resource_type}"
            )
        if func_name in indexed_write_helpers:
            if resource_name in {"Buffer", "StructuredBuffer", "ByteAddressBuffer"}:
                raise DirectXRefusal(
                    f"DirectX buffer helper '{func_name}' cannot write readonly "
                    f"{resource_type}"
                )
            if resource_name not in indexed_write_resources:
                raise DirectXRefusal(
                    f"DirectX buffer helper '{func_name}' requires a resource with "
                    f"indexed write support, got {resource_type}"
                )
        vector_helper_width = self.hlsl_byte_address_vector_helper_width(func_name)
        if vector_helper_width is not None:
            byte_address_resources = {
                "ByteAddressBuffer",
                "RWByteAddressBuffer",
                "RasterizerOrderedByteAddressBuffer",
            }
            if resource_name not in byte_address_resources:
                raise DirectXRefusal(
                    f"DirectX buffer helper '{func_name}' requires "
                    "ByteAddressBuffer, RWByteAddressBuffer, or "
                    "RasterizerOrderedByteAddressBuffer resource, got "
                    f"{resource_type}"
                )
            if func_name.startswith("buffer_store") and len(args) >= 3:
                expected_value_type = f"uint{vector_helper_width}"
                value_type = self.type_name_string(self.expression_result_type(args[2]))
                mapped_value_type = self.map_type(value_type) if value_type else None
                if mapped_value_type != expected_value_type:
                    actual_value_type = mapped_value_type or value_type or "unknown"
                    raise DirectXRefusal(
                        f"DirectX buffer helper '{func_name}' requires "
                        f"{expected_value_type} value, got {actual_value_type}"
                    )
        if func_name == "buffer_append" and resource_name != "AppendStructuredBuffer":
            raise DirectXRefusal(
                "DirectX buffer helper 'buffer_append' requires "
                f"AppendStructuredBuffer, got {resource_type}"
            )
        if func_name == "buffer_append" and len(args) >= 2:
            element_type = self.hlsl_typed_buffer_element_type(
                resource_type, {"AppendStructuredBuffer"}
            )
            value_type = self.expression_result_type(args[1])
            if (
                element_type is not None
                and value_type is not None
                and not self.hlsl_value_shape_matches(element_type, value_type)
            ):
                raise DirectXRefusal(
                    "DirectX buffer helper 'buffer_append' requires value matching "
                    "AppendStructuredBuffer element type "
                    f"{self.hlsl_value_type_display(element_type)}, got "
                    f"{self.hlsl_value_type_display(value_type)}"
                )
        if func_name == "buffer_consume" and resource_name != "ConsumeStructuredBuffer":
            raise DirectXRefusal(
                "DirectX buffer helper 'buffer_consume' requires "
                f"ConsumeStructuredBuffer, got {resource_type}"
            )
        if func_name == "buffer_consume":
            self.validate_hlsl_buffer_consume_result_shape(resource_type)
        if (
            func_name in {"buffer_increment_counter", "buffer_decrement_counter"}
            and resource_name != "RWStructuredBuffer"
        ):
            raise DirectXRefusal(
                f"DirectX buffer helper '{func_name}' requires "
                f"RWStructuredBuffer, got {resource_type}"
            )

    def generate_buffer_call(
        self, func_name, args, *, rendered_value=None, statement_context=False
    ):
        """Render canonical CrossGL buffer operations as HLSL resource methods."""
        if func_name not in self.hlsl_buffer_helper_names():
            return None

        self.validate_buffer_call_access(func_name, args)
        pointer_call = hlsl_pointer_buffer_call(
            self,
            func_name,
            args,
            rendered_value=rendered_value,
            statement_context=statement_context,
        )
        if pointer_call is not None:
            return pointer_call

        vector_load_methods = {
            "buffer_load2": "Load2",
            "buffer_load3": "Load3",
            "buffer_load4": "Load4",
        }
        if func_name in vector_load_methods and len(args) >= 2:
            buffer = self.generate_expression(args[0])
            index = self.generate_expression(args[1])
            index = self.hlsl_resource_pointer_offset_index(buffer, index, args[0])
            resource_type = self.hlsl_buffer_helper_resource_type(args[0])
            index = self.hlsl_resource_index_expression(
                args[0],
                args[1],
                index,
                rendered_resource=buffer,
                resource_type=resource_type,
            )
            return f"{buffer}.{vector_load_methods[func_name]}({index})"

        vector_store_methods = {
            "buffer_store2": "Store2",
            "buffer_store3": "Store3",
            "buffer_store4": "Store4",
        }
        if func_name in vector_store_methods and len(args) >= 3:
            buffer = self.generate_expression(args[0])
            index = self.generate_expression(args[1])
            index = self.hlsl_resource_pointer_offset_index(buffer, index, args[0])
            resource_type = self.hlsl_buffer_helper_resource_type(args[0])
            index = self.hlsl_resource_index_expression(
                args[0],
                args[1],
                index,
                rendered_resource=buffer,
                resource_type=resource_type,
            )
            value = self.generate_expression(args[2])
            return f"{buffer}.{vector_store_methods[func_name]}({index}, {value})"

        if func_name == "buffer_load" and len(args) >= 2:
            buffer = self.generate_expression(args[0])
            index = self.generate_expression(args[1])
            index = self.hlsl_resource_pointer_offset_index(buffer, index, args[0])
            resource_type = self.hlsl_buffer_helper_resource_type(args[0])
            index = self.hlsl_resource_index_expression(
                args[0],
                args[1],
                index,
                rendered_resource=buffer,
                resource_type=resource_type,
            )
            rendered = f"{buffer}.Load({index})"
            logical = self.hlsl_half_storage.binding_type(
                self.hlsl_resource_pointer_binding(args[0])
            )
            return (
                self.hlsl_half_storage.convert(logical, rendered)
                if logical is not None
                else rendered
            )
        if func_name == "buffer_store" and len(args) >= 3:
            buffer = self.generate_expression(args[0])
            index = self.generate_expression(args[1])
            rendered_value = self.hlsl_precise_buffer_store_value(
                args[2], rendered_value
            )
            index = self.hlsl_resource_pointer_offset_index(buffer, index, args[0])
            resource_type = self.hlsl_buffer_helper_resource_type(args[0])
            source_element_type = self.hlsl_typed_buffer_element_type(
                resource_type,
                {
                    "RWBuffer",
                    "RWStructuredBuffer",
                    "RasterizerOrderedBuffer",
                    "RasterizerOrderedStructuredBuffer",
                },
            )
            binding = self.hlsl_resource_pointer_binding(args[0])
            logical_half = self.hlsl_half_storage.binding_type(binding)
            if logical_half is not None:
                value = (
                    rendered_value
                    if rendered_value is not None
                    else self.generate_expression_with_expected(args[2], logical_half)
                )
                value = self.hlsl_half_storage.convert(logical_half, value, encode=True)
                storage_access = self.generate_hlsl_resource_pointer_access(
                    args[0], args[1], require_write=True, physical=True
                )
                store = f"{storage_access} = {value}"
                return (
                    store
                    if statement_context
                    else self.hlsl_half_storage.convert(logical_half, f"({store})")
                )
            if binding is not None and self.is_hlsl_bfloat16_type(
                binding.get("source_element_type")
            ):
                source_element_type = binding["source_element_type"]
            byte_store_type = next(
                (
                    candidate
                    for candidate in (
                        source_element_type,
                        self.hlsl_resource_pointer_assignment_type(binding),
                    )
                    if self.hlsl_byte_integer_shape(candidate) is not None
                ),
                None,
            )
            value_source_type = self.hlsl_source_expression_type(args[2])
            logical_bfloat_type = next(
                (
                    value_type
                    for value_type in (source_element_type, value_source_type)
                    if self.is_hlsl_bfloat16_type(value_type)
                ),
                None,
            )
            physical_resource_type = self.map_type(
                self.directx_resource_declaration_type(resource_type)
            )
            physical_element_type = self.hlsl_typed_buffer_element_type(
                physical_resource_type,
                {
                    "RWBuffer",
                    "RWStructuredBuffer",
                    "RasterizerOrderedBuffer",
                    "RasterizerOrderedStructuredBuffer",
                },
            )
            if rendered_value is not None:
                value = (
                    self.hlsl_byte_conversion_expression(
                        rendered_value, byte_store_type
                    )
                    or rendered_value
                )
            elif (
                logical_bfloat_type is not None and physical_element_type == "uint16_t"
            ):
                value = self.generate_expression(args[2])
                storage_type = self.hlsl_bfloat16_storage_type(
                    logical_bfloat_type,
                    operation="buffer store",
                    source_location=getattr(args[2], "source_location", None),
                )
                value = f"{storage_type}({value})"
            else:
                value = self.generate_expression_with_expected(
                    args[2],
                    byte_store_type or physical_element_type or source_element_type,
                )
            index = self.hlsl_resource_index_expression(
                args[0],
                args[1],
                index,
                rendered_resource=buffer,
                resource_type=resource_type,
            )
            typed_buffers = {
                "RWBuffer",
                "RWStructuredBuffer",
                "RasterizerOrderedBuffer",
                "RasterizerOrderedStructuredBuffer",
            }
            resource_name = self.hlsl_resource_type_name(resource_type)
            if resource_type is None and binding is not None:
                # A helper's ``device T*`` parameter is the buffer its binding
                # declares; a typed or structured buffer has no Store method.
                resource_name = self.hlsl_resource_type_name(
                    binding.get("resource_type")
                )
            if resource_name in typed_buffers:
                return f"{buffer}[{index}] = {value}"
            return f"{buffer}.Store({index}, {value})"
        if func_name == "buffer_append" and len(args) >= 2:
            buffer = self.generate_expression(args[0])
            logical = self.hlsl_typed_buffer_element_type(
                self.hlsl_buffer_helper_resource_type(args[0]),
                {"AppendStructuredBuffer"},
            )
            value = self.hlsl_precise_buffer_store_value(
                args[1], None
            ) or self.generate_expression_with_expected(args[1], logical)
            if logical is not None and self.hlsl_half_storage.lowered(logical):
                value = self.hlsl_half_storage.convert(logical, value, encode=True)
            return f"{buffer}.Append({value})"
        if func_name == "buffer_consume" and args:
            buffer = self.generate_expression(args[0])
            logical = self.hlsl_typed_buffer_element_type(
                self.hlsl_buffer_helper_resource_type(args[0]),
                {"ConsumeStructuredBuffer"},
            )
            rendered = f"{buffer}.Consume()"
            return (
                self.hlsl_half_storage.convert(logical, rendered)
                if logical is not None and self.hlsl_half_storage.lowered(logical)
                else rendered
            )
        if func_name == "buffer_increment_counter" and args:
            buffer = self.generate_expression(args[0])
            return f"{buffer}.IncrementCounter()"
        if func_name == "buffer_decrement_counter" and args:
            buffer = self.generate_expression(args[0])
            return f"{buffer}.DecrementCounter()"
        if func_name == "buffer_dimensions" and len(args) >= 2:
            return self.hlsl_buffer_dimension_outputs(args[0], args[1:])
        if func_name == "buffer_dimensions" and len(args) == 1:
            return self.hlsl_buffer_element_count(args[0])
        return None

    def hlsl_buffer_dimension_outputs(self, buffer_expr, output_exprs):
        """``buffer_dimensions(b, out...)``: the outputs written through the GetDimensions ``b``'s resource declares."""
        buffer = self.generate_expression(buffer_expr)
        outputs = [self.generate_expression(arg) for arg in output_exprs]
        resource_type = self.hlsl_buffer_helper_resource_type(buffer_expr)
        resource_name = self.hlsl_resource_type_name(resource_type)
        rendered = self.hlsl_buffer_length.dimension_outputs(
            buffer,
            resource_name,
            self.map_type(resource_type) if resource_type else None,
            outputs,
        )
        if rendered is None:
            raise BufferLengthUnsupportedError(
                "directx",
                f"buffer_dimensions({buffer}, {', '.join(outputs)}) passes {len(outputs)} "
                "output(s), and the GetDimensions of "
                f"{resource_type or 'a non-buffer argument'} takes no such list",
            )
        return rendered

    def hlsl_buffer_element_count(self, buffer_expr):
        """``buffer_dimensions(b)``: the element count of a structured or typed buffer."""
        resource_type = self.hlsl_buffer_helper_resource_type(buffer_expr)
        resource_name = self.hlsl_resource_type_name(resource_type)
        rendered = self.hlsl_buffer_length.element_count(
            self.generate_expression(buffer_expr),
            resource_name,
            self.map_type(resource_type) if resource_type else None,
        )
        if rendered is None:
            raise BufferLengthUnsupportedError(
                "directx",
                f"buffer_dimensions({self.generate_expression(buffer_expr)}) needs a "
                "structured or typed buffer, whose GetDimensions reports an element "
                f"count; the argument is {resource_type or 'not a buffer resource'}",
            )
        return rendered

    def hlsl_resource_pointer_length(self, pointer_expr):
        """``length()`` of a helper's ``buffer T*`` parameter: its buffer's elements from the offset on."""
        # A buffer parameter is passed as its buffer plus an element offset, so its
        # length is that buffer's element count less the offset, not a member call.
        if not isinstance(pointer_expr, (IdentifierNode, VariableNode)):
            # An element such as data[3] is a value, not a view, even though its
            # binding is the view moved to that element.
            return None
        binding = self.hlsl_resource_pointer_binding(pointer_expr)
        if (
            binding is None
            or binding.get("kind") == "workgroup-pointer"
            or "pointer_reinterpretation" in binding
            or "byte_offset" in binding
            or not binding.get("root")
        ):
            return None
        resource_type = binding.get("resource_type")
        count = self.hlsl_buffer_length.element_count(
            binding["root"],
            self.hlsl_resource_type_name(resource_type),
            self.map_type(resource_type) if resource_type else None,
        )
        if count is None:
            return None
        offset = binding.get("offset")
        if offset is None or str(offset) in {"0", "0u"}:
            return f"int({count})"
        return f"(int({count}) - int({offset}))"

    def generate_glsl_buffer_block_length_call(self, func_expr, args):
        """``block.member.length()`` for a GLSL buffer block lowered to a byte-address buffer."""
        if not (
            isinstance(func_expr, MemberAccessNode)
            and getattr(func_expr, "member", None) == "length"
            and not args
        ):
            return None
        object_expr = getattr(
            func_expr, "object_expr", getattr(func_expr, "object", None)
        )
        if isinstance(object_expr, (IdentifierNode, VariableNode, ArrayAccessNode)):
            # 'buffer T* name' is a runtime-sized element view of a structured
            # buffer, and 'bufs[i]' an element of a resource array: their
            # length() is GetDimensions' element count.
            resource_type = self.hlsl_buffer_helper_resource_type(object_expr)
            if resource_type is not None:
                return f"int({self.hlsl_buffer_element_count(object_expr)})"
            pointer_length = self.hlsl_resource_pointer_length(object_expr)
            if pointer_length is not None:
                return pointer_length
        access = self.glsl_buffer_block_member_access(
            getattr(func_expr, "object_expr", getattr(func_expr, "object", None))
        )
        if access is None or not access.get("is_array"):
            return None
        size = access.get("array_size")
        if size is not None:
            return f"int({self.generate_expression(size)})"
        resource_name = (
            "ByteAddressBuffer" if access.get("readonly") else "RWByteAddressBuffer"
        )
        rendered = self.hlsl_buffer_length.runtime_array_length(
            access["buffer"], resource_name, access.get("offset"), access.get("stride")
        )
        if rendered is None:
            raise BufferLengthUnsupportedError(
                "directx",
                f"length() of runtime array '{access.get('member')}' has no constant "
                "offset and stride in its byte-address buffer",
            )
        return rendered

    def generate_hlsl_byteaddress_member_call(self, func_expr, args):
        if not isinstance(func_expr, MemberAccessNode):
            return None

        method_name = str(getattr(func_expr, "member", ""))
        if method_name not in {
            "Load",
            "Load2",
            "Load3",
            "Load4",
            "Store",
            "Store2",
            "Store3",
            "Store4",
        }:
            return None

        object_expr = getattr(
            func_expr, "object", getattr(func_expr, "object_expr", None)
        )
        resource_type = self.hlsl_buffer_helper_resource_type(object_expr)
        resource_name = self.hlsl_resource_type_name(resource_type)
        if method_name == "Load" and resource_name in {
            "Buffer",
            "StructuredBuffer",
            "RWBuffer",
            "RWStructuredBuffer",
            "RasterizerOrderedBuffer",
            "RasterizerOrderedStructuredBuffer",
        }:
            if not args:
                return None
            buffer = self.generate_expression(object_expr)
            rendered_args = [self.generate_expression(arg) for arg in args]
            rendered_args[0] = self.hlsl_resource_index_expression(
                object_expr,
                args[0],
                rendered_args[0],
                rendered_resource=buffer,
                resource_type=resource_type,
            )
            return f"{buffer}.{method_name}({', '.join(rendered_args)})"
        if resource_name not in {
            "ByteAddressBuffer",
            "RWByteAddressBuffer",
            "RasterizerOrderedByteAddressBuffer",
        }:
            return None
        if method_name.startswith("Store") and resource_name == "ByteAddressBuffer":
            return None
        if not args:
            return None

        buffer = self.generate_expression(object_expr)
        rendered_args = [self.generate_expression(arg) for arg in args]
        rendered_args[0] = self.hlsl_resource_index_expression(
            object_expr,
            args[0],
            rendered_args[0],
            rendered_resource=buffer,
            resource_type=resource_type,
        )
        return f"{buffer}.{method_name}({', '.join(rendered_args)})"
