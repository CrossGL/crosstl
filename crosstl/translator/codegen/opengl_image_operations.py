"""Image operations in the GLSL generator: texture gathers and compares, compile-time offsets, texel stores and storage image component kinds."""

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
)
from ..refusals import OpenGLRefusal
from ..validation import (
    expression_debug_name,
    texture_offset_argument_indices,
)
from .array_utils import split_array_type_suffix
from .glsl_shadow_lod import (
    SHADOW_LOD_EXTENSION_SAMPLERS,
    is_zero_lod_literal,
    zero_lod_compare_gradient_call,
)
from .glsl_version_requirements import (
    VARIABLE_GATHER_OFFSET,
    parse_version_line,
)
from .image_access_contracts import (
    image_format_component_kind,
    integer_coordinate_dimension_from_type_name,
    is_glsl_float_image_resource,
    is_glsl_integer_image_type,
    is_projected_texture_compare_operation,
    is_scalar_image_format,
    is_texture_compare_basic_operation,
    is_texture_compare_grad_offset_operation,
    is_texture_compare_grad_operation,
    is_texture_compare_lod_offset_operation,
    is_texture_compare_lod_operation,
    is_texture_compare_offset_operation,
    is_texture_gather_basic_operation,
    is_texture_gather_multi_offset_operation,
    is_texture_gather_offset_operation,
    is_texture_gather_single_offset_operation,
    is_two_component_image_format,
    operation_argument_type_error,
    operation_dimension_argument_error,
    projected_texture_compare_reference,
    storage_image_atomic_zero_value,
    storage_image_format_store_constructor,
    storage_image_store_constructors,
    storage_image_two_component_store_expression,
    storage_image_zero_values,
    texture_compare_argument_error,
    texture_compare_coordinate_error,
    texture_compare_extra_argument_count_error,
    texture_compare_offset_capability_error,
    texture_compare_projected_coordinate_error,
    texture_compare_projected_lod_array_error,
    texture_coordinate_arguments_error,
    texture_gather_capability_error,
    texture_gather_compare_extra_argument_count_error,
    texture_gather_component_count_error,
    texture_gather_component_literal_error,
    texture_gather_offset_argument_count_error,
    texture_gather_offset_capability_error,
    texture_gather_offsets_argument_count_error,
    texture_gather_operation_error,
    unsupported_texture_compare_operation_error,
    unsupported_texture_compare_scalar_expression,
    unsupported_texture_gather_compare_call_expression,
)
from .storage_image_texels import (
    leading_components_suffix,
    texel_store_value,
    texel_vector_type,
)


class OpenGLImageOperationMixin:
    """Image operation members of ``GLSLCodeGen``."""

    def generate_texture_gather_call(self, func_name, args):
        parts = self.texture_call_parts(args)
        if parts is None:
            return self.unsupported_texture_gather_call(
                func_name, texture_coordinate_arguments_error()
            )

        texture_name, coord, extra_args = parts
        texture_type = self.texture_resource_type(args[0])
        if self.is_multisample_texture_resource_type(texture_type):
            return self.unsupported_multisample_texture_call(func_name, texture_type)
        if is_texture_gather_basic_operation(
            func_name
        ) and not self.texture_gather_supported(texture_type):
            return self.unsupported_texture_gather_call(
                func_name, texture_gather_capability_error()
            )
        if is_texture_gather_offset_operation(
            func_name
        ) and not self.texture_gather_offset_supported(texture_type):
            return self.unsupported_texture_gather_call(
                func_name, texture_gather_offset_capability_error()
            )

        offset_args = []
        component_arg = None

        if is_texture_gather_basic_operation(func_name):
            if len(extra_args) > 1:
                return self.unsupported_texture_gather_call(
                    func_name, texture_gather_component_count_error()
                )
            if extra_args:
                component_arg = extra_args[0]
        elif is_texture_gather_single_offset_operation(func_name):
            if len(extra_args) not in {1, 2}:
                return self.unsupported_texture_gather_call(
                    func_name, texture_gather_offset_argument_count_error()
                )
            offset_args = [extra_args[0]]
            if len(extra_args) == 2:
                component_arg = extra_args[1]
        elif is_texture_gather_multi_offset_operation(func_name):
            offset_args, component_arg = self.texture_gather_offsets_args(extra_args)
            if offset_args is None:
                return self.unsupported_texture_gather_call(
                    func_name, texture_gather_offsets_argument_count_error()
                )
        else:
            return self.unsupported_texture_gather_call(
                func_name, texture_gather_operation_error()
            )

        # Multi-offset gathers are emitted as one textureGatherOffset per offset.
        if offset_args and not self.glsl_gather_offsets_are_expressible(offset_args):
            return self.unsupported_dynamic_texel_offset_gather_call(
                func_name, texture_type
            )

        component = self.texture_gather_component_value(component_arg)
        if component is not None:
            if component not in {0, 1, 2, 3}:
                return self.unsupported_texture_gather_call(
                    func_name, texture_gather_component_literal_error()
                )
            if is_texture_gather_multi_offset_operation(func_name):
                return self.texture_gather_offsets_expression(
                    texture_name, coord, offset_args, component, texture_type
                )
            offset_arg = (
                self.generate_expression(offset_args[0]) if offset_args else None
            )
            function_name = (
                "textureGatherOffset"
                if is_texture_gather_single_offset_operation(func_name)
                else "textureGather"
            )
            return self.texture_gather_call_expression(
                function_name, texture_name, coord, offset_arg, component
            )

        if component_arg is None:
            if is_texture_gather_multi_offset_operation(func_name):
                return self.texture_gather_offsets_expression(
                    texture_name, coord, offset_args, None, texture_type
                )
            offset_arg = (
                self.generate_expression(offset_args[0]) if offset_args else None
            )
            function_name = (
                "textureGatherOffset"
                if is_texture_gather_single_offset_operation(func_name)
                else "textureGather"
            )
            return self.texture_gather_call_expression(
                function_name, texture_name, coord, offset_arg
            )

        component_expr = self.generate_expression(component_arg)
        if is_texture_gather_multi_offset_operation(func_name):
            return self.texture_gather_dynamic_component_expression(
                lambda option: self.texture_gather_offsets_expression(
                    texture_name, coord, offset_args, option, texture_type
                ),
                component_expr,
            )

        offset_arg = self.generate_expression(offset_args[0]) if offset_args else None
        function_name = (
            "textureGatherOffset"
            if is_texture_gather_single_offset_operation(func_name)
            else "textureGather"
        )
        return self.texture_gather_dynamic_component_expression(
            lambda option: self.texture_gather_call_expression(
                function_name, texture_name, coord, offset_arg, option
            ),
            component_expr,
        )

    def texture_compare_coordinate(self, texture_type, coord, compare):
        texture_type = self.resource_base_type(texture_type)
        if texture_type == "samplerCubeArrayShadow":
            return None
        if texture_type == "sampler1DShadow":
            # GLSL 4.60 section 8.9.2: texture(sampler1DShadow, vec3 P)
            # takes (u, unused, Dref); the second component is unused.
            return f"vec3({coord}, 0.0, {compare})"
        constructor = (
            "vec4"
            if texture_type in {
                "sampler2DArrayShadow",
                "samplerCubeShadow",
            }
            else "vec3"
        )
        return f"{constructor}({coord}, {compare})"

    def texture_compare_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["compare_offset"]

    def shadow_lod_extension_compare_call(
        self, texture_type, texture_name, coord, compare, lod_arg, offset=None
    ):
        """An explicit-level compare through GL_EXT_texture_shadow_lod, or None when it adds no overload."""
        base_type = self.resource_base_type(texture_type)
        has_offset = SHADOW_LOD_EXTENSION_SAMPLERS.get(base_type)
        if not self.GLSL_SHADOW_LOD_EXTENSION_SUPPORTED or has_offset is None:
            return None
        if offset is not None and not has_offset:
            return None
        lod = self.generate_expression(lod_arg)
        if base_type == "samplerCubeArrayShadow":
            call = f"textureLod({texture_name}, {coord}, {compare}, {lod})"
        else:
            compare_coord = self.texture_compare_coordinate(base_type, coord, compare)
            if compare_coord is None:
                return None
            if offset is None:
                call = f"textureLod({texture_name}, {compare_coord}, {lod})"
            else:
                call = f"textureLodOffset({texture_name}, {compare_coord}, {lod}, {offset})"
        self.glsl_shadow_lod_extension_used = True
        return call

    def zero_lod_compare_gradient_call(
        self, texture_type, texture_name, coord, compare, lod_arg, offset=None
    ):
        """A level-zero compare on a shadow sampler without textureLod, as textureGrad (glsl_shadow_lod)."""
        base_type = self.resource_base_type(texture_type)
        if not is_zero_lod_literal(lod_arg):
            return None
        compare_coord = self.texture_compare_coordinate(base_type, coord, compare)
        if compare_coord is None:
            return None
        return zero_lod_compare_gradient_call(
            base_type, texture_name, compare_coord, lod_arg, offset
        )

    def texture_compare_lod_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["compare_lod"]

    def texture_compare_grad_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["compare_grad"]

    def texture_compare_lod_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["compare_lod_offset"]

    def texture_compare_grad_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["compare_grad_offset"]

    def texture_compare_projected_coordinate(
        self, texture_type, coord_arg, coord, compare
    ):
        texture_type = self.resource_base_type(texture_type)
        coord_type = self.resource_base_type(self.expression_result_type(coord_arg))

        if texture_type == "sampler2DShadow":
            if coord_type in {"vec3", "float3"}:
                divisor = self.vector_component(coord, "z")
            elif coord_type in {"vec4", "float4"}:
                divisor = self.vector_component(coord, "w")
            else:
                return None
            projected_coord = f"{self.vector_component(coord, 'xy')} / {divisor}"
            reference = projected_texture_compare_reference(compare, divisor)
            return f"vec3({projected_coord}, {reference})"

        if texture_type == "samplerCubeShadow":
            if coord_type not in {"vec4", "float4"}:
                return None
            divisor = self.vector_component(coord, "w")
            projected_coord = f"{self.vector_component(coord, 'xyz')} / {divisor}"
            reference = projected_texture_compare_reference(compare, divisor)
            return f"vec4({projected_coord}, {reference})"

        if texture_type != "sampler2DArrayShadow" or coord_type not in {
            "vec4",
            "float4",
        }:
            return None

        divisor = self.vector_component(coord, "w")
        projected_coord = f"{self.vector_component(coord, 'xy')} / {divisor}"
        layer = self.vector_component(coord, "z")
        reference = projected_texture_compare_reference(compare, divisor)
        return f"vec4({projected_coord}, {layer}, {reference})"

    def unsupported_texture_compare_call(self, func_name, reason):
        return unsupported_texture_compare_scalar_expression("GLSL", func_name, reason)

    def glsl_integer_vector_type_names(self):
        return {
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
            # CrossGL also spells integer vectors as HLSL does (the SPIR-V importer's spelling).
            "int2",
            "int3",
            "int4",
            "uint2",
            "uint3",
            "uint4",
        }

    def glsl_compile_time_int_constants(self):
        if self.current_compile_time_int_constants is not None:
            return self.current_compile_time_int_constants
        return self.literal_int_constants

    def glsl_compile_time_int_vector_constants(self):
        if self.current_compile_time_int_vector_constants is not None:
            return self.current_compile_time_int_vector_constants
        return self.literal_int_vector_constants

    def glsl_compile_time_int_vector_array_constants(self):
        if self.current_compile_time_int_vector_array_constants is not None:
            return self.current_compile_time_int_vector_array_constants
        return self.literal_int_vector_array_constants

    def glsl_integer_vector_literal_value(
        self,
        expr,
        int_constants=None,
        vector_constants=None,
        vector_array_constants=None,
    ):
        int_constants = (
            self.glsl_compile_time_int_constants()
            if int_constants is None
            else int_constants
        )
        if isinstance(expr, ArrayAccessNode):
            vector_array_constants = (
                self.glsl_compile_time_int_vector_array_constants()
                if vector_array_constants is None
                else vector_array_constants
            )
            array_name = self.expression_name(
                getattr(expr, "array", getattr(expr, "array_expr", None))
            )
            values = vector_array_constants.get(array_name)
            index = self.literal_int_value(
                getattr(expr, "index", getattr(expr, "index_expr", None)),
                int_constants,
            )
            if (
                values is not None
                and isinstance(index, int)
                and not isinstance(index, bool)
                and 0 <= index < len(values)
            ):
                return values[index]
            return None

        vector_constants = (
            self.glsl_compile_time_int_vector_constants()
            if vector_constants is None
            else vector_constants
        )
        expr_name = self.expression_name(expr)
        if expr_name in vector_constants:
            return vector_constants[expr_name]

        class_name = expr.__class__.__name__ if expr is not None else ""
        if "FunctionCall" in class_name:
            constructor = getattr(expr, "function", getattr(expr, "name", None))
            arguments = getattr(expr, "arguments", getattr(expr, "args", []))
        elif "Constructor" in class_name:
            constructor = getattr(expr, "constructor_type", None)
            arguments = getattr(expr, "arguments", [])
        else:
            return None

        constructor_name = self.type_name_string(constructor)
        if constructor_name not in self.glsl_integer_vector_type_names():
            return None

        arguments = list(arguments or [])
        if not arguments:
            return None

        values = []
        for argument in arguments:
            literal_argument = self.literal_int_value(argument, int_constants)
            if not isinstance(literal_argument, int) or isinstance(
                literal_argument, bool
            ):
                return None
            values.append(literal_argument)

        vector_width = int(constructor_name[-1])
        if len(values) == 1:
            return tuple(values * vector_width)
        if len(values) == vector_width:
            return tuple(values)
        return None

    def collect_literal_int_vector_constants_from_variables(
        self, variables, int_constants
    ):
        vector_constants = {}
        for variable in variables or []:
            name = getattr(variable, "name", None)
            if not name or "const" not in getattr(variable, "qualifiers", []):
                continue
            var_type = self.type_name_string(
                getattr(variable, "var_type", getattr(variable, "vtype", None))
            )
            if var_type not in self.glsl_integer_vector_type_names():
                continue
            value = self.glsl_integer_vector_literal_value(
                getattr(variable, "initial_value", None),
                int_constants,
                vector_constants,
            )
            if value is not None:
                vector_constants[name] = value
        return vector_constants

    def glsl_integer_vector_array_literal_value(
        self, expr, int_constants=None, vector_constants=None
    ):
        if not isinstance(expr, ArrayLiteralNode):
            return None

        int_constants = (
            self.glsl_compile_time_int_constants()
            if int_constants is None
            else int_constants
        )
        vector_constants = (
            self.glsl_compile_time_int_vector_constants()
            if vector_constants is None
            else vector_constants
        )
        values = []
        for element in expr.elements:
            value = self.glsl_integer_vector_literal_value(
                element,
                int_constants,
                vector_constants,
            )
            if value is None:
                return None
            values.append(value)
        return tuple(values)

    def collect_literal_int_vector_array_constants_from_variables(
        self, variables, int_constants, vector_constants
    ):
        vector_array_constants = {}
        for variable in variables or []:
            name = getattr(variable, "name", None)
            if not name or "const" not in getattr(variable, "qualifiers", []):
                continue

            var_type = getattr(variable, "var_type", getattr(variable, "vtype", None))
            if "ArrayType" not in var_type.__class__.__name__:
                continue
            element_type = self.type_name_string(
                getattr(var_type, "element_type", None)
            )
            if element_type not in self.glsl_integer_vector_type_names():
                continue

            value = self.glsl_integer_vector_array_literal_value(
                getattr(variable, "initial_value", None),
                int_constants,
                vector_constants,
            )
            if value is None:
                continue

            size = self.literal_int_value(
                getattr(var_type, "size", None), int_constants
            )
            if size is not None and len(value) != size:
                continue
            vector_array_constants[name] = value
        return vector_array_constants

    def update_compile_time_offset_constants(self, variable, var_type):
        name = getattr(variable, "name", None)
        if not name:
            return
        self.current_compile_time_int_constants.pop(name, None)
        self.current_compile_time_int_vector_constants.pop(name, None)
        self.current_compile_time_int_vector_array_constants.pop(name, None)
        if "const" not in getattr(variable, "qualifiers", []):
            return

        initial_value = getattr(variable, "initial_value", None)
        raw_var_type = var_type
        var_type = self.type_name_string(var_type)
        scalar_integer_types = {
            "bool",
            "int",
            "uint",
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int32",
            "uint32",
            "int64",
            "uint64",
        }
        narrow_contract = self.glsl_narrow_integer_contract(raw_var_type)
        if var_type in scalar_integer_types or (
            narrow_contract is not None and narrow_contract["width"] == 1
        ):
            value = self.literal_int_value(
                initial_value, self.current_compile_time_int_constants
            )
            if isinstance(value, int) and not isinstance(value, bool):
                if narrow_contract is not None and narrow_contract["width"] == 1:
                    value = self.glsl_coerce_compile_time_integer(
                        value, narrow_contract
                    )
                self.current_compile_time_int_constants[name] = value
            return

        if var_type not in self.glsl_integer_vector_type_names():
            if "[" not in str(var_type):
                return
            base_type, _ = split_array_type_suffix(var_type)
            if base_type not in self.glsl_integer_vector_type_names():
                return
            value = self.glsl_integer_vector_array_literal_value(
                initial_value,
                self.current_compile_time_int_constants,
                self.current_compile_time_int_vector_constants,
            )
            if value is None:
                return
            size = self.literal_int_value(
                getattr(raw_var_type, "size", None),
                self.current_compile_time_int_constants,
            )
            if size is not None and len(value) != size:
                return
            self.current_compile_time_int_vector_array_constants[name] = value
            return

        value = self.glsl_integer_vector_literal_value(
            initial_value,
            self.current_compile_time_int_constants,
            self.current_compile_time_int_vector_constants,
        )
        if value is not None:
            self.current_compile_time_int_vector_constants[name] = value

    def glsl_narrow_compile_time_integer_initializer(self, variable, var_type):
        if "const" not in getattr(variable, "qualifiers", []):
            return None
        contract = self.glsl_narrow_integer_contract(var_type)
        if contract is None or contract["width"] != 1:
            return None
        value = self.current_compile_time_int_constants.get(
            getattr(variable, "name", None)
        )
        if value is None:
            return None
        suffix = "" if contract["signed"] else "u"
        return f"{value}{suffix}"

    def glsl_coerce_compile_time_integer(self, value, contract):
        bits = int(contract["bits"])
        modulus = 1 << bits
        coerced = int(value) % modulus
        if contract["signed"] and coerced >= (1 << (bits - 1)):
            coerced -= modulus
        return coerced

    def glsl_texel_offset_is_compile_time_constant(self, offset_arg):
        offset_type = self.type_name_string(self.expression_result_type(offset_arg))
        scalar_integer_types = {"int", "uint"}
        vector_integer_types = self.glsl_integer_vector_type_names()
        int_constants = self.glsl_compile_time_int_constants()

        literal_value = self.literal_int_value(offset_arg, int_constants)
        if isinstance(literal_value, int) and not isinstance(literal_value, bool):
            return offset_type is None or offset_type in scalar_integer_types

        if offset_type is not None and offset_type not in vector_integer_types:
            return False
        return (
            self.glsl_integer_vector_literal_value(offset_arg, int_constants)
            is not None
        )

    def glsl_version_supports_variable_gather_offsets(self):
        """Return whether ``textureGatherOffset`` accepts a non-constant offset.

        GLSL 4.00 and GLSL ES 3.20 let the gather offset be variable (GLSL 4.60 and
        GLSL ES 3.20, section 8.9.4); ``textureOffset`` and the other offset forms
        still require constant expressions.
        """
        number, profile = parse_version_line(
            getattr(self, "current_glsl_version_line", None)
        )
        if number is None:
            return True
        return VARIABLE_GATHER_OFFSET.satisfied_by(
            number, profile == "es", self.current_glsl_extensions
        )

    def glsl_gather_offsets_are_expressible(self, offset_args):
        """Return whether ``textureGatherOffset`` can take these offsets.

        A non-constant offset below the version that allows one is a version requirement, so the program is
        emitted at that version; a profile whose version is fixed refuses it.
        """
        if self.glsl_version_supports_variable_gather_offsets():
            return True
        if self.glsl_texel_offsets_are_compile_time_constants(offset_args):
            return True
        if self.glsl_version_line_is_fixed():
            return False
        self.glsl_emission_requirements.append(VARIABLE_GATHER_OFFSET)
        return True

    def glsl_texel_offsets_are_compile_time_constants(self, offset_args):
        return all(
            self.glsl_texel_offset_is_compile_time_constant(offset_arg)
            for offset_arg in offset_args
        )

    def glsl_texture_call_offsets_are_compile_time_constants(self, func_name, args):
        if self.glsl_texture_offset_non_const_enabled():
            return True

        offset_indices = texture_offset_argument_indices(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        return self.glsl_texel_offsets_are_compile_time_constants(
            args[offset_index] for offset_index in offset_indices
        )

    def dynamic_texel_offset_reason(self):
        return "texel offsets must be compile-time integer constants"

    def unsupported_dynamic_texel_offset_compare_call(self, func_name):
        return self.unsupported_texture_compare_call(
            func_name, self.dynamic_texel_offset_reason()
        )

    def unsupported_dynamic_texel_offset_sample_call(self, func_name):
        return self.unsupported_texture_sample_offset_call(
            func_name, self.dynamic_texel_offset_reason()
        )

    def unsupported_dynamic_texel_offset_projected_call(self, func_name, texture_type):
        return self.unsupported_texture_projected_call(
            func_name, self.dynamic_texel_offset_reason(), texture_type
        )

    def unsupported_dynamic_texel_offset_gather_call(self, func_name, texture_type):
        zero_value = self.texture_sample_vector_zero_value(texture_type)
        return self.refuse_lowering(
            f"texture gather: {func_name} {self.dynamic_texel_offset_reason()}",
            zero_value,
        )

    def unsupported_dynamic_texel_offset_gather_compare_call(self, func_name):
        return self.unsupported_texture_gather_compare_call(
            func_name, self.dynamic_texel_offset_reason()
        )

    def unsupported_dynamic_texel_fetch_offset_call(self, func_name, texture_type):
        zero_value = self.texture_sample_vector_zero_value(texture_type)
        return self.refuse_lowering(
            f"texel fetch offset: {func_name} {self.dynamic_texel_offset_reason()}",
            zero_value,
        )

    def generate_texture_compare_call(self, func_name, args):
        parts = self.texture_call_parts(args)
        if parts is None:
            return self.unsupported_texture_compare_call(
                func_name, texture_coordinate_arguments_error()
            )

        texture_name, coord, extra_args = parts
        if not extra_args:
            return self.unsupported_texture_compare_call(
                func_name, texture_compare_argument_error()
            )

        # Struct members and other non-identifier resources resolve through their expression type.
        texture_type = self.texture_argument_resource_type(args[0])
        if self.is_multisample_texture_resource_type(texture_type):
            return self.unsupported_multisample_texture_compare_call(
                func_name, texture_type
            )

        compare = self.generate_expression(extra_args[0])
        if is_projected_texture_compare_operation(func_name):
            coord_index = 2 if self.is_explicit_sampler_argument(args) else 1
            compare_coord = self.texture_compare_projected_coordinate(
                texture_type, args[coord_index], coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_projected_coordinate_error("GLSL")
                )

            if is_texture_compare_basic_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                return f"texture({texture_name}, {compare_coord})"

            if is_texture_compare_offset_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if not self.texture_compare_offset_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name, texture_compare_offset_capability_error("GLSL")
                    )
                offset_arg = extra_args[1]
                if not self.glsl_texel_offset_is_compile_time_constant(offset_arg):
                    return self.unsupported_dynamic_texel_offset_compare_call(func_name)
                offset = self.generate_expression(offset_arg)
                return f"textureOffset({texture_name}, {compare_coord}, {offset})"

            if is_texture_compare_lod_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if self.resource_base_type(texture_type) == "sampler2DArrayShadow":
                    return self.unsupported_texture_compare_call(
                        func_name, texture_compare_projected_lod_array_error()
                    )
                if not self.texture_compare_lod_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name, "explicit LOD requires 2D shadow samplers"
                    )
                lod = self.generate_expression(extra_args[1])
                return f"textureLod({texture_name}, {compare_coord}, {lod})"

            if is_texture_compare_lod_offset_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if self.resource_base_type(texture_type) == "sampler2DArrayShadow":
                    return self.unsupported_texture_compare_call(
                        func_name, texture_compare_projected_lod_array_error()
                    )
                if not self.texture_compare_lod_offset_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name, "explicit LOD offsets require 2D shadow samplers"
                    )
                lod = self.generate_expression(extra_args[1])
                offset_arg = extra_args[2]
                if not self.glsl_texel_offset_is_compile_time_constant(offset_arg):
                    return self.unsupported_dynamic_texel_offset_compare_call(func_name)
                offset = self.generate_expression(offset_arg)
                return (
                    f"textureLodOffset({texture_name}, {compare_coord}, "
                    f"{lod}, {offset})"
                )

            if is_texture_compare_grad_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if not self.texture_compare_grad_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name,
                        "explicit gradients require 2D, 2D-array, or cube shadow samplers",
                    )
                ddx = self.generate_expression(extra_args[1])
                ddy = self.generate_expression(extra_args[2])
                return f"textureGrad({texture_name}, {compare_coord}, {ddx}, {ddy})"

            if is_texture_compare_grad_offset_operation(func_name):
                count_error = texture_compare_extra_argument_count_error(
                    func_name, len(extra_args)
                )
                if count_error:
                    return self.unsupported_texture_compare_call(func_name, count_error)
                if not self.texture_compare_grad_offset_supported(texture_type):
                    return self.unsupported_texture_compare_call(
                        func_name,
                        "explicit gradient offsets require 2D or 2D-array shadow samplers",
                    )
                ddx = self.generate_expression(extra_args[1])
                ddy = self.generate_expression(extra_args[2])
                offset_arg = extra_args[3]
                if not self.glsl_texel_offset_is_compile_time_constant(offset_arg):
                    return self.unsupported_dynamic_texel_offset_compare_call(func_name)
                offset = self.generate_expression(offset_arg)
                return (
                    f"textureGradOffset({texture_name}, {compare_coord}, "
                    f"{ddx}, {ddy}, {offset})"
                )

            return self.unsupported_texture_compare_call(
                func_name, unsupported_texture_compare_operation_error(projected=True)
            )

        if is_texture_compare_basic_operation(func_name):
            if texture_type == "samplerCubeArrayShadow":
                return f"texture({texture_name}, {coord}, {compare})"
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_coordinate_error()
                )
            return f"texture({texture_name}, {compare_coord})"

        if is_texture_compare_offset_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_offset_capability_error("GLSL")
                )
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_coordinate_error()
                )
            offset_arg = extra_args[1]
            if not self.glsl_texel_offset_is_compile_time_constant(offset_arg):
                return self.unsupported_dynamic_texel_offset_compare_call(func_name)
            offset = self.generate_expression(offset_arg)
            return f"textureOffset({texture_name}, {compare_coord}, {offset})"

        if is_texture_compare_lod_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_lod_supported(texture_type):
                gradient_call = self.zero_lod_compare_gradient_call(
                    texture_type, texture_name, coord, compare, extra_args[1]
                ) or self.shadow_lod_extension_compare_call(
                    texture_type, texture_name, coord, compare, extra_args[1]
                )
                if gradient_call is not None:
                    return gradient_call
                return self.unsupported_texture_compare_call(
                    func_name, "explicit LOD requires 2D shadow samplers"
                )
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_coordinate_error()
                )
            lod = self.generate_expression(extra_args[1])
            return f"textureLod({texture_name}, {compare_coord}, {lod})"

        if is_texture_compare_lod_offset_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_lod_offset_supported(texture_type):
                offset_arg = extra_args[2]
                gradient_call = None
                if self.glsl_texel_offset_is_compile_time_constant(offset_arg):
                    offset = self.generate_expression(offset_arg)
                    gradient_call = self.zero_lod_compare_gradient_call(
                        texture_type,
                        texture_name,
                        coord,
                        compare,
                        extra_args[1],
                        offset,
                    ) or self.shadow_lod_extension_compare_call(
                        texture_type,
                        texture_name,
                        coord,
                        compare,
                        extra_args[1],
                        offset,
                    )
                if gradient_call is not None:
                    return gradient_call
                return self.unsupported_texture_compare_call(
                    func_name, "explicit LOD offsets require 2D shadow samplers"
                )
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_coordinate_error()
                )
            lod = self.generate_expression(extra_args[1])
            offset_arg = extra_args[2]
            if not self.glsl_texel_offset_is_compile_time_constant(offset_arg):
                return self.unsupported_dynamic_texel_offset_compare_call(func_name)
            offset = self.generate_expression(offset_arg)
            return f"textureLodOffset({texture_name}, {compare_coord}, {lod}, {offset})"

        if is_texture_compare_grad_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_grad_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name,
                    "explicit gradients require 2D, 2D-array, or cube shadow samplers",
                )
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_coordinate_error()
                )
            ddx = self.generate_expression(extra_args[1])
            ddy = self.generate_expression(extra_args[2])
            return f"textureGrad({texture_name}, {compare_coord}, {ddx}, {ddy})"

        if is_texture_compare_grad_offset_operation(func_name):
            count_error = texture_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_compare_call(func_name, count_error)
            if not self.texture_compare_grad_offset_supported(texture_type):
                return self.unsupported_texture_compare_call(
                    func_name,
                    "explicit gradient offsets require 2D or 2D-array shadow samplers",
                )
            compare_coord = self.texture_compare_coordinate(
                texture_type, coord, compare
            )
            if compare_coord is None:
                return self.unsupported_texture_compare_call(
                    func_name, texture_compare_coordinate_error()
                )
            ddx = self.generate_expression(extra_args[1])
            ddy = self.generate_expression(extra_args[2])
            offset_arg = extra_args[3]
            if not self.glsl_texel_offset_is_compile_time_constant(offset_arg):
                return self.unsupported_dynamic_texel_offset_compare_call(func_name)
            offset = self.generate_expression(offset_arg)
            return (
                f"textureGradOffset({texture_name}, {compare_coord}, "
                f"{ddx}, {ddy}, {offset})"
            )

        return None

    def unsupported_texture_gather_compare_call(self, func_name, reason):
        return unsupported_texture_gather_compare_call_expression(
            "GLSL", func_name, reason
        )

    def texture_gather_compare_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather_compare_offset"]

    def validate_texture_gather_compare_offsets_argument(
        self, func_name, texture_type, offset_arg
    ):
        offset_type = self.expression_result_type(offset_arg)
        if offset_type is None:
            return
        if not self.is_integer_coordinate_type(offset_type):
            raise OpenGLRefusal(
                operation_argument_type_error(
                    "OpenGL",
                    "resource",
                    func_name,
                    "an integer",
                    "offset",
                    expression_debug_name(offset_arg),
                    self.type_name_string(offset_type),
                )
            )
        expected_dimension = self.resource_offset_dimension(
            "textureGatherCompareOffset", texture_type
        )
        offset_dimension = integer_coordinate_dimension_from_type_name(
            self.type_name_string(offset_type), self.map_type
        )
        if offset_dimension is None or offset_dimension == expected_dimension:
            return
        raise OpenGLRefusal(
            operation_dimension_argument_error(
                "OpenGL",
                "resource",
                func_name,
                expected_dimension,
                "integer",
                "offset",
                self.resource_base_type(texture_type),
                expression_debug_name(offset_arg),
                self.type_name_string(offset_type),
            )
        )

    def generate_texture_gather_compare_offsets_call(self, func_name, args):
        parts = self.texture_call_parts(args)
        if parts is None:
            return self.unsupported_texture_gather_compare_call(
                func_name, texture_coordinate_arguments_error()
            )

        texture_name, coord, extra_args = parts
        texture_type = self.texture_resource_type(args[0])
        if self.is_multisample_texture_resource_type(texture_type):
            return self.unsupported_multisample_texture_gather_compare_call(
                func_name, texture_type
            )

        compare_arg, offset_args = self.texture_gather_compare_offsets_args(extra_args)
        if offset_args is None:
            return self.unsupported_texture_gather_compare_call(
                func_name, "requires compare and typed offsets array or four offsets"
            )
        if not self.texture_gather_compare_offset_supported(texture_type):
            return self.unsupported_texture_gather_compare_call(
                func_name, texture_compare_offset_capability_error("GLSL")
            )

        compare_type = self.expression_result_type(compare_arg)
        if compare_type is not None and not self.is_scalar_floating_type(compare_type):
            raise OpenGLRefusal(
                operation_argument_type_error(
                    "OpenGL",
                    "texture compare",
                    func_name,
                    "a scalar floating",
                    "compare",
                    expression_debug_name(compare_arg),
                    self.type_name_string(compare_type),
                )
            )
        for offset_arg in offset_args:
            self.validate_texture_gather_compare_offsets_argument(
                func_name, texture_type, offset_arg
            )
        if not self.glsl_gather_offsets_are_expressible(offset_args):
            return self.unsupported_dynamic_texel_offset_gather_compare_call(func_name)

        compare = self.generate_expression(compare_arg)
        return self.texture_gather_compare_offsets_expression(
            texture_name, coord, compare, offset_args
        )

    def generate_texture_gather_compare_call(self, func_name, args):
        parts = self.texture_call_parts(args)
        if parts is None:
            return self.unsupported_texture_gather_compare_call(
                func_name, texture_coordinate_arguments_error()
            )

        texture_name, coord, extra_args = parts
        if not extra_args:
            return self.unsupported_texture_gather_compare_call(
                func_name, texture_compare_argument_error()
            )

        texture_type = self.texture_resource_type(args[0])
        if self.is_multisample_texture_resource_type(texture_type):
            return self.unsupported_multisample_texture_gather_compare_call(
                func_name, texture_type
            )

        compare = self.generate_expression(extra_args[0])
        if func_name == "textureGatherCompare":
            count_error = texture_gather_compare_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_gather_compare_call(
                    func_name, count_error
                )
            return f"textureGather({texture_name}, {coord}, {compare})"

        count_error = texture_gather_compare_extra_argument_count_error(
            func_name, len(extra_args)
        )
        if count_error:
            return self.unsupported_texture_gather_compare_call(func_name, count_error)
        if not self.texture_gather_compare_offset_supported(texture_type):
            return self.unsupported_texture_gather_compare_call(
                func_name, texture_compare_offset_capability_error("GLSL")
            )
        offset_arg = extra_args[1]
        if not self.glsl_gather_offsets_are_expressible([offset_arg]):
            return self.unsupported_dynamic_texel_offset_gather_compare_call(func_name)
        offset = self.generate_expression(offset_arg)
        return f"textureGatherOffset({texture_name}, {coord}, {compare}, {offset})"

    def image_resource_format(self, texture_arg):
        return self.resource_metadata(
            texture_arg,
            self.current_image_format_parameters,
            self.image_variable_formats,
        )

    def image_resource_access(self, texture_arg):
        return self.resource_metadata(
            texture_arg,
            self.current_image_access_parameters,
            self.image_variable_accesses,
        )

    def resource_metadata(self, resource_arg, parameter_metadata, variable_metadata):
        resource_name = self.expression_name(resource_arg)
        if not resource_name:
            return None
        parameter_value = parameter_metadata.get(resource_name)
        if parameter_value is not None:
            return parameter_value
        emitted_name = self.current_identifier_aliases.get(resource_name)
        if emitted_name and emitted_name != resource_name:
            return parameter_metadata.get(
                emitted_name,
                variable_metadata.get(emitted_name),
            )
        return variable_metadata.get(resource_name)

    def image_atomic_parameter_requires_diagnostic(self, texture_arg):
        texture_name = self.expression_name(texture_arg)
        if texture_name is None:
            return False
        if texture_name in self.current_resource_aliases:
            return False
        return (
            texture_name in self.current_texture_parameters
            and texture_name not in self.texture_variable_types
        )

    def structured_buffer_resource_access(self, buffer_arg):
        buffer_name = self.expression_name(buffer_arg)
        if not buffer_name:
            return None
        parameter_access = self.current_structured_buffer_access_parameters.get(
            buffer_name
        )
        if parameter_access is not None:
            return parameter_access
        emitted_name = self.current_identifier_aliases.get(buffer_name)
        if emitted_name and emitted_name != buffer_name:
            return self.current_structured_buffer_access_parameters.get(
                emitted_name,
                self.structured_buffer_variable_accesses.get(emitted_name),
            )
        return self.structured_buffer_variable_accesses.get(buffer_name)

    def image_atomic_zero_value(self, texture_type, image_format):
        component_kind = image_format_component_kind(image_format)
        if component_kind is None:
            texture_type = self.resource_base_type(texture_type)
            if texture_type and texture_type.startswith("uimage"):
                component_kind = "uint"
            elif texture_type and texture_type.startswith("image"):
                component_kind = "float"
            else:
                component_kind = "int"
        return storage_image_atomic_zero_value(component_kind)

    def image_atomic_parameter_fallback_call(self, func_name, args):
        texture_type = self.texture_argument_resource_type(args[0])
        image_format = self.image_resource_format(args[0])
        zero_value = self.image_atomic_zero_value(texture_type, image_format)
        texture_type = self.resource_base_type(texture_type) or expression_debug_name(
            args[0]
        )
        # GLSL image parameters carry no format qualifier, so image atomics need
        # a clone specialized for a concrete resource.
        return self.refuse_in_generic_resource_helper(
            f"image atomic resource call: {func_name} on {texture_type} parameter",
            zero_value,
        )

    def refuse_in_generic_resource_helper(self, reason, placeholder):
        """Refuse a construct of a helper whose resource parameters are bound by specialization.

        The generic copy of such a helper is still emitted, while every call that
        passes a concrete resource calls a specialized clone. A construct only the
        clones can lower is refused if emitted code still calls the generic copy.
        """
        key = self.current_glsl_function_key
        source_name = self.current_glsl_source_function_name
        if (
            key
            and key[1] == source_name
            and source_name in self.glsl_resource_specialized_source_names
        ):
            self.glsl_generic_helper_refusals.setdefault(source_name, []).append(reason)
            return placeholder
        return self.refuse_lowering(reason, placeholder)

    def finish_generic_resource_helper_refusals(self, code):
        """Drop the dead generic copies of specialized helpers and refuse the live ones.

        A generic copy is dead when no emitted function outside the other dead
        generic copies calls it. A dead copy that needed a refusal loses its body;
        its prototype stays, which GLSL accepts for a function that is never
        called. A refusal in a generic copy that is still called stands.
        """
        dead = set(self.glsl_resource_specialized_source_names)
        while True:
            called = {
                target_key[1]
                for source_key, target_keys in self.glsl_function_call_graph.items()
                if source_key[1] not in dead
                for target_key in target_keys
            }
            live = dead & called
            if not live:
                break
            dead -= live
        for name, reasons in sorted(self.glsl_generic_helper_refusals.items()):
            function_code = self.glsl_generic_helper_code.get(name)
            if name in dead and function_code and code.count(function_code) == 1:
                code = code.replace(function_code, "", 1)
                continue
            for reason in reasons:
                self.refuse_lowering(reason)
        return code

    def generate_image_atomic_call(self, func_name, args):
        if self.image_atomic_parameter_requires_diagnostic(args[0]):
            return self.image_atomic_parameter_fallback_call(func_name, args)
        rendered = [self.generate_expression(arg) for arg in args]
        if len(args) >= 2:
            rendered[1] = self.glsl_signed_coordinate_expression(args[1], rendered[1])
        return "{}({})".format(func_name, ", ".join(rendered))

    def is_integer_image_type(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return is_glsl_integer_image_type(texture_type) or texture_type in {
            "iimageCube",
            "iimageCubeArray",
            "uimageCube",
            "uimageCubeArray",
        }

    def is_scalar_image_format(self, image_format):
        return is_scalar_image_format(image_format)

    def is_two_component_image_format(self, image_format):
        return is_two_component_image_format(image_format)

    def is_scalar_integer_image_resource(self, texture_type, image_format):
        if image_format is not None:
            return self.is_scalar_image_format(image_format)
        return self.is_integer_image_type(texture_type)

    def is_float_image_resource(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return is_glsl_float_image_resource(texture_type) or texture_type in {
            "imageCube",
            "imageCubeArray",
        }

    def image_store_constructors_by_kind(self):
        return storage_image_store_constructors("vec4", "ivec4", "uvec4")

    def image_store_zero_values_by_kind(self):
        return storage_image_zero_values()

    def image_load_component_suffix(self, texture_type, image_format):
        # imageLoad yields four components; a narrower context takes the leading ones.
        return leading_components_suffix(self.expected_component_count())

    def image_format_store_constructor(self, image_format):
        return storage_image_format_store_constructor(
            image_format, self.image_store_constructors_by_kind()
        )

    def integer_image_store_constructor(self, texture_type):
        if texture_type in {
            "iimage1D",
            "iimage1DArray",
            "iimage2D",
            "iimage3D",
            "iimageCube",
            "iimageCubeArray",
            "iimage2DArray",
            "iimage2DMS",
            "iimage2DMSArray",
        }:
            return "ivec4"
        if texture_type in {
            "uimage1D",
            "uimage1DArray",
            "uimage2D",
            "uimage3D",
            "uimageCube",
            "uimageCubeArray",
            "uimage2DArray",
            "uimage2DMS",
            "uimage2DMSArray",
        }:
            return "uvec4"
        return None

    def two_component_image_store_expression(
        self, image_format, value, value_type=None
    ):
        return storage_image_two_component_store_expression(
            image_format,
            value,
            self.is_scalar_value_type(value_type),
            self.image_store_constructors_by_kind(),
            self.image_store_zero_values_by_kind(),
        )

    def image_store_value_expression(
        self, texture_type, image_format, value, value_type=None
    ):
        component_kind = (
            image_format_component_kind(image_format)
            if image_format
            else self.glsl_storage_image_component_kind(texture_type)
        )
        constructor = texel_vector_type(
            component_kind, {"float": "vec", "int": "ivec", "uint": "uvec"}
        )
        if constructor is None:
            return value
        return texel_store_value(
            constructor, component_kind, value, self.value_component_count(value_type)
        )

    def glsl_storage_image_component_kind(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if texture_type.startswith("uimage"):
            return "uint"
        if texture_type.startswith("iimage"):
            return "int"
        if texture_type.startswith("image"):
            return "float"
        return None
