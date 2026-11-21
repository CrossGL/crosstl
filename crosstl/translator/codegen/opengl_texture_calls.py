"""Texture calls in the GLSL generator: image atomic arguments, gathers, queries, dimensions and coordinates, and refusals of unsupported texel fetches."""

from ..ast import (
    ArrayAccessNode,
    LiteralNode,
    PrimitiveType,
)
from ..refusals import OpenGLRefusal
from ..validation import (
    IMAGE_RESOURCE_INTRINSIC_NAMES,
    INTEGER_COORDINATE_INTRINSIC_NAMES,
    expression_debug_name,
    texture_bias_argument_index,
    texture_compare_argument_index,
    texture_gather_component_argument_index,
    texture_gradient_argument_indices,
    texture_lod_argument_index,
    texture_mip_level_argument_index,
    texture_offset_argument_indices,
    texture_query_lod_coordinate_argument_index,
    texture_sample_index_argument_index,
)
from .generic_function_utils import generic_function_parameters
from .image_access_contracts import (
    floating_coordinate_dimension_from_type_name,
    image_access_diagnostic_name,
    image_access_requirement_label,
    image_access_satisfies_requirement,
    image_atomic_explicit_format_component_kind,
    image_atomic_format_error,
    image_atomic_result_kind_error,
    image_atomic_result_kind_mismatch,
    image_atomic_value_kind_error,
    image_atomic_value_kind_mismatch,
    image_multisample_sample_argument_index,
    image_multisample_sample_type_error,
    image_multisample_sample_type_mismatch,
    integer_coordinate_dimension_from_type_name,
    is_floating_scalar_type_name,
    is_image_atomic_operation,
    is_integer_coordinate_type_name,
    is_integer_scalar_type_name,
    is_numeric_scalar_type_name,
    is_projected_texture_basic_offset_operation,
    is_projected_texture_basic_operation,
    is_projected_texture_grad_offset_operation,
    is_projected_texture_grad_operation,
    is_projected_texture_lod_offset_operation,
    is_projected_texture_lod_operation,
    is_storage_image_texture_comparison_operation,
    is_storage_image_texture_operation,
    is_texture_samples_query_operation,
    numeric_scalar_type_kind,
    operation_argument_type_error,
    operation_dimension_argument_error,
    projected_texture_extra_argument_count_error,
    requires_integer_coordinate,
)
from .image_access_contracts import (
    texture_argument_diagnostic_type as shared_texture_argument_diagnostic_type,
)
from .image_access_contracts import (
    texture_coordinate_arguments_error,
    texture_image_resource_operation_names,
    texture_multisample_sample_type_error,
    texture_query_levels_multisample_expression,
    texture_query_lod_coordinate_dimension_error,
    texture_query_lod_coordinate_swizzle,
    texture_query_lod_coordinate_type_error,
    texture_resource_dimension_descriptor,
    texture_resource_offset_dimension_key,
    texture_sample_offset_capability_error,
    texture_samples_query_expression,
    unsupported_multisample_texture_compare_scalar_expression,
    unsupported_multisample_texture_gather_compare_vector_expression,
    unsupported_multisample_texture_query_lod_expression,
    unsupported_projected_texture_operation_error,
    unsupported_storage_image_texture_comparison_scalar_expression,
    unsupported_storage_image_texture_operation_vector_expression,
    unsupported_texture_gather_call_expression,
    unsupported_texture_query_levels_expression,
    unsupported_texture_query_lod_expression,
    unsupported_texture_samples_query_call_expression,
    validate_texture_operation_arity,
)


class OpenGLTextureCallMixin:
    """Texture call members of ``GLSLCodeGen``."""

    def validate_image_atomic_value_argument_types(
        self, func_name, args, component_kind, image_format
    ):
        mismatch = image_atomic_value_kind_mismatch(
            func_name,
            self.image_atomic_value_arguments(func_name, args),
            component_kind,
            self.scalar_expression_kind,
        )
        if mismatch is None:
            return
        value_arg, value_kind = mismatch
        raise OpenGLRefusal(
            image_atomic_value_kind_error(
                "OpenGL",
                func_name,
                image_format,
                component_kind,
                expression_debug_name(value_arg),
                value_kind,
            )
        )

    def scalar_expected_kind(self):
        return numeric_scalar_type_kind(
            self.current_expression_expected_type,
            self.type_name_string,
            self.map_type,
        )

    def validate_image_atomic_result_type(
        self, func_name, component_kind, image_format
    ):
        expected_kind = image_atomic_result_kind_mismatch(
            self.scalar_expected_kind(), component_kind
        )
        if expected_kind is None:
            return
        raise OpenGLRefusal(
            image_atomic_result_kind_error(
                "OpenGL", func_name, image_format, component_kind, expected_kind
            )
        )

    def validate_image_atomic_format_argument(self, func_name, args):
        if not is_image_atomic_operation(func_name) or not args:
            return
        image_format = self.image_resource_format(args[0])
        if image_format is None:
            return
        component_kind = image_atomic_explicit_format_component_kind(
            func_name, image_format
        )
        if component_kind is not None:
            self.validate_image_atomic_value_argument_types(
                func_name,
                args,
                component_kind,
                image_format,
            )
            self.validate_image_atomic_result_type(
                func_name,
                component_kind,
                image_format,
            )
            return
        raise OpenGLRefusal(
            image_atomic_format_error("OpenGL", func_name, image_format)
        )

    def storage_image_effective_format(self, image_type, explicit_format=None):
        if explicit_format is not None:
            return explicit_format
        image_type = self.resource_base_type(image_type)
        if not self.is_storage_image_type(image_type):
            return None
        return self.image_format_qualifier(image_type)

    def storage_image_parameter_contract(self, param):
        raw_type = getattr(param, "param_type", getattr(param, "vtype", None))
        if not self.is_storage_image_type(raw_type):
            return None

        base_type = self.resource_base_type(raw_type)
        expected_type = self.resource_base_type(
            self.map_resource_type_with_format(base_type, param)
        )
        explicit_format = self.explicit_glsl_image_format(param)
        access_choices = self.resource_access_metadata_choices(param)
        self.validate_resource_access_metadata_consistency(param, access_choices)
        return {
            "type": expected_type,
            "format": explicit_format,
            "access": access_choices[0][1] if access_choices else None,
        }

    def storage_image_argument_contract(self, arg):
        actual_type = self.texture_argument_resource_type(arg)
        if not self.is_storage_image_type(actual_type):
            return None
        explicit_format = self.image_resource_format(arg)
        return {
            "type": self.resource_base_type(actual_type),
            "format": self.storage_image_effective_format(actual_type, explicit_format),
            "access": self.image_resource_access(arg),
        }

    def storage_image_binding_contract(self, binding):
        if binding is None:
            return None
        actual_type = binding.get("type")
        if not self.is_storage_image_type(actual_type):
            return None
        explicit_format = binding.get("format")
        return {
            "type": self.resource_base_type(actual_type),
            "format": self.storage_image_effective_format(actual_type, explicit_format),
            "access": binding.get("access"),
        }

    def validate_storage_image_parameter_contract(
        self,
        func_name,
        param,
        arg,
        index,
        actual,
    ):
        expected = self.storage_image_parameter_contract(param)
        if expected is None or actual is None:
            return

        param_name = getattr(param, "name", None) or f"arg{index}"
        actual_name = expression_debug_name(arg)
        expected_format = expected.get("format")
        if expected_format is not None and actual.get("format") != expected_format:
            actual_format = actual.get("format") or "<unspecified>"
            raise OpenGLRefusal(
                f"OpenGL function call '{func_name}' requires "
                f"{expected_format} storage image format for argument "
                f"{actual_name} passed to parameter {param_name}: got "
                f"{actual_format}"
            )

        expected_type = expected.get("type")
        if expected_type is not None and actual.get("type") != expected_type:
            actual_type = actual.get("type") or "<unknown>"
            raise OpenGLRefusal(
                f"OpenGL function call '{func_name}' requires "
                f"{expected_type} storage image for argument {actual_name} "
                f"passed to parameter {param_name}: got {actual_type}"
            )

        expected_access = expected.get("access")
        actual_access = actual.get("access")
        if expected_access is not None and not image_access_satisfies_requirement(
            expected_access,
            actual_access,
        ):
            required_label = image_access_requirement_label(expected_access)
            actual_label = image_access_diagnostic_name(actual_access)
            raise OpenGLRefusal(
                f"OpenGL function call '{func_name}' requires "
                f"{required_label} storage image access for argument "
                f"{actual_name} passed to parameter {param_name}: got "
                f"{actual_label}"
            )

    def validate_function_image_parameter_contract_arguments(self, func_name, args):
        callee = self.function_definitions.get(func_name)
        if callee is None:
            return
        # A generic definition's parameters name type parameters, not concrete
        # image types; its specializations are validated under their own names.
        # This matters when a concrete overload of the same name takes the call
        # and the generic definition shadows it in function_definitions.
        if generic_function_parameters(callee):
            return

        params = list(getattr(callee, "parameters", getattr(callee, "params", [])))
        for index, param in enumerate(params):
            if index >= len(args):
                break
            self.validate_storage_image_parameter_contract(
                func_name,
                param,
                args[index],
                index,
                self.storage_image_argument_contract(args[index]),
            )

    def validate_function_image_access_arguments(self, func_name, args):
        callee_requirements = self.function_image_access_requirements.get(func_name)
        if not callee_requirements:
            return
        param_names = self.function_parameter_names.get(func_name, [])
        for index, param_name in enumerate(param_names):
            required_access = callee_requirements.get(param_name)
            if required_access is None or index >= len(args):
                continue
            actual_access = self.image_resource_access(args[index])
            if image_access_satisfies_requirement(required_access, actual_access):
                continue
            actual_name = expression_debug_name(args[index])
            required_label = image_access_requirement_label(required_access)
            actual_label = image_access_diagnostic_name(actual_access)
            raise OpenGLRefusal(
                f"OpenGL function call '{func_name}' requires {required_label} "
                f"storage image access for argument {actual_name} passed to "
                f"parameter {param_name}: got {actual_label}"
            )

    def validate_function_structured_buffer_access_arguments(self, func_name, args):
        callee_requirements = self.function_structured_buffer_access_requirements.get(
            func_name
        )
        if not callee_requirements:
            return
        param_names = self.function_parameter_names.get(func_name, [])
        for index, param_name in enumerate(param_names):
            required_access = callee_requirements.get(param_name)
            if required_access is None or index >= len(args):
                continue
            actual_access = self.structured_buffer_resource_access(args[index])
            if image_access_satisfies_requirement(required_access, actual_access):
                continue
            actual_name = expression_debug_name(args[index])
            required_label = image_access_requirement_label(required_access)
            actual_label = image_access_diagnostic_name(actual_access)
            raise OpenGLRefusal(
                f"OpenGL function call '{func_name}' requires {required_label} "
                f"SSBO access for argument {actual_name} passed to parameter "
                f"{param_name}: got {actual_label}"
            )

    def is_integer_coordinate_type(self, vtype):
        type_name = self.type_name_string(vtype)
        base_type = self.resource_base_type(type_name)
        mapped_type = self.map_type(base_type)
        if is_integer_coordinate_type_name(base_type, mapped_type):
            return True
        value_info = self.glsl_value_type_info(base_type)
        return bool(
            value_info is not None
            and value_info["family"] in {"int", "uint"}
            and value_info["width"] in {1, 2, 3, 4}
        )

    def texture_dimension_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        sampling = self.texture_sampling_capabilities(texture_type)
        is_multisample = self.is_multisample_texture_resource_type(texture_type)
        is_storage_image = self.is_storage_image_type(texture_type)

        coordinate_dimension = None
        if texture_type and "Cube" in texture_type and is_storage_image:
            coordinate_dimension = 3
        elif texture_type and "Cube" not in texture_type:
            for prefix in ("iimage", "uimage", "image"):
                if texture_type.startswith(f"{prefix}2DMSArray"):
                    coordinate_dimension = 3
                    break
                if texture_type.startswith(f"{prefix}2DMS"):
                    coordinate_dimension = 2
                    break
                if texture_type.startswith(f"{prefix}2DArray"):
                    coordinate_dimension = 3
                    break
                if texture_type.startswith(f"{prefix}3D"):
                    coordinate_dimension = 3
                    break
                if texture_type.startswith(f"{prefix}2D"):
                    coordinate_dimension = 2
                    break
                if texture_type.startswith(f"{prefix}1DArray"):
                    coordinate_dimension = 2
                    break
                if texture_type.startswith(f"{prefix}1D"):
                    coordinate_dimension = 1
                    break
            if coordinate_dimension is None:
                coordinate_dimension = {
                    "sampler1D": 1,
                    "sampler1DArray": 2,
                    "sampler2D": 2,
                    "samplerExternalOES": 2,
                    "__samplerExternal2DY2YEXT": 2,
                    "sampler2DArray": 3,
                    "sampler2DMS": 2,
                    "sampler2DMSArray": 3,
                    "sampler3D": 3,
                    "isampler1D": 1,
                    "isampler1DArray": 2,
                    "isampler2D": 2,
                    "isampler2DArray": 3,
                    "isampler2DMS": 2,
                    "isampler2DMSArray": 3,
                    "isampler3D": 3,
                    "usampler1D": 1,
                    "usampler1DArray": 2,
                    "usampler2D": 2,
                    "usampler2DArray": 3,
                    "usampler2DMS": 2,
                    "usampler2DMSArray": 3,
                    "usampler3D": 3,
                }.get(texture_type)

        offset_dimension = None
        if texture_type and "Cube" not in texture_type:
            offset_dimension = {
                "sampler1D": 1,
                "sampler1DArray": 1,
                "sampler1DShadow": 1,
                "sampler1DArrayShadow": 1,
                "sampler2D": 2,
                "sampler2DArray": 2,
                "sampler2DShadow": 2,
                "sampler2DArrayShadow": 2,
                "sampler3D": 3,
                "isampler1D": 1,
                "isampler1DArray": 1,
                "isampler2D": 2,
                "isampler2DArray": 2,
                "isampler3D": 3,
                "usampler1D": 1,
                "usampler1DArray": 1,
                "usampler2D": 2,
                "usampler2DArray": 2,
                "usampler3D": 3,
            }.get(texture_type)

        gradient_dimension = None
        if texture_type and not is_multisample and not is_storage_image:
            if "Cube" in texture_type:
                gradient_dimension = 3
            else:
                gradient_dimension = {
                    "sampler1D": 1,
                    "sampler1DArray": 1,
                    "sampler1DShadow": 1,
                    "sampler1DArrayShadow": 1,
                    "sampler2D": 2,
                    "sampler2DArray": 2,
                    "sampler2DShadow": 2,
                    "sampler2DArrayShadow": 2,
                    "sampler3D": 3,
                    "isampler1D": 1,
                    "isampler1DArray": 1,
                    "isampler2D": 2,
                    "isampler2DArray": 2,
                    "isampler3D": 3,
                    "usampler1D": 1,
                    "usampler1DArray": 1,
                    "usampler2D": 2,
                    "usampler2DArray": 2,
                    "usampler3D": 3,
                }.get(texture_type)

        query_lod_coordinate_dimension = None
        if texture_type and not is_multisample and not is_storage_image:
            query_lod_coordinate_dimension = {
                "sampler1D": 1,
                "sampler1DArray": 2,
                "sampler1DShadow": 1,
                "sampler1DArrayShadow": 2,
                "sampler2D": 2,
                "sampler2DArray": 3,
                "sampler2DShadow": 2,
                "sampler2DArrayShadow": 3,
                "sampler3D": 3,
                "samplerCube": 3,
                "samplerCubeArray": 4,
                "samplerCubeShadow": 3,
                "samplerCubeArrayShadow": 4,
                "isampler1D": 1,
                "isampler1DArray": 2,
                "isampler2D": 2,
                "isampler2DArray": 3,
                "isampler3D": 3,
                "isamplerCube": 3,
                "isamplerCubeArray": 4,
                "usampler1D": 1,
                "usampler1DArray": 2,
                "usampler2D": 2,
                "usampler2DArray": 3,
                "usampler3D": 3,
                "usamplerCube": 3,
                "usamplerCubeArray": 4,
            }.get(texture_type)

        return texture_resource_dimension_descriptor(
            texture_type,
            sampling,
            coordinate_dimension=coordinate_dimension,
            offset_dimension=offset_dimension,
            gradient_dimension=gradient_dimension,
            query_lod_coordinate_dimension=query_lod_coordinate_dimension,
            is_multisample=is_multisample,
        )

    def resource_coordinate_dimension(self, texture_type):
        return self.texture_dimension_descriptor(texture_type)["coordinate_dimension"]

    def resource_offset_dimension(self, func_name, texture_type):
        descriptor = self.texture_dimension_descriptor(texture_type)
        return descriptor[texture_resource_offset_dimension_key(func_name)]

    def resource_gradient_dimension(self, func_name, texture_type):
        return self.texture_dimension_descriptor(texture_type)["gradient_dimension"]

    def resource_query_lod_coordinate_dimension(self, texture_type):
        return self.texture_dimension_descriptor(texture_type)[
            "query_lod_coordinate_dimension"
        ]

    def validate_integer_coordinate_argument(self, func_name, args):
        if (
            not requires_integer_coordinate(
                func_name, INTEGER_COORDINATE_INTRINSIC_NAMES
            )
            or len(args) < 2
        ):
            return
        coord_type = self.expression_result_type(args[1])
        if coord_type is None or self.is_integer_coordinate_type(coord_type):
            return
        raise OpenGLRefusal(
            operation_argument_type_error(
                "OpenGL",
                "resource",
                func_name,
                "an integer",
                "coordinate",
                expression_debug_name(args[1]),
                self.type_name_string(coord_type),
            )
        )

    def validate_coordinate_dimension_argument(self, func_name, args):
        if (
            not requires_integer_coordinate(
                func_name, INTEGER_COORDINATE_INTRINSIC_NAMES
            )
            or len(args) < 2
        ):
            return
        texture_type = self.texture_argument_resource_type(args[0])
        expected_dimension = self.resource_coordinate_dimension(texture_type)
        if expected_dimension is None:
            return
        coord_type = self.expression_result_type(args[1])
        coord_dimension = integer_coordinate_dimension_from_type_name(
            self.type_name_string(coord_type), self.map_type
        )
        if coord_dimension is None or coord_dimension == expected_dimension:
            return
        raise OpenGLRefusal(
            operation_dimension_argument_error(
                "OpenGL",
                "resource",
                func_name,
                expected_dimension,
                "integer",
                "coordinate",
                self.resource_base_type(texture_type),
                expression_debug_name(args[1]),
                self.type_name_string(coord_type),
            )
        )

    def validate_offset_dimension_argument(self, func_name, args):
        offset_indices = texture_offset_argument_indices(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if not offset_indices:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        expected_dimension = self.resource_offset_dimension(func_name, texture_type)
        if expected_dimension is None:
            return
        for offset_index in offset_indices:
            offset_type = self.expression_result_type(args[offset_index])
            if offset_type is None:
                continue
            if not self.is_integer_coordinate_type(offset_type):
                raise OpenGLRefusal(
                    operation_argument_type_error(
                        "OpenGL",
                        "resource",
                        func_name,
                        "an integer",
                        "offset",
                        expression_debug_name(args[offset_index]),
                        self.type_name_string(offset_type),
                    )
                )
            offset_dimension = integer_coordinate_dimension_from_type_name(
                self.type_name_string(offset_type), self.map_type
            )
            if offset_dimension is None or offset_dimension == expected_dimension:
                continue
            raise OpenGLRefusal(
                operation_dimension_argument_error(
                    "OpenGL",
                    "resource",
                    func_name,
                    expected_dimension,
                    "integer",
                    "offset",
                    self.resource_base_type(texture_type),
                    expression_debug_name(args[offset_index]),
                    self.type_name_string(offset_type),
                )
            )

    def gradient_argument_dimension(self, vtype):
        type_name = self.resource_base_type(self.type_name_string(vtype))
        return floating_coordinate_dimension_from_type_name(type_name, self.map_type)

    def query_lod_coordinate_dimension(self, vtype):
        type_name = self.resource_base_type(self.type_name_string(vtype))
        return floating_coordinate_dimension_from_type_name(type_name, self.map_type)

    def validate_query_lod_coordinate_argument(self, func_name, args):
        coord_index = texture_query_lod_coordinate_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if coord_index is None:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        expected_dimension = self.resource_query_lod_coordinate_dimension(texture_type)
        if expected_dimension is None:
            return
        coord_type = self.expression_result_type(args[coord_index])
        if coord_type is None:
            return
        coord_dimension = self.query_lod_coordinate_dimension(coord_type)
        if coord_dimension is None:
            raise OpenGLRefusal(
                texture_query_lod_coordinate_type_error(
                    "OpenGL",
                    func_name,
                    expression_debug_name(args[coord_index]),
                    self.type_name_string(coord_type),
                )
            )
        if coord_dimension == expected_dimension:
            return
        raise OpenGLRefusal(
            texture_query_lod_coordinate_dimension_error(
                "OpenGL",
                func_name,
                expected_dimension,
                self.resource_base_type(texture_type),
                expression_debug_name(args[coord_index]),
                self.type_name_string(coord_type),
            )
        )

    def validate_gradient_dimension_arguments(self, func_name, args):
        gradient_indices = texture_gradient_argument_indices(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if not gradient_indices:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        expected_dimension = self.resource_gradient_dimension(func_name, texture_type)
        if expected_dimension is None:
            return
        for gradient_index in gradient_indices:
            gradient_type = self.expression_result_type(args[gradient_index])
            if gradient_type is None:
                continue
            gradient_dimension = self.gradient_argument_dimension(gradient_type)
            if gradient_dimension is None:
                raise OpenGLRefusal(
                    operation_argument_type_error(
                        "OpenGL",
                        "resource",
                        func_name,
                        "a floating",
                        "gradient",
                        expression_debug_name(args[gradient_index]),
                        self.type_name_string(gradient_type),
                    )
                )
            if gradient_dimension == expected_dimension:
                continue
            raise OpenGLRefusal(
                operation_dimension_argument_error(
                    "OpenGL",
                    "resource",
                    func_name,
                    expected_dimension,
                    "floating",
                    "gradient",
                    self.resource_base_type(texture_type),
                    expression_debug_name(args[gradient_index]),
                    self.type_name_string(gradient_type),
                )
            )

    def is_scalar_floating_type(self, vtype):
        return is_floating_scalar_type_name(self.type_name_string(vtype), self.map_type)

    def is_scalar_numeric_type(self, vtype):
        return is_numeric_scalar_type_name(self.type_name_string(vtype), self.map_type)

    def is_scalar_integer_type(self, vtype):
        return is_integer_scalar_type_name(self.type_name_string(vtype), self.map_type)

    def texture_argument_diagnostic_type(self, arg):
        return shared_texture_argument_diagnostic_type(
            arg,
            self.texture_resource_type,
            self.expression_name,
            self.expression_result_type,
            self.sampler_variables | self.current_sampler_parameters,
        )

    def validate_compare_argument(self, func_name, args):
        compare_index = texture_compare_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if compare_index is None:
            return
        compare_type = self.expression_result_type(args[compare_index])
        if compare_type is None or self.is_scalar_floating_type(compare_type):
            return
        raise OpenGLRefusal(
            operation_argument_type_error(
                "OpenGL",
                "texture compare",
                func_name,
                "a scalar floating",
                "compare",
                expression_debug_name(args[compare_index]),
                self.type_name_string(compare_type),
            )
        )

    def validate_lod_argument(self, func_name, args):
        lod_index = texture_lod_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if lod_index is None:
            return
        lod_type = self.texture_argument_diagnostic_type(args[lod_index])
        if lod_type is None or self.is_scalar_numeric_type(lod_type):
            return
        raise OpenGLRefusal(
            operation_argument_type_error(
                "OpenGL",
                "texture LOD",
                func_name,
                "a scalar numeric",
                "lod",
                expression_debug_name(args[lod_index]),
                self.type_name_string(lod_type),
            )
        )

    def validate_bias_argument(self, func_name, args):
        bias_index = texture_bias_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if bias_index is None:
            return
        bias_type = self.texture_argument_diagnostic_type(args[bias_index])
        if bias_type is None or self.is_scalar_numeric_type(bias_type):
            return
        raise OpenGLRefusal(
            operation_argument_type_error(
                "OpenGL",
                "texture bias",
                func_name,
                "a scalar numeric",
                "bias",
                expression_debug_name(args[bias_index]),
                self.type_name_string(bias_type),
            )
        )

    def validate_mip_level_argument(self, func_name, args):
        level_index = texture_mip_level_argument_index(func_name, len(args))
        if level_index is None:
            return
        level_type = self.texture_argument_diagnostic_type(args[level_index])
        if level_type is None or self.is_scalar_integer_type(level_type):
            return
        raise OpenGLRefusal(
            operation_argument_type_error(
                "OpenGL",
                "resource",
                func_name,
                "a scalar integer",
                "mip/sample level",
                expression_debug_name(args[level_index]),
                self.type_name_string(level_type),
            )
        )

    def validate_sample_index_argument(self, func_name, args):
        sample_index = texture_sample_index_argument_index(func_name, len(args))
        if sample_index is None:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        if not self.is_multisample_texture_resource_type(texture_type):
            return
        sample_type = self.texture_argument_diagnostic_type(args[sample_index])
        if sample_type is None or self.is_scalar_integer_type(sample_type):
            return
        raise OpenGLRefusal(
            texture_multisample_sample_type_error(
                "OpenGL",
                func_name,
                expression_debug_name(args[sample_index]),
                self.type_name_string(sample_type),
            )
        )

    def validate_image_multisample_arguments(self, func_name, args):
        if not args:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        sample_index = image_multisample_sample_argument_index(
            func_name,
            len(args),
            self.is_multisample_storage_image_type(texture_type),
            "OpenGL",
        )
        if sample_index is None:
            return
        sample_type = self.texture_argument_diagnostic_type(args[sample_index])
        sample_type = image_multisample_sample_type_mismatch(
            sample_type, self.is_scalar_integer_type
        )
        if sample_type is None:
            return
        raise OpenGLRefusal(
            image_multisample_sample_type_error(
                "OpenGL",
                func_name,
                expression_debug_name(args[sample_index]),
                self.type_name_string(sample_type),
            )
        )

    def validate_gather_component_argument(self, func_name, args):
        component_index = texture_gather_component_argument_index(
            func_name,
            self.texture_call_uses_explicit_sampler(args),
            len(args),
        )
        if component_index is None:
            return
        component_type = self.texture_argument_diagnostic_type(args[component_index])
        if component_type is None or self.is_scalar_integer_type(component_type):
            return
        raise OpenGLRefusal(
            operation_argument_type_error(
                "OpenGL",
                "texture gather",
                func_name,
                "a scalar integer",
                "component",
                expression_debug_name(args[component_index]),
                self.type_name_string(component_type),
            )
        )

    def validate_texture_call_arity(self, func_name, args):
        validate_texture_operation_arity(
            "OpenGL",
            func_name,
            args,
            self.texture_resource_operation_names(),
            self.texture_call_uses_explicit_sampler,
        )

    def texture_size_expected_argument_count(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if not texture_type:
            return None
        if self.is_storage_image_type(texture_type):
            return None
        if self.is_multisample_texture_resource_type(texture_type):
            return 1
        if texture_type.endswith(self.TEXTURE_SIZE_NO_LOD_SUFFIXES):
            return 1
        if texture_type.startswith(("sampler", "isampler", "usampler")):
            return 2
        return None

    def validate_texture_size_argument_count(self, func_name, args):
        if func_name != "textureSize" or not args:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        expected_count = self.texture_size_expected_argument_count(texture_type)
        if expected_count is None or len(args) == expected_count:
            return
        texture_type = self.resource_base_type(texture_type)
        if len(args) < expected_count:
            raise OpenGLRefusal(
                f"OpenGL texture operation 'textureSize' requires "
                f"{expected_count} argument(s) for {texture_type}, got {len(args)}"
            )
        raise OpenGLRefusal(
            f"OpenGL texture operation 'textureSize' accepts "
            f"{expected_count} argument(s) for {texture_type}, got {len(args)}"
        )

    def texture_resource_operation_names(self):
        return texture_image_resource_operation_names(
            IMAGE_RESOURCE_INTRINSIC_NAMES
        ) | {"textureGatherCompareOffsets"}

    def vector_component(self, expression, component):
        if all(char.isalnum() or char in "_.[]" for char in expression):
            return f"{expression}.{component}"
        return f"({expression}).{component}"

    def texture_query_lod_coordinate(self, texture_type, coord):
        texture_type = self.resource_base_type(texture_type)
        swizzle = texture_query_lod_coordinate_swizzle("GLSL", texture_type)
        if swizzle:
            return self.vector_component(coord, swizzle)
        return coord

    def is_array_expression(self, node):
        type_name = self.type_name_string(self.expression_result_type(node))
        return isinstance(type_name, str) and "[" in type_name and "]" in type_name

    def texture_gather_offsets_args(self, extra_args):
        if len(extra_args) in {1, 2} and self.is_array_expression(extra_args[0]):
            offset_args = [
                ArrayAccessNode(
                    extra_args[0],
                    LiteralNode(index, PrimitiveType("int")),
                )
                for index in range(4)
            ]
            component_arg = extra_args[1] if len(extra_args) == 2 else None
            return offset_args, component_arg

        if len(extra_args) in {4, 5}:
            component_arg = extra_args[4] if len(extra_args) == 5 else None
            return extra_args[:4], component_arg

        return None, None

    def texture_gather_component_value(self, component_arg):
        if component_arg is None:
            return None
        return self.literal_int_value(component_arg, self.literal_int_constants)

    def texture_gather_call_expression(
        self, function_name, texture_name, coord, offset_arg=None, component=None
    ):
        args = [texture_name, coord]
        if offset_arg is not None:
            args.append(offset_arg)
        if component is not None:
            args.append(str(component))
        return f"{function_name}({', '.join(args)})"

    def texture_gather_offsets_expression(
        self, texture_name, coord, offset_args, component, texture_type=None
    ):
        """Lower ``textureGatherOffsets`` to one ``textureGatherOffset`` per offset.

        Component ``i`` of ``textureGatherOffsets`` is texel ``T_i0_j0`` of the
        footprint at offset ``i`` (GLSL 4.60, section 8.9.4), and
        ``textureGatherOffset`` returns that texel in ``w``.
        """
        component_values = []
        for offset_arg in offset_args:
            gather = self.texture_gather_call_expression(
                "textureGatherOffset",
                texture_name,
                coord,
                self.generate_expression(offset_arg),
                component,
            )
            component_values.append(f"{gather}.w")
        vector_type = self.texture_gather_result_vector_type(texture_type)
        return f"{vector_type}({', '.join(component_values)})"

    def texture_gather_result_vector_type(self, texture_type):
        texture_type = self.resource_base_type(texture_type) or ""
        if texture_type.startswith("isampler"):
            return "ivec4"
        if texture_type.startswith("usampler"):
            return "uvec4"
        return "vec4"

    def texture_gather_compare_offsets_args(self, extra_args):
        if not extra_args:
            return None, None

        compare_arg = extra_args[0]
        offset_args, component_arg = self.texture_gather_offsets_args(extra_args[1:])
        if component_arg is not None:
            return None, None
        return compare_arg, offset_args

    def texture_gather_compare_offsets_expression(
        self, texture_name, coord, compare, offset_args
    ):
        """Lower a shadow ``textureGatherOffsets``; see ``texture_gather_offsets_expression``."""
        component_values = []
        for offset_arg in offset_args:
            offset = self.generate_expression(offset_arg)
            gather = (
                f"textureGatherOffset({texture_name}, {coord}, {compare}, {offset})"
            )
            component_values.append(f"{gather}.w")
        return f"vec4({', '.join(component_values)})"

    def texture_gather_dynamic_component_expression(self, build_expression, component):
        component_calls = [build_expression(index) for index in range(4)]
        return (
            f"({component} == 0 ? {component_calls[0]} : "
            f"{component} == 1 ? {component_calls[1]} : "
            f"{component} == 2 ? {component_calls[2]} : {component_calls[3]})"
        )

    def unsupported_texture_gather_call(self, func_name, reason):
        return unsupported_texture_gather_call_expression("GLSL", func_name, reason)

    def texture_sampling_capabilities(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        gather_types = {
            "sampler2D",
            "sampler2DArray",
            "samplerCube",
            "samplerCubeArray",
            "isampler2D",
            "isampler2DArray",
            "isamplerCube",
            "isamplerCubeArray",
            "usampler2D",
            "usampler2DArray",
            "usamplerCube",
            "usamplerCubeArray",
        }
        gather_offset_types = {
            "sampler2D",
            "sampler2DArray",
            "isampler2D",
            "isampler2DArray",
            "usampler2D",
            "usampler2DArray",
        }
        sample_offset_types = {
            "sampler1D",
            "sampler1DArray",
            "sampler2D",
            "sampler3D",
            "sampler2DArray",
            "isampler1D",
            "isampler1DArray",
            "isampler2D",
            "isampler3D",
            "isampler2DArray",
            "usampler1D",
            "usampler1DArray",
            "usampler2D",
            "usampler3D",
            "usampler2DArray",
            "sampler2DShadow",
            "sampler2DArrayShadow",
        }
        # GLSL 4.60 section 8.9.2: textureOffset and textureGradOffset take
        # every 1D and 2D shadow sampler; textureGatherOffset only the 2D ones.
        compare_offset_types = {
            "sampler1DShadow",
            "sampler1DArrayShadow",
            "sampler2DShadow",
            "sampler2DArrayShadow",
        }
        gather_compare_offset_types = {"sampler2DShadow", "sampler2DArrayShadow"}
        compare_lod_types = {
            "sampler1DShadow",
            "sampler1DArrayShadow",
            "sampler2DShadow",
        }
        compare_grad_types = {
            "sampler1DShadow",
            "sampler1DArrayShadow",
            "sampler2DShadow",
            "sampler2DArrayShadow",
            "samplerCubeShadow",
        }
        return {
            "texture_type": texture_type,
            "gather": texture_type in gather_types,
            "gather_offset": texture_type in gather_offset_types,
            "sample_offset": texture_type in sample_offset_types,
            "compare_offset": texture_type in compare_offset_types,
            "compare_lod": texture_type in compare_lod_types,
            "compare_grad": texture_type in compare_grad_types,
            "compare_lod_offset": texture_type in compare_lod_types,
            "compare_grad_offset": texture_type in compare_offset_types,
            "gather_compare_offset": texture_type in gather_compare_offset_types,
        }

    def texture_gather_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather"]

    def texture_gather_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["gather_offset"]

    def texture_sample_vector_zero_value(self, texture_type=None):
        expected_type = self.map_type(self.current_expression_expected_type)
        if expected_type in {"ivec4", "uvec4", "vec4"}:
            return self.zero_value_expression(expected_type)

        texture_type = self.resource_base_type(texture_type)
        if texture_type.startswith("isampler"):
            return "ivec4(0)"
        if texture_type.startswith("usampler"):
            return "uvec4(0u)"
        return "vec4(0.0)"

    def texture_sampling_result_type(self, texture_arg, func_name=None):
        texture_type = self.resource_base_type(self.texture_resource_type(texture_arg))
        if texture_type.startswith("isampler"):
            return "ivec4"
        if texture_type.startswith("usampler"):
            return "uvec4"
        if "Shadow" in texture_type and func_name not in {
            "textureGather",
            "textureGatherOffset",
            "textureGatherOffsets",
        }:
            return "float"
        return "vec4"

    def unsupported_texture_projected_call(self, func_name, reason, texture_type=None):
        zero_value = self.texture_sample_vector_zero_value(texture_type)
        return self.refuse_lowering(
            f"projected texture: {func_name} {reason}", zero_value
        )

    def projected_cube_texture_coordinate(self, texture_type, coord_arg, coord):
        texture_type = self.resource_base_type(texture_type)
        if texture_type not in {"samplerCube", "isamplerCube", "usamplerCube"}:
            return None

        coord_type = self.resource_base_type(self.expression_result_type(coord_arg))
        if coord_type not in {"vec4", "float4"}:
            return None
        return (
            f"{self.vector_component(coord, 'xyz')} / "
            f"{self.vector_component(coord, 'w')}"
        )

    def generate_projected_cube_texture_call(self, func_name, args):
        parts = self.texture_call_parts(args)
        if parts is None:
            return self.unsupported_texture_projected_call(
                func_name, texture_coordinate_arguments_error()
            )

        texture_name, coord, extra_args = parts
        coord_index = 2 if self.is_explicit_sampler_argument(args) else 1
        texture_type = self.texture_resource_type(args[0])
        projected_coord = self.projected_cube_texture_coordinate(
            texture_type, args[coord_index], coord
        )
        if projected_coord is None:
            return self.unsupported_texture_projected_call(
                func_name,
                "requires 1D, 2D, 2D-array, 3D, or cube projection coordinates",
                texture_type,
            )

        if is_projected_texture_basic_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(
                    func_name, count_error, texture_type
                )
            mapped_args = [texture_name, projected_coord]
            if extra_args:
                mapped_args.append(self.generate_expression(extra_args[0]))
            return f"texture({', '.join(mapped_args)})"

        if is_projected_texture_lod_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(
                    func_name, count_error, texture_type
                )
            lod = self.generate_expression(extra_args[0])
            return f"textureLod({texture_name}, {projected_coord}, {lod})"

        if is_projected_texture_grad_operation(func_name):
            count_error = projected_texture_extra_argument_count_error(
                func_name, len(extra_args)
            )
            if count_error:
                return self.unsupported_texture_projected_call(
                    func_name, count_error, texture_type
                )
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            return f"textureGrad({texture_name}, {projected_coord}, {ddx}, {ddy})"

        if (
            is_projected_texture_basic_offset_operation(func_name)
            or is_projected_texture_lod_offset_operation(func_name)
            or is_projected_texture_grad_offset_operation(func_name)
        ):
            return self.unsupported_texture_projected_call(
                func_name, texture_sample_offset_capability_error("GLSL"), texture_type
            )

        return self.unsupported_texture_projected_call(
            func_name, unsupported_projected_texture_operation_error(), texture_type
        )

    def texture_sample_offset_supported(self, texture_type):
        return self.texture_sampling_capabilities(texture_type)["sample_offset"]

    def unsupported_texture_sample_offset_call(self, func_name, reason):
        zero_value = self.texture_sample_vector_zero_value()
        return self.refuse_lowering(f"texture offset: {func_name} {reason}", zero_value)

    def is_multisample_texture_resource_type(self, texture_type):
        return self.resource_base_type(texture_type) in {
            "sampler2DMS",
            "sampler2DMSArray",
            "isampler2DMS",
            "isampler2DMSArray",
            "usampler2DMS",
            "usampler2DMSArray",
            "image2DMS",
            "image2DMSArray",
            "iimage2DMS",
            "iimage2DMSArray",
            "uimage2DMS",
            "uimage2DMSArray",
        }

    def is_multisample_storage_image_type(self, texture_type):
        return self.resource_base_type(texture_type) in {
            "image2DMS",
            "image2DMSArray",
            "iimage2DMS",
            "iimage2DMSArray",
            "uimage2DMS",
            "uimage2DMSArray",
        }

    def unsupported_multisample_texture_call(self, func_name, texture_type):
        texture_type = self.resource_base_type(texture_type)
        zero_value = self.texture_sample_vector_zero_value(texture_type)
        return self.refuse_lowering(
            f"multisample texture call: {func_name} on {texture_type}", zero_value
        )

    def unsupported_multisample_texture_compare_call(self, func_name, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return unsupported_multisample_texture_compare_scalar_expression(
            "GLSL", func_name, texture_type
        )

    def unsupported_multisample_texture_gather_compare_call(
        self, func_name, texture_type
    ):
        texture_type = self.resource_base_type(texture_type)
        return unsupported_multisample_texture_gather_compare_vector_expression(
            "GLSL", func_name, texture_type
        )

    def unsupported_multisample_texture_query_lod_call(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return unsupported_multisample_texture_query_lod_expression(
            "GLSL", texture_type
        )

    def unsupported_texture_query_levels_call(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return unsupported_texture_query_levels_expression("GLSL", texture_type)

    def unsupported_texture_query_lod_call(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return unsupported_texture_query_lod_expression("GLSL", texture_type)

    def storage_image_texture_operation_expression(self, func_name, texture_type):
        if not self.is_storage_image_type(texture_type):
            return None

        texture_type = self.resource_base_type(texture_type)
        if is_storage_image_texture_comparison_operation(func_name):
            return unsupported_storage_image_texture_comparison_scalar_expression(
                "GLSL", func_name, texture_type
            )

        if is_storage_image_texture_operation(func_name):
            return unsupported_storage_image_texture_operation_vector_expression(
                "GLSL", func_name, texture_type
            )

        return None

    def unsupported_texture_samples_query_call(self):
        return unsupported_texture_samples_query_call_expression("GLSL")

    def image_size_expression(self, image_arg):
        image_name = self.generate_expression(image_arg)
        return f"imageSize({image_name})"

    def texture_query_size_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        if not texture_type:
            return None
        return {
            "function_name": (
                "imageSize"
                if self.is_storage_image_type(texture_type)
                else "textureSize"
            ),
        }

    def texture_query_resource_descriptor(self, texture_arg):
        texture_type = self.texture_resource_type(texture_arg)
        return {
            "texture_type": texture_type,
            "storage_image": self.is_storage_image_type(texture_type),
            "multisample": self.is_multisample_texture_resource_type(texture_type),
            "size_descriptor": self.texture_query_size_descriptor(texture_type),
        }

    def texture_query_levels_expression(self, texture_arg):
        descriptor = self.texture_query_resource_descriptor(texture_arg)
        texture_type = descriptor["texture_type"]
        if descriptor["storage_image"]:
            return self.unsupported_texture_query_levels_call(texture_type)
        if descriptor["multisample"]:
            return texture_query_levels_multisample_expression()
        return None

    def texture_samples_expression(self, func_name, texture_arg):
        descriptor = self.texture_query_resource_descriptor(texture_arg)
        if descriptor["texture_type"] and not descriptor["multisample"]:
            return self.unsupported_texture_samples_query_call()
        if (
            not is_texture_samples_query_operation(func_name)
            and descriptor["multisample"]
            and not descriptor["storage_image"]
        ):
            texture_name = self.generate_expression(texture_arg)
            return texture_samples_query_expression("GLSL", texture_name)
        return None

    def texture_size_query_expression(self, texture_arg):
        descriptor = self.texture_query_resource_descriptor(texture_arg)
        size_descriptor = descriptor["size_descriptor"]
        if size_descriptor and size_descriptor["function_name"] == "imageSize":
            return self.image_size_expression(texture_arg)
        return None

    def unsupported_multisample_texel_fetch_offset_call(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        zero_value = self.texture_sample_vector_zero_value(texture_type)
        return self.refuse_lowering(
            f"texel fetch offset: multisample texture {texture_type} does not "
            "support offsets",
            zero_value,
        )

    def is_cube_texture_resource_type(self, texture_type):
        return self.resource_base_type(texture_type) in {
            "samplerCube",
            "samplerCubeArray",
            "samplerCubeShadow",
            "samplerCubeArrayShadow",
            "isamplerCube",
            "isamplerCubeArray",
            "usamplerCube",
            "usamplerCubeArray",
        }

    def unsupported_cube_texel_fetch_call(self, func_name, texture_type):
        texture_type = self.resource_base_type(texture_type)
        zero_value = self.texture_sample_vector_zero_value(texture_type)
        return self.refuse_lowering(
            f"texel fetch: {func_name} on {texture_type}", zero_value
        )
