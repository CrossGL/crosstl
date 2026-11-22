"""Image and texture arguments in the HLSL generator: components and kinds, implicit samplers, atomics, scalars, LOD, dimensions, counts, query results and coordinates."""

import re
import struct

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    BinaryOpNode,
    ConstructorNode,
    FunctionCallNode,
    LiteralNode,
    MemberAccessNode,
    ReferenceType,
    SwizzleNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from ..refusals import DirectXRefusal
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
from .directx_values import (
    DirectXContextualConversionError,
    DirectXMappedOverloadError,
    DirectXUnresolvedSourceTypeError,
)
from .for_in_utils import validate_range_reference_argument
from .hlsl_image_formats import strip_texel_modifier
from .image_access_contracts import (
    default_storage_image_channel_count,
    floating_coordinate_dimension_from_type_name,
    image_access_diagnostic_name,
    image_access_requirement_label,
    image_access_satisfies_requirement,
    image_atomic_result_kind_error,
    image_atomic_result_kind_mismatch,
)
from .image_access_contracts import (
    image_atomic_value_arguments as shared_image_atomic_value_arguments,
)
from .image_access_contracts import (
    image_atomic_value_kind_error,
    image_atomic_value_kind_mismatch,
    image_format_component_kind,
    image_format_component_type,
    image_format_or_default_channel_count,
    image_multisample_sample_argument_index,
    image_multisample_sample_type_error,
    image_multisample_sample_type_mismatch,
    image_resource_metadata,
    integer_coordinate_dimension_from_type_name,
    is_floating_scalar_type_name,
    is_image_atomic_operation,
    is_image_resource_operation,
    is_integer_scalar_type_name,
    is_numeric_scalar_type_name,
    is_texture_query_lod_operation,
    numeric_component_count_from_type,
    numeric_component_kind_from_type,
    numeric_expression_component_count,
    numeric_expression_component_kind,
    numeric_scalar_expression_kind,
    numeric_scalar_type_kind,
    operation_argument_type_error,
    operation_dimension_argument_error,
    requires_integer_coordinate,
    resolve_image_atomic_component_kind,
    storage_image_store_vector_constructor,
    storage_image_zero_values,
)
from .image_access_contracts import (
    texture_argument_diagnostic_type as shared_texture_argument_diagnostic_type,
)
from .image_access_contracts import (
    texture_image_resource_operation_names,
    texture_multisample_sample_type_error,
    texture_query_lod_coordinate_dimension_error,
    texture_query_lod_coordinate_type_error,
    validate_texture_operation_arity,
)
from .reference_aliases import references_provably_disjoint
from .storage_image_texels import texel_vector_type


class DirectXImageArgumentMixin:
    """Image argument members of ``HLSLCodeGen``."""

    def generate_call_arguments(
        self,
        func_name,
        args,
        type_func_name=None,
        *,
        call_node=None,
        resolved_overload=None,
    ):
        parameter_lookup_name = type_func_name or func_name
        parameter_types = self.function_parameter_types.get(parameter_lookup_name)
        if not parameter_types and type_func_name != func_name:
            parameter_types = self.function_parameter_types.get(func_name)
        if resolved_overload is None:

            def resolved_parameter_overload(function_name):
                try:
                    return self.resolve_hlsl_function_overload(
                        function_name,
                        args,
                        call_node=call_node,
                    )
                except DirectXMappedOverloadError:
                    # HLSL can still resolve source overloads whose mapped target
                    # signatures remain distinct. Only mapped-collision overloads
                    # require CrossTL to choose a target declaration itself.
                    if function_name in self.hlsl_mapped_overload_names:
                        raise
                    return None

            resolved_overload = resolved_parameter_overload(parameter_lookup_name)
            if resolved_overload is None and parameter_lookup_name != func_name:
                resolved_overload = resolved_parameter_overload(func_name)
        parameters = (
            getattr(
                resolved_overload,
                "parameters",
                getattr(resolved_overload, "params", []),
            )
            or []
        )
        if resolved_overload is not None:
            parameter_types = [
                self.function_parameter_type_name(parameter) for parameter in parameters
            ]
        self.validate_hlsl_byte_reference_arguments(args, parameters, call_node)
        parameter_types = parameter_types or []
        workgroup_pointer_func_name = type_func_name or func_name
        workgroup_pointer_indices = (
            self.function_hlsl_workgroup_pointer_parameter_indices.get(
                workgroup_pointer_func_name, {}
            )
        )
        resource_pointer_indices = (
            self.hlsl_resource_pointer_parameter_indices_for_function(
                workgroup_pointer_func_name
            )
        )
        resource_pointer_array_indices = (
            self.hlsl_resource_pointer_array_parameter_indices_for_function(
                workgroup_pointer_func_name
            )
        )
        private_pointer_indices = self.function_private_pointer_parameter_indices.get(
            type_func_name or func_name, {}
        )
        range_reference_parameters = []
        if getattr(self, "current_for_in_reference_bindings", {}):
            candidates = self.hlsl_function_overloads_by_name.get(
                parameter_lookup_name, ()
            ) or [self.current_hlsl_available_functions.get(parameter_lookup_name)]
            range_reference_parameters = [
                list(getattr(candidate, "parameters", []) or [])
                for candidate in candidates
                if candidate is not None
            ]
        rendered_args = []
        argument_assignments = []
        for index, arg in enumerate(args):
            expected_type = (
                parameter_types[index] if index < len(parameter_types) else None
            )
            reference_qualifiers = {
                qualifier
                for parameters in range_reference_parameters
                if index < len(parameters)
                for qualifier in self.hlsl_parameter_qualifiers(parameters[index])
            }
            validate_range_reference_argument(
                self, arg, expected_type, reference_qualifiers
            )
            writable_parameter = index < len(parameters) and bool(
                set(self.hlsl_parameter_qualifiers(parameters[index])).intersection(
                    {"out", "inout"}
                )
            )
            if isinstance(arg, ArrayLiteralNode) and self.hlsl_outer_array_type(
                expected_type
            ):
                array_value = (
                    None
                    if writable_parameter
                    else self.hlsl_array_value_arguments.argument(arg, expected_type)
                )
                if array_value is not None:
                    rendered_args.append(array_value)
                    continue
                self.hlsl_fixed_array_aggregate_initializer(arg, expected_type)
                self.hlsl_aggregate_error(
                    arg,
                    expected_type=self.map_type(expected_type),
                    reason="array-argument-unsupported",
                    detail=(
                        "HLSL has no fixed-array literal expression for a call "
                        "argument; a materialized array temporary is required"
                    ),
                )
            if (
                index in workgroup_pointer_indices
                or index in resource_pointer_indices
                or index in resource_pointer_array_indices
                or index in private_pointer_indices
            ):
                rendered_args.append("")
            elif index < len(parameters) and set(
                self.hlsl_parameter_qualifiers(parameters[index])
            ).intersection({"out", "inout"}):
                # Value conversion would turn a writable argument into a temporary.
                # Capture the location for both copy-in and copy-out, including
                # aggregate receivers and indices changed inside the callee.
                if self.hlsl_array_value_arguments.writable_lowered_value(
                    arg, expected_type
                ):
                    rendered_args.append("")
                    continue
                byte = self.hlsl_byte_integer_shape(expected_type) is not None
                assignments, arg = self.hlsl_stabilize_inout_lvalue(
                    call_node or arg,
                    arg,
                    self.map_type(expected_type),
                    operation=(
                        "byte-reference argument" if byte else "reference argument"
                    ),
                    reason=(
                        "byte-reference-unstable-lvalue"
                        if byte
                        else "reference-unstable-lvalue"
                    ),
                )
                argument_assignments.extend(assignments)
                self.hlsl_tall_matrix_rows.mark_chain(arg)
                rendered_args.append(self.generate_expression_with_expected(arg, None))
            else:
                array_value = self.hlsl_array_value_arguments.argument(
                    arg, expected_type
                )
                rendered_args.append(
                    array_value
                    if array_value is not None
                    else self.generate_expression_with_expected(arg, expected_type)
                )
        return (
            self.generate_call_arguments_from_rendered(
                func_name,
                args,
                rendered_args,
                private_pointer_func_name=type_func_name or func_name,
                workgroup_pointer_func_name=workgroup_pointer_func_name,
            ),
            argument_assignments,
        )

    def validate_hlsl_byte_reference_arguments(self, args, parameters, call_node):
        references = []
        for arg, parameter in zip(args, parameters):
            qualifiers = set(self.hlsl_parameter_qualifiers(parameter))
            writable = bool(qualifiers.intersection({"out", "inout"}))
            source_qualifiers = set(getattr(parameter, "qualifiers", []) or [])
            raw_type = self.hlsl_parameter_raw_type(parameter)
            if not (
                writable
                or isinstance(raw_type, ReferenceType)
                or "thread" in source_qualifiers
            ):
                continue
            value_type = self.function_parameter_type_name(parameter)
            byte = self.hlsl_byte_integer_shape(value_type) is not None
            owner = arg
            while isinstance(owner, (ArrayAccessNode, MemberAccessNode, SwizzleNode)):
                if isinstance(owner, ArrayAccessNode):
                    owner = owner.array
                elif isinstance(owner, MemberAccessNode):
                    owner = owner.object_expr
                else:
                    owner = owner.vector_expr
            name = self.expression_name(owner)
            if not name:
                continue
            for (
                previous_arg,
                previous_name,
                previous_byte,
                previous_writable,
            ) in references:
                if (
                    name == previous_name
                    and (byte or previous_byte)
                    and (writable or previous_writable)
                    and not references_provably_disjoint(
                        previous_arg,
                        arg,
                        lambda value: self.literal_int_value(value, {}),
                    )
                ):
                    raise DirectXContextualConversionError(
                        "DirectX cannot preserve potentially overlapping byte "
                        "references through HLSL parameter copy-in/copy-out; "
                        "shared storage requires alias-aware call lowering",
                        source_type=value_type,
                        target_type=self.map_type(value_type),
                        reason="byte-reference-alias-unsupported",
                        source_location=getattr(call_node, "source_location", None),
                    )
            references.append((arg, name, byte, writable))

    def generate_call_arguments_from_rendered(
        self,
        func_name,
        args,
        rendered_args,
        private_pointer_func_name=None,
        workgroup_pointer_func_name=None,
    ):
        generated_args = []
        implicit_samplers = self.implicit_texture_sampler_parameters.get(func_name, {})
        param_names = self.function_parameter_names.get(func_name, [])
        private_pointer_func_name = private_pointer_func_name or func_name
        workgroup_pointer_func_name = workgroup_pointer_func_name or func_name
        pointer_array_bindings = {}
        resource_bindings = {}
        for index, arg in enumerate(args):
            if self.hlsl_resource_pointer_call_parameter_is_omitted(
                workgroup_pointer_func_name, index
            ):
                continue
            pointer_array_binding = (
                self.hlsl_resource_pointer_array_call_argument_binding(
                    workgroup_pointer_func_name, index, arg
                )
            )
            if pointer_array_binding is not None:
                pointer_array_bindings[index] = pointer_array_binding
                continue
            resource_binding = self.hlsl_resource_pointer_call_argument_binding(
                workgroup_pointer_func_name, index, arg
            )
            if resource_binding is not None:
                resource_bindings[index] = resource_binding
        self.record_hlsl_resource_access_call(
            workgroup_pointer_func_name, resource_bindings
        )
        self.hlsl_validate_resource_pointer_call_aliases(
            workgroup_pointer_func_name,
            args,
            pointer_array_bindings,
            resource_bindings,
        )

        for index, arg in enumerate(args):
            if self.hlsl_resource_pointer_call_parameter_is_omitted(
                workgroup_pointer_func_name, index
            ):
                continue
            pointer_array_binding = pointer_array_bindings.get(index)
            if pointer_array_binding is not None:
                generated_args.append(pointer_array_binding["root"])
                for offset in pointer_array_binding["offset_expressions"]:
                    generated_args.append(
                        offset
                        if pointer_array_binding["mutated"]
                        else f"int64_t({offset})"
                    )
                generated_args.append(
                    f"int64_t({pointer_array_binding.get('base_index') or '0'})"
                )
            else:
                workgroup_binding = self.hlsl_workgroup_pointer_call_argument_binding(
                    workgroup_pointer_func_name, index, arg
                )
                if workgroup_binding is not None:
                    generated_args.append(
                        f"int({workgroup_binding.get('offset', '0')})"
                    )
                else:
                    resource_binding = resource_bindings.get(index)
                    if resource_binding is not None:
                        resource_offset = self.hlsl_resource_pointer_call_offset(
                            resource_binding
                        )
                        if resource_binding.get("call_offset_direction") is None:
                            resource_offset = f"int64_t({resource_offset})"
                        generated_args.extend(
                            [
                                resource_binding["root"],
                                resource_offset,
                            ]
                        )
                    else:
                        private_binding = (
                            self.hlsl_private_pointer_call_argument_binding(
                                private_pointer_func_name, index, arg
                            )
                        )
                        if private_binding is None:
                            generated_args.append(rendered_args[index])
                        elif private_binding.get("scalar"):
                            generated_args.append(private_binding["backing"])
                        else:
                            generated_args.extend(
                                [
                                    private_binding["backing"],
                                    private_binding["offset"],
                                ]
                            )
            if index >= len(param_names):
                continue
            texture_param = param_names[index]
            if texture_param not in implicit_samplers:
                continue
            sampler_info = implicit_samplers[texture_param]
            if sampler_info["synthetic"]:
                generated_args.append(self.generate_implicit_sampler_argument(arg))
            regular_sampler_name = self.implicit_regular_sampler_name(
                texture_param, sampler_info
            )
            if (
                sampler_info.get("regular")
                and sampler_info.get("comparison")
                and regular_sampler_name != sampler_info["sampler_name"]
                and regular_sampler_name not in param_names
            ):
                generated_args.append(
                    self.generate_implicit_regular_sampler_argument(arg, sampler_info)
                )
            query_sampler_name = self.implicit_query_lod_sampler_name(
                texture_param, sampler_info
            )
            if (
                sampler_info.get("query_lod")
                and query_sampler_name != sampler_info["sampler_name"]
                and query_sampler_name not in param_names
            ):
                generated_args.append(
                    self.generate_implicit_query_lod_sampler_argument(arg, sampler_info)
                )

        for parameter in self.required_hlsl_stage_parameters(func_name):
            name = getattr(parameter, "name", None)
            if not name or name in param_names:
                continue
            generated_args.append(name)

        for builtin_name in self.required_hlsl_fragment_builtin_names(func_name):
            argument = self.current_identifier_aliases.get(builtin_name)
            if argument is None:
                # The caller has no parameter or local bound to the built-in (a vertex or compute caller, or a
                # helper whose built-in has no HLSL parameter type), and its CrossGL name is not an HLSL value.
                argument = self.refuse_lowering(
                    f"'{func_name}' reads fragment built-in '{builtin_name}', which its caller "
                    "has no HLSL value for",
                    builtin_name,
                )
            generated_args.append(argument)

        return generated_args

    def generate_implicit_sampler_argument(self, texture_arg):
        member_sampler = self.hlsl_implicit_member_sampler_name(texture_arg)
        if member_sampler:
            return member_sampler

        texture_name = self.expression_name(texture_arg)
        if texture_name in self.current_implicit_texture_samplers:
            return self.current_implicit_texture_samplers[texture_name]
        if texture_name:
            return f"{texture_name}Sampler"

        texture_expr = self.generate_expression(texture_arg)
        return f"{texture_expr}Sampler"

    def implicit_call_uses_regular_sampler(self, func_name):
        return (
            func_name in self.regular_texture_function_names()
            and not is_texture_query_lod_operation(func_name)
        )

    def implicit_regular_sampler_name(self, texture_name, sampler_info):
        if sampler_info.get("regular") and sampler_info.get("comparison"):
            return f"{texture_name}RegularSampler"
        return sampler_info["sampler_name"]

    def generate_implicit_regular_sampler_argument(
        self, texture_arg, sampler_info=None
    ):
        member_sampler = self.hlsl_implicit_member_sampler_name(
            texture_arg, sampler_info
        )
        if member_sampler:
            return member_sampler

        texture_name = self.expression_name(texture_arg)
        if texture_name in self.current_implicit_texture_regular_samplers:
            return self.current_implicit_texture_regular_samplers[texture_name]
        if texture_name in self.global_implicit_texture_regular_samplers:
            return self.global_implicit_texture_regular_samplers[texture_name]
        if texture_name:
            if sampler_info and sampler_info.get("comparison"):
                return f"{texture_name}RegularSampler"
            return f"{texture_name}Sampler"

        texture_expr = self.generate_expression(texture_arg)
        if sampler_info and sampler_info.get("comparison"):
            return f"{texture_expr}RegularSampler"
        return f"{texture_expr}Sampler"

    def implicit_query_lod_sampler_name(self, texture_name, sampler_info):
        if sampler_info.get("query_lod") and sampler_info.get("comparison"):
            return f"{texture_name}QuerySampler"
        return sampler_info["sampler_name"]

    def generate_implicit_query_lod_sampler_argument(
        self, texture_arg, sampler_info=None
    ):
        member_sampler = self.hlsl_implicit_member_query_lod_sampler_name(texture_arg)
        if member_sampler:
            return member_sampler

        texture_name = self.expression_name(texture_arg)
        if texture_name in self.current_implicit_texture_query_lod_samplers:
            return self.current_implicit_texture_query_lod_samplers[texture_name]
        if texture_name in self.global_implicit_texture_query_lod_samplers:
            return self.global_implicit_texture_query_lod_samplers[texture_name]
        if texture_name:
            if sampler_info and sampler_info.get("comparison"):
                return f"{texture_name}QuerySampler"
            return f"{texture_name}Sampler"

        texture_expr = self.generate_expression(texture_arg)
        if sampler_info and sampler_info.get("comparison"):
            return f"{texture_expr}QuerySampler"
        return f"{texture_expr}Sampler"

    def texture_resource_type(self, texture_arg):
        texture_name = self.expression_name(texture_arg)
        if texture_name:
            local_type = self.local_variable_types.get(texture_name)
            if local_type is not None and not self.is_texture_or_image_type(local_type):
                return None
            texture_type = self.current_texture_parameters.get(
                texture_name, self.texture_variable_types.get(texture_name)
            )
            if texture_type is not None:
                return texture_type

        arg_type = self.expression_result_type(texture_arg)
        if arg_type is None or not self.is_texture_or_image_type(arg_type):
            return None
        return self.map_resource_type_with_format(self.resource_base_type(arg_type))

    def texture_argument_resource_type(self, texture_arg):
        return self.texture_resource_type(texture_arg)

    def image_resource_access(self, texture_arg):
        return image_resource_metadata(
            texture_arg,
            self.expression_name,
            self.current_image_access_parameters,
            self.image_variable_accesses,
        )

    def image_resource_format(self, texture_arg):
        return image_resource_metadata(
            texture_arg,
            self.expression_name,
            self.current_image_format_parameters,
            self.image_variable_formats,
        )

    def validate_texture_resource_argument(self, func_name, args):
        if not args or func_name not in self.texture_resource_operation_names():
            return
        if self.texture_resource_type(args[0]) is not None:
            return
        arg_type = self.expression_result_type(args[0])
        if arg_type is None:
            raise self.unresolved_source_resource_type_error(
                func_name, args[0], "texture or image resource"
            )
        if arg_type is not None and self.is_texture_or_image_type(arg_type):
            return

        texture_name = self.expression_name(args[0]) or str(args[0])
        raise DirectXRefusal(
            f"DirectX texture operation '{func_name}' requires a declared "
            f"texture or image resource argument: {texture_name}"
        )

    def unresolved_source_resource_type_error(self, func_name, arg, resource_role):
        expression_name = expression_debug_name(arg)
        function_name = self.current_function_name or "<global>"
        return DirectXUnresolvedSourceTypeError(
            "DirectX translation could not resolve source type metadata for "
            f"{resource_role} argument '{expression_name}' used by operation "
            f"'{func_name}' in function '{function_name}'. Declare or annotate "
            "the source resource type before translating to DirectX."
        )

    def validate_image_resource_argument(self, func_name, args):
        if not args or not is_image_resource_operation(
            func_name, IMAGE_RESOURCE_INTRINSIC_NAMES
        ):
            return
        texture_type = self.texture_argument_resource_type(args[0])
        if self.is_storage_image_resource_type(texture_type):
            return
        arg_type = self.expression_result_type(args[0])
        if texture_type is None and arg_type is None:
            raise self.unresolved_source_resource_type_error(
                func_name, args[0], "storage image resource"
            )
        texture_name = self.expression_name(args[0]) or str(args[0])
        raise DirectXRefusal(
            f"DirectX image operation '{func_name}' requires a storage "
            f"image resource argument: {texture_name}"
        )

    def validate_image_access_argument(self, func_name, args):
        if not args or not is_image_resource_operation(
            func_name, IMAGE_RESOURCE_INTRINSIC_NAMES
        ):
            return
        access = self.image_resource_access(args[0])
        if access is None or access == "read_write":
            return
        texture_name = expression_debug_name(args[0])
        if func_name == "imageLoad" and access == "write":
            raise DirectXRefusal(
                f"DirectX image operation '{func_name}' requires read-capable "
                f"storage image access for {texture_name}: got writeonly"
            )
        if func_name == "imageStore" and access == "read":
            raise DirectXRefusal(
                f"DirectX image operation '{func_name}' requires write-capable "
                f"storage image access for {texture_name}: got readonly"
            )
        if self.image_atomic_intrinsic(func_name):
            access_name = "readonly" if access == "read" else "writeonly"
            raise DirectXRefusal(
                f"DirectX image operation '{func_name}' requires read-write "
                f"storage image access for {texture_name}: got {access_name}"
            )

    def hlsl_storage_image_component_type(self, image_type):
        image_type = self.resource_base_type(image_type)
        if "<" not in image_type or ">" not in image_type:
            return None
        # A unorm or snorm element has the shape of its float type.
        return strip_texel_modifier(image_type.split("<", 1)[1].split(">", 1)[0])

    def hlsl_storage_image_component_kind(self, image_type):
        component_type = self.hlsl_storage_image_component_type(image_type)
        component_kind = self.vector_component_type(component_type)
        if component_kind in {"float", "int", "uint"}:
            return component_kind
        return None

    def image_atomic_value_arguments(self, func_name, args, image_type):
        has_sample = self.is_multisample_storage_image_resource_type(image_type)
        return shared_image_atomic_value_arguments(func_name, args, has_sample)

    def scalar_expression_kind(self, expr):
        return numeric_scalar_expression_kind(
            expr,
            self.expression_result_type,
            self.type_name_string,
            self.map_type,
        )

    def validate_image_atomic_value_argument_types(
        self, func_name, args, image_type, component_kind, image_format
    ):
        mismatch = image_atomic_value_kind_mismatch(
            func_name,
            self.image_atomic_value_arguments(func_name, args, image_type),
            component_kind,
            self.scalar_expression_kind,
        )
        if mismatch is None:
            return
        value_arg, value_kind = mismatch
        format_label = image_format or self.resource_base_type(image_type)
        raise DirectXRefusal(
            image_atomic_value_kind_error(
                "DirectX",
                func_name,
                format_label,
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

    def hlsl_atomic_result_expected_label(self, expected_type):
        type_name = self.type_name_string(expected_type)
        if not type_name:
            return None
        expected_kind = numeric_scalar_type_kind(
            type_name,
            self.type_name_string,
            self.map_type,
        )
        if expected_kind is not None:
            return expected_kind
        return self.map_type(type_name)

    def validate_image_atomic_result_type(
        self, func_name, image_type, component_kind, image_format
    ):
        expected_label = self.hlsl_atomic_result_expected_label(
            self.current_expression_expected_type
        )
        if expected_label is None:
            return
        expected_kind = image_atomic_result_kind_mismatch(
            expected_label, component_kind
        )
        if expected_kind is None and expected_label == component_kind:
            return
        format_label = image_format or self.resource_base_type(image_type)
        raise DirectXRefusal(
            image_atomic_result_kind_error(
                "DirectX",
                func_name,
                format_label,
                component_kind,
                expected_kind or expected_label,
            )
        )

    def validate_image_atomic_format_argument(self, func_name, args):
        if not is_image_atomic_operation(func_name) or not args:
            return
        image_type = self.texture_argument_resource_type(args[0])
        image_format = self.image_resource_format(args[0])
        component_kind = resolve_image_atomic_component_kind(
            func_name,
            image_format,
            self.hlsl_storage_image_component_type(image_type),
            "DirectX",
            self.resource_base_type(image_type),
        )
        self.validate_image_atomic_value_argument_types(
            func_name,
            args,
            image_type,
            component_kind,
            image_format,
        )
        self.validate_image_atomic_result_type(
            func_name,
            image_type,
            component_kind,
            image_format,
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
            raise DirectXRefusal(
                f"DirectX function call '{func_name}' requires {required_label} "
                f"storage image access for argument {actual_name} passed to "
                f"parameter {param_name}: got {actual_label}"
            )

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
        raise DirectXRefusal(
            operation_argument_type_error(
                "DirectX",
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
        raise DirectXRefusal(
            operation_dimension_argument_error(
                "DirectX",
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
                raise DirectXRefusal(
                    operation_argument_type_error(
                        "DirectX",
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
            raise DirectXRefusal(
                operation_dimension_argument_error(
                    "DirectX",
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
            raise DirectXRefusal(
                texture_query_lod_coordinate_type_error(
                    "DirectX",
                    func_name,
                    expression_debug_name(args[coord_index]),
                    self.type_name_string(coord_type),
                )
            )
        if coord_dimension == expected_dimension:
            return
        raise DirectXRefusal(
            texture_query_lod_coordinate_dimension_error(
                "DirectX",
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
                raise DirectXRefusal(
                    operation_argument_type_error(
                        "DirectX",
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
            raise DirectXRefusal(
                operation_dimension_argument_error(
                    "DirectX",
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
        raise DirectXRefusal(
            operation_argument_type_error(
                "DirectX",
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
        raise DirectXRefusal(
            operation_argument_type_error(
                "DirectX",
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
        raise DirectXRefusal(
            operation_argument_type_error(
                "DirectX",
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
        raise DirectXRefusal(
            operation_argument_type_error(
                "DirectX",
                "resource",
                func_name,
                "a scalar integer",
                "mip/sample level",
                expression_debug_name(args[level_index]),
                self.type_name_string(level_type),
            )
        )

    def validate_sample_index_argument(self, func_name, args):
        sample_index = (
            1
            if func_name == "textureSamplePosition" and len(args) > 1
            else texture_sample_index_argument_index(func_name, len(args))
        )
        if sample_index is None:
            return
        texture_type = self.texture_argument_resource_type(args[0])
        if not self.is_multisample_texture_resource_type(texture_type):
            return
        sample_type = self.texture_argument_diagnostic_type(args[sample_index])
        if sample_type is None or self.is_scalar_integer_type(sample_type):
            return
        if func_name == "textureSamplePosition":
            raise DirectXRefusal(
                operation_argument_type_error(
                    "DirectX",
                    "texture sample-position query",
                    func_name,
                    "a scalar integer",
                    "sample index",
                    expression_debug_name(args[sample_index]),
                    self.type_name_string(sample_type),
                )
            )
        raise DirectXRefusal(
            texture_multisample_sample_type_error(
                "DirectX",
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
            self.is_multisample_storage_image_resource_type(texture_type),
            "DirectX",
        )
        if sample_index is None:
            return
        sample_type = self.texture_argument_diagnostic_type(args[sample_index])
        sample_type = image_multisample_sample_type_mismatch(
            sample_type, self.is_scalar_integer_type
        )
        if sample_type is None:
            return
        raise DirectXRefusal(
            image_multisample_sample_type_error(
                "DirectX",
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
        raise DirectXRefusal(
            operation_argument_type_error(
                "DirectX",
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
            "DirectX",
            func_name,
            args,
            self.texture_resource_operation_names(),
            self.texture_call_uses_explicit_sampler,
        )

    def texture_resource_operation_names(self):
        return texture_image_resource_operation_names(IMAGE_RESOURCE_INTRINSIC_NAMES)

    def four_component_image_store_constructor(self, texture_type):
        component_type = self.hlsl_storage_image_component_type(texture_type)
        return storage_image_store_vector_constructor(
            component_type,
            4,
            self.hlsl_storage_image_component_kind(texture_type),
        )

    def image_store_zero_values_by_kind(self):
        return storage_image_zero_values()

    def two_component_image_store_constructor(self, texture_type):
        component_type = self.hlsl_storage_image_component_type(texture_type)
        return storage_image_store_vector_constructor(
            component_type,
            2,
            self.hlsl_storage_image_component_kind(texture_type),
            zero_values_by_kind=self.image_store_zero_values_by_kind(),
        )

    def image_store_expected_value_type(self, image_type, image_format, value_arg=None):
        if value_arg is not None and self.is_vector_value_type(
            self.expression_result_type(value_arg)
        ):
            return None
        if image_format:
            return image_format_component_kind(image_format)
        component_kind = self.hlsl_storage_image_component_kind(image_type)
        if component_kind in {"float", "int", "uint"}:
            return component_kind
        return None

    def image_load_result_type(self, image_arg):
        # imageLoad yields four components whatever the format.
        image_format = self.image_resource_format(image_arg)
        if image_format:
            component_kind = image_format_component_kind(image_format)
        else:
            image_type = self.texture_argument_resource_type(image_arg)
            component_kind = self.hlsl_storage_image_component_kind(image_type)
        return texel_vector_type(
            component_kind, {"float": "float", "int": "int", "uint": "uint"}
        )

    def hlsl_storage_image_element_count(self, image_type):
        """Components of the HLSL element type of a storage image, or ``None``."""
        component_type = self.hlsl_storage_image_component_type(image_type)
        if component_type is None:
            return None
        return self.value_component_count(component_type)

    def image_atomic_result_type(self, func_name, image_arg):
        image_format = self.image_resource_format(image_arg)
        if image_format:
            return image_format_component_type(image_format)
        image_type = self.texture_argument_resource_type(image_arg)
        component_type = self.hlsl_storage_image_component_type(image_type)
        component_kind = self.vector_component_type(component_type)
        if component_kind in {"float", "int", "uint"}:
            return component_kind
        return component_type

    def image_load_component_kind(self, image_type, image_format):
        if image_format:
            return image_format_component_kind(image_format)
        return self.hlsl_storage_image_component_kind(image_type)

    def image_load_channel_count(self, image_type, image_format):
        component_type = self.hlsl_storage_image_component_type(image_type)
        component_count = self.value_component_count(component_type)
        if image_format is None and component_count == 1:
            default_count = default_storage_image_channel_count(
                self.hlsl_storage_image_component_kind(image_type)
            )
            if default_count is not None:
                return default_count
        return image_format_or_default_channel_count(
            image_format,
            component_count,
        )

    def expected_component_kind(self):
        return numeric_component_kind_from_type(
            self.current_expression_expected_type,
            self.type_name_string,
            self.map_type,
            self.vector_component_type,
        )

    def expected_component_count(self):
        return self.value_component_count(self.current_expression_expected_type)

    def expression_component_kind(self, expr):
        return numeric_expression_component_kind(
            expr,
            self.expression_result_type,
            self.type_name_string,
            self.map_type,
            self.vector_component_type,
        )

    def value_component_count(self, vtype):
        return numeric_component_count_from_type(
            vtype,
            self.type_name_string,
            self.map_type,
            self.vector_component_type,
            scalar_types={
                "float",
                "float16_t",
                "half",
                "min16float",
                "min10float",
                "double",
                "int64_t",
                "uint64_t",
                "int16_t",
                "uint16_t",
                "int",
                "min16int",
                "min12int",
                "uint",
                "min16uint",
                "bool",
            },
            excluded_type_markers=("x",),
        )

    def expression_component_count(self, expr):
        return numeric_expression_component_count(
            expr,
            self.expression_result_type,
            self.type_name_string,
            self.map_type,
            self.vector_component_type,
            scalar_types={
                "float",
                "float16_t",
                "half",
                "min16float",
                "min10float",
                "double",
                "int64_t",
                "uint64_t",
                "int16_t",
                "uint16_t",
                "int",
                "min16int",
                "min12int",
                "uint",
                "min16uint",
                "bool",
            },
            excluded_type_markers=("x",),
        )

    def hlsl_expression_is_repeatable(self, expr):
        if expr is None:
            return False
        if isinstance(expr, ConstructorNode):
            return all(
                self.hlsl_expression_is_repeatable(arg)
                for arg in getattr(expr, "arguments", None) or []
            )
        if isinstance(expr, (int, float, str)):
            return True
        if hasattr(expr, "__class__") and "Literal" in str(expr.__class__):
            return True
        if hasattr(expr, "__class__") and "Identifier" in str(expr.__class__):
            return True
        if isinstance(expr, VariableNode):
            return True
        if isinstance(expr, MemberAccessNode):
            return self.hlsl_expression_is_repeatable(
                getattr(expr, "object", getattr(expr, "object_expr", None))
            )
        if isinstance(expr, ArrayAccessNode):
            index = getattr(expr, "index", None)
            indices = getattr(expr, "indices", None)
            if indices is None and index is not None:
                indices = [index]
            return self.hlsl_expression_is_repeatable(
                getattr(expr, "array", None)
            ) and all(
                self.hlsl_expression_is_repeatable(item) for item in indices or []
            )
        if isinstance(expr, UnaryOpNode):
            if self.map_operator(getattr(expr, "op", None)) in {"++", "--"}:
                return False
            return self.hlsl_expression_is_repeatable(getattr(expr, "operand", None))
        if isinstance(expr, BinaryOpNode):
            return self.hlsl_expression_is_repeatable(
                getattr(expr, "left", None)
            ) and self.hlsl_expression_is_repeatable(getattr(expr, "right", None))
        if isinstance(expr, TernaryOpNode) or (
            hasattr(expr, "__class__") and "TernaryOp" in str(expr.__class__)
        ):
            return all(
                self.hlsl_expression_is_repeatable(getattr(expr, name, None))
                for name in ("condition", "true_expr", "false_expr")
            )
        if isinstance(expr, FunctionCallNode) or (
            hasattr(expr, "__class__") and "FunctionCall" in str(expr.__class__)
        ):
            func_expr = getattr(expr, "function", getattr(expr, "name", None))
            func_name = getattr(func_expr, "name", func_expr)
            args = getattr(expr, "arguments", getattr(expr, "args", []))
            return (
                self.is_scalar_value_type(func_name)
                or self.hlsl_type_constructor_name(func_name) is not None
                or self.hlsl_crossgl_vector_constructor(func_name) is not None
            ) and all(self.hlsl_expression_is_repeatable(arg) for arg in args)
        return False

    def hlsl_crossgl_vector_constructor(self, name):
        """The HLSL vector type spelled by a CrossGL constructor such as ``vec3``, or ``None``."""
        if not isinstance(name, str):
            return None
        mapped = self.map_type(name)
        if isinstance(mapped, str) and re.fullmatch(
            r"(?:float|half|int|uint|bool|double)[234]", mapped
        ):
            return mapped
        return None

    def hlsl_scalar_splat_cast(self, constructor_type, rendered_arg):
        return f"(({self.map_type(constructor_type)})({rendered_arg}))"

    def hlsl_half_literal_is_exact(self, expression):
        if isinstance(expression, UnaryOpNode) and self.map_operator(expression.op) in {
            "+",
            "-",
        }:
            return self.hlsl_half_literal_is_exact(expression.operand)
        if not isinstance(expression, LiteralNode):
            return False
        value = expression.value
        if not isinstance(value, (int, float)):
            return False
        try:
            return struct.unpack("<e", struct.pack("<e", value))[0] == value
        except (OverflowError, struct.error):
            return False
