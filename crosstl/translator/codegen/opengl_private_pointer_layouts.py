"""Private pointer layouts in the GLSL generator: image store value types, component counts, sampler parameters, scalar struct views and contract keys."""

from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    ArrayType,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    PointerReinterpretNode,
    PointerType,
    ReferenceType,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from ..refusals import OpenGLRefusal
from ..validation import (
    collect_non_resource_global_resource_shadows,
    expression_debug_name,
)
from .array_utils import (
    get_array_size_from_node,
    split_array_type_suffix,
)
from .generic_function_utils import (
    generic_function_call_name,
    generic_function_value_arguments,
)
from .image_access_contracts import (
    default_storage_image_channel_count,
    image_format_component_kind,
    image_format_or_default_channel_count,
    image_load_result_kind_error,
    image_load_result_kind_mismatch,
    image_load_result_shape_error,
    image_store_value_kind_error,
    image_store_value_kind_mismatch,
    image_store_value_shape_error,
    numeric_component_count_from_type,
    numeric_component_kind_from_type,
    numeric_expression_component_count,
    numeric_expression_component_kind,
    should_validate_image_load_result_shape,
)
from .opengl_values import OpenGLPrivatePointerParameterError
from .pointer_reinterpret import (
    PointerReinterpretationError,
    scalar_storage_layout,
)
from .private_array_rows import array_row_shape
from .stage_utils import collect_stage_local_variables
from .storage_image_texels import (
    load_context_allowed,
    store_value_allowed,
    texel_vector_type,
)


class OpenGLPrivatePointerLayoutMixin:
    """Private pointer layout members of ``GLSLCodeGen``."""

    def image_store_expected_value_type(
        self, texture_type, image_format, value_arg=None
    ):
        if value_arg is not None and self.is_vector_value_type(
            self.expression_result_type(value_arg)
        ):
            return None
        if image_format:
            return image_format_component_kind(image_format)
        component_kind = self.glsl_storage_image_component_kind(texture_type)
        if component_kind in {"float", "int", "uint"}:
            return component_kind
        return None

    def image_load_result_type(self, image_arg):
        image_format = self.image_resource_format(image_arg)
        if image_format:
            component_kind = image_format_component_kind(image_format)
        else:
            image_type = self.texture_argument_resource_type(image_arg)
            component_kind = self.glsl_storage_image_component_kind(image_type)
        return texel_vector_type(
            component_kind, {"float": "vec", "int": "ivec", "uint": "uvec"}
        )

    def image_load_component_kind(self, texture_type, image_format):
        if image_format:
            return image_format_component_kind(image_format)
        return self.glsl_storage_image_component_kind(texture_type)

    def image_load_channel_count(self, texture_type, image_format):
        return image_format_or_default_channel_count(
            image_format,
            default_storage_image_channel_count(
                self.glsl_storage_image_component_kind(texture_type)
            ),
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
            scalar_types={"float", "double", "int", "uint", "bool"},
            excluded_type_markers=("mat",),
        )

    def expression_component_count(self, expr):
        return numeric_expression_component_count(
            expr,
            self.expression_result_type,
            self.type_name_string,
            self.map_type,
            self.vector_component_type,
            scalar_types={"float", "double", "int", "uint", "bool"},
            excluded_type_markers=("mat",),
        )

    def image_store_channel_count(self, texture_type, image_format):
        return image_format_or_default_channel_count(
            image_format,
            default_storage_image_channel_count(
                self.glsl_storage_image_component_kind(texture_type)
            ),
        )

    def validate_image_store_value_shape(self, texture_type, image_format, value_arg):
        expected_channels = self.image_store_channel_count(texture_type, image_format)
        value_channels = self.expression_component_count(value_arg)
        if store_value_allowed(expected_channels, value_channels):
            return
        format_label = image_format or self.resource_base_type(texture_type)
        raise OpenGLRefusal(
            image_store_value_shape_error(
                "OpenGL",
                format_label,
                expression_debug_name(value_arg),
                expected_channels,
                value_channels,
            )
        )

    def validate_image_store_value_type(self, texture_type, image_format, value_arg):
        value_type = self.expression_result_type(value_arg)
        if self.is_matrix_value_type(value_type):
            format_label = image_format or self.resource_base_type(texture_type)
            mapped_type = self.map_type(self.type_name_string(value_type))
            raise OpenGLRefusal(
                "OpenGL image store operation 'imageStore' requires scalar or "
                f"vector value for {format_label} images: "
                f"{expression_debug_name(value_arg)} has type {mapped_type}"
            )
        self.validate_image_store_value_shape(texture_type, image_format, value_arg)
        expected_kind = self.image_store_expected_value_type(texture_type, image_format)
        value_kind = image_store_value_kind_mismatch(
            expected_kind, self.expression_component_kind(value_arg)
        )
        if value_kind is None:
            return
        format_label = image_format or self.resource_base_type(texture_type)
        raise OpenGLRefusal(
            image_store_value_kind_error(
                "OpenGL",
                format_label,
                expression_debug_name(value_arg),
                expected_kind,
                value_kind,
            )
        )

    def validate_image_load_result_type(self, texture_type, image_format):
        if self.is_matrix_value_type(self.current_expression_expected_type):
            format_label = image_format or self.resource_base_type(texture_type)
            expected_type = self.map_type(self.current_expression_expected_type)
            raise OpenGLRefusal(
                "OpenGL image load operation 'imageLoad' requires scalar or "
                f"vector result context for {format_label} images: "
                f"got {expected_type}"
            )
        expected_kind = self.expected_component_kind()
        component_kind = self.image_load_component_kind(texture_type, image_format)
        format_label = image_format or self.resource_base_type(texture_type)
        if not should_validate_image_load_result_shape(expected_kind, component_kind):
            return
        expected_kind = image_load_result_kind_mismatch(expected_kind, component_kind)
        if expected_kind is not None:
            raise OpenGLRefusal(
                image_load_result_kind_error(
                    "OpenGL", format_label, component_kind, expected_kind
                )
            )
        expected_channels = self.expected_component_count()
        loaded_channels = self.image_load_channel_count(texture_type, image_format)
        if load_context_allowed(loaded_channels, expected_channels):
            return
        raise OpenGLRefusal(
            image_load_result_shape_error(
                "OpenGL", format_label, loaded_channels, expected_channels
            )
        )

    def collect_function_sampler_parameter_indices(self, root):
        sampler_indices = {}
        visited = set()

        def visit(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    visit(item)
                return
            value_id = id(value)
            if value_id in visited:
                return
            visited.add(value_id)

            name = getattr(value, "name", None)
            params = getattr(value, "parameters", getattr(value, "params", []))
            if name and params:
                indices = []
                for index, param in enumerate(params):
                    param_type = getattr(
                        param, "param_type", getattr(param, "vtype", None)
                    )
                    if self.is_sampler_type(param_type):
                        indices.append(index)
                if indices:
                    sampler_indices[name] = set(indices)

            if hasattr(value, "__dict__"):
                for child in vars(value).values():
                    visit(child)

        visit(root)
        return sampler_indices

    def collect_global_resource_names(self, root):
        resource_names = set()
        for node in getattr(root, "global_variables", []) or []:
            var_type = getattr(node, "var_type", getattr(node, "vtype", "float"))
            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            base_type = self.resource_base_type(var_type)
            mapped_type = self.map_resource_type_with_format(base_type, node)
            if var_name and (
                self.is_sampler_type(base_type)
                or self.is_opaque_resource_type(mapped_type)
            ):
                resource_names.add(var_name)
        return resource_names

    def validate_global_resource_shadows(self, ast):
        conflicts = collect_non_resource_global_resource_shadows(
            ast,
            self.collect_global_resource_names(ast),
            self.is_inferable_resource_type,
        )
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise OpenGLRefusal(
                "Non-resource local declaration(s) shadow OpenGL global resource(s): "
                f"{names}"
            )

    def filter_sampler_arguments(self, func_name, args):
        skipped_indices = self.skipped_function_parameter_indices(func_name)
        if not skipped_indices:
            return args
        return [arg for index, arg in enumerate(args) if index not in skipped_indices]

    def should_preserve_sampler_resource(self, name):
        return name in getattr(self, "preserved_sampler_variables", set())

    def should_preserve_sampler_parameter(self, func_name, param_name):
        if not func_name:
            return False
        return param_name in getattr(self, "preserved_sampler_parameters", {}).get(
            func_name, set()
        )

    def expression_uses_nonuniform_ext(self, expr):
        for node in self.walk_ast(expr):
            if (
                isinstance(node, FunctionCallNode)
                and self.function_call_name(node) == "nonuniformEXT"
            ):
                return True
        return False

    def collect_preserved_nonuniform_sampler_resources(self, ast, functions):
        global_samplers = set()
        for node in [
            *(getattr(ast, "global_variables", []) or []),
            *collect_stage_local_variables(ast),
        ]:
            name = self.resource_node_name(node)
            if name and self.is_sampler_type(self.resource_node_type(node)):
                global_samplers.add(name)

        sampler_parameters = {}
        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            sampler_parameters[func_name] = {
                getattr(param, "name", None)
                for param in getattr(func, "parameters", getattr(func, "params", []))
                if self.is_sampler_type(
                    getattr(param, "param_type", getattr(param, "vtype", None))
                )
            }

        preserved_globals = set()
        preserved_params = {name: set() for name in sampler_parameters}

        changed = True
        while changed:
            changed = False
            for func in functions or []:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue

                for node in self.walk_ast(getattr(func, "body", [])):
                    index_expr = getattr(
                        node, "index", getattr(node, "index_expr", None)
                    )
                    if isinstance(
                        node, ArrayAccessNode
                    ) and self.expression_uses_nonuniform_ext(index_expr):
                        array_name = self.expression_name(
                            getattr(node, "array", getattr(node, "array_expr", None))
                        )
                        if array_name in sampler_parameters.get(func_name, set()):
                            if array_name not in preserved_params[func_name]:
                                preserved_params[func_name].add(array_name)
                                changed = True
                        elif array_name in global_samplers:
                            if array_name not in preserved_globals:
                                preserved_globals.add(array_name)
                                changed = True

                    if not isinstance(node, FunctionCallNode):
                        continue
                    callee_name = self.function_call_name(node)
                    callee_preserved = preserved_params.get(callee_name)
                    if not callee_preserved:
                        continue
                    callee_params = self.function_parameter_names.get(callee_name, [])
                    args = list(getattr(node, "arguments", getattr(node, "args", [])))
                    for index, arg in enumerate(args):
                        if index >= len(callee_params):
                            break
                        if callee_params[index] not in callee_preserved:
                            continue
                        arg_name = self.expression_name(arg)
                        if arg_name in sampler_parameters.get(func_name, set()):
                            if arg_name not in preserved_params[func_name]:
                                preserved_params[func_name].add(arg_name)
                                changed = True
                        elif arg_name in global_samplers:
                            if arg_name not in preserved_globals:
                                preserved_globals.add(arg_name)
                                changed = True

        self.preserved_sampler_variables = preserved_globals
        self.preserved_sampler_parameters = {
            func_name: names for func_name, names in preserved_params.items() if names
        }

    def glsl_private_pointer_propagated_parameter_intervals(
        self,
        functions_by_key,
        source_candidates,
        pointer_parameters,
    ):
        scalar_parameters = {}
        for function_key in pointer_parameters:
            function = functions_by_key.get(function_key)
            if function is None:
                continue
            names = []
            for parameter in (
                getattr(function, "parameters", getattr(function, "params", [])) or []
            ):
                parameter_type = getattr(
                    parameter,
                    "param_type",
                    getattr(parameter, "vtype", None),
                )
                if self.is_scalar_integer_type(parameter_type):
                    name = getattr(parameter, "name", None)
                    if name:
                        names.append(name)
            if names:
                scalar_parameters[function_key] = tuple(names)
        if not scalar_parameters:
            return {}

        propagated = {}
        missing = object()
        converged = False
        for _iteration in range(len(functions_by_key) + 1):
            for function_key, function in functions_by_key.items():
                previous = getattr(
                    function,
                    "_glsl_workgroup_proof_parameter_intervals",
                    missing,
                )
                inherited = {} if previous is missing else dict(previous or {})
                inherited.update(propagated.get(function_key, {}))
                if inherited:
                    function._glsl_workgroup_proof_parameter_intervals = inherited
                try:
                    self.annotate_glsl_control_flow_intervals(function)
                finally:
                    if previous is missing:
                        if hasattr(
                            function,
                            "_glsl_workgroup_proof_parameter_intervals",
                        ):
                            delattr(
                                function,
                                "_glsl_workgroup_proof_parameter_intervals",
                            )
                    else:
                        function._glsl_workgroup_proof_parameter_intervals = previous

            call_counts = {function_key: 0 for function_key in scalar_parameters}
            call_intervals = {
                function_key: {name: [] for name in names}
                for function_key, names in scalar_parameters.items()
            }
            blocked = set()
            for caller in functions_by_key.values():
                for call in self.walk_glsl_reachable_ast(getattr(caller, "body", [])):
                    if not isinstance(call, FunctionCallNode):
                        continue
                    arguments = list(
                        getattr(call, "arguments", getattr(call, "args", [])) or []
                    )
                    raw_name = self.function_call_name(call)
                    specialized_name = generic_function_call_name(
                        self,
                        raw_name,
                        arguments,
                    )
                    normalized_name = (
                        self.glsl_private_pointer_materialized_function_name(raw_name)
                    )
                    possible_targets = set()
                    for candidate_name in (
                        raw_name,
                        normalized_name,
                        specialized_name,
                    ):
                        if candidate_name in pointer_parameters:
                            possible_targets.add(candidate_name)
                        possible_targets.update(
                            candidate
                            for candidate in source_candidates.get(candidate_name, ())
                            if candidate in pointer_parameters
                        )
                    target_key = self.glsl_private_pointer_call_target_key(
                        call,
                        source_candidates,
                    )
                    if target_key is None:
                        blocked.update(possible_targets)
                        continue
                    if target_key not in scalar_parameters:
                        continue
                    callee = functions_by_key.get(target_key)
                    if callee is None:
                        blocked.add(target_key)
                        continue
                    value_arguments = arguments
                    if specialized_name == target_key:
                        value_arguments = list(
                            generic_function_value_arguments(
                                self,
                                raw_name,
                                arguments,
                            )
                        )
                    intervals = self.glsl_workgroup_call_parameter_intervals(
                        callee,
                        value_arguments,
                    )
                    call_counts[target_key] += 1
                    for name in scalar_parameters[target_key]:
                        call_intervals[target_key][name].append(intervals.get(name))

            next_propagated = {}
            for function_key, names in scalar_parameters.items():
                call_count = call_counts[function_key]
                if function_key in blocked or call_count == 0:
                    continue
                for name in names:
                    values = call_intervals[function_key][name]
                    if len(values) != call_count or any(
                        not (
                            isinstance(interval, tuple)
                            and len(interval) == 2
                            and all(
                                isinstance(bound, int) and not isinstance(bound, bool)
                                for bound in interval
                            )
                            and interval[0] <= interval[1]
                        )
                        for interval in values
                    ):
                        continue
                    next_propagated.setdefault(function_key, {})[name] = (
                        min(interval[0] for interval in values),
                        max(interval[1] for interval in values),
                    )
            if next_propagated == propagated:
                converged = True
                break
            propagated = next_propagated

        if converged:
            return propagated
        for function in functions_by_key.values():
            self.annotate_glsl_control_flow_intervals(function)
        return {}

    @staticmethod
    def glsl_private_pointer_byte_view_variant(value):
        return (
            isinstance(value, tuple)
            and len(value) == 8
            and value[0] == "private-word-array-view"
        )

    def glsl_private_pointer_variant_extent(self, value):
        if self.glsl_private_pointer_byte_view_variant(value):
            return value[4]
        return value

    def glsl_private_pointer_variant_sort_key(self, variant):
        return tuple(
            (
                (1, repr(shape))
                if self.glsl_private_pointer_byte_view_variant(shape)
                else (0, int(shape))
            )
            for shape in variant
        )

    def glsl_private_pointer_byte_view_error(
        self,
        expression,
        source_type,
        target_type,
        reason,
        detail,
        *,
        access="read",
    ):
        target_layout = scalar_storage_layout(target_type)
        raise PointerReinterpretationError(
            "OpenGL cannot preserve a private pointer reinterpretation from "
            f"'{source_type or 'unknown'}' to '{target_type or 'unknown'}': {detail}",
            source_type=source_type,
            target_type=target_type,
            address_space="thread",
            alignment=target_layout.byte_width if target_layout else None,
            access=access,
            target_backend="opengl",
            reason=reason,
            source_location=getattr(expression, "source_location", None),
        )

    def glsl_private_scalar_struct_reinterpret_node(self, expression):
        if not (
            isinstance(expression, UnaryOpNode)
            and self.map_operator(expression.op) == "*"
            and isinstance(expression.operand, PointerReinterpretNode)
        ):
            return None
        reinterpretation = expression.operand
        pointer_type = getattr(reinterpretation, "target_type", None)
        target_type_node = getattr(pointer_type, "pointee_type", None)
        target_type = self.glsl_normalized_source_type(target_type_node)
        if not target_type or scalar_storage_layout(target_type) is not None:
            return None
        return (
            reinterpretation
            if self.glsl_struct_node(self.map_type(target_type)) is not None
            else None
        )

    def glsl_private_scalar_struct_reinterpret_error(
        self,
        expression,
        *,
        source_type,
        target_type,
        reason,
        detail,
        alignment=None,
        access="read",
    ):
        raise PointerReinterpretationError(
            "OpenGL cannot preserve a private scalar-to-struct view from "
            f"'{source_type or 'unknown'}' to '{target_type or 'unknown'}': {detail}",
            source_type=source_type,
            target_type=target_type,
            address_space=(
                getattr(getattr(expression, "target_type", None), "address_space", None)
                or "thread"
            ),
            alignment=alignment,
            access=access,
            target_backend="opengl",
            reason=reason,
            source_location=getattr(expression, "source_location", None),
        )

    def glsl_private_scalar_struct_reinterpret_contract(
        self,
        expression,
        destination_type=None,
    ):
        reinterpretation = self.glsl_private_scalar_struct_reinterpret_node(expression)
        if reinterpretation is None:
            return None

        pointer_type = getattr(reinterpretation, "target_type", None)
        target_type_node = getattr(pointer_type, "pointee_type", None)
        target_type = self.glsl_normalized_source_type(target_type_node)
        mapped_target_type = self.map_type(target_type)
        struct = self.glsl_struct_node(mapped_target_type)

        def reject(reason, detail, *, source_type=None, alignment=None, access="read"):
            self.glsl_private_scalar_struct_reinterpret_error(
                reinterpretation,
                source_type=source_type,
                target_type=target_type,
                reason=reason,
                detail=detail,
                alignment=alignment,
                access=access,
            )

        address_space = str(getattr(pointer_type, "address_space", None) or "thread")
        if address_space.lower() not in {"thread", "private", "function"}:
            reject(
                "unsupported-private-scalar-struct-address-space",
                f"address space '{address_space}' is not private storage",
            )

        source_address = getattr(reinterpretation, "expression", None)
        source_value = getattr(source_address, "operand", None)
        source_name = getattr(source_value, "name", None)
        if (
            not isinstance(source_address, UnaryOpNode)
            or self.map_operator(source_address.op) != "&"
            or not isinstance(source_value, (IdentifierNode, VariableNode))
            or not source_name
        ):
            reject(
                "private-scalar-struct-backing-unresolved",
                "the source is not the stable address of one scalar identifier",
            )

        source_type_node = (
            self.local_variable_source_types.get(source_name)
            or self.local_variable_types.get(source_name)
            or getattr(source_value, "var_type", getattr(source_value, "vtype", None))
        )
        source_type = self.glsl_normalized_source_type(source_type_node)
        source_layout = scalar_storage_layout(source_type)
        if source_layout is None:
            reject(
                "unsupported-private-scalar-struct-source-layout",
                "the addressed value is not a scalar with a known storage layout",
                source_type=source_type,
            )

        members = [
            member
            for member in getattr(struct, "members", []) or []
            if not self.glsl_static_struct_member(member)
        ]
        if (
            len(members) != 1
            or getattr(struct, "generic_params", None)
            or getattr(struct, "inheritance", None)
            or getattr(struct, "attributes", None)
            or getattr(struct, "is_union", False)
        ):
            reject(
                "unsupported-private-scalar-struct-layout",
                "the destination must be a canonical attribute-free struct with "
                "exactly one instance member",
                source_type=source_type,
                alignment=source_layout.byte_width,
            )

        member = members[0]
        member_type_node = getattr(
            member,
            "member_type",
            getattr(member, "vtype", getattr(member, "element_type", None)),
        )
        if (
            isinstance(member, ArrayNode)
            or isinstance(member_type_node, (ArrayType, PointerType, ReferenceType))
            or getattr(member, "attributes", None)
            or getattr(member, "resource_qualifiers", None)
        ):
            reject(
                "unsupported-private-scalar-struct-layout",
                "the sole destination member is not an unqualified scalar value",
                source_type=source_type,
                alignment=source_layout.byte_width,
            )

        member_type = self.glsl_normalized_source_type(member_type_node)
        member_layout = scalar_storage_layout(member_type)
        if (
            member_layout is None
            or source_layout != member_layout
            or source_layout.bit_width not in {8, 16, 32}
        ):
            reject(
                "private-scalar-struct-layout-mismatch",
                "the source scalar and sole destination member do not have the "
                "same supported logical storage layout",
                source_type=source_type,
                alignment=member_layout.byte_width if member_layout else None,
            )

        normalized_destination = self.glsl_normalized_source_type(destination_type)
        if (
            normalized_destination is not None
            and self.map_type(normalized_destination) != mapped_target_type
        ):
            reject(
                "private-scalar-struct-destination-mismatch",
                f"the destination type '{normalized_destination}' does not match "
                f"'{target_type}'",
                source_type=source_type,
                alignment=member_layout.byte_width,
            )

        rendered_value = self.generate_expression_with_expected(
            source_value,
            member_type_node,
        )
        if (
            member_layout.kind == "integer"
            and not member_layout.signed
            and member_layout.bit_width < 32
        ):
            mask = (1 << member_layout.bit_width) - 1
            rendered_value = f"({rendered_value} & 0x{mask:x}u)"

        return {
            "reinterpretation": reinterpretation,
            "target_type": target_type,
            "mapped_target_type": mapped_target_type,
            "member": member,
            "member_type": member_type_node,
            "member_layout": member_layout,
            "source_type": source_type,
            "rendered_value": rendered_value,
        }

    def generate_glsl_private_scalar_struct_reinterpret_read(self, expression):
        contract = self.glsl_private_scalar_struct_reinterpret_contract(expression)
        if contract is None:
            return None
        return f"{contract['mapped_target_type']}({contract['rendered_value']})"

    def glsl_private_pointer_struct_word_array_layout(
        self,
        source_type_node,
        constants,
        expression,
        target_type,
    ):
        source_type = self.glsl_normalized_source_type(source_type_node)
        struct = self.glsl_struct_node(self.map_type(source_type))
        if struct is None:
            self.glsl_private_pointer_byte_view_error(
                expression,
                source_type,
                target_type,
                "unsupported-private-byte-backing",
                "the backing value is not a concrete struct",
            )
        members = [
            member
            for member in getattr(struct, "members", []) or []
            if not self.glsl_static_struct_member(member)
        ]
        if len(members) != 1 or getattr(struct, "inheritance", None):
            self.glsl_private_pointer_byte_view_error(
                expression,
                source_type,
                target_type,
                "unsupported-private-byte-layout",
                "only a padding-free struct with one homogeneous scalar-array "
                "member is supported",
            )
        alignment = 1
        attributes = list(getattr(struct, "attributes", None) or [])
        if attributes:
            attribute = attributes[0] if len(attributes) == 1 else None
            arguments = (
                list(getattr(attribute, "arguments", None) or [])
                if attribute is not None
                else []
            )
            if (
                attribute is None
                or getattr(attribute, "name", None) != "metal_alignas"
                or len(arguments) != 1
            ):
                self.glsl_private_pointer_byte_view_error(
                    expression,
                    source_type,
                    target_type,
                    "unsupported-private-byte-layout",
                    "the scalar-array struct has unsupported layout attributes",
                )
            alignment = self.literal_int_value(arguments[0], constants)
            if (
                not isinstance(alignment, int)
                or isinstance(alignment, bool)
                or alignment <= 0
                or alignment & (alignment - 1)
            ):
                self.glsl_private_pointer_byte_view_error(
                    expression,
                    source_type,
                    target_type,
                    "private-byte-layout-unresolved",
                    "the struct alignment is not a fixed positive power of two",
                )
        member = members[0]
        if getattr(member, "attributes", None) or getattr(
            member, "resource_qualifiers", None
        ):
            self.glsl_private_pointer_byte_view_error(
                expression,
                source_type,
                target_type,
                "unsupported-private-byte-layout",
                "the scalar-array member has unsupported attributes or qualifiers",
            )
        member_type = getattr(member, "member_type", None)
        if not isinstance(member_type, ArrayType):
            self.glsl_private_pointer_byte_view_error(
                expression,
                source_type,
                target_type,
                "unsupported-private-byte-layout",
                "the struct member is not a fixed scalar array",
            )
        extent = self.literal_int_value(member_type.size, constants)
        if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
            self.glsl_private_pointer_byte_view_error(
                expression,
                source_type,
                target_type,
                "private-byte-layout-unresolved",
                "the word-array extent is not a fixed positive integer",
            )
        source_element_type = self.glsl_normalized_source_type(member_type.element_type)
        source_layout = scalar_storage_layout(source_element_type)
        if (
            source_layout is None
            or source_layout.bit_width not in {8, 16, 32}
            or (source_layout.kind == "floating" and source_layout.bit_width != 32)
        ):
            self.glsl_private_pointer_byte_view_error(
                expression,
                source_type,
                target_type,
                "unsupported-private-byte-layout",
                "the array element does not have a bit-preserving 8-, 16-, "
                "or 32-bit scalar layout",
            )
        source_bytes = extent * source_layout.byte_width
        if source_bytes % alignment != 0:
            self.glsl_private_pointer_byte_view_error(
                expression,
                source_type,
                target_type,
                "unsupported-private-byte-layout",
                "the explicit struct alignment would add unsupported tail padding",
            )
        return (
            source_type,
            self.map_type(source_type_node),
            self.glsl_struct_member_name(source_type, member.name),
            extent,
            source_layout,
        )

    def glsl_private_pointer_reinterpret_binding(
        self,
        expression,
        binding,
        constants,
    ):
        pointer_type = getattr(expression, "target_type", None)
        target_type_node = getattr(pointer_type, "pointee_type", None)
        target_type = self.strip_metal_parameter_indirection(
            self.type_name_string(target_type_node)
        )
        target_layout = scalar_storage_layout(target_type)
        address_space = str(getattr(pointer_type, "address_space", None) or "").lower()
        source_type_node = binding.get("source_type")
        source_type = self.glsl_normalized_source_type(source_type_node)
        if address_space not in {"", "thread", "private"}:
            self.glsl_private_pointer_byte_view_error(
                expression,
                source_type,
                target_type,
                "unsupported-private-byte-address-space",
                f"address space '{address_space}' is not private storage",
            )
        if (
            target_layout is None
            or target_layout.bit_width not in {8, 16, 32}
            or (target_layout.kind == "floating" and target_layout.bit_width != 32)
            or isinstance(target_type_node, (PointerType, ArrayType))
        ):
            self.glsl_private_pointer_byte_view_error(
                expression,
                source_type,
                target_type,
                "unsupported-private-byte-view",
                "the destination is not a supported 8-, 16-, or 32-bit scalar view",
            )
        if (
            binding.get("root_kind") != "scalar"
            or binding.get("member_view")
            or binding.get("offset") not in {0, "0", "0u"}
        ):
            self.glsl_private_pointer_byte_view_error(
                expression,
                source_type,
                target_type,
                "unsupported-private-byte-backing",
                "the view does not start at one local struct value",
            )
        (
            source_type,
            mapped_source_type,
            member_name,
            word_count,
            source_layout,
        ) = self.glsl_private_pointer_struct_word_array_layout(
            source_type_node,
            constants,
            expression,
            target_type,
        )
        source_bytes = word_count * source_layout.byte_width
        if source_bytes % target_layout.byte_width != 0:
            self.glsl_private_pointer_byte_view_error(
                expression,
                source_type,
                target_type,
                "unsupported-private-byte-alignment",
                "the source extent is not aligned to the destination element width",
            )
        view_extent = source_bytes // target_layout.byte_width
        variant = (
            "private-word-array-view",
            source_type,
            mapped_source_type,
            member_name,
            view_extent,
            word_count,
            source_layout.name,
            target_layout.name,
        )
        return {
            **binding,
            "root_kind": "byte_view",
            "extent": view_extent,
            "offset": binding.get("offset", 0),
            "offset_interval": (0, 0),
            "element_type": self.map_type(target_type),
            "byte_view": variant,
        }

    def glsl_private_pointer_parameter_element_type(self, parameter, function_name):
        parameter_type = getattr(
            parameter, "param_type", getattr(parameter, "vtype", None)
        )
        pointee_type = getattr(parameter_type, "pointee_type", None)
        if pointee_type is None or isinstance(pointee_type, PointerType):
            display_name = self.glsl_private_pointer_display_name(function_name)
            raise OpenGLPrivatePointerParameterError(
                "OpenGL private pointer parameter "
                f"'{display_name}.{getattr(parameter, 'name', None)}' has an "
                "unsupported pointee type",
                function_name=display_name,
                parameter_name=getattr(parameter, "name", None),
                reason="unsupported-pointee-type",
                source_location=getattr(parameter, "source_location", None),
            )
        return self.map_type(pointee_type)

    def glsl_private_pointer_local_arrays(self, function):
        constants = self.visible_literal_int_constants(function)
        arrays = {}
        ambiguous = set()
        # CrossGL keeps a fixed private array parameter as a whole array, so a
        # view into it is backed by that array just as a local one is.
        for parameter in getattr(function, "parameters", []) or []:
            if self.glsl_private_pointer_array_parameter(parameter):
                info = self.glsl_private_pointer_local_array_node_info(
                    parameter, constants
                )
                if info is not None:
                    arrays[parameter.name] = info
        for node in self.walk_ast(getattr(function, "body", [])):
            if not isinstance(node, (VariableNode, ArrayNode)):
                continue
            name = getattr(node, "name", None)
            if not name:
                continue
            info = self.glsl_private_pointer_local_array_node_info(node, constants)
            if info is None:
                continue
            previous = arrays.get(name)
            if previous is not None and previous != info:
                ambiguous.add(name)
            else:
                arrays[name] = info
        for name in ambiguous:
            arrays[name]["extent"] = None
        return arrays

    @staticmethod
    def glsl_private_pointer_array_parameter(parameter):
        parameter_type = getattr(
            parameter, "param_type", getattr(parameter, "vtype", None)
        )
        if not isinstance(parameter_type, ArrayType) or isinstance(
            getattr(parameter_type, "element_type", None), (PointerType, ArrayType)
        ):
            return False
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", []) or []
        }
        return bool(qualifiers.intersection({"thread", "private", "function"}))

    def glsl_private_pointer_local_array_node_info(self, node, constants):
        node_type = getattr(
            node,
            "var_type",
            getattr(node, "vtype", getattr(node, "param_type", None)),
        )
        if isinstance(node_type, ArrayType):
            element_type = node_type.element_type
            size_expression = node_type.size
        elif isinstance(node, ArrayNode):
            element_type = getattr(node, "element_type", None)
            size_expression = get_array_size_from_node(node)
        else:
            return None
        extent = self.literal_int_value(size_expression, constants)
        info = {
            "extent": extent if isinstance(extent, int) and extent > 0 else None,
            "element_type": (
                self.map_type(element_type) if element_type is not None else None
            ),
            "source_location": getattr(node, "source_location", None),
        }
        rows = array_row_shape(
            node_type, lambda size: self.literal_int_value(size, constants)
        )
        if rows is not None:
            info["rows"] = (rows[0], self.map_type(rows[1]))
        return info

    def glsl_private_pointer_local_scalar_node_info(self, node):
        node_type = getattr(
            node,
            "var_type",
            getattr(node, "param_type", getattr(node, "vtype", None)),
        )
        if node_type is None or isinstance(node, ArrayNode):
            return None
        if isinstance(node_type, (ArrayType, PointerType, ReferenceType)):
            return None
        type_name = self.type_name_string(node_type)
        if not type_name or type_name.rstrip().endswith(("*", "&")):
            return None
        _base_type, array_suffix = split_array_type_suffix(type_name)
        if array_suffix:
            return None
        return {
            "element_type": self.map_type(node_type),
            "source_type": node_type,
            "source_location": getattr(node, "source_location", None),
        }

    def glsl_private_pointer_affine_form(self, expression, forms, constants):
        value = self.literal_int_value(expression, constants)
        if value is not None:
            return value, ()
        if expression is None:
            return None
        if isinstance(expression, str):
            return forms.get(expression)
        if isinstance(expression, (IdentifierNode, VariableNode)):
            return forms.get(getattr(expression, "name", None))
        if isinstance(expression, UnaryOpNode):
            if expression.op in {"++", "--"}:
                return None
            operand = self.glsl_private_pointer_affine_form(
                expression.operand, forms, constants
            )
            if operand is None:
                return None
            if expression.op == "+":
                return operand
            if expression.op == "-":
                return self.glsl_private_pointer_scale_affine_form(operand, -1)
            return None
        if isinstance(expression, BinaryOpNode):
            if expression.op in {"+", "-"}:
                left = self.glsl_private_pointer_affine_form(
                    expression.left, forms, constants
                )
                right = self.glsl_private_pointer_affine_form(
                    expression.right, forms, constants
                )
                return self.glsl_private_pointer_combine_affine_forms(
                    left,
                    right,
                    -1 if expression.op == "-" else 1,
                )
            if expression.op != "*":
                return None
            left_coefficient = self.literal_int_value(expression.left, constants)
            right_coefficient = self.literal_int_value(expression.right, constants)
            if left_coefficient is not None:
                affine = self.glsl_private_pointer_affine_form(
                    expression.right, forms, constants
                )
                return self.glsl_private_pointer_scale_affine_form(
                    affine, left_coefficient
                )
            if right_coefficient is not None:
                affine = self.glsl_private_pointer_affine_form(
                    expression.left, forms, constants
                )
                return self.glsl_private_pointer_scale_affine_form(
                    affine, right_coefficient
                )
            return None
        if isinstance(expression, FunctionCallNode):
            function_name = self.function_call_name(expression)
            arguments = list(
                getattr(expression, "arguments", getattr(expression, "args", [])) or []
            )
            if (
                function_name in {
                    "int",
                    "uint",
                    "short",
                    "ushort",
                    "int16",
                    "uint16",
                    "int32_t",
                    "uint32_t",
                    "size_t",
                    "ptrdiff_t",
                }
                and len(arguments) == 1
            ):
                return self.glsl_private_pointer_affine_form(
                    arguments[0], forms, constants
                )
        return None

    def glsl_private_pointer_combine_affine_forms(self, left, right, right_scale=1):
        if left is None or right is None:
            return None
        coefficients = dict(left[1])
        for symbol, coefficient in right[1]:
            coefficients[symbol] = coefficients.get(symbol, 0) + (
                right_scale * coefficient
            )
            if coefficients[symbol] == 0:
                del coefficients[symbol]
        return (
            left[0] + right_scale * right[0],
            tuple(sorted(coefficients.items())),
        )

    def glsl_private_pointer_scale_affine_form(self, form, coefficient):
        if form is None:
            return None
        return (
            form[0] * coefficient,
            tuple(
                (symbol, value * coefficient)
                for symbol, value in form[1]
                if value * coefficient != 0
            ),
        )

    def glsl_private_pointer_contract_expression_key(
        self, expression, value_keys, constants
    ):
        value = self.literal_int_value(expression, constants)
        if value is not None:
            return "literal", value
        if expression is None:
            return None
        if isinstance(expression, str):
            return value_keys.get(expression, ("name", expression))
        if isinstance(expression, (IdentifierNode, VariableNode)):
            name = getattr(expression, "name", None)
            return value_keys.get(name, ("name", name))
        if isinstance(expression, UnaryOpNode):
            if expression.op in {"++", "--"}:
                return None
            operand = self.glsl_private_pointer_contract_expression_key(
                expression.operand, value_keys, constants
            )
            return None if operand is None else ("unary", expression.op, operand)
        if isinstance(expression, BinaryOpNode):
            left = self.glsl_private_pointer_contract_expression_key(
                expression.left, value_keys, constants
            )
            right = self.glsl_private_pointer_contract_expression_key(
                expression.right, value_keys, constants
            )
            if left is None or right is None:
                return None
            return "binary", expression.op, left, right
        if isinstance(expression, TernaryOpNode):
            condition = self.glsl_private_pointer_contract_expression_key(
                expression.condition, value_keys, constants
            )
            true_value = self.glsl_private_pointer_contract_expression_key(
                expression.true_expr, value_keys, constants
            )
            false_value = self.glsl_private_pointer_contract_expression_key(
                expression.false_expr, value_keys, constants
            )
            if condition is None or true_value is None or false_value is None:
                return None
            return "ternary", condition, true_value, false_value
        if isinstance(expression, FunctionCallNode):
            function_name = self.function_call_name(expression)
            arguments = list(
                getattr(expression, "arguments", getattr(expression, "args", [])) or []
            )
            if (
                function_name in {
                    "int",
                    "uint",
                    "short",
                    "ushort",
                    "int16",
                    "uint16",
                    "int32_t",
                    "uint32_t",
                    "size_t",
                    "ptrdiff_t",
                }
                and len(arguments) == 1
            ):
                argument = self.glsl_private_pointer_contract_expression_key(
                    arguments[0], value_keys, constants
                )
                return None if argument is None else ("cast", function_name, argument)
        return None
