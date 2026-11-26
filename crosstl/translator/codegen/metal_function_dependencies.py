"""Function dependencies in the Metal generator: buffer resource parameters, the constant buffer, stage parameter and global resource dependencies of functions, packed dot products, literal integer constants, image formats and binding attributes."""

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    FunctionCallNode,
    MemberAccessNode,
    RayTracingOpNode,
    VariableNode,
)
from ..refusals import MetalRefusal
from ..validation import collect_non_resource_global_resource_shadows
from .array_utils import (
    evaluate_literal_int_expression,
    split_array_type_suffix,
)
from .generic_function_utils import (
    generic_function_call_name,
    generic_function_definition_name,
)
from .image_access_contracts import (
    image_format_channel_count,
    image_format_component_type,
    is_projected_texture_basic_offset_operation,
    is_projected_texture_compare_operation,
    is_projected_texture_grad_offset_operation,
    is_projected_texture_lod_offset_operation,
    is_projected_texture_operation,
    is_texture_compare_offset_operation,
    is_texture_compare_operation,
    is_texture_gather_compare_operation,
    is_texture_query_lod_operation,
    is_texture_sample_offset_operation,
    is_texture_sample_operation,
    is_texture_sampling_operation,
    supported_image_formats,
)
from .metal_buffer_resource_types import (
    metal_buffer_resource_is_readonly,
    metal_buffer_resource_scope_type,
)
from .metal_reference_parameters import binds_buffer_reference, is_program_scope_object
from .metal_render_walks import (
    names_identifier_class,
    names_resource_reference_class,
)
from .metal_restrict import metal_restrict_suffix
from .stage_utils import (
    normalize_stage_name,
    stage_matches,
)


class MetalFunctionDependencyMixin:
    """Function dependency members of ``MetalCodeGen``."""

    def is_stage_local_resource_variable(self, node):
        vtype = self.type_name_string(
            getattr(node, "var_type", getattr(node, "vtype", "float"))
        )
        return (
            self.metal_buffer_resource_address_space(node, vtype) is not None
            or self.is_resource_parameter_type(vtype)
            or self.is_glsl_buffer_block_variable(node, vtype)
        )

    def metal_buffer_resource_address_space(self, node, vtype=None):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        if not qualifiers & {"uniform", "buffer"}:
            return None

        type_name = self.type_name_string(
            vtype if vtype is not None else getattr(node, "var_type", None)
        )
        if (
            self.is_resource_parameter_type(type_name)
            or self.is_structured_buffer_type(type_name)
            or self.is_glsl_buffer_block_variable(node, type_name)
        ):
            return None

        if "uniform" in qualifiers:
            return "constant"
        return "device"

    def metal_buffer_resource_binding_count(self, node, array_size=None):
        return 1

    def format_metal_buffer_resource_parameter(
        self, raw_type, name, array_size=None, address_space="device", node=None
    ):
        mapped_type = self.map_resource_type_with_format(raw_type)
        memory = self.resource_memory_qualifier_prefix(node, raw_type) if node else ""
        if metal_buffer_resource_is_readonly(node, raw_type, address_space):
            address_space = f"const {address_space}"
        if str(mapped_type).endswith("*"):
            if array_size is not None:
                array_size = array_size or "1"
                return (
                    f"array<{memory}{address_space} {mapped_type}, {array_size}> {name}"
                )
            restrict = metal_restrict_suffix(node, raw_type)
            return f"{memory}{address_space} {mapped_type}{restrict} {name}"
        if array_size is not None:
            restrict = metal_restrict_suffix(node, raw_type)
            return f"{memory}{address_space} {mapped_type}*{restrict} {name}"
        return f"{memory}{address_space} {mapped_type}& {name}"

    def register_metal_buffer_resource_parameter_scope(
        self, func_name, include_all=False
    ):
        if include_all:
            resources = [
                (node, buffer_type, array_size, address_space)
                for node, _, buffer_type, array_size, address_space in (
                    self.metal_buffer_resource_variables
                )
            ]
            structured_resources = [
                (node, buffer_type)
                for node, _, buffer_type, _ in self.structured_buffer_variables
            ]
        else:
            resources = self.required_function_metal_buffer_resources(func_name)
            structured_resources = [
                (node, buffer_type)
                for node, buffer_type, _ in self.required_function_structured_buffers(
                    func_name
                )
            ]

        for node, buffer_type, array_size, address_space in resources:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            if not name:
                continue
            self.local_variable_types[name] = metal_buffer_resource_scope_type(
                self, buffer_type, array_size
            )
            self.current_address_space_variables[name] = address_space
            if address_space == "constant":
                self.current_readonly_metal_parameters.add(name)
                self.current_readonly_metal_parameter_reasons[name] = (
                    "constant address space"
                )
            elif metal_buffer_resource_is_readonly(node, buffer_type, address_space):
                self.current_readonly_metal_parameters.add(name)
                self.current_readonly_metal_parameter_reasons[name] = "readonly"

        for node, buffer_type in structured_resources:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            if not name:
                continue
            rendered_space = self.structured_buffer_address_space(buffer_type, node)
            address_space = "constant" if rendered_space == "constant" else "device"
            self.local_variable_types[name] = self.type_name_string(buffer_type)
            self.current_address_space_variables[name] = address_space
            if self.structured_buffer_type_name(buffer_type) == "StructuredBuffer":
                self.current_readonly_metal_parameters.add(name)
                self.current_readonly_metal_parameter_reasons[name] = (
                    f"{address_space} read-only resource"
                )

    def validate_global_resource_shadows(self, ast):
        conflicts = collect_non_resource_global_resource_shadows(
            ast,
            self.collect_global_resource_names(ast),
            self.is_resource_parameter_type,
        )
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise MetalRefusal(
                "Non-resource local declaration(s) shadow Metal global resource(s): "
                f"{names}"
            )

    def collect_function_cbuffer_dependencies(self, functions):
        direct_dependencies = {}
        function_calls = {}
        for func in functions:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            direct_dependencies[func_name] = self.direct_cbuffer_dependencies(func)
            function_calls[func_name] = self.called_user_function_names(func)

        dependencies = {name: set(deps) for name, deps in direct_dependencies.items()}
        changed = True
        while changed:
            changed = False
            for func_name, calls in function_calls.items():
                before = set(dependencies.get(func_name, set()))
                for called_name in calls:
                    dependencies.setdefault(func_name, set()).update(
                        dependencies.get(called_name, set())
                    )
                if dependencies.get(func_name, set()) != before:
                    changed = True
        return dependencies

    def direct_cbuffer_dependencies(self, func):
        local_names = {
            getattr(param, "name", None)
            for param in getattr(func, "parameters", getattr(func, "params", []))
            if getattr(param, "name", None)
        }
        for node in self.metal_render_walks.body_nodes_of_type(func, VariableNode):
            if getattr(node, "name", None):
                local_names.add(node.name)

        member_to_cbuffer = {}
        for cbuffer in self.cbuffer_variables:
            cbuffer_name = getattr(cbuffer, "name", None)
            if not cbuffer_name:
                continue
            for member in getattr(cbuffer, "members", []) or []:
                member_name = getattr(member, "name", None)
                if member_name:
                    member_to_cbuffer[member_name] = cbuffer_name

        dependencies = set()
        for node in self.metal_render_walks.body_nodes_with_class(
            func, names_identifier_class
        ):
            name = getattr(node, "name", None)
            if not name or name in local_names:
                continue
            cbuffer_name = member_to_cbuffer.get(name)
            if cbuffer_name:
                dependencies.add(cbuffer_name)
        return dependencies

    def called_user_function_names(self, func):
        called_names = set()
        for node in self.metal_render_walks.body_nodes_of_type(func, FunctionCallNode):
            func_name = self.function_call_name(node)
            if not func_name:
                continue
            specialized_name = generic_function_call_name(
                self,
                func_name,
                getattr(node, "arguments", getattr(node, "args", [])),
            )
            if specialized_name:
                candidates = (specialized_name,)
            else:
                base_name = generic_function_definition_name(self, func_name)
                candidates = (func_name, base_name)
            for candidate in candidates:
                if (
                    candidate
                    and candidate in self.user_function_names
                    and candidate != getattr(func, "name", None)
                ):
                    called_names.add(candidate)
        return called_names

    def collect_function_stage_parameter_dependencies(self, ast, target_stage):
        direct_dependencies = {}
        function_calls = {}
        parameter_nodes = {}

        for func in self.all_functions(ast):
            func_name = getattr(func, "name", None)
            if func_name:
                direct_dependencies.setdefault(func_name, set())
                function_calls.setdefault(
                    func_name, self.called_user_function_names(func)
                )

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if not stage_matches(target_stage, stage_name):
                continue
            entry_point = getattr(stage, "entry_point", None)
            stage_parameters = {
                parameter.name: parameter
                for parameter in (
                    getattr(
                        entry_point, "parameters", getattr(entry_point, "params", [])
                    )
                    or []
                )
                if getattr(parameter, "name", None)
            }
            for parameter in stage_parameters.values():
                parameter_nodes[parameter.name] = parameter

            stage_functions = list(getattr(stage, "local_functions", []) or [])
            if entry_point is not None:
                stage_functions.append(entry_point)
            for func in stage_functions:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue
                direct_dependencies[func_name] = (
                    self.direct_stage_parameter_dependencies(func, stage_parameters)
                )
                function_calls[func_name] = self.called_user_function_names(func)

        dependencies = {name: set(deps) for name, deps in direct_dependencies.items()}
        changed = True
        while changed:
            changed = False
            for func_name, calls in function_calls.items():
                before = set(dependencies.get(func_name, set()))
                for called_name in calls:
                    dependencies.setdefault(func_name, set()).update(
                        dependencies.get(called_name, set())
                    )
                if dependencies.get(func_name, set()) != before:
                    changed = True

        return {
            func_name: [
                parameter_nodes[name]
                for name in sorted(dependency_names)
                if name in parameter_nodes
            ]
            for func_name, dependency_names in dependencies.items()
            if dependency_names
        }

    def direct_stage_parameter_dependencies(self, func, stage_parameters):
        local_names = {
            getattr(param, "name", None)
            for param in getattr(func, "parameters", getattr(func, "params", []))
            if getattr(param, "name", None)
        }
        for node in self.metal_render_walks.body_nodes_of_type(func, VariableNode):
            if getattr(node, "name", None):
                local_names.add(node.name)

        dependencies = set()
        for node in self.metal_render_walks.body_nodes_with_class(
            func, names_identifier_class
        ):
            name = getattr(node, "name", None)
            if name and name in stage_parameters and name not in local_names:
                dependencies.add(name)
        return dependencies

    def required_function_stage_parameters(self, func_name):
        return self.function_stage_parameter_dependencies.get(func_name, [])

    def required_function_stage_parameter_argument_names(self, func_name):
        return [
            parameter.name
            for parameter in self.required_function_stage_parameters(func_name)
        ]

    def collect_function_stage_output_dependencies(self, ast, target_stage):
        direct_dependencies = {}
        function_calls = {}
        output_types = {}

        for func in self.all_functions(ast):
            func_name = getattr(func, "name", None)
            if func_name:
                direct_dependencies.setdefault(func_name, None)
                function_calls.setdefault(
                    func_name, self.called_user_function_names(func)
                )

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if not stage_matches(target_stage, stage_name):
                continue
            entry_point = getattr(stage, "entry_point", None)
            output_type = getattr(entry_point, "return_type", None)
            if output_type is None or self.type_name_string(output_type) == "void":
                continue
            stage_functions = list(getattr(stage, "local_functions", []) or [])
            if entry_point is not None:
                stage_functions.append(entry_point)
            for func in stage_functions:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue
                output_types[func_name] = output_type
                if self.direct_stage_output_dependency(func):
                    direct_dependencies[func_name] = output_type
                function_calls[func_name] = self.called_user_function_names(func)

        dependencies = dict(direct_dependencies)
        changed = True
        while changed:
            changed = False
            for func_name, calls in function_calls.items():
                if dependencies.get(func_name) is not None:
                    continue
                for called_name in calls:
                    if dependencies.get(called_name) is not None:
                        dependencies[func_name] = output_types.get(
                            func_name
                        ) or dependencies.get(called_name)
                        changed = True
                        break

        return {
            func_name: output_type
            for func_name, output_type in dependencies.items()
            if output_type is not None
        }

    def direct_stage_output_dependency(self, func):
        parameter_names = {
            getattr(param, "name", None)
            for param in getattr(func, "parameters", getattr(func, "params", [])) or []
        }
        if "output" in parameter_names:
            return False
        local_names = set(parameter_names)
        for node in self.metal_render_walks.body_nodes_of_type(func, VariableNode):
            if getattr(node, "name", None):
                local_names.add(node.name)
        if "output" in local_names:
            return False
        for node in self.iter_ast_nodes(getattr(func, "body", [])):
            if isinstance(node, MemberAccessNode):
                if self.expression_name(getattr(node, "object", None)) == "output":
                    return True
            if getattr(node, "name", None) == "output":
                return True
        return False

    def required_function_stage_output_type(self, func_name):
        return self.function_stage_output_dependencies.get(func_name)

    def required_function_stage_output_argument_names(self, func_name):
        if self.required_function_stage_output_type(func_name) is None:
            return []
        return ["output"]

    def required_function_cbuffers(self, func_name):
        dependencies = self.function_cbuffer_dependencies.get(func_name, set())
        return [
            cbuffer
            for cbuffer in self.cbuffer_variables
            if getattr(cbuffer, "name", None) in dependencies
        ]

    def collect_function_global_resource_dependencies(self, functions):
        direct_dependencies = {}
        function_calls = {}
        for func in functions:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            direct_dependencies[func_name] = self.direct_global_resource_dependencies(
                func
            )
            function_calls[func_name] = self.called_user_function_names(func)

        dependencies = {name: set(deps) for name, deps in direct_dependencies.items()}
        changed = True
        while changed:
            changed = False
            for func_name, calls in function_calls.items():
                before = set(dependencies.get(func_name, set()))
                for called_name in calls:
                    dependencies.setdefault(func_name, set()).update(
                        dependencies.get(called_name, set())
                    )
                if dependencies.get(func_name, set()) != before:
                    changed = True
        return dependencies

    def direct_global_resource_dependencies(self, func):
        local_names = {
            getattr(param, "name", None)
            for param in getattr(func, "parameters", getattr(func, "params", []))
            if getattr(param, "name", None)
        }
        for node in self.metal_render_walks.body_nodes_of_type(func, VariableNode):
            if getattr(node, "name", None):
                local_names.add(node.name)

        texture_names = self.global_texture_names()
        acceleration_structure_names = self.global_acceleration_structure_names()
        visible_function_table_names = self.global_visible_function_table_names()
        intersection_function_table_names = (
            self.global_intersection_function_table_names()
        )
        buffer_names = (
            self.global_structured_buffer_names()
            | self.global_metal_buffer_resource_names()
            | self.global_glsl_buffer_block_names()
        )
        sampler_names = self.global_sampler_names()
        texture_alias_sources = self.local_texture_alias_sources(func, texture_names)
        acceleration_structure_alias_types = (
            self.local_acceleration_structure_dependency_types(
                func, acceleration_structure_names
            )
        )
        dependencies = set()

        for node in self.metal_render_walks.body_nodes_with_class(
            func, names_resource_reference_class
        ):
            if names_identifier_class(node.__class__):
                name = getattr(node, "name", None)
                if (
                    name
                    and name not in local_names
                    and (
                        name in texture_names
                        or name in acceleration_structure_names
                        or name in visible_function_table_names
                        or name in intersection_function_table_names
                        or name in buffer_names
                        or name in sampler_names
                    )
                ):
                    dependencies.add(name)

            if isinstance(node, FunctionCallNode):
                self.add_texture_call_resource_dependencies(
                    node,
                    local_names,
                    texture_names,
                    sampler_names,
                    dependencies,
                    texture_alias_sources,
                )
                self.add_buffer_call_resource_dependencies(
                    node, local_names, buffer_names, dependencies
                )
            if isinstance(node, RayTracingOpNode):
                self.add_ray_tracing_resource_dependencies(
                    node,
                    local_names,
                    visible_function_table_names,
                    intersection_function_table_names,
                    acceleration_structure_alias_types,
                    dependencies,
                )

        return dependencies

    def local_acceleration_structure_dependency_types(
        self, func, acceleration_structure_names
    ):
        aliases = {}
        for param in getattr(func, "parameters", getattr(func, "params", [])):
            param_name = getattr(param, "name", None)
            param_type = self.type_name_string(
                getattr(
                    param,
                    "param_type",
                    getattr(param, "var_type", getattr(param, "vtype", None)),
                )
            )
            if (
                param_name
                and param_type
                and self.is_acceleration_structure_type(param_type)
                and not self.metal_type_is_pointer_like(param_type)
            ):
                aliases[param_name] = self.map_resource_type_with_format(param_type)

        for node in self.metal_render_walks.body_nodes_of_type(func, VariableNode):
            name = getattr(node, "name", None)
            if not name:
                continue
            initial_type = self.acceleration_structure_dependency_argument_type(
                getattr(node, "initial_value", None),
                aliases,
                acceleration_structure_names,
            )
            if initial_type is None:
                continue

            declared_type = self.local_variable_type_node(node)
            declared_type_name = self.type_name_string(declared_type)
            if not declared_type_name:
                aliases[name] = initial_type
                continue
            if not self.is_acceleration_structure_type(declared_type_name):
                continue

            declared_resource_type = self.map_resource_type_with_format(
                declared_type_name, node
            )
            if declared_resource_type == initial_type:
                aliases[name] = declared_resource_type
        return aliases

    def acceleration_structure_dependency_argument_type(
        self, argument, alias_types=None, acceleration_structure_names=None
    ):
        if argument is None:
            return None

        argument_type = self.expression_result_type(argument)
        if argument_type and self.is_acceleration_structure_type(argument_type):
            if self.metal_type_is_pointer_like(argument_type):
                return None
            return self.map_resource_type_with_format(argument_type)

        argument_name = self.expression_name(argument)
        if not argument_name:
            return None
        if alias_types and argument_name in alias_types:
            return alias_types[argument_name]

        for (
            acceleration_structure_variable,
            _,
            mapped_type,
            array_size,
        ) in self.acceleration_structure_variables:
            if getattr(acceleration_structure_variable, "name", None) != argument_name:
                continue
            if (
                acceleration_structure_names is not None
                and argument_name not in acceleration_structure_names
            ):
                return None
            if array_size is not None and not isinstance(argument, ArrayAccessNode):
                return None
            return mapped_type
        return None

    def local_texture_alias_sources(self, func, texture_names):
        aliases = {}
        for node in self.metal_render_walks.body_nodes_of_type(func, VariableNode):
            name = getattr(node, "name", None)
            initial_value = getattr(node, "initial_value", None)
            source_name = self.expression_name(initial_value)
            if (
                name
                and source_name
                and (source_name in texture_names or source_name in aliases)
            ):
                aliases[name] = initial_value
        return aliases

    def texture_alias_dependency_source(self, texture_name, texture_alias_sources):
        source_arg = texture_alias_sources.get(texture_name)
        seen = {texture_name}
        while source_arg is not None:
            source_name = self.expression_name(source_arg)
            if not source_name or source_name in seen:
                return source_arg
            next_source_arg = texture_alias_sources.get(source_name)
            if next_source_arg is None:
                return source_arg
            seen.add(source_name)
            source_arg = next_source_arg
        return None

    def add_ray_tracing_resource_dependencies(
        self,
        call,
        local_names,
        visible_function_table_names,
        intersection_function_table_names,
        acceleration_structure_alias_types,
        dependencies,
    ):
        operation = getattr(call, "operation", None)
        if operation == "TraceRay":
            args = getattr(call, "arguments", [])
            if not args:
                return
            acceleration_structure_type = (
                self.acceleration_structure_dependency_argument_type(
                    args[0], acceleration_structure_alias_types
                )
                or self.metal_acceleration_structure_argument_type(args[0])
            )
            table = self.default_intersection_function_table(
                acceleration_structure_type
            )
            if (
                table is not None
                and table["name"] in intersection_function_table_names
                and table["name"] not in local_names
            ):
                dependencies.add(table["name"])
            return

        if operation != "CallShader":
            return
        args = getattr(call, "arguments", [])
        if len(args) == 3:
            table_name = self.expression_name(args[0])
            if (
                table_name in visible_function_table_names
                and table_name not in local_names
            ):
                dependencies.add(table_name)
            return
        if len(args) == 2 and len(self.visible_function_table_variables) == 1:
            table_name = getattr(
                self.visible_function_table_variables[0][0], "name", None
            )
            if table_name and table_name not in local_names:
                dependencies.add(table_name)

    def add_texture_call_resource_dependencies(
        self,
        call,
        local_names,
        texture_names,
        sampler_names,
        dependencies,
        texture_alias_sources=None,
    ):
        func_name = self.function_call_name(call)
        if not func_name or not str(func_name).startswith(("texture", "image")):
            return
        args = getattr(call, "arguments", getattr(call, "args", []))
        if not args:
            return

        texture_name = self.expression_name(args[0])
        if texture_name in texture_names and texture_name not in local_names:
            dependencies.add(texture_name)

        if len(args) >= 3:
            sampler_name = self.expression_name(args[1])
            if sampler_name in sampler_names and sampler_name not in local_names:
                dependencies.add(sampler_name)
                return

        if not self.texture_call_needs_implicit_sampler_dependency(func_name, args):
            return
        implicit_sampler_name = f"{texture_name}Sampler" if texture_name else None
        if (
            implicit_sampler_name in sampler_names
            and implicit_sampler_name not in local_names
        ):
            dependencies.add(implicit_sampler_name)
            return

        source_arg = self.texture_alias_dependency_source(
            texture_name, texture_alias_sources or {}
        )
        source_name = self.expression_name(source_arg)
        source_sampler_name = f"{source_name}Sampler" if source_name else None
        if (
            source_sampler_name in sampler_names
            and source_sampler_name not in local_names
        ):
            dependencies.add(source_sampler_name)

    def add_buffer_call_resource_dependencies(
        self, call, local_names, buffer_names, dependencies
    ):
        func_name = self.function_call_name(call)
        if func_name not in {
            "buffer_load",
            "buffer_store",
            "buffer_append",
            "buffer_consume",
            "buffer_dimensions",
        }:
            return
        args = getattr(call, "arguments", getattr(call, "args", []))
        if not args:
            return
        buffer_name = self.expression_name(args[0])
        if buffer_name in buffer_names and buffer_name not in local_names:
            dependencies.add(buffer_name)

    def texture_sampling_uses_implicit_sampler(self, func_name):
        # A depth comparison of a combined sampler samples through its sampler too.
        return (
            is_texture_sampling_operation(func_name)
            or is_texture_query_lod_operation(func_name)
            or is_texture_compare_operation(func_name)
            or is_texture_gather_compare_operation(func_name)
            or is_projected_texture_compare_operation(func_name)
        )

    def texture_call_needs_implicit_sampler_dependency(self, func_name, args):
        if not self.texture_sampling_uses_implicit_sampler(func_name):
            return False

        texture_type = self.texture_resource_type(args[0])
        if self.storage_image_texture_operation_expression(func_name, texture_type):
            return False

        if is_texture_sample_operation(func_name):
            return not self.is_multisample_texture_resource(texture_type)

        if (
            is_texture_compare_operation(func_name)
            or is_texture_gather_compare_operation(func_name)
            or is_projected_texture_compare_operation(func_name)
        ):
            if self.is_multisample_texture_resource(texture_type):
                return False
            if is_texture_compare_offset_operation(func_name) or str(
                func_name
            ).startswith("textureGatherCompareOffset"):
                # Offsets the texture cannot take are refused, and need no sampler.
                return self.texture_sample_supports_offset(texture_type)
            return True

        if is_texture_query_lod_operation(func_name):
            return not (
                self.is_multisample_texture_resource(texture_type)
                or self.is_storage_image_resource(texture_type)
            )

        if is_texture_sample_offset_operation(func_name):
            return self.texture_sample_supports_offset(texture_type)

        if is_projected_texture_operation(func_name):
            texture_type = self.resource_base_type(texture_type)
            if (
                is_projected_texture_basic_offset_operation(func_name)
                or is_projected_texture_lod_offset_operation(func_name)
                or is_projected_texture_grad_offset_operation(func_name)
            ):
                return texture_type.startswith("texture2d<") or texture_type.startswith(
                    "texture2d_array<"
                )
            return (
                texture_type.startswith("texture1d<")
                or texture_type.startswith("texture2d<")
                or texture_type.startswith("texture2d_array<")
                or texture_type.startswith("texture3d<")
                or texture_type.startswith("texturecube<")
            )

        return True

    def global_texture_names(self):
        return {
            texture_variable.name
            for texture_variable, _, _, _ in self.texture_variables
            if getattr(texture_variable, "name", None)
        }

    def global_acceleration_structure_names(self):
        return {
            acceleration_structure_variable.name
            for acceleration_structure_variable, _, _, _ in (
                self.acceleration_structure_variables
            )
            if getattr(acceleration_structure_variable, "name", None)
        }

    def global_visible_function_table_names(self):
        return {
            visible_function_table_variable.name
            for visible_function_table_variable, _, _, _ in (
                self.visible_function_table_variables
            )
            if getattr(visible_function_table_variable, "name", None)
        }

    def global_intersection_function_table_names(self):
        return {
            intersection_function_table_variable.name
            for intersection_function_table_variable, _, _, _ in (
                self.intersection_function_table_variables
            )
            if getattr(intersection_function_table_variable, "name", None)
        }

    def global_structured_buffer_names(self):
        return {
            buffer_variable.name
            for buffer_variable, _, _, _ in self.structured_buffer_variables
            if getattr(buffer_variable, "name", None)
        }

    def global_metal_buffer_resource_names(self):
        return {
            buffer_variable.name
            for buffer_variable, _, _, _, _ in self.metal_buffer_resource_variables
            if getattr(buffer_variable, "name", None)
        }

    def structured_buffer_requires_length(self, name):
        return bool(name and name in self.global_structured_buffer_length_dependencies)

    def structured_buffer_parameter_requires_length(self, func_name, name):
        return bool(
            func_name
            and name
            and name in self.function_structured_buffer_length_dependencies.get(
                func_name, set()
            )
        )

    def global_structured_buffer_requires_counter(self, name):
        if not name:
            return False
        for buffer_variable, _, buffer_type, _ in self.structured_buffer_variables:
            if getattr(buffer_variable, "name", None) == name:
                return self.structured_buffer_requires_counter(buffer_type)
        return False

    def global_glsl_buffer_block_names(self):
        return {
            buffer_variable.name
            for buffer_variable, _, _, _ in self.glsl_buffer_block_variables
            if getattr(buffer_variable, "name", None)
        }

    def global_sampler_names(self):
        return {
            sampler_variable.name
            for sampler_variable, _, _ in self.sampler_variables
            if getattr(sampler_variable, "name", None)
        }

    def required_function_textures(self, func_name):
        dependencies = self.function_global_resource_dependencies.get(func_name, set())
        return [
            (texture_variable, texture_type, array_size)
            for texture_variable, _, texture_type, array_size in self.texture_variables
            if getattr(texture_variable, "name", None) in dependencies
        ]

    def required_function_acceleration_structures(self, func_name):
        dependencies = self.function_global_resource_dependencies.get(func_name, set())
        return [
            (
                acceleration_structure_variable,
                acceleration_structure_type,
                array_size,
            )
            for (
                acceleration_structure_variable,
                _,
                acceleration_structure_type,
                array_size,
            ) in self.acceleration_structure_variables
            if getattr(acceleration_structure_variable, "name", None) in dependencies
        ]

    def required_function_visible_function_tables(self, func_name):
        dependencies = self.function_global_resource_dependencies.get(func_name, set())
        return [
            (
                visible_function_table_variable,
                visible_function_table_type,
                array_size,
            )
            for (
                visible_function_table_variable,
                _,
                visible_function_table_type,
                array_size,
            ) in self.visible_function_table_variables
            if getattr(visible_function_table_variable, "name", None) in dependencies
        ]

    def required_function_intersection_function_tables(self, func_name):
        dependencies = self.function_global_resource_dependencies.get(func_name, set())
        return [
            (
                intersection_function_table_variable,
                intersection_function_table_type,
                array_size,
            )
            for (
                intersection_function_table_variable,
                _,
                intersection_function_table_type,
                array_size,
            ) in self.intersection_function_table_variables
            if getattr(intersection_function_table_variable, "name", None)
            in dependencies
        ]

    def required_function_samplers(self, func_name):
        dependencies = self.function_global_resource_dependencies.get(func_name, set())
        return [
            (sampler_variable, array_size)
            for sampler_variable, _, array_size in self.sampler_variables
            if getattr(sampler_variable, "name", None) in dependencies
        ]

    def required_function_structured_buffers(self, func_name):
        dependencies = self.function_global_resource_dependencies.get(func_name, set())
        return [
            (buffer_variable, buffer_type, array_size)
            for (
                buffer_variable,
                _,
                buffer_type,
                array_size,
            ) in self.structured_buffer_variables
            if getattr(buffer_variable, "name", None) in dependencies
        ]

    def required_function_metal_buffer_resources(self, func_name):
        dependencies = self.function_global_resource_dependencies.get(func_name, set())
        return [
            (buffer_variable, buffer_type, array_size, address_space)
            for (
                buffer_variable,
                _,
                buffer_type,
                array_size,
                address_space,
            ) in self.metal_buffer_resource_variables
            if getattr(buffer_variable, "name", None) in dependencies
        ]

    def required_function_glsl_buffer_blocks(self, func_name):
        dependencies = self.function_global_resource_dependencies.get(func_name, set())
        return [
            (buffer_variable, block, array_size)
            for (
                buffer_variable,
                _,
                block,
                array_size,
            ) in self.glsl_buffer_block_variables
            if getattr(buffer_variable, "name", None) in dependencies
        ]

    # HLSL dot4add_i8packed(a, b, acc) and dot4add_u8packed(a, b, acc) add the
    # dot product of the four 8-bit lanes packed in a and b to acc, with byte 0
    # the least significant lane. Metal has no packed integer dot product, and
    # its dot() takes only floating-point vectors, so each operand is
    # reinterpreted as a char4 or uchar4 (byte 0 lands in .x on the
    # little-endian Apple GPUs) and the lanes go through a float dot(). Every
    # lane product and their sum stay below 2^24 in magnitude, so the float
    # dot product is exact. HLSL wraps the accumulator addition, but signed
    # overflow is undefined in MSL, so the signed sum is taken in uint.
    PACKED_DOT4ADD_LANES = {
        "dot4add_i8packed": (
            "char4",
            "as_type<int>(as_type<uint>(int({accumulator})) + "
            "as_type<uint>(int({dot})))",
        ),
        "dot4add_u8packed": ("uchar4", "(uint({accumulator}) + uint({dot}))"),
    }

    def generate_packed_dot4add_call(self, func_name, args):
        lowering = self.PACKED_DOT4ADD_LANES.get(func_name)
        if lowering is None or func_name in self.user_function_names:
            return None
        if len(args) != 3:
            raise MetalRefusal(
                f"Metal {func_name} alias requires 3 arguments; got {len(args)}"
            )

        lane_type, total = lowering
        left, right = (
            f"float4(as_type<{lane_type}>(uint({self.generate_expression(arg)})))"
            for arg in args[:2]
        )
        return total.format(
            accumulator=self.generate_expression(args[2]),
            dot=f"dot({left}, {right})",
        )

    def generate_function_call_arguments(self, func_name, call_args):
        call_args = list(call_args or [])
        parameter_infos = self.function_parameter_infos.get(func_name, [])
        if len(call_args) != len(parameter_infos):
            parameter_infos = []
        skipped_indices = self.skipped_function_parameter_indices(func_name)
        args = []
        for index, arg in enumerate(call_args):
            if index in skipped_indices:
                continue
            param_name, param_type = (
                parameter_infos[index] if index < len(parameter_infos) else (None, None)
            )
            if self.is_unsupported_metal_ray_function_table_array_parameter(
                func_name, index
            ):
                continue
            if self.is_unsupported_metal_acceleration_structure_array_parameter(
                func_name, index
            ):
                continue
            array_copy = self.metal_array_arguments.argument(
                func_name, index, arg, call_args
            )
            if array_copy is not None:
                args.append(array_copy)
                continue
            if (
                isinstance(arg, ArrayLiteralNode)
                and split_array_type_suffix(self.type_name_string(param_type) or "")[1]
            ):
                # A C array parameter is a pointer, and no Metal expression
                # yields the address of an array literal.
                args.append(
                    self.refuse_lowering(
                        f"array literal argument to '{func_name}': Metal has no array literal "
                        "expression for an array parameter; a materialized array temporary is "
                        "required"
                    )
                )
                continue
            native_type = self.metal_native_narrow_bitcast_storage_type(param_type)
            parameters = self.function_parameter_nodes.get(func_name, [])
            qualifiers = (
                set(getattr(parameters[index], "qualifiers", []) or [])
                if index < len(parameters)
                else set()
            )
            if native_type is not None and binds_buffer_reference(
                self, func_name, index
            ):
                # A `constant uchar&` parameter binds the stored object itself.
                rendered = (
                    self.refuse_lowering(
                        f"argument {index + 1} of '{func_name}' is a program-scope "
                        f"{native_type} object in the widened representation, which "
                        f"cannot bind to the {native_type}& buffer reference parameter"
                    )
                    if is_program_scope_object(self, arg)
                    else self.generate_expression(arg)
                )
            elif (
                native_type is not None
                and not qualifiers.intersection({"out", "inout"})
                and len(self.function_overloads_by_name.get(func_name, [])) <= 1
            ):
                rendered = self.generate_expression(arg)
                # Native aggregate vectors need an explicit value conversion
                # when the callee uses the widened arithmetic representation.
                converted = self.metal_narrow_integer_conversion_expression(
                    rendered, param_type
                )
                if converted is not None:
                    rendered = converted
                elif native_type[-1:] in {"2", "3", "4"}:
                    rendered = f"{self.map_type(param_type)}({rendered})"
            else:
                rendered = self.generate_function_value_argument(
                    func_name, index, arg, param_type
                )
            args.append(rendered)
            if self.structured_buffer_parameter_requires_length(func_name, param_name):
                length = self.structured_buffer_length_data_argument(arg)
                if length is not None:
                    args.append(length)
            if param_type is not None and self.structured_buffer_requires_counter(
                param_type
            ):
                counter = self.structured_buffer_counter_data_argument(arg)
                if counter is not None:
                    args.append(counter)
        return args

    def is_unsupported_metal_ray_function_table_array_parameter(self, func_name, index):
        parameter_nodes = self.function_parameter_nodes.get(func_name, [])
        if index >= len(parameter_nodes):
            return False
        parameter = parameter_nodes[index]
        raw_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        previous_function_name = self.current_function_name
        self.current_function_name = func_name
        try:
            return (
                self.metal_ray_function_table_array_parameter_kind(raw_type, parameter)
                is not None
            )
        finally:
            self.current_function_name = previous_function_name

    def is_unsupported_metal_acceleration_structure_array_parameter(
        self, func_name, index
    ):
        parameter_nodes = self.function_parameter_nodes.get(func_name, [])
        if index >= len(parameter_nodes):
            return False
        parameter = parameter_nodes[index]
        raw_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        previous_function_name = self.current_function_name
        self.current_function_name = func_name
        try:
            return (
                self.metal_acceleration_structure_array_parameter_kind(
                    raw_type, parameter
                )
                is not None
            )
        finally:
            self.current_function_name = previous_function_name

    def required_function_resource_argument_names(self, func_name):
        names = [
            texture_variable.name
            for texture_variable, _, _ in self.required_function_textures(func_name)
        ]
        names.extend(
            acceleration_structure_variable.name
            for acceleration_structure_variable, _, _ in (
                self.required_function_acceleration_structures(func_name)
            )
        )
        names.extend(
            visible_function_table_variable.name
            for visible_function_table_variable, _, _ in (
                self.required_function_visible_function_tables(func_name)
            )
        )
        names.extend(
            intersection_function_table_variable.name
            for intersection_function_table_variable, _, _ in (
                self.required_function_intersection_function_tables(func_name)
            )
        )
        for (
            buffer_variable,
            buffer_type,
            _array_size,
            _address_space,
        ) in self.required_function_metal_buffer_resources(func_name):
            buffer_name = getattr(buffer_variable, "name", None)
            if buffer_name:
                names.append(buffer_name)
        for (
            buffer_variable,
            buffer_type,
            _array_size,
        ) in self.required_function_structured_buffers(func_name):
            buffer_name = getattr(buffer_variable, "name", None)
            if not buffer_name:
                continue
            names.append(buffer_name)
            if self.structured_buffer_requires_length(buffer_name):
                names.append(self.structured_buffer_length_parameter_name(buffer_name))
            if self.structured_buffer_requires_counter(buffer_type):
                names.append(self.structured_buffer_counter_parameter_name(buffer_name))
        names.extend(
            buffer_variable.name
            for buffer_variable, _, _ in self.required_function_glsl_buffer_blocks(
                func_name
            )
        )
        names.extend(
            sampler_variable.name
            for sampler_variable, _ in self.required_function_samplers(func_name)
        )
        return names

    def iter_ast_nodes(self, node):
        return self.metal_render_walks.walk(node)

    def literal_int_value(self, expr, constants=None):
        return evaluate_literal_int_expression(expr, constants)

    def initial_literal_int_constants(self, func):
        visible_constants = dict(self.literal_int_constants)
        for param in getattr(func, "parameters", []) or []:
            visible_constants.pop(getattr(param, "name", None), None)
        return visible_constants

    def visible_literal_int_constants(self, func):
        visible_constants = self.initial_literal_int_constants(func)

        for node in self.metal_render_walks.body_nodes_of_type(func, VariableNode):
            name = getattr(node, "name", None)
            if not name:
                continue

            visible_constants.pop(name, None)
            if "const" not in getattr(node, "qualifiers", []):
                continue

            value = self.literal_int_value(
                getattr(node, "initial_value", None), visible_constants
            )
            if value is not None:
                visible_constants[name] = value

        return visible_constants

    def function_call_name(self, call):
        func_expr = getattr(call, "function", None)
        if func_expr is None:
            func_expr = getattr(call, "name", None)
        if isinstance(func_expr, str):
            return func_expr
        if hasattr(func_expr, "name") and isinstance(func_expr.name, str):
            return func_expr.name
        return None

    def member_access_is_image_load_component(self, expr):
        if not isinstance(getattr(expr, "object", None), FunctionCallNode):
            return False
        if self.function_call_name(expr.object) != "imageLoad":
            return False
        member = str(getattr(expr, "member", ""))
        return bool(member) and all(char in "xyzwrgba" for char in member)

    def supported_image_formats(self):
        return supported_image_formats()

    def scalar_image_format_components(self):
        return {
            image_format: image_format_component_type(image_format)
            for image_format in supported_image_formats()
            if image_format_channel_count(image_format) == 1
        }

    def vector_image_format_components(self):
        return {
            image_format: image_format_component_type(image_format)
            for image_format in supported_image_formats()
            if image_format_channel_count(image_format) in {2, 3, 4}
        }

    def attribute_value_to_string(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        name = getattr(value, "name", None)
        if name is not None:
            return str(name)
        if hasattr(value, "value"):
            return str(value.value).strip('"')
        return str(value)

    def is_resource_binding_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        return str(attr_name).lower() in {
            "binding",
            "buffer",
            "group",
            "packoffset",
            "register",
            "sampler",
            "set",
            "space",
            "texture",
        }

    def binding_index_value(self, value, prefixes=()):
        if hasattr(value, "value") and value.value is not None:
            raw_value = value.value
        elif hasattr(value, "name") and value.name is not None:
            raw_value = value.name
        else:
            raw_value = self.attribute_value_to_string(value)
        if raw_value is None:
            return None
        raw_value = str(raw_value).strip().lower()
        if raw_value.isdigit():
            return int(raw_value)
        for prefix in prefixes:
            if raw_value.startswith(prefix) and raw_value[len(prefix) :].isdigit():
                return int(raw_value[len(prefix) :])
        return None

    def explicit_resource_binding_index(
        self, node, attribute_names=(), register_prefixes=()
    ):
        if not hasattr(node, "attributes"):
            return None
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            arguments = getattr(attr, "arguments", []) or []
            if not attr_name or not arguments:
                continue
            attr_name = str(attr_name).lower()
            if attr_name in attribute_names:
                binding = self.binding_index_value(arguments[0])
            elif attr_name == "register":
                binding = self.binding_index_value(arguments[0], register_prefixes)
            else:
                binding = None
            if binding is not None:
                return binding
        return None
