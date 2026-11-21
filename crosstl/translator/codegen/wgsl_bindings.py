"""Attributes and resource bindings in the WGSL generator: locations, groups, explicit bindings and function resource parameters."""

import re

from ..ast import IdentifierNode
from ..refusals import WGSLRefusal
from .stage_io_semantic_locations import semantic_location as shared_semantic_location
from .wgsl_stage_io import is_system_semantic_key


class WGSLBindingMixin:
    """Binding members of ``WGSLCodeGen``."""

    def stage_return_attributes(self, stage_name, function):
        attributes = getattr(function, "attributes", []) or []
        if not attributes and stage_name == "vertex":
            return ()
        if not attributes and stage_name == "fragment":
            return_type = getattr(function, "return_type", None)
            if (
                self.scalar_type_name(return_type) is not None
                or self.vector_shape(return_type) is not None
            ):
                return ("@location(0)",)
        return_attributes = self.wgsl_attributes(
            attributes, direction="target" if stage_name == "fragment" else "out"
        )
        return (return_attributes,) if return_attributes else ()

    def wgsl_attributes(self, attributes, direction="generic"):
        rendered = []
        for attr in attributes or []:
            semantic = str(getattr(attr, "name", attr))
            explicit_attribute = self.explicit_wgsl_attribute(attr, semantic)
            if explicit_attribute:
                rendered.append(explicit_attribute)
                continue

            key = self.semantic_key(semantic)
            builtin = self.BUILTIN_SEMANTICS.get(key)
            if builtin:
                rendered.append(f"@builtin({builtin})")
                continue

            location = self.attribute_location(attr, semantic, direction)
            if location is not None:
                rendered.append(f"@location({location})")

        return " ".join(rendered)

    def explicit_wgsl_attribute(self, attr, name):
        key = self.semantic_key(name)
        if key not in {
            "builtin",
            "invariant",
            "interpolate",
            "location",
        }:
            return None

        arguments = getattr(attr, "arguments", []) or []
        if not arguments:
            if key in {"builtin", "interpolate", "location"}:
                return None
            return f"@{key}"

        rendered_args = ", ".join(
            self.generate_attribute_argument(arg) for arg in arguments
        )
        return f"@{key}({rendered_args})"

    def generate_attribute_argument(self, argument):
        if isinstance(argument, IdentifierNode):
            return argument.name
        return self.generate_expression(argument)

    def semantic_key(self, semantic):
        return re.sub(r"[^a-z0-9_]+", "", semantic.lower())

    def semantic_location(self, semantic, direction):
        """The location of a value with ``semantic`` (:mod:`.stage_io_semantic_locations`).

        ``direction`` is ``"target"`` for a fragment output, whose ``COLORn`` names colour attachment ``n``.
        """
        key = self.semantic_key(semantic)
        # Out_Color<n>, an older spelling of SV_Target<n>, names attachment n too.
        target = re.fullmatch(r"(?:sv_target|gl_fragcolor|out_color)(\d*)", key)
        if target is None and direction == "target":
            target = re.fullmatch(r"color(\d*)", key)
        if target is not None:
            return int(target.group(1) or 0)
        if is_system_semantic_key(key):
            # System values are built-ins or refused, never user locations.
            return None
        return shared_semantic_location(semantic)

    def next_binding_attributes(self, group="0", *, start=None):
        group = str(group)
        implicit = start is None
        if implicit:
            start = self._next_implicit_binding_by_group.get(group, 0)
        binding = self.next_available_binding(group, start)
        if implicit:
            # Explicit resources and companion samplers must not consume lower free slots.
            self._next_implicit_binding_by_group[group] = binding + 1
        self.mark_allocated_binding(group, binding)
        return f"@group({group}) @binding({binding})"

    def binding_group_from_attributes(self, attributes):
        match = re.search(r"@group\(([^)]+)\)", attributes)
        return match.group(1) if match else "0"

    def sampler_binding_attributes_for_texture(self, texture_attributes, *, node=None):
        group = self.binding_group_from_attributes(texture_attributes)
        if node is not None:
            reserved_sampler = self._reserved_sampler_bindings.get(id(node))
            if reserved_sampler is not None:
                group, binding = reserved_sampler
                self.mark_allocated_binding(group, binding)
                return f"@group({group}) @binding({binding})"
        match = re.search(r"@binding\(([^)]+)\)", texture_attributes)
        if not match:
            return self.next_binding_attributes(group=group)
        try:
            start = int(str(match.group(1)), 0) + 1
        except ValueError:
            return self.next_binding_attributes(group=group)
        binding = self.next_available_binding(group, start)
        self.mark_allocated_binding(group, binding)
        return f"@group({group}) @binding({binding})"

    def next_available_binding(self, group, start):
        binding = start
        reserved = self._reserved_bindings_by_group.get(str(group), set())
        allocated = self._allocated_bindings_by_group.get(str(group), set())
        while binding in reserved or binding in allocated:
            binding += 1
        return binding

    def mark_allocated_binding(self, group, binding):
        self._allocated_bindings_by_group.setdefault(str(group), set()).add(binding)

    def reserve_binding(self, group, binding):
        self._reserved_bindings_by_group.setdefault(str(group), set()).add(binding)

    def resource_binding_diagnostic_name(self, node):
        name = getattr(node, "name", None)
        if name:
            return name
        return getattr(
            node, "member_type", getattr(node, "var_type", type(node).__name__)
        )

    def numeric_binding_components(self, node):
        group, binding = self.resolved_explicit_binding_components(node)
        try:
            binding = int(str(binding), 0)
        except (TypeError, ValueError):
            return None
        return str(group), binding

    def reserve_explicit_binding_for_node(self, node):
        group, binding, register_class = self.explicit_binding_info(node)
        try:
            numeric_binding = int(str(binding), 0)
        except (TypeError, ValueError):
            return None
        group = str(group)
        if register_class is not None:
            source_key = (group, register_class, numeric_binding)
            owner = self._hlsl_register_source_owners.get(source_key)
            if owner is not None:
                raise WGSLRefusal(
                    "WGSL target resource binding collision: "
                    f"{self.resource_binding_diagnostic_name(node)} and "
                    f"{self.resource_binding_diagnostic_name(owner)} both declare "
                    f"@group({group}) @binding({numeric_binding})"
                )
            self._hlsl_register_source_owners[source_key] = node
            owner = self._explicit_binding_owners.setdefault(group, {}).get(
                numeric_binding
            )
            if owner is None:
                target_binding = numeric_binding
            else:
                owner_class = self._explicit_binding_register_classes.get(
                    (group, numeric_binding)
                )
                if owner_class is None:
                    raise WGSLRefusal(
                        "WGSL target resource binding collision: "
                        f"{self.resource_binding_diagnostic_name(node)} and "
                        f"{self.resource_binding_diagnostic_name(owner)} "
                        f"both declare @group({group}) @binding({numeric_binding})"
                    )
                target_binding = self.next_available_binding(group, numeric_binding + 1)
            self._hlsl_register_binding_allocations[source_key] = target_binding
            self._explicit_binding_owners[group][target_binding] = node
            self._explicit_binding_register_classes[(group, target_binding)] = (
                register_class
            )
            self.reserve_binding(group, target_binding)
            return group, target_binding

        owner = self._explicit_binding_owners.setdefault(group, {}).get(numeric_binding)
        if owner is not None:
            raise WGSLRefusal(
                "WGSL target resource binding collision: "
                f"{self.resource_binding_diagnostic_name(node)} and "
                f"{self.resource_binding_diagnostic_name(owner)} both declare "
                f"@group({group}) @binding({numeric_binding})"
            )
        self._explicit_binding_owners[group][numeric_binding] = node
        self._explicit_binding_register_classes[(group, numeric_binding)] = None
        self.reserve_binding(group, numeric_binding)
        return group, numeric_binding

    def reserve_explicit_sampler_for_texture_node(self, node):
        components = self.numeric_binding_components(node)
        if components is None:
            return
        group, texture_binding = components
        sampler_binding = self.next_available_binding(group, texture_binding + 1)
        self.reserve_binding(group, sampler_binding)
        self._reserved_sampler_bindings[id(node)] = (group, sampler_binding)

    def declared_resource_type(self, node):
        return getattr(
            node,
            "param_type",
            getattr(node, "var_type", getattr(node, "member_type", None)),
        )

    def explicit_resource_binding_nodes(
        self, cbuffers, global_variables, stage_resource_parameters=()
    ):
        for node in cbuffers:
            yield node
        for node in global_variables:
            yield node
            for info in self.resource_paths_for_type(self.declared_resource_type(node)):
                yield info["member"]
        for node in stage_resource_parameters:
            yield node
            for info in self.resource_paths_for_type(self.declared_resource_type(node)):
                yield info["member"]

    def reserve_explicit_resource_bindings(
        self, cbuffers, global_variables, stage_resource_parameters=()
    ):
        resource_nodes = list(
            self.explicit_resource_binding_nodes(
                cbuffers, global_variables, stage_resource_parameters
            )
        )
        for node in resource_nodes:
            self.reserve_explicit_binding_for_node(node)
        for node in resource_nodes:
            resource_type = self.declared_resource_type(node)
            if self.sampled_texture_type(resource_type) is not None:
                self.reserve_explicit_sampler_for_texture_node(node)

    def explicit_binding_components(self, node):
        group, binding, _register_class = self.explicit_binding_info(node)
        return group, binding

    def resolved_explicit_binding_components(self, node):
        group, binding, register_class = self.explicit_binding_info(node)
        if binding is None or register_class is None:
            return group, binding
        try:
            numeric_binding = int(str(binding), 0)
        except (TypeError, ValueError):
            return group, binding
        source_key = (str(group), register_class, numeric_binding)
        target_binding = self._hlsl_register_binding_allocations.get(source_key)
        if target_binding is None:
            return group, binding
        return group, target_binding

    def explicit_binding_info(self, node):
        group = "0"
        binding = None
        register_class = None
        for attr in getattr(node, "attributes", []) or []:
            key = self.semantic_key(str(getattr(attr, "name", attr)))
            arguments = getattr(attr, "arguments", []) or []
            if key in {"group", "set", "space"} and arguments:
                group = self.generate_attribute_argument(arguments[0])
            elif key in {"binding", "buffer"} and arguments:
                binding = self.generate_attribute_argument(arguments[0])
                register_class = None
            elif key == "register" and arguments:
                binding, group, register_class = self.register_attribute_components(
                    arguments, group
                )
        return group, binding, register_class

    def explicit_binding_attributes(self, node):
        group, binding = self.resolved_explicit_binding_components(node)
        if binding is None:
            return ""
        try:
            numeric_binding = int(str(binding), 0)
            self.mark_allocated_binding(group, numeric_binding)
        except ValueError:
            pass
        return f"@group({group}) @binding({binding})"

    def register_attribute_binding(self, arguments, default_group):
        binding, group, _register_class = self.register_attribute_components(
            arguments, default_group
        )
        return binding, group

    def register_attribute_components(self, arguments, default_group):
        binding_text = self.generate_attribute_argument(arguments[0])
        group = default_group
        register_class = None
        binding_match = re.fullmatch(r"\s*([A-Za-z]+)?\s*(\d+)\s*", str(binding_text))
        if binding_match:
            register_class = (
                binding_match.group(1).lower() if binding_match.group(1) else None
            )
            binding = binding_match.group(2)
        else:
            fallback_match = re.search(r"\d+", str(binding_text))
            binding = fallback_match.group(0) if fallback_match else str(binding_text)
        if len(arguments) > 1:
            space_text = self.generate_attribute_argument(arguments[1])
            group_match = re.search(r"\d+", str(space_text))
            group = group_match.group(0) if group_match else str(space_text)
        return binding, group, register_class

    def cbuffer_instance_name(self, node):
        return self.safe_wgsl_identifier(f"_{self.type_identifier_name(node.name)}")

    def cbuffer_member_accesses(self, cbuffers):
        accesses = {}
        for cbuffer in cbuffers:
            instance_name = self.cbuffer_instance_name(cbuffer)
            for member in getattr(cbuffer, "members", []) or []:
                member_name = getattr(member, "name", "")
                if not member_name:
                    continue
                if member_name in accesses:
                    raise WGSLRefusal(
                        "WGSL target cannot flatten duplicate cbuffer member "
                        f"name: {member_name}"
                    )
                accesses[member_name] = (
                    f"{instance_name}."
                    f"{self.struct_member_identifier_name(cbuffer.name, member_name)}"
                )
        return accesses

    def cbuffer_member_types(self, cbuffers):
        member_types = {}
        for cbuffer in cbuffers:
            for member in getattr(cbuffer, "members", []) or []:
                member_name = getattr(member, "name", "")
                if member_name:
                    member_types[member_name] = getattr(member, "member_type", None)
        return member_types

    def function_texture_parameters(self, ast, target_stage):
        functions = list(self._helper_functions(ast, target_stage))
        functions.extend(
            stage_node.entry_point
            for stage_node in self._stage_nodes(ast, target_stage)
            if getattr(stage_node, "entry_point", None) is not None
        )
        texture_parameters = {}
        for function in functions:
            indices = [
                index
                for index, parameter in enumerate(
                    getattr(function, "parameters", []) or []
                )
                if self.sampled_texture_type(parameter.param_type) is not None
            ]
            if indices:
                for key in self.function_metadata_keys(function):
                    texture_parameters[key] = tuple(indices)
        return texture_parameters

    def function_resource_member_parameters(self, ast, target_stage):
        functions = list(self._helper_functions(ast, target_stage))
        functions.extend(
            stage_node.entry_point
            for stage_node in self._stage_nodes(ast, target_stage)
            if getattr(stage_node, "entry_point", None) is not None
        )
        resource_parameters = {}
        for function in functions:
            if getattr(function, "generic_params", None):
                # A generic signature (``Box<T>``) names no concrete structure; its specializations are concrete.
                continue
            parameter_paths = {}
            for index, parameter in enumerate(
                getattr(function, "parameters", []) or []
            ):
                paths = self.resource_paths_for_type(parameter.param_type)
                if paths:
                    parameter_paths[index] = tuple(paths)
            if parameter_paths:
                for key in self.function_metadata_keys(function):
                    resource_parameters[key] = parameter_paths
        return resource_parameters

    def function_buffer_pointer_parameters(self, ast, target_stage):
        functions = list(self._helper_functions(ast, target_stage))
        functions.extend(
            stage_node.entry_point
            for stage_node in self._stage_nodes(ast, target_stage)
            if getattr(stage_node, "entry_point", None) is not None
        )
        pointer_parameters = {}
        for function in functions:
            indices = [
                index
                for index, parameter in enumerate(
                    getattr(function, "parameters", []) or []
                )
                if self.is_buffer_pointer_type(
                    parameter.param_type, getattr(parameter, "qualifiers", [])
                )
            ]
            if indices:
                for key in self.function_metadata_keys(function):
                    pointer_parameters[key] = tuple(indices)
        return pointer_parameters

    def function_metadata_keys(self, function):
        function_name = getattr(function, "name", "")
        keys = [self.function_signature_key(function)]
        if len(self._function_overloads_by_name.get(function_name, ())) <= 1:
            keys.append(function_name)
        return tuple(key for key in keys if key)

    def buffer_pointer_parameter_names(self, function):
        return tuple(
            getattr(parameter, "name", "")
            for parameter in getattr(function, "parameters", []) or []
            if self.is_buffer_pointer_type(
                parameter.param_type, getattr(parameter, "qualifiers", [])
            )
        )

    def has_binding_attribute(self, node):
        _group, binding = self.explicit_binding_components(node)
        return binding is not None

    def is_stage_resource_parameter(self, parameter):
        param_type = getattr(parameter, "param_type", None)
        if self.structured_buffer_element_type(param_type) is not None:
            return True
        if self.stage_uniform_parameter_type(parameter) is not None:
            return True
        return False
